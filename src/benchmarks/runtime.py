"""
5.4  Runtime and memory.

latency   : time each of `n` single-instance explanations (one row per call, as in a live pipeline) and report the
            median and 95th percentile in milliseconds. A short warm-up is discarded.
memory    : measured separately from latency. The reported figure is `rss_probe`: in a FRESH process (so the
            allocator holds nothing left over from earlier explainers), after the model, data and explainer are
            built, run the explainer on n single instances and report the PEAK growth of the process's resident
            memory (sampled every 2 ms; includes torch's native allocations). `peak_memory` (tracemalloc + RSS in the
            current process) is kept as a cross-check only: tracemalloc cannot see torch tensors, and in a long-lived
            process freed memory is reused, so it under-reports.
Targets from the proposal: LRD < 50 ms per instance; the whole system < 100 ms.
"""
import threading
import time
import tracemalloc

import numpy as np
import psutil


def latency(call, n, warmup=3):
    for i in range(warmup):
        call(i)
    ts = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        call(i)
        ts[i] = (time.perf_counter() - t0) * 1000.0
    return {"n": int(n), "p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)),
            "mean_ms": float(ts.mean()), "max_ms": float(ts.max())}


def peak_memory(call, n):
    proc = psutil.Process()
    stop = threading.Event()
    peak = [proc.memory_info().rss]

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.005)

    call(0)                                    # warm-up outside the measurement (first call allocates caches)
    rss0 = proc.memory_info().rss
    peak[0] = rss0
    th = threading.Thread(target=sample, daemon=True)
    tracemalloc.start()
    th.start()
    for i in range(n):
        call(i)
    _, py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stop.set()
    th.join()
    return {"peak_python_mb": py_peak / 1e6, "peak_rss_delta_mb": max(0.0, (peak[0] - rss0) / 1e6)}


def rss_probe(call, n):
    """Peak resident-memory growth (MB) over n calls, in the CURRENT process, with no warm-up (meant for a fresh child process)."""
    import gc
    gc.collect()
    proc = psutil.Process()
    base = proc.memory_info().rss
    stop = threading.Event()
    peak = [base]

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.002)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    for i in range(n):
        call(i)
    peak[0] = max(peak[0], proc.memory_info().rss)
    stop.set()
    th.join()
    return {"baseline_rss_mb": base / 1e6, "peak_rss_growth_mb": max(0.0, (peak[0] - base) / 1e6), "n": int(n)}
