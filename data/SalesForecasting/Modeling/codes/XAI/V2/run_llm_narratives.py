"""
run_llm_narratives.py — Standalone LLM narrative generation for SalesForecasting XAI V2.

Generates 5 narrative types (Model Overview, Global Explanation, Model Drift,
Data Lineage, Model Comparison) at Global/Decile/Segment levels using Google Gemini
with a RAG architecture for token efficiency.

Usage:
  export GOOGLE_API_KEY="your-key"
  python3 -u run_llm_narratives.py --narratives all
  python3 -u run_llm_narratives.py --narratives overview comparison --sleep 3
  python3 -u run_llm_narratives.py --narratives overview --evaluate
"""

import os
import sys
import time
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.llm_utils import (
    NARRATIVE_PERSONAS,
    build_knowledge_base,
    save_knowledge_base,
    load_knowledge_base,
    is_cache_valid,
    generate_narrative,
    evaluate_narratives,
)

ALL_NARRATIVE_TYPES = ["overview", "explanation", "drift", "lineage", "comparison", "consensus"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _section(title: str):
    print(f"\n{'─'*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'─'*60}", flush=True)


def _save(output_dir: str, subfolder: str, filename: str, content: str):
    dirpath = os.path.join(output_dir, "llm_narratives", subfolder)
    os.makedirs(dirpath, exist_ok=True)
    filepath = os.path.join(dirpath, filename)
    tmp = filepath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, filepath)
    print(f"  Saved: {os.path.join(subfolder, filename)}", flush=True)
    return filepath


def _run_personas(kb, narrative_type: str, context_chunks: list, params: dict,
                  client, sleep_s: float, ragas_records: list) -> tuple:
    """
    Call generate_narrative for each persona in this narrative type.
    Per-persona failures are caught gracefully: a [GENERATION_FAILED] placeholder
    section is written so the file still saves with the successful personas intact.
    Returns (markdown_str, n_failures).
    """
    personas = NARRATIVE_PERSONAS[narrative_type]
    sections = []
    n_failures = 0
    for persona in personas:
        t0 = time.time()
        print(f"    [{persona}]... ", end="", flush=True)
        try:
            narrative = generate_narrative(
                narrative_type, context_chunks, params, client, persona, sleep_s
            )
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s)", flush=True)
            sections.append(f"## {persona}\n\n{narrative}\n\n---\n")
            ragas_records.append({
                "narrative_type": narrative_type,
                "model":          params.get("model", ""),
                "dataset":        params.get("eval_dataset", params.get("dataset", "")),
                "level":          params.get("level", "Global"),
                "persona":        persona,
                "query":          f"{narrative_type} for {params}",
                "answer":         narrative,
                "contexts":       [str(c) for c in context_chunks],
            })
        except RuntimeError as e:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.1f}s)", flush=True)
            print(f"      [ERROR] {e}", flush=True)
            sections.append(
                f"## {persona}\n\n"
                f"> **[GENERATION_FAILED]** Generation failed after all retries. "
                f"Rerun this narrative type to regenerate this section.\n\n---\n"
            )
            n_failures += 1
    return "\n".join(sections), n_failures


# ── Narrative runners ──────────────────────────────────────────────────────────

def run_overview(kb, models: list, champion_model: str,
                 output_dir: str, client, sleep_s: float,
                 ragas_records: list) -> int:
    _section("Model Overview")
    n_failures = 0
    for model in models:
        try:
            is_champ = (model == champion_model)
            mfst_chunks = kb.filter(type="manifest", model=model)
            perf_chunks = kb.filter(type="performance", model=model,
                                     dataset="DEV", group_var="Global", group_value="Global")
            context = mfst_chunks + perf_chunks
            params = {
                "model":    model,
                "champion": "Y" if is_champ else "N",
                "level":    "Global",
            }
            header = [
                f"# Model Overview: {model}",
                f"> Champion Model: {'Yes' if is_champ else 'No'}  |  "
                f"Benchmark: DEV\n",
            ]
            print(f"\n  [{model}]  champion={is_champ}", flush=True)
            body, pf = _run_personas(kb, "overview", context, params, client, sleep_s, ragas_records)
            n_failures += pf
            content = "\n".join(header) + "\n" + body
            _save(output_dir, "model_overview", f"{model}_overview.md", content)
        except Exception as e:
            print(f"  [ERROR] overview for {model} failed: {e}", flush=True)
            n_failures += 1
    return n_failures


def run_explanation(kb, models: list, champion_model: str,
                    output_dir: str, client, sleep_s: float,
                    ragas_records: list) -> int:
    _section("Global Model Explanation")
    n_failures = 0
    for model in models:
        is_champ = (model == champion_model)
        datasets  = ["DEV", "OOS", "OOT_01"]

        for level in ["Global", "Decile", "Segment"]:
            try:
                print(f"\n  [{model}] level={level}", flush=True)
                if level == "Global":
                    context = []
                    for ds in datasets:
                        for fw in ["SHAP", "LIME", "Surrogate"]:
                            context += kb.filter(type="importance", model=model, dataset=ds,
                                                 group_var="Global", group_value="Global",
                                                 framework=fw)
                    context += kb.filter(type="performance", model=model,
                                          group_var="Global", group_value="Global")
                else:
                    context = []
                    for fw in ["SHAP", "LIME", "Surrogate"]:
                        context += kb.filter(type="importance_pivot", model=model,
                                              dataset="DEV", framework=fw, pivot_var=level)
                        for ds in ["OOS", "OOT_01"]:
                            context += kb.filter(type="importance_pivot", model=model,
                                                  dataset=ds, framework=fw, pivot_var=level)

                params = {
                    "model":    model,
                    "level":    level,
                    "champion": "Y" if is_champ else "N",
                }
                header = [
                    f"# Global Explanation: {model} — {level} Level",
                    f"> Champion Model: {'Yes' if is_champ else 'No'}  |  "
                    f"Benchmark: DEV\n",
                ]
                body, pf = _run_personas(kb, "explanation", context, params, client, sleep_s, ragas_records)
                n_failures += pf
                content = "\n".join(header) + "\n" + body
                _save(output_dir, "global_explanation", f"{model}_{level}_explanation.md", content)
            except Exception as e:
                print(f"  [ERROR] explanation {model}/{level} failed: {e}", flush=True)
                n_failures += 1
    return n_failures


def run_drift(kb, models: list, champion_model: str, eval_datasets: list,
              output_dir: str, client, sleep_s: float, ragas_records: list) -> int:
    _section("Model Drift")
    n_failures = 0
    for model in models:
        is_champ = (model == champion_model)
        for eval_ds in eval_datasets:
            try:
                drift_chunks = kb.filter(type="drift", model=model, eval_dataset=eval_ds)
                if not drift_chunks:
                    print(f"  [WARN] No drift data for {model} vs {eval_ds} — skipping", flush=True)
                    continue
                desc_dev  = kb.filter(type="descriptive", dataset="DEV")
                desc_eval = kb.filter(type="descriptive", dataset=eval_ds)
                perf_dev  = kb.filter(type="performance", model=model, dataset="DEV",
                                       group_var="Global", group_value="Global")
                perf_eval = kb.filter(type="performance", model=model, dataset=eval_ds,
                                       group_var="Global", group_value="Global")
                context = drift_chunks + desc_dev + desc_eval + perf_dev + perf_eval

                params = {
                    "model":        model,
                    "benchmark":    "DEV",
                    "eval_dataset": eval_ds,
                    "champion":     "Y" if is_champ else "N",
                    "level":        "Global",
                }
                print(f"\n  [{model}] DEV → {eval_ds}", flush=True)
                header = [
                    f"# Model Drift Report: {model}",
                    f"> Benchmark: DEV  |  Evaluation: {eval_ds}  |  "
                    f"Champion Model: {'Yes' if is_champ else 'No'}\n",
                ]
                body, pf = _run_personas(kb, "drift", context, params, client, sleep_s, ragas_records)
                n_failures += pf
                content = "\n".join(header) + "\n" + body
                _save(output_dir, "model_drift", f"{model}_DEV_vs_{eval_ds}_drift.md", content)
            except Exception as e:
                print(f"  [ERROR] drift {model}/DEV→{eval_ds} failed: {e}", flush=True)
                n_failures += 1
    return n_failures


def run_lineage(kb, datasets: list, output_dir: str, client, sleep_s: float,
                ragas_records: list) -> int:
    _section("Data Lineage")
    n_failures = 0
    for dataset in datasets:
        try:
            desc_chunks = kb.filter(type="descriptive", dataset=dataset)
            if dataset != "DEV":
                desc_chunks += kb.filter(type="descriptive", dataset="DEV")
            drift_chunks = [c for c in kb.chunks
                            if c.get("type") == "drift" and c.get("eval_dataset") == dataset]
            context = desc_chunks + drift_chunks

            params = {
                "dataset": dataset,
                "level":   "Global",
            }
            print(f"\n  [{dataset}]", flush=True)
            header = [
                f"# Data Lineage Report: {dataset} Dataset",
                f"> Reference Benchmark: DEV\n",
            ]
            body, pf = _run_personas(kb, "lineage", context, params, client, sleep_s, ragas_records)
            n_failures += pf
            content = "\n".join(header) + "\n" + body
            _save(output_dir, "data_lineage", f"{dataset}_lineage.md", content)
        except Exception as e:
            print(f"  [ERROR] lineage for {dataset} failed: {e}", flush=True)
            n_failures += 1
    return n_failures


def run_comparison(kb, models: list, champion_model: str, datasets: list,
                   output_dir: str, client, sleep_s: float, ragas_records: list) -> int:
    _section("Model Comparison")
    n_failures = 0
    for dataset in datasets:
        try:
            context = []
            for model in models:
                context += kb.filter(type="performance", model=model, dataset=dataset,
                                      group_var="Global", group_value="Global")
                context += kb.filter(type="performance", model=model, dataset="DEV",
                                      group_var="Global", group_value="Global")
                context += kb.filter(type="manifest", model=model)

            params = {
                "dataset":        dataset,
                "champion_model": champion_model,
                "level":          "Global",
            }
            print(f"\n  [{dataset}]  champion={champion_model}", flush=True)
            header = [
                f"# Model Comparison Report: {dataset} Dataset",
                f"> Champion Model: {champion_model}  |  Baseline: DEV\n",
            ]
            body, pf = _run_personas(kb, "comparison", context, params, client, sleep_s, ragas_records)
            n_failures += pf
            content = "\n".join(header) + "\n" + body
            _save(output_dir, "model_comparison", f"{dataset}_comparison.md", content)
        except Exception as e:
            print(f"  [ERROR] comparison for {dataset} failed: {e}", flush=True)
            n_failures += 1
    return n_failures


def run_consensus(kb, models: list, champion_model: str,
                  output_dir: str, client, sleep_s: float,
                  ragas_records: list) -> int:
    _section("Explanation Consensus & Triangulation")
    n_failures = 0
    for model in models:
        is_champ = (model == champion_model)

        for level in ["Global", "Decile", "Segment"]:
            try:
                print(f"\n  [{model}] level={level}", flush=True)

                if level == "Global":
                    context = []
                    for ds in ["DEV", "OOS", "OOT_01"]:
                        context += kb.filter(type="consensus", model=model, dataset=ds,
                                             group_var="Global", group_value="Global")
                    for fw in ["SHAP", "LIME", "Surrogate"]:
                        context += kb.filter(type="importance", model=model, dataset="DEV",
                                              group_var="Global", group_value="Global",
                                              framework=fw)
                else:
                    context = []
                    for ds in ["DEV", "OOS"]:
                        context += [c for c in kb.chunks
                                    if c.get("type") == "consensus"
                                    and c.get("model") == model
                                    and c.get("dataset") == ds
                                    and c.get("group_var") == level]
                    for fw in ["SHAP", "LIME", "Surrogate"]:
                        context += kb.filter(type="importance_pivot", model=model,
                                              dataset="DEV", framework=fw, pivot_var=level)

                params = {
                    "model":    model,
                    "level":    level,
                    "champion": "Y" if is_champ else "N",
                }
                header = [
                    f"# Explanation Consensus & Triangulation: {model} — {level} Level",
                    f"> Champion Model: {'Yes' if is_champ else 'No'}  |  Benchmark: DEV",
                    f"> Frameworks: SHAP, LIME, Surrogate Regression\n",
                ]
                body, pf = _run_personas(kb, "consensus", context, params,
                                         client, sleep_s, ragas_records)
                n_failures += pf
                content = "\n".join(header) + "\n" + body
                _save(output_dir, "explanation_consensus",
                      f"{model}_{level}_consensus.md", content)
            except Exception as e:
                print(f"  [ERROR] consensus {model}/{level} failed: {e}", flush=True)
                n_failures += 1
    return n_failures


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM narrative generation for SalesForecasting XAI V2"
    )
    parser.add_argument("--output",     default="Outputs",     help="Outputs directory")
    parser.add_argument("--manifest",   default="manifest.csv", help="Manifest CSV path")
    parser.add_argument("--narratives", nargs="+", default=["all"],
                        help="Narrative types: all | overview explanation drift lineage comparison")
    parser.add_argument("--top-k",      type=int, default=10,  help="Top N features per chunk")
    parser.add_argument("--sleep",      type=float, default=5.0, help="Sleep between LLM calls (s)")
    parser.add_argument("--evaluate",     action="store_true", help="Run RAGAS evaluation after generation")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuild of KB cache even if up-to-date")
    args = parser.parse_args()

    # ── API key ────────────────────────────────────────────────────────────────
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set.", flush=True)
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=api_key)

    # ── Manifest ───────────────────────────────────────────────────────────────
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.manifest)
    if not os.path.exists(manifest_path):
        print(f"ERROR: manifest not found at {manifest_path}", flush=True)
        sys.exit(1)
    manifest_df = pd.read_csv(manifest_path)
    active_df   = manifest_df[manifest_df["active"] == "Y"]

    # Derive active models (unique model_names) and champion
    active_models = list(active_df["model_name"].unique())
    champ_rows    = manifest_df[manifest_df.get("champion", pd.Series(dtype=str)).str.upper() == "Y"]
    # Safer champion detection
    champ_col     = manifest_df["champion"].str.strip().str.upper() if "champion" in manifest_df.columns else pd.Series(["N"] * len(manifest_df))
    champion_rows = manifest_df[champ_col == "Y"]
    champion_model = champion_rows["model_name"].iloc[0] if len(champion_rows) else active_models[0]

    # Active eval datasets (non-benchmark datasets)
    eval_datasets = [
        ds for ds in active_df["dataset"].unique() if ds != "DEV"
    ]
    all_datasets  = list(active_df["dataset"].unique())

    # ── Steps ─────────────────────────────────────────────────────────────────
    narratives_req = (
        ALL_NARRATIVE_TYPES if "all" in args.narratives
        else [n for n in args.narratives if n in ALL_NARRATIVE_TYPES]
    )
    unknown = [n for n in args.narratives if n not in ALL_NARRATIVE_TYPES + ["all"]]
    if unknown:
        print(f"[WARN] Unknown narrative types ignored: {unknown}", flush=True)

    # ── Header ─────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}", flush=True)
    print(f"  XAI LLM Narrative Generation — SalesForecasting V2", flush=True)
    print(f"  Manifest   : {manifest_path}", flush=True)
    print(f"  Output     : {args.output}", flush=True)
    print(f"  Narratives : {narratives_req}", flush=True)
    print(f"  Models     : {active_models}", flush=True)
    print(f"  Champion   : {champion_model}", flush=True)
    print(f"  Sleep/call : {args.sleep}s", flush=True)
    print(f"  RAGAS eval : {args.evaluate}", flush=True)
    print(f"{'═'*60}", flush=True)

    # ── Build / load knowledge base ────────────────────────────────────────────
    _section("Building Knowledge Base")
    cache_path = os.path.join(args.output, "llm_narratives", "kb_cache.pkl")
    force_rebuild = args.rebuild_cache
    if not force_rebuild and is_cache_valid(cache_path, args.output, manifest_path):
        print("  Cache is up-to-date — loading from disk (skipping re-embedding).", flush=True)
        kb = load_knowledge_base(cache_path)   # returns None if version mismatch
    else:
        kb = None

    if kb is None:
        if force_rebuild:
            print("  --rebuild-cache flag set — forcing full rebuild.", flush=True)
        elif os.path.exists(cache_path):
            print("  Source files changed or version mismatch — rebuilding cache.", flush=True)
        else:
            print("  No cache found — building and embedding (first run only).", flush=True)
        kb = build_knowledge_base(args.output, manifest_df, top_n=args.top_k, client=client)
        save_knowledge_base(kb, cache_path)

    # ── Run narratives ─────────────────────────────────────────────────────────
    t_total       = time.time()
    ragas_records = []
    failures      = 0

    if "overview" in narratives_req:
        failures += run_overview(kb, active_models, champion_model,
                                 args.output, client, args.sleep, ragas_records)

    if "explanation" in narratives_req:
        failures += run_explanation(kb, active_models, champion_model,
                                    args.output, client, args.sleep, ragas_records)

    if "drift" in narratives_req:
        failures += run_drift(kb, active_models, champion_model, eval_datasets,
                              args.output, client, args.sleep, ragas_records)

    if "lineage" in narratives_req:
        failures += run_lineage(kb, all_datasets, args.output, client, args.sleep, ragas_records)

    if "comparison" in narratives_req:
        failures += run_comparison(kb, active_models, champion_model, all_datasets,
                                   args.output, client, args.sleep, ragas_records)

    if "consensus" in narratives_req:
        failures += run_consensus(kb, active_models, champion_model,
                                  args.output, client, args.sleep, ragas_records)

    # ── RAGAS evaluation ───────────────────────────────────────────────────────
    if args.evaluate and ragas_records:
        _section("RAGAS Evaluation")
        print(f"  Evaluating {len(ragas_records)} narratives...", flush=True)
        scores_df = evaluate_narratives(ragas_records, client, api_key)
        if not scores_df.empty:
            ragas_path = os.path.join(args.output, "llm_narratives", "ragas_scores.csv")
            scores_df.to_csv(ragas_path, index=False)
            print(f"  Saved: ragas_scores.csv  ({len(scores_df)} rows)", flush=True)
            if "faithfulness" in scores_df.columns:
                print(f"  Avg faithfulness      : {scores_df['faithfulness'].mean():.3f}", flush=True)
            if "answer_relevancy" in scores_df.columns:
                print(f"  Avg answer_relevancy  : {scores_df['answer_relevancy'].mean():.3f}", flush=True)

    # ── File count + summary ───────────────────────────────────────────────────
    elapsed = time.time() - t_total
    narrative_dir = os.path.join(args.output, "llm_narratives")
    n_files = sum(1 for _, _, files in os.walk(narrative_dir) for f in files if f.endswith(".md"))

    print(f"\n{'═'*60}", flush=True)
    if failures:
        print(f"  LLM Narratives complete with {failures} failure(s)  [{elapsed:.1f}s]", flush=True)
        print(f"  Partial files contain [GENERATION_FAILED] markers — rerun specific types to fix.", flush=True)
    else:
        print(f"  LLM Narratives complete  [{elapsed:.1f}s]", flush=True)
    print(f"  Files created : {n_files} markdown reports", flush=True)
    print(f"  Output dir    : {narrative_dir}", flush=True)
    print(f"{'═'*60}\n", flush=True)
    # Always exit 0 — individual failures are isolated and reported above.
    # The pipeline detects [GENERATION_FAILED] markers via content scan.


if __name__ == "__main__":
    main()
