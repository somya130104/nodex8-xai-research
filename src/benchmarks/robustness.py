"""
5.3  Stability and adversarial robustness.

STABILITY. Add Gaussian noise of 1% of each feature's standard deviation (in the preprocessed space) to an
instance, explain again, and compare with the original explanation using the weighted Kendall tau of
Phase 4 (so a flip at the top of the ranking costs more than one in the tail).
Stochastic explainers (the LIME variants, KernelSHAP, GradientSHAP) are re-run with a new random seed, so
their noise-stability includes their own sampling noise. The CONTROL re-explains the UNCHANGED input with the
new seed: tau(original, same input, new seed). It is the ceiling for what noise-stability could be. Deterministic
explainers have control = 1 by construction.

ADVERSARIAL SEARCH. For an instance, hill-climb over small perturbations delta (|delta_j| <= eps_in) that keep the
model's output essentially unchanged (|f(x+delta) - f(x)| <= eps_f) to find the one whose explanation differs
most (lowest weighted tau against the original explanation). Every explainer gets the same budget and the same
constraints. A stochastic explainer is searched with ONE fixed seed (otherwise the search would just chase random
draws) and the best perturbation is then CONFIRMED under a different seed. The reported robustness is the lowest
confirmed tau (higher = harder to fool), against a control: tau under random perturbations of the same size.
"""
import numpy as np

from src.consensus import weighted_kendall_tau


def _tau_rows(A, B):
    return np.array([weighted_kendall_tau(a, b) for a, b in zip(A, B)], dtype=float)


def noise_stability(suite, name, U, sigma, rel=0.01, seed=0):
    """Returns dict(tau_noise (n,), tau_control (n,)). sigma: per-feature std (p,) of the normal population."""
    U = np.asarray(U, dtype=np.float32)
    rng = np.random.RandomState(seed + 7)
    Un = (U + rel * sigma[None, :] * rng.normal(size=U.shape)).astype(np.float32)
    A0 = suite.explain(name, U, seed=1000)
    An = suite.explain(name, Un, seed=2000)
    tn = _tau_rows(A0, An)
    if suite.is_stochastic(name):
        tc = _tau_rows(A0, suite.explain(name, U, seed=2000))
    else:
        tc = np.ones(len(U))
    return {"tau_noise": tn, "tau_control": tc}


def adversarial_search(suite, name, U_inst, eps_in=0.05, eps_f=None, iters=30, cands=4, seed=0, n_random_ctrl=6):
    """
    Hill-climbing search per instance. Returns per-instance arrays:
      tau_search  lowest tau found with the search seed
      tau_confirmed  tau of that perturbation re-evaluated with a different seed
      tau_random  mean tau under random perturbations of up to the same size, shrunk until the output constraint holds (the control)
      top1_changed  whether the best perturbation changed the top-ranked feature (confirmed)
      feasible_share  share of proposals that satisfied the output constraint
    """
    f_np = suite.f_np
    U_inst = np.asarray(U_inst, dtype=np.float32)
    rng = np.random.RandomState(seed)
    n, p = U_inst.shape
    S0, S1 = 5000, 9000
    out = {k: [] for k in ("tau_search", "tau_confirmed", "tau_random", "top1_changed", "feasible_share")}
    for i in range(n):
        x = U_inst[i:i + 1]
        fx = float(f_np(x)[0])
        a0 = suite.explain(name, x, seed=S0, same_seed=True)[0]
        best_tau, best_delta = 1.0, np.zeros(p, dtype=np.float32)
        n_prop = n_feas = 0
        for _ in range(iters):
            prop = np.clip(best_delta[None, :] + 0.5 * eps_in * rng.normal(size=(cands, p)), -eps_in, eps_in).astype(np.float32)
            Xp = x + prop
            ok = np.abs(f_np(Xp) - fx) <= eps_f
            n_prop += cands
            n_feas += int(ok.sum())
            if not ok.any():
                continue
            Ap = suite.explain(name, Xp[ok], seed=S0, same_seed=True)
            taus = _tau_rows(np.repeat(a0[None, :], len(Ap), 0), Ap)
            j = np.nanargmin(taus) if np.isfinite(taus).any() else None
            if j is not None and taus[j] < best_tau:
                best_tau, best_delta = float(taus[j]), prop[ok][j]
        # confirm under a different random seed (same seed for original and perturbed so only the input differs)
        b0 = suite.explain(name, x, seed=S1, same_seed=True)[0]
        b1 = suite.explain(name, x + best_delta[None, :], seed=S1, same_seed=True)[0]
        conf = weighted_kendall_tau(b0, b1)
        # control: random perturbations of up to the same size, each SHRUNK (x1, 1/2, 1/4, 1/8, 1/16) until it satisfies
        # the same output constraint. This is the fair comparison: how much does an UNDIRECTED perturbation that is
        # equally invisible to the model change the explanation?
        rp = []
        for _ in range(n_random_ctrl):
            r = (eps_in * rng.choice([-1.0, 1.0], size=p)).astype(np.float32)
            for s in (1.0, 0.5, 0.25, 0.125, 0.0625):
                if abs(float(f_np(x + s * r[None, :])[0]) - fx) <= eps_f:
                    rp.append(s * r)
                    break
        if rp:
            Ar = suite.explain(name, x + np.array(rp, dtype=np.float32), seed=S0, same_seed=True)
            ctrl = float(np.nanmean(_tau_rows(np.repeat(a0[None, :], len(Ar), 0), Ar)))
        else:
            ctrl = np.nan
        out["tau_search"].append(best_tau)
        out["tau_confirmed"].append(conf)
        out["tau_random"].append(ctrl)
        out["top1_changed"].append(float(np.argmax(np.abs(b0)) != np.argmax(np.abs(b1))))
        out["feasible_share"].append(n_feas / max(n_prop, 1))
    return {k: np.array(v, dtype=float) for k, v in out.items()}
