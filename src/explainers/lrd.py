"""
The proprietary method: Latent Reconstruction Decomposition (LRD).

Problem it targets (proposal, Section 1, failure mode 1): an autoencoder
outputs no probability, only a reconstruction. KernelSHAP needs thousands of
model evaluations per instance. LRD instead decomposes the reconstruction
error ANALYTICALLY, through the network's bottleneck, with no sampling and no
reference input.

NOTE ON PROVENANCE. The proposal describes LRD only in one sentence ("back-
projects reconstruction error gradients through bottleneck latent activations
directly to input features"). Whether NodeX8 already has a written
specification is an open question (ARCHITECTURE.md Section 8, question 2).
This module implements the concrete formulation proposed in ROADMAP.md
Phase 3.4, with ONE correction described below; if NodeX8 supplies an
existing specification, that takes precedence.

Notation. x: preprocessed input (p features). z = enc(x): bottleneck code
(K units; for a VAE, z = mu(x), deterministic). x_hat = dec(z).
L = sum_j r_j^2 with r = x - x_hat: the total reconstruction error.

Step A — which latent units are responsible for the error?
    g = dL/dz, holding x fixed (a backward pass through the DECODER only —
    it stops at the bottleneck), and  c_k = z_k * g_k.
Step B — which inputs drive each latent unit?
    J[k, j] = d z_k / d x_j   (the encoder Jacobian, K x p).
Step C — distribute each unit's responsibility over the inputs in proportion
    to that unit's dependence on each input:
        a[k, j] = |J[k, j] * x_j| / sum_j' |J[k, j'] * x_j'|        (sums to 1 over j)
        w[k]    = |c_k| / sum_k' |c_k'|                              (sums to 1 over k)
        LRD_latent[j] = L * sum_k w[k] * a[k, j]
    By construction  sum_j LRD_latent[j] = L  (completeness) and every
    attribution is non-negative (magnitude of responsibility; sign is not
    meaningful for the later rank-based consensus step).
Step D — blend with the direct per-feature error r_j^2 (which also sums to L):
        LRD[j] = lam * r_j^2 + (1 - lam) * LRD_latent[j]
    Completeness therefore holds for ANY lam in [0, 1].

CORRECTION to ROADMAP.md Step D. The roadmap said to choose lam "so that
sum_j LRD_j ~= L". With the roadmap's un-normalised latent term (whose total
is S = sum_k c_k, generally != L) the constraint lam*L + (1-lam)*S = L has
only the solution lam = 1 (or S = L): i.e. it degenerates into the trivial
per-feature-error baseline. Normalising the latent term to total L, as above,
makes completeness automatic and leaves lam as a genuinely free blend. lam is
therefore FIXED IN ADVANCE (primary: 0.5) and openly ablated over {0, 0.5, 1}
— lam = 1 IS the trivial per-feature-error baseline, so the ablation also
shows whether routing through the latent code adds anything beyond it.

Why this differs from Adapted SHAP: no reference/background input and no path
integral — one decoder backward pass plus one encoder Jacobian.

VAE option (`sigma_weighting`): divide each latent unit's responsibility by
its posterior standard deviation sigma_k, so units the model is itself
uncertain about count for less.
"""
import numpy as np
import torch


class LRDExplainer:
    def __init__(self, model, kind: str = "ae", lam: float = 0.5,
                 sigma_weighting: bool = False, eps: float = 1e-12):
        assert kind in ("ae", "vae")
        assert 0.0 <= lam <= 1.0
        self.model, self.kind = model.eval(), kind
        self.lam, self.sigma_weighting, self.eps = lam, sigma_weighting, eps
        for p in self.model.parameters():
            p.requires_grad_(False)

    # deterministic latent code (z for AE, mu for VAE) as a function of a SINGLE row
    def _latent_single(self, x1):
        if self.kind == "ae":
            return self.model.encode(x1.unsqueeze(0))[0]
        return self.model.encode(x1.unsqueeze(0))[0][0]

    def components(self, X):
        """
        All quantities LRD is built from, for a batch X (B, p):
        returns dict(loss (B,), direct (B,p), latent_path (B,p), c (B,K), J (B,K,p), sigma (B,K)|None)
        """
        X = torch.as_tensor(X, dtype=torch.float32)

        with torch.no_grad():
            if self.kind == "ae":
                z = self.model.encode(X)
                sigma = None
            else:
                z, logvar = self.model.encode(X)
                sigma = torch.exp(0.5 * logvar)

        # Step A: decoder-only backward pass, stopping at the bottleneck
        z_req = z.clone().requires_grad_(True)
        x_hat = self.model.decode(z_req)
        loss = ((X - x_hat) ** 2).sum(dim=1)
        g = torch.autograd.grad(loss.sum(), z_req)[0]
        c = (z * g).detach()
        direct = ((X - x_hat) ** 2).detach()

        # Step B: encoder Jacobian dz/dx
        J = torch.func.vmap(torch.func.jacrev(self._latent_single))(X).detach()

        # Step C: distribute latent responsibility over inputs
        A = (J * X.unsqueeze(1)).abs()
        a = A / (A.sum(dim=2, keepdim=True) + self.eps)
        resp = c.abs()
        if self.sigma_weighting and sigma is not None:
            resp = resp / (sigma + self.eps)
        w = resp / (resp.sum(dim=1, keepdim=True) + self.eps)
        latent_path = loss.detach().unsqueeze(1) * (w.unsqueeze(2) * a).sum(dim=1)

        # Edge cases that would silently break completeness: a latent unit whose
        # Jacobian-times-input is zero for every feature (its share cannot be
        # distributed), or no unit carrying any responsibility at all. Re-normalise
        # each row to total L; if nothing is distributable, fall back to the direct
        # per-feature error for that row.
        total = latent_path.sum(dim=1, keepdim=True)
        ok = total > self.eps
        latent_path = torch.where(ok, latent_path * (loss.detach().unsqueeze(1) / total.clamp_min(self.eps)), direct)

        return dict(loss=loss.detach(), direct=direct, latent_path=latent_path, c=c, J=J, sigma=sigma)

    def explain(self, X, lam=None):
        """LRD attributions (B, p) as a numpy array."""
        lam = self.lam if lam is None else lam
        comp = self.components(X)
        out = lam * comp["direct"] + (1.0 - lam) * comp["latent_path"]
        return out.numpy()

    def explain_variants(self, X, lams=(0.0, 0.5, 1.0)):
        """One components() call, several lam blends: {lam: (B,p)}."""
        comp = self.components(X)
        return {lam: (lam * comp["direct"] + (1.0 - lam) * comp["latent_path"]).numpy() for lam in lams}, comp["loss"].numpy()
