"""
Importable definitions of TradeFraud's two autoencoder-family networks.

These used to be defined inline inside train_model4_autoencoder.py and
train_model9_vae.py. Those scripts run training at import time, so nothing
else (Phase 3's explainers) could load the saved weights without
re-running training. The classes now live here and both training scripts
import them — the architectures are unchanged, so existing saved
state_dicts (model4_ae_state.pt, model9_vae_state.pt) load as-is.

Both networks operate in the PREPROCESSED space produced by
src.autoencoder_preprocessing.AutoencoderPreprocessor (signed-log1p ->
standardise -> clip to [-5, 5]).
"""
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    encode(x) -> z: all layers except the last get BatchNorm+ReLU+Dropout;
    the LAST encoder layer (the bottleneck, z) is a plain nn.Linear with no
    activation and no regularisation, so z is a clean, deterministic,
    differentiable function of x — required for LRD, which back-projects a
    Jacobian through exactly this layer.

    decode(z) -> x_hat: mirrors the encoder in reverse.
    """
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        enc = []
        prev = input_dim
        for h in hidden_dims[:-1]:
            enc += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        enc.append(nn.Linear(prev, hidden_dims[-1]))  # bottleneck: plain Linear only
        self.encoder = nn.Sequential(*enc)

        dec = []
        prev = hidden_dims[-1]
        for h in reversed(hidden_dims[:-1]):
            dec += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


class VAE(nn.Module):
    """
    encode(x) -> (mu, logvar): the distribution over the latent code.
    reparameterize(mu, logvar) -> z: the sampled code used during training.
    decode(z) -> x_hat.

    For SCORING and for LRD, z = mu is used deterministically (no sampling)
    — see score() — so a given input always gets the same reconstruction
    error and the same Jacobian.
    """
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        self.encoder_body = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        h = self.encoder_body(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def score(self, x):
        """Deterministic reconstruction error using z = mu (no sampling)."""
        mu, _ = self.encode(x)
        x_hat = self.decode(mu)
        return torch.mean((x - x_hat) ** 2, dim=1)
