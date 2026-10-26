"""
Background ("baseline") references for gradient-based attribution.

Integrated Gradients and SHAP-family methods explain an input by comparing
it against a reference input — "how different is this client from a
neutral one, and which features account for that difference?". The choice of
reference changes the answer, which is exactly the research question the
proposal poses ("How sensitive are these attributions to the choice of
background/baseline reference data?"). Four references are provided:

  zeros           the origin of the preprocessed space. Because the
                  preprocessing standardises each feature on NORMAL rows, the
                  origin is (approximately) the average normal client in the
                  transformed space.
  raw_mean        the arithmetic mean of the normal rows in RAW units,
                  pushed through the same preprocessing. This is NOT the same
                  point as `zeros`: raw features are heavy-tailed, so their
                  arithmetic mean sits far above the typical client, and
                  after the log transform it lands away from the origin. It
                  is included precisely because a user who says "use the
                  mean of the training data" usually means this one.
  kmeans          k cluster centres of normal rows (default k=10). Used as a
                  SET: attributions are computed against each centre and
                  averaged (an "expected gradients" style reference
                  distribution).
  nearest_neighbor  for each instance, the single closest normal row —
                  an instance-specific "what would a normal version of this
                  client look like" reference.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def zeros_baseline(n_features: int) -> np.ndarray:
    return np.zeros((1, n_features), dtype=np.float32)


def raw_mean_baseline(X_raw_normal, preprocessor) -> np.ndarray:
    """Arithmetic mean of normal rows in raw units, mapped into preprocessed space."""
    import pandas as pd
    mean_raw = pd.DataFrame(np.asarray(X_raw_normal).mean(axis=0, keepdims=True),
                            columns=preprocessor.feat_cols)
    return preprocessor.transform(mean_raw).astype(np.float32)


def kmeans_baselines(U_normal: np.ndarray, k: int = 10, seed: int = 42,
                     n_subsample: int = 20000) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if len(U_normal) > n_subsample:
        U_normal = U_normal[rng.choice(len(U_normal), n_subsample, replace=False)]
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(U_normal)
    return km.cluster_centers_.astype(np.float32)


def nearest_neighbor_baselines(U_query: np.ndarray, U_reference: np.ndarray,
                               n_subsample: int = 20000, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if len(U_reference) > n_subsample:
        U_reference = U_reference[rng.choice(len(U_reference), n_subsample, replace=False)]
    nn = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(U_reference)
    _, idx = nn.kneighbors(U_query)
    return U_reference[idx[:, 0]].astype(np.float32)
