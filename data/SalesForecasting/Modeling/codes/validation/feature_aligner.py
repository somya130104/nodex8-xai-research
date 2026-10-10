from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureAlignmentError(ValueError):
    pass


def align(df: pd.DataFrame, features: list) -> np.ndarray:
    missing = [f for f in features if str(f) not in df.columns]
    if missing:
        raise FeatureAlignmentError(
            f"{len(missing)} features missing from CSV: {[str(f) for f in missing[:5]]}…"
        )
    cols = [str(f) for f in features]
    return df[cols].fillna(0).values.astype(float)