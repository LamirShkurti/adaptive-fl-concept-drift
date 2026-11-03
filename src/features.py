"""
Feature engineering per parashikim temperature -- e njejta metodologji e
perdorur ne punimin e meparshem (lag-e, rolling mean/std, delta, kodim
ciklik kohe), tani e riperdorur per secilin klient FL veç e veç.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def build_features(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    n_lags: int,
    rolling_windows: List[int],
    horizon_steps: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Kthen (X, y) ku y = target_col nderruar horizon_steps perpara (shape e barabarte)."""
    out = pd.DataFrame(index=df.index)

    for col in feature_cols:
        s = df[col]
        out[f"{col}_t0"] = s
        for lag in range(1, n_lags + 1):
            out[f"{col}_lag{lag}"] = s.shift(lag)
        for w in rolling_windows:
            out[f"{col}_rollmean{w}"] = s.rolling(w).mean()
            out[f"{col}_rollstd{w}"] = s.rolling(w).std()
        out[f"{col}_delta"] = s.diff()

    # Kodim ciklik kohe (ore e dites, dite e vitit).
    hour = df.index.hour + df.index.minute / 60.0
    doy = df.index.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    y = df[target_col].shift(-horizon_steps)
    y.name = "target"

    valid = out.notna().all(axis=1) & y.notna()
    return out[valid], y[valid]


def build_minimal_features(df: pd.DataFrame, target_col: str, lite_cols: List[str], horizon_steps: int):
    """Variant 'lite' (vetem vlerat aktuale, 2 features -- temp+humidity, si
    modeli i eksportueshem per mikrokontrollues ne punimin e meparshem)."""
    out = df[lite_cols].copy()
    y = df[target_col].shift(-horizon_steps)
    y.name = "target"
    valid = out.notna().all(axis=1) & y.notna()
    return out[valid], y[valid]
