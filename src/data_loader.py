
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import DatasetPreset, ExperimentConfig


# Kufij fizike plauzibël per secilin sensor -- jashte ketyre trajtohen si
# lexime te deshtuara (njesoj si filtrat fizike te punimit te parashikimit
# te temperatures: humiteti s'mund te jete >100%, etj.).
PHYSICAL_BOUNDS = {
    "temp": (-20.0, 60.0),
    "humidity": (0.0, 100.0),
    "co": (0.0, 1.0),       # dataset-i i publikuar jep co/lpg/smoke si raporte te normalizuara ppm (jo ppm bruto)
    "lpg": (0.0, 1.0),
    "smoke": (0.0, 1.0),
}


def load_raw(data_path: str, preset: DatasetPreset) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Nuk gjeta '{data_path}'. Shiko data/README.md per te shkarkuar "
            f"dataset-in nga Kaggle dhe ta vendosesh ne vendin/emrin e duhur."
        )
    df = pd.read_csv(data_path)
    missing = [c for c in [preset.time_col, preset.device_col, preset.target_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Skedari '{data_path}' s'ka kolonat e pritura {missing}. "
            f"Kolonat e gjetura: {list(df.columns)}"
        )
    return df


def clean_and_split_by_device(df: pd.DataFrame, preset: DatasetPreset) -> Dict[str, pd.DataFrame]:
    """Kthen {device_id: DataFrame indeksuar me kohe, i renditur kronologjikisht, i pastruar}."""
    df = df.copy()

    if preset.time_is_unix_seconds:
        df[preset.time_col] = pd.to_datetime(df[preset.time_col], unit="s", errors="coerce")
    else:
        df[preset.time_col] = pd.to_datetime(df[preset.time_col], format=preset.time_format, errors="coerce")
    df = df.dropna(subset=[preset.time_col])

    # Kufij fizike per çdo kolone qe i kemi te njohura.
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col in df.columns:
            bad = (df[col] < lo) | (df[col] > hi)
            if bad.any():
                df.loc[bad, col] = np.nan

    keep_cols = [preset.time_col, preset.device_col] + sorted(set(preset.feature_cols) | {preset.target_col})
    keep_cols = [c for c in dict.fromkeys(keep_cols) if c in df.columns]
    df = df[keep_cols]

    per_device: Dict[str, pd.DataFrame] = {}
    for dev_id, g in df.groupby(preset.device_col):
        g = g.drop(columns=[preset.device_col]).set_index(preset.time_col).sort_index()
        g = g[~g.index.duplicated(keep="first")]
        # Interpolim i kufizuar (vrima te shkurtra), jo per vrima te gjata.
        g = g.interpolate(method="time", limit=6).dropna()
        per_device[str(dev_id)] = g

    return per_device


def resample_devices(per_device: Dict[str, pd.DataFrame], freq: str) -> Dict[str, pd.DataFrame]:
    out = {}
    for dev_id, g in per_device.items():
        r = g.resample(freq).mean().dropna()
        out[dev_id] = r
    return out


def load_clients(cfg: ExperimentConfig, data_path: str) -> Dict[str, pd.DataFrame]:
    preset = cfg.get_preset()
    raw = load_raw(data_path, preset)
    per_device = clean_and_split_by_device(raw, preset)
    per_device = resample_devices(per_device, cfg.resample_freq)
    return per_device


def device_report(per_device: Dict[str, pd.DataFrame]) -> List[dict]:
    """Permbledhje per raportim (perdoret ne main/README dhe ne punim)."""
    rows = []
    for dev_id, g in per_device.items():
        rows.append({
            "device": dev_id,
            "n_rows": len(g),
            "start": g.index.min(),
            "end": g.index.max(),
            "temp_mean": g["temp"].mean() if "temp" in g else None,
            "temp_std": g["temp"].std() if "temp" in g else None,
        })
    return rows
