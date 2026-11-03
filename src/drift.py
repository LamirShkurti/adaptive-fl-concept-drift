"""
Injektim dhe zbulim i concept drift.

Dataset-i Kaggle mbulon vetem ~1 jave, pra s'ka drift stinor afatgjate si ne
Tetovo (Tetor kunder Janar). Per te vleresuar kontrolluesin nen drift,
INJEKTOJME shprehimisht dy lloj drift-i te njohur ne literature (taksonomia
Gama et al. 2014) mbi te dhenat REALE gjate fazes "live" te simulimit (jo gjate
trajnimit fillestar) -- deklarohet qarte ne punim si skenar vleresimi, jo si
nje fenomen i zbuluar organikisht ne dataset.

  - "gradual_sensor_bias": bias aditiv qe rritet ngadale (si ngrohja diellore
    e panelit te node4/UB ne punimin e meparshem) -- shtohet NE TE GJITHA
    kolonat qe permbajne 'temp' (target + features perkatese), qe sensori te
    mbetet i vet-konsistent (ashtu si nje defekt fizik real i kalibrimit).
  - "abrupt_relationship_shift": ne nje moment te vetem, shkallezohet marrdhenia
    humidity->temp (p.sh. sensori i lagështise defektohet/ndryshon montimi) --
    drift "i papritur" klasik, qe prish korrelacionin e mesuar nga modeli.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DriftScenario:
    kind: str  # "none" | "gradual_sensor_bias" | "abrupt_relationship_shift"
    start_round: int
    magnitude: float = 1.0  # shkalla e drift-it (°C per rast gradual; faktor shumezimi per rast abrupt)
    ramp_rounds: int = 30   # per gradual: sa raunde deri sa te arrihet 'magnitude' e plote


def apply_drift_to_frame(df: pd.DataFrame, scenario: DriftScenario, round_index: np.ndarray) -> pd.DataFrame:
    """Aplikon drift mbi nje DataFrame (indeksuar me kohe) qe korrespondon me
    raundet "live". round_index[k] eshte NUMRI I RAUNDIT (0-indexed) te cilit i
    perket rreshti i k-te (rresht i panderprere/raw), JO vetë rreshti.

    (Fiks: me pare kalohej `df.index` dhe `enumerate()` mbi te trajtonte çdo
    RRESHT RAW si nje "raund" te vetem -- meqe ka ~6-8 rreshta raw per raund
    (resample_freq=5min pjesetuar ne n_live_rounds), kjo bente qe
    scenario.start_round=40 te aplikohej efektivisht ne RRESHTIN raw #40, pra
    ne rreth raundin ~5-6, shume perpara se cfare synohej. round_index duhet
    te jete nje array/Series e gjatesise len(df) me numrin e saktë te raundit
    per secilin rresht, llogaritur NE TE NJEJTEN menyre si round_slices_live
    me vone ne prepare_clients -- shih thirrjen ne fl_simulation.py.)
    """
    if scenario.kind == "none":
        return df

    out = df.copy()
    temp_cols = [c for c in out.columns if "temp" in c]
    round_index = np.asarray(round_index)
    assert len(round_index) == len(out), (
        f"round_index ({len(round_index)}) duhet te kete te njejten gjatesi si df ({len(out)})"
    )

    for pos in range(len(out)):
        i = int(round_index[pos])
        if i < scenario.start_round:
            continue
        ts = out.index[pos]

        if scenario.kind == "gradual_sensor_bias":
            progress = min(1.0, (i - scenario.start_round) / max(1, scenario.ramp_rounds))
            bias = scenario.magnitude * progress
            for c in temp_cols:
                out.loc[ts, c] = out.loc[ts, c] + bias

        elif scenario.kind == "abrupt_relationship_shift":
            if "humidity" in out.columns:
                out.loc[ts, "humidity"] = out.loc[ts, "humidity"] * scenario.magnitude

    return out


class DriftDetector:
    """EWMA i gabimit lokal kunder nje mesatare afatgjate -- nje raport i larte
    sinjalizon concept drift (modeli po gabon papritur me shume se zakonisht)."""

    def __init__(self, alpha: float = 0.3, long_window: int = 20):
        self.alpha = alpha
        self.long_window = long_window
        self.ewma: Optional[float] = None
        self.history: list = []

    def update(self, error: float) -> float:
        self.ewma = error if self.ewma is None else self.alpha * error + (1 - self.alpha) * self.ewma
        self.history.append(error)
        if len(self.history) > self.long_window:
            self.history.pop(0)
        baseline = np.mean(self.history) if self.history else error
        if baseline < 1e-6:
            return 1.0
        return float(self.ewma / baseline)
