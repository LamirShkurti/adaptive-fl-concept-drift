"""
Kompresim i update-it te modelit (vendimi #4 i kontrolluesit adaptiv):
  - "full": dergo vektorin e peshave float64 pa ndryshim.
  - "compressed": delta ndaj raundit te fundit te dërguar + kuantizim 8-bit
    (linear, min-max per-vektor) -- qasje standarde dhe e thjeshte per
    krahasim real bajte-per-bajte, jo vetem teorike.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CompressedUpdate:
    quantized: np.ndarray  # uint8
    scale: float
    zero_point: float
    base_weights: np.ndarray  # referenca kunder se ciles eshte marre delta

    def num_bytes(self) -> int:
        # uint8 payload + 2 float64 per (scale, zero_point). Nuk numerojme
        # 'base_weights' sepse ai mbahet nga marresi (serveri) nga raundi
        # i fundit i pranuar -- nuk ridergohet.
        return self.quantized.nbytes + 16

    def decode(self) -> np.ndarray:
        delta = self.quantized.astype(np.float64) * self.scale + self.zero_point
        return self.base_weights + delta


def compress_update(new_weights: np.ndarray, base_weights: np.ndarray) -> CompressedUpdate:
    delta = new_weights - base_weights
    lo, hi = float(delta.min()), float(delta.max())
    if hi - lo < 1e-12:
        scale = 1.0
        zero_point = lo
        q = np.zeros_like(delta, dtype=np.uint8)
    else:
        scale = (hi - lo) / 255.0
        zero_point = lo
        q = np.clip(np.round((delta - zero_point) / scale), 0, 255).astype(np.uint8)
    return CompressedUpdate(quantized=q, scale=scale, zero_point=zero_point, base_weights=base_weights.copy())
