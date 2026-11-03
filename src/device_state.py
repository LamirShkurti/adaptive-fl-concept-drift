"""
Gjendja e simuluar e nje klienti FL ndermjet raundeve: bateri, CPU/RAM/rrjet
te mostruara nga profili i klases (shih config.DEVICE_CLASSES), dhe sinjali
i "te dhenave te reja" per raund. Bateria evoluon REALISHT ne kete simulim
(nuk eshte vetem nje numer i rastesishem çdo raund) -- zbret sipas rrymes se
tërhequr (idle kundrejt training) dhe kohezgjatjes se raundit, dhe mund te
rimbushet periodikisht (modelon power-bank te ringarkuar / node me panel diellor)
nese 'recharge_every_n_rounds' > 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import DeviceClassProfile


@dataclass
class RoundResourceSample:
    cpu_busy_pct: float
    cpu_temp_c: float
    ram_free_mb: float
    bandwidth_mbps: float
    latency_ms: float
    battery_pct: float
    battery_constrained: bool
    new_samples: int


@dataclass
class DeviceState:
    client_id: str
    profile: DeviceClassProfile
    round_minutes: float
    recharge_every_n_rounds: int = 0  # 0 = asnjehere
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    battery_pct: float = field(init=False, default=100.0)
    round_idx: int = field(init=False, default=0)

    def __post_init__(self):
        self.battery_pct = 100.0

    def sample_resources(self, n_new_samples: int) -> RoundResourceSample:
        p = self.profile
        cpu = float(np.clip(self.rng.normal(p.cpu_busy_mean, p.cpu_busy_std), 1.0, 100.0))
        temp = float(max(20.0, self.rng.normal(p.cpu_temp_mean, p.cpu_temp_std)))
        ram = float(max(50.0, self.rng.normal(p.ram_free_mb_mean, p.ram_free_mb_std)))
        bw = float(max(0.01, self.rng.normal(p.bandwidth_mbps_mean, p.bandwidth_mbps_std)))
        lat = float(max(0.0, self.rng.normal(p.latency_ms_mean, p.latency_ms_std)))
        return RoundResourceSample(
            cpu_busy_pct=cpu, cpu_temp_c=temp, ram_free_mb=ram,
            bandwidth_mbps=bw, latency_ms=lat,
            battery_pct=self.battery_pct, battery_constrained=p.battery_constrained,
            new_samples=n_new_samples,
        )

    def apply_energy_cost(self, trained: bool, epochs: int, sent_update: bool, compressed: bool):
        """Zbret baterine sipas vendimeve te marra kete raund. Pajisjet jo-bateri
        (battery_constrained=False) mbeten gjithmone ne 100%."""
        p = self.profile
        if not p.battery_constrained:
            self.battery_pct = 100.0
            self.round_idx += 1
            return

        hours = self.round_minutes / 60.0
        if trained:
            # Rryma rritet afersisht proporcionalisht me epochs (me teper pune = me shume energji).
            ma = p.current_draw_idle_ma + (p.current_draw_training_ma - p.current_draw_idle_ma) * min(epochs, 6) / 6.0
        else:
            ma = p.current_draw_idle_ma
        # Dergimi i update-it (radio ne) shton nje kosto fikse te vogel shtese,
        # me pak nese eshte i kompresuar (me pak bajte = me pak kohe transmetimi).
        if sent_update:
            ma += (15.0 if compressed else 35.0)

        mah_used = ma * hours
        pct_used = 100.0 * mah_used / p.battery_capacity_mah
        self.battery_pct = max(0.0, self.battery_pct - pct_used)

        self.round_idx += 1
        if self.recharge_every_n_rounds and self.round_idx % self.recharge_every_n_rounds == 0:
            self.battery_pct = 100.0
