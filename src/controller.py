"""
Kontrolluesi adaptiv -- kontributi qendror i ri. Per çdo klient, çdo raund,
lexon 9 sinjale dhe vendos 6 veprime (shih README per shpjegim te plote te
çdo pragu, qe te gjitha vijne nga config.ControllerThresholds).

Sinjalet hyrese (InputSignals):
  1. battery_pct            -- niveli i baterise (0-100, 100 = jo-bateri/prize)
  2. cpu_busy_pct            -- shfrytezim CPU gjate nje raundi tipik trajnimi
  3. ram_free_mb             -- RAM e lire
  4. bandwidth_mbps          -- cilesia e rrjetit (bandwidth)
  5. latency_ms              -- vonesa e rrjetit
  6. new_samples             -- sasia e te dhenave te reja te mbledhura kete raund
  7. distribution_skew       -- sa larg eshte shperndarja lokale e fundit nga
                                 shperndarja historike e ketij klienti (0 = asnje ndryshim)
  8. recent_rmse_trend       -- (rmse_raund_fundit - rmse_mesatare_e_fundit) / rmse_mesatare
                                 (negativ = po permiresohet, pozitiv = po keqesohet)
  9. drift_ratio             -- raporti EWMA/baseline nga DriftDetector (>1 = drift i mundshem)

Veprimet dalese (Decision):
  1. should_train            -- a trajnon fare kete raund
  2. local_epochs             -- sa epochs lokale (0 nese should_train=False)
  3. should_send_update        -- a e dergon update-in kete raund (mund te trajnoje
                                 lokalisht pa dërguar, per te kursyer rrjet/bateri)
  4. update_mode               -- "full" | "compressed"
  5. model_active              -- nese False, klienti sherben parashikimin e fundit
                                 te mire te njohur ne vend qe te perdor modelin aktual
  6. model_variant              -- "full" | "lite" (kalim ne modelin me 2 features)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import AblationFlags, ControllerThresholds


@dataclass
class InputSignals:
    battery_pct: float
    cpu_busy_pct: float
    ram_free_mb: float
    bandwidth_mbps: float
    latency_ms: float
    new_samples: int
    distribution_skew: float
    recent_rmse_trend: float
    drift_ratio: float


@dataclass
class Decision:
    should_train: bool
    local_epochs: int
    should_send_update: bool
    update_mode: str          # "full" | "compressed"
    model_active: bool
    model_variant: str        # "full" | "lite"
    reason: str                # shpjegim i shkurter, per logging/debug/punim


class AdaptiveController:
    def __init__(self, thresholds: ControllerThresholds, ablate: Optional[AblationFlags] = None):
        self.t = thresholds
        self.ablate = ablate or AblationFlags()
        self.rounds_since_last_send = 0
        self.consecutive_stress_rounds = 0
        self.current_variant = "full"
        self.model_was_deactivated = False

    def decide(self, s: InputSignals) -> Decision:
        t = self.t
        ab = self.ablate
        reasons = []

        # --- (1) Trajno apo kapërce raundin -------------------------------
        if ab.disable_train_skip:
            should_train = True
            reasons.append("train=yes (ablation: bosht-1 i çaktivizuar)")
        else:
            should_train = True
            if s.battery_pct < t.min_battery_pct_to_train:
                should_train = False
                reasons.append(f"battery {s.battery_pct:.1f}%<{t.min_battery_pct_to_train} -> skip train")
            elif s.new_samples < t.min_new_samples_to_train:
                should_train = False
                reasons.append(f"new_samples {s.new_samples}<{t.min_new_samples_to_train} -> skip train")

        # --- (2) Epochs lokale ---------------------------------------------
        if not should_train:
            local_epochs = 0
        elif ab.disable_epoch_adapt:
            local_epochs = ab.fixed_epochs_for_ablation
            reasons.append(f"epochs={local_epochs} (ablation: bosht-2 i çaktivizuar, fiks)")
        else:
            budget = 1.0
            if s.cpu_busy_pct > t.stress_cpu_pct:
                budget *= 0.5
            if s.battery_pct < t.stress_battery_pct:
                budget *= 0.5
            if s.recent_rmse_trend > 0.05:       # po keqesohet -> trajno me fort
                budget *= 1.5
            if s.drift_ratio > t.drift_error_ratio_threshold:
                budget *= 1.5                     # drift i zbuluar -> me shume epochs per t'u rikuperuar
            local_epochs = int(round(max(t.epochs_min, min(t.epochs_max, t.epochs_max * budget))))
            reasons.append(f"epochs={local_epochs} (cpu={s.cpu_busy_pct:.0f}%, batt={s.battery_pct:.0f}%, "
                            f"trend={s.recent_rmse_trend:+.2f}, drift={s.drift_ratio:.2f})")

        # --- (3) Frekuenca e dergimit ---------------------------------------
        self.rounds_since_last_send += 1
        if ab.disable_send_hold:
            should_send_update = should_train
            if should_send_update:
                self.rounds_since_last_send = 0
            reasons.append("send=" + ("yes" if should_send_update else "no")
                            + " (ablation: bosht-3 i çaktivizuar, dergo gjithmone kur trajnon)")
        else:
            resource_constrained_send = (s.battery_pct < t.compress_if_battery_below_pct
                                          or s.bandwidth_mbps < t.compress_if_bandwidth_below_mbps)
            must_send = (self.rounds_since_last_send >= t.max_rounds_without_send) or (s.drift_ratio > t.drift_error_ratio_threshold)
            should_send_update = should_train and (must_send or not resource_constrained_send)
            if should_send_update:
                self.rounds_since_last_send = 0
                reasons.append("send=yes" + (" (forced: max-delay/drift)" if must_send else ""))
            else:
                reasons.append("send=no (holding update, resource constrained)")

        # --- (4) Full vs i kompresuar ----------------------------------------
        update_mode = "full"
        if not ab.disable_compression:
            if should_send_update and (s.bandwidth_mbps < t.compress_if_bandwidth_below_mbps
                                        or s.battery_pct < t.compress_if_battery_below_pct):
                update_mode = "compressed"
                reasons.append("update_mode=compressed")
        elif should_send_update:
            reasons.append("update_mode=full (ablation: bosht-4 i çaktivizuar)")

        # --- (5) Aktivizo/çaktivizo modelin lokal -----------------------------
        if ab.disable_deactivation:
            model_active = True
        else:
            model_active = s.battery_pct >= t.deactivate_if_battery_below_pct
            if not model_active:
                reasons.append(f"model DEACTIVATED (battery {s.battery_pct:.1f}%<{t.deactivate_if_battery_below_pct})")

        # --- (6) Kalim ne modelin 'lite' ----------------------------------------
        if ab.disable_lite_switch:
            model_variant = "full"
        else:
            is_stress_round = (s.cpu_busy_pct > t.stress_cpu_pct) or (s.battery_pct < t.stress_battery_pct)
            if is_stress_round:
                self.consecutive_stress_rounds += 1
            else:
                self.consecutive_stress_rounds = 0

            if self.consecutive_stress_rounds >= t.switch_to_lite_after_consecutive_stress_rounds:
                self.current_variant = "lite"
            elif self.current_variant == "lite" and not is_stress_round and s.battery_pct > t.stress_battery_pct + 20:
                # rikuperim i qarte -> kthehu te modeli i plote
                self.current_variant = "full"
            model_variant = self.current_variant
            if model_variant == "lite":
                reasons.append(f"variant=lite (stress_rounds={self.consecutive_stress_rounds})")

        return Decision(
            should_train=should_train,
            local_epochs=local_epochs,
            should_send_update=should_send_update,
            update_mode=update_mode,
            model_active=model_active,
            model_variant=model_variant,
            reason="; ".join(reasons),
        )


class NonAdaptiveController:
    """Baseline jo-adaptiv: trajno gjithmone, epochs fikse, dergo gjithmone
    update te plote, modeli gjithmone aktiv, gjithmone varianti 'full' --
    analoge me UseCase1_NOAdaptive ne projektin e FL-se."""

    def __init__(self, fixed_epochs: int = 3):
        self.fixed_epochs = fixed_epochs

    def decide(self, s: InputSignals) -> Decision:
        return Decision(
            should_train=True,
            local_epochs=self.fixed_epochs,
            should_send_update=True,
            update_mode="full",
            model_active=True,
            model_variant="full",
            reason="non-adaptive baseline",
        )
