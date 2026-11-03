"""
Orkestrimi i simulimit Federated Learning: lidh data_loader + features +
models + device_state + drift + controller + compression ne nje lak
raund-pas-raundi, per politiken "adaptive" ose "non_adaptive".

Protokolli per raund r, per çdo klient c (online/walk-forward, pa leakage):
  1. VLERËSO: parashiko chunk_r me peshat qe klienti ka TASHME (para se te
     shohe chunk_r) -- kjo eshte gabimi real "si do te performonte sot".
  2. Mostro gjendjen e burimeve (bateri/CPU/RAM/rrjet) nga profili i klases.
  3. Kontrolluesi (adaptiv ose jo-adaptiv) vendos 6 veprimet duke pare
     sinjalet (perfshi gabimin e sapo-llogaritur te hapit 1 dhe EWMA i drift-it).
  4. Nese should_train: terheq peshat globale (nese ekzistojne), trajno E
     epochs lokale mbi chunk_r (variant full ose lite, sipas vendimit #6).
  5. Nese should_send_update: paketo peshat (full ose te kompresuara) dhe
     "dergoji" serverit -> FedAvg i rifreskon peshat globale (te atij varianti).
  6. Kosto energjie aplikohet gjithmone (trajnim + dergim ndikojne baterine).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import features as feat_mod
from .compression import compress_update
from .config import AblationFlags, DEVICE_CLASS_ASSIGNMENT_ORDER, DEVICE_CLASSES, ControllerThresholds, ExperimentConfig
from .controller import AdaptiveController, Decision, InputSignals, NonAdaptiveController
from .device_state import DeviceState
from .drift import DriftDetector, DriftScenario, apply_drift_to_frame
from .models import LinearFedModel, rmse as rmse_fn


LITE_COLS = ["temp", "humidity"]


@dataclass
class RoundLog:
    round: int
    client: str
    policy: str
    rmse: float
    battery_pct: float
    cpu_busy_pct: float
    bandwidth_mbps: float
    should_train: bool
    local_epochs: int
    should_send_update: bool
    update_mode: str
    model_active: bool
    model_variant: str
    bytes_sent: int
    new_samples: int
    drift_ratio: float
    reason: str


@dataclass
class ClientRuntime:
    client_id: str
    full_X: np.ndarray
    full_y: np.ndarray
    lite_X: np.ndarray
    lite_y: np.ndarray
    round_slices: List[slice]
    device_state: DeviceState
    full_model: LinearFedModel
    lite_model: LinearFedModel
    rmse_history: List[float] = field(default_factory=list)
    drift_detector: DriftDetector = field(default_factory=lambda: DriftDetector())
    last_active_prediction: Optional[float] = None
    target_running_mean: float = 0.0
    target_running_std: float = 1.0


def _chronological_round_slices(n_rows: int, n_rounds: int) -> List[slice]:
    edges = np.linspace(0, n_rows, n_rounds + 1).astype(int)
    return [slice(edges[i], edges[i + 1]) for i in range(n_rounds) if edges[i + 1] > edges[i]]


def prepare_clients(
    per_device: Dict[str, pd.DataFrame],
    cfg: ExperimentConfig,
    n_live_rounds: int,
    round_minutes: float,
    drift_scenarios: Optional[Dict[str, DriftScenario]] = None,
    seed: int = 42,
    n_warmup_epochs: int = 150,
    warmup_lr: float = 0.02,
    min_new_samples_to_train: Optional[int] = None,
) -> Dict[str, ClientRuntime]:
    preset = cfg.get_preset()
    drift_scenarios = drift_scenarios or {}
    clients: Dict[str, ClientRuntime] = {}

    for i, (dev_id, df) in enumerate(sorted(per_device.items())):
        n = len(df)
        n_train = int(n * cfg.train_frac)
        n_val = int(n * cfg.val_frac)
        history_df = df.iloc[: n_train + n_val]
        live_df = df.iloc[n_train + n_val:]

        scenario = drift_scenarios.get(dev_id, DriftScenario(kind="none", start_round=10 ** 9))
        if scenario.kind != "none" and len(live_df) > 0:
            # FIKS: round_index ketu duhet te jete numri i RAUNDIT per secilin
            # rresht raw te live_df (jo vete rreshti) -- perpara kesaj thirrej
            # apply_drift_to_frame(live_df, scenario, live_df.index) dhe
            # enumerate() brenda saj trajtonte çdo rresht raw si "raund",
            # cka bente scenario.start_round=40 te aplikohej efektivisht ne
            # rreth raundin ~5-6 (meqe ka ~n_live_rows/n_live_rounds rreshta
            # raw per raund, zakonisht 6-8 me resample_freq=5min). Kjo
            # perafrim (bazuar ne len(live_df) para feature engineering)
            # mund te dalloje per 1 raund nga kufijte finale te round_slices
            # (llogaritur me poshte ne n_live_rows post-feature-engineering,
            # qe mund te heqe pak rreshta nga fillimi per shkak te NaN nga
            # lags/rolling) -- i pranueshem, shume me i vogel se gabimi
            # origjinal (~7x).
            raw_round_slices = _chronological_round_slices(len(live_df), n_live_rounds)
            row_round_idx = np.zeros(len(live_df), dtype=int)
            for r, sl in enumerate(raw_round_slices):
                row_round_idx[sl] = r
            live_df = apply_drift_to_frame(live_df, scenario, row_round_idx)

        full_df = pd.concat([history_df, live_df])

        X_full, y_full = feat_mod.build_features(
            full_df, preset.target_col, preset.feature_cols, cfg.n_lags, cfg.rolling_windows, cfg.horizon_steps,
        )
        X_lite, y_lite = feat_mod.build_minimal_features(
            full_df, preset.target_col, LITE_COLS, cfg.horizon_steps,
        )

        common_idx = X_full.index.intersection(X_lite.index)
        X_full, y_full = X_full.loc[common_idx], y_full.loc[common_idx]
        X_lite, y_lite = X_lite.loc[common_idx], y_lite.loc[common_idx]

        live_mask = common_idx.isin(live_df.index)
        n_history_rows = int((~live_mask).sum())
        n_live_rows = int(live_mask.sum())
        if n_live_rows < n_live_rounds:
            raise ValueError(
                f"Klienti {dev_id} ka vetem {n_live_rows} rreshta 'live' per {n_live_rounds} raunde -- "
                f"zvogelo n_live_rounds ose resample_freq."
            )

        # REGRESION (zbuluar tetor 2026, duke testuar vete --rounds 200 te
        # rekomanduar per boshtin e çaktivizimit): nese mesatarja e rreshtave
        # 'live' per raund bie nen ControllerThresholds.min_new_samples_to_train
        # (parazgjedhje 6), axis-i (1) ("new_samples < min -> skip train") e
        # kapercen TRAJNIMIN NE ÇDO RAUND, GJITHE XHIRIMIN -- pa asnje gabim a
        # paralajmerim, thjesht CSV me rounds_trained=0 kudo qe duket si
        # rezultat "normal" nese dikush s'e kontrollon kete kolone shprehimisht.
        # Kjo ndodh lehte: p.sh. --rounds 200 me --resample-freq=5min
        # (parazgjedhja) jep vetem ~2.9 rreshta/raund (<6) per dataset-in
        # Kaggle -- ndersa --rounds 80 jep ~7.2 (>6), prandaj bug-u s'u pa
        # deri sa dikush provoi 200 direkt. Zgjidhje per perdoruesin: zvogelo
        # --rounds, ose zvogelo --resample-freq (p.sh. '2min' ose '1min', qe
        # jep me shume rreshta 'live' gjithsej) -- shih README.md §9.1.
        avg_new_samples_per_round = n_live_rows / n_live_rounds
        if min_new_samples_to_train is not None and avg_new_samples_per_round < min_new_samples_to_train:
            raise ValueError(
                f"Klienti {dev_id}: mesatarisht {avg_new_samples_per_round:.2f} rreshta 'live' per raund, "
                f"nen min_new_samples_to_train={min_new_samples_to_train} -- ÇDO raund do kapercente "
                f"trajnimin (axis 1), xhirimi do te prodhonte rounds_trained=0 ne te gjithe CSV-ne, pa "
                f"asnje gabim/paralajmerim tjeter. Zgjidhje: zvogelo --rounds (max i sigurt tani: "
                f"{int(n_live_rows // min_new_samples_to_train)}), ose zvogelo --resample-freq (p.sh. "
                f"'2min'/'1min') qe te kesh me shume rreshta 'live' gjithsej -- shih README.md §9.1."
            )

        round_slices_live = _chronological_round_slices(n_live_rows, n_live_rounds)
        round_slices = [slice(s.start + n_history_rows, s.stop + n_history_rows) for s in round_slices_live]

        class_name = DEVICE_CLASS_ASSIGNMENT_ORDER[i % len(DEVICE_CLASS_ASSIGNMENT_ORDER)]
        profile = DEVICE_CLASSES[class_name]
        dstate = DeviceState(
            client_id=dev_id, profile=profile, round_minutes=round_minutes,
            recharge_every_n_rounds=0,
            rng=np.random.default_rng(seed + i),
        )

        full_model = LinearFedModel(n_features=X_full.shape[1])
        full_model.set_scaler(X_full.values[:n_history_rows] if n_history_rows > 0 else X_full.values)
        lite_model = LinearFedModel(n_features=X_lite.shape[1])
        lite_model.set_scaler(X_lite.values[:n_history_rows] if n_history_rows > 0 else X_lite.values)

        # Warm-start: çdo klient trajnohet lokalisht mbi dritaren e tij te
        # historise (train_frac+val_frac) PARA se te fillojne raundet "live".
        # Kjo pasqyron nje vendosje realiste FL (modelet NUK vendosen nga e
        # para/zero-init ne prodhim) dhe eshte e domosdoshme praktikisht: me
        # vetem disa dhjetra mostra te reja per raund, gradient descent nga
        # zero s'ka kohe te mjaftueshme te konvergjoje brenda numrit tipik
        # te raundeve -- u verifikua empirikisht (shih tests/).
        if n_history_rows >= 30:
            hist_Xf = X_full.values[:n_history_rows]
            hist_yf = y_full.values[:n_history_rows]
            hist_Xl = X_lite.values[:n_history_rows]
            hist_yl = y_lite.values[:n_history_rows]
            full_model.local_train(hist_Xf, hist_yf, epochs=n_warmup_epochs, lr=warmup_lr,
                                    l2=1e-2, batch_size=64, rng=np.random.default_rng(seed + i))
            lite_model.local_train(hist_Xl, hist_yl, epochs=n_warmup_epochs, lr=warmup_lr,
                                    l2=1e-2, batch_size=64, rng=np.random.default_rng(seed + i))

        hist_y = y_full.values[:n_history_rows] if n_history_rows > 0 else y_full.values
        clients[dev_id] = ClientRuntime(
            client_id=dev_id,
            full_X=X_full.values, full_y=y_full.values,
            lite_X=X_lite.values, lite_y=y_lite.values,
            round_slices=round_slices,
            device_state=dstate,
            full_model=full_model, lite_model=lite_model,
            target_running_mean=float(np.mean(hist_y)) if len(hist_y) else 0.0,
            target_running_std=float(np.std(hist_y)) + 1e-6 if len(hist_y) else 1.0,
        )

    return clients


def run_simulation(
    clients: Dict[str, ClientRuntime],
    policy: str,                       # "adaptive" | "non_adaptive"
    thresholds: ControllerThresholds,
    n_rounds: int,
    local_lr: float = 0.01,
    non_adaptive_epochs: int = 3,
    ablate: Optional[AblationFlags] = None,
) -> List[RoundLog]:
    logs: List[RoundLog] = []

    controllers = {}
    for cid in clients:
        controllers[cid] = (AdaptiveController(thresholds, ablate=ablate) if policy == "adaptive"
                             else NonAdaptiveController(fixed_epochs=non_adaptive_epochs))

    global_full = LinearFedModel(n_features=next(iter(clients.values())).full_model.n_features)
    global_lite = LinearFedModel(n_features=next(iter(clients.values())).lite_model.n_features)
    global_full.set_scaler(next(iter(clients.values())).full_X)
    global_lite.set_scaler(next(iter(clients.values())).lite_X)

    # Global fillestare = FedAvg i peshave TE WARM-STARTUARA te klienteve (jo
    # zero) -- shih shenimin ne prepare_clients() per arsyen.
    n_each = [len(c.full_X) for c in clients.values()]
    total_n0 = sum(n_each) or 1
    global_full.set_weights(sum(c.full_model.get_weights() * len(c.full_X) for c in clients.values()) / total_n0)
    global_lite.set_weights(sum(c.lite_model.get_weights() * len(c.lite_X) for c in clients.values()) / total_n0)

    for r in range(n_rounds):
        pending_full: List[tuple] = []  # (weights, n_samples)
        pending_lite: List[tuple] = []

        for cid, c in clients.items():
            if r >= len(c.round_slices):
                continue
            sl = c.round_slices[r]
            Xf, yf = c.full_X[sl], c.full_y[sl]
            Xl, yl = c.lite_X[sl], c.lite_y[sl]
            n_new = len(yf)
            if n_new == 0:
                continue

            # --- (1) Evaluate with current weights, BEFORE training ---------
            pred_full = c.full_model.predict(Xf)
            pred_lite = c.lite_model.predict(Xl)
            err_full = rmse_fn(yf, pred_full)
            c.rmse_history.append(err_full)
            trend_window = c.rmse_history[-6:-1]
            recent_trend = 0.0 if not trend_window else (err_full - np.mean(trend_window)) / (np.mean(trend_window) + 1e-6)

            chunk_mean = float(np.mean(yf))
            skew = abs(chunk_mean - c.target_running_mean) / c.target_running_std
            c.target_running_mean = 0.95 * c.target_running_mean + 0.05 * chunk_mean

            drift_ratio = c.drift_detector.update(err_full)

            res = c.device_state.sample_resources(n_new)
            signals = InputSignals(
                battery_pct=res.battery_pct, cpu_busy_pct=res.cpu_busy_pct,
                ram_free_mb=res.ram_free_mb, bandwidth_mbps=res.bandwidth_mbps,
                latency_ms=res.latency_ms, new_samples=n_new,
                distribution_skew=skew, recent_rmse_trend=recent_trend,
                drift_ratio=drift_ratio,
            )
            decision: Decision = controllers[cid].decide(signals)

            active_model = c.full_model if decision.model_variant == "full" else c.lite_model
            active_pred = pred_full if decision.model_variant == "full" else pred_lite
            if decision.model_active:
                c.last_active_prediction = float(np.mean(active_pred))
                reported_rmse = err_full if decision.model_variant == "full" else rmse_fn(yl, pred_lite)
            else:
                fallback = c.last_active_prediction if c.last_active_prediction is not None else float(np.mean(yf))
                reported_rmse = rmse_fn(yf, np.full_like(yf, fallback))

            bytes_sent = 0
            if decision.should_train:
                if decision.model_variant == "full":
                    c.full_model.set_weights(global_full.get_weights())
                    c.full_model.local_train(Xf, yf, epochs=decision.local_epochs, lr=local_lr)
                else:
                    c.lite_model.set_weights(global_lite.get_weights())
                    c.lite_model.local_train(Xl, yl, epochs=decision.local_epochs, lr=local_lr)

                if decision.should_send_update:
                    if decision.model_variant == "full":
                        if decision.update_mode == "full":
                            bytes_sent = c.full_model.num_bytes_full()
                            pending_full.append((c.full_model.get_weights(), n_new))
                        else:
                            cu = compress_update(c.full_model.get_weights(), global_full.get_weights())
                            bytes_sent = cu.num_bytes()
                            pending_full.append((cu.decode(), n_new))
                    else:
                        if decision.update_mode == "full":
                            bytes_sent = c.lite_model.num_bytes_full()
                            pending_lite.append((c.lite_model.get_weights(), n_new))
                        else:
                            cu = compress_update(c.lite_model.get_weights(), global_lite.get_weights())
                            bytes_sent = cu.num_bytes()
                            pending_lite.append((cu.decode(), n_new))

            c.device_state.apply_energy_cost(
                trained=decision.should_train, epochs=decision.local_epochs,
                sent_update=decision.should_send_update, compressed=(decision.update_mode == "compressed"),
            )

            logs.append(RoundLog(
                round=r, client=cid, policy=policy, rmse=reported_rmse,
                battery_pct=res.battery_pct, cpu_busy_pct=res.cpu_busy_pct,
                bandwidth_mbps=res.bandwidth_mbps,
                should_train=decision.should_train, local_epochs=decision.local_epochs,
                should_send_update=decision.should_send_update, update_mode=decision.update_mode,
                model_active=decision.model_active, model_variant=decision.model_variant,
                bytes_sent=bytes_sent, new_samples=n_new, drift_ratio=drift_ratio,
                reason=decision.reason,
            ))

        if pending_full:
            total_n = sum(n for _, n in pending_full)
            avg_w = sum(w * n for w, n in pending_full) / total_n
            global_full.set_weights(avg_w)
            # Klientet qe derguan kete raund e terheqin global_full ne raundin
            # tjeter qe trajnojne (shih "should_train" me siper); klientet qe
            # s'kane derguar mbeten me peshat e tyre lokale te fundit.
        if pending_lite:
            total_n = sum(n for _, n in pending_lite)
            avg_w = sum(w * n for w, n in pending_lite) / total_n
            global_lite.set_weights(avg_w)

    return logs
