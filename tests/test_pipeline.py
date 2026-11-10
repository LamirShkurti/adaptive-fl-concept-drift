
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import AblationFlags, ControllerThresholds, ExperimentConfig
from src.data_loader import device_report, load_clients
from src.drift import DriftScenario
from src.fl_simulation import prepare_clients, run_simulation

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "tiny_iot_telemetry.csv")


def _load_tiny_clients(n_rounds=10):
    cfg = ExperimentConfig()
    per_device = load_clients(cfg, FIXTURE)
    assert len(per_device) == 3, f"Prisja 3 pajisje, gjeta {len(per_device)}"
    clients = prepare_clients(per_device, cfg, n_live_rounds=n_rounds, round_minutes=5.0,
                               seed=1, n_warmup_epochs=40)
    return cfg, clients


def test_data_loader_finds_three_devices():
    cfg = ExperimentConfig()
    per_device = load_clients(cfg, FIXTURE)
    report = device_report(per_device)
    assert len(report) == 3
    for row in report:
        assert row["n_rows"] > 20


def test_adaptive_policy_runs_without_nan_or_inf():
    # n_rounds i vogel qe çdo raund te kete >= min_new_samples_to_train rreshta
    # (fixture-i eshte i vogel qellimisht, per shpejtesi ne CI).
    cfg, clients = _load_tiny_clients(n_rounds=3)
    logs = run_simulation(clients, policy="adaptive", thresholds=ControllerThresholds(), n_rounds=3)
    assert len(logs) > 0
    rmses = [l.rmse for l in logs]
    assert all(np.isfinite(r) for r in rmses), "RMSE jo-finite -- gjase divergjence ne local_train"
    assert any(l.should_train for l in logs), "Asnje raund s'ka trajnuar -- kontrollo pragjet"


def test_non_adaptive_always_trains_and_sends_full():
    cfg, clients = _load_tiny_clients(n_rounds=10)
    logs = run_simulation(clients, policy="non_adaptive", thresholds=ControllerThresholds(), n_rounds=10)
    assert all(l.should_train for l in logs)
    assert all(l.should_send_update for l in logs)
    assert all(l.update_mode == "full" for l in logs)
    assert all(l.model_variant == "full" for l in logs)


def test_drift_injection_raises_error_on_targeted_client():
    # Krahasim BRENDA TE NJEJTIT klient (para kunder GJATE kalimit/ramp-it te
    # drift-it) -- me i qendrueshem se krahasimi mes klienteve te ndryshem.
    #
    # SHENIM METODOLOGJIK (vlen edhe per punim): nje bias gradual QE
    # STABILIZOHET ne nje konstante (si keti) ESHTE LARGSISHT I "PERTHITHSHEM"
    # nga nje model autoregresiv lineare pasi te perfundoje ramp-i -- i njejti
    # offset shfaqet ne lag-features DHE ne target, pra modeli vazhdon te
    # parashikoje mire. Rritja REALE, e matshme e gabimit ndodh GJATE vete
    # ramp-it (kur offset-i i inputeve te vjetra ndryshon nga ai i targetit te
    # ri) -- jo pas tij. Prandaj testi krahason "para" me "gjate ramp-it", jo
    # me "shume kohe pas". Per nje drift qe mbetet i vrojtueshem edhe PAS
    # stabilizimit (jo vetem tranzitor), perdor 'abrupt_relationship_shift'.
    cfg = ExperimentConfig()
    per_device = load_clients(cfg, FIXTURE)
    target = sorted(per_device.keys())[0]
    scenario = DriftScenario(kind="gradual_sensor_bias", start_round=3, magnitude=8.0, ramp_rounds=2)
    clients = prepare_clients(per_device, cfg, n_live_rounds=10, round_minutes=5.0,
                               drift_scenarios={target: scenario}, seed=1, n_warmup_epochs=150)
    logs = run_simulation(clients, policy="adaptive", thresholds=ControllerThresholds(), n_rounds=10)
    pre_drift = [l.rmse for l in logs if l.client == target and l.round < 3]
    during_ramp = [l.rmse for l in logs if l.client == target and 3 <= l.round <= 5]
    assert np.mean(during_ramp) > np.mean(pre_drift), (
        "Drift-i i injektuar nuk e rriti gabimin gjate ramp-it -- kontrollo drift.py"
    )


def test_drift_never_touches_rounds_before_start_round():
    # REGRESION per nje bug te gjetur ne analizen e eksperimenteve reale te
    # perdoruesit (tetor 2026): apply_drift_to_frame merrte RRESHTA RAW
    # (live_df.index, para feature engineering) dhe i trajtonte si "raunde" me
    # enumerate() -- meqe ne te dhenat reale ka ~6-8 rreshta raw per raund
    # (resample_freq=5min / n_live_rounds), nje scenario.start_round=40
    # (synuar si RAUNDI 40) perfundonte duke u aplikuar ne RRESHTIN raw #40,
    # pra rreth raundi ~5-6 -- ~7x me shpejt se sa duhej. Ky test eshte i
    # ndjeshem ndaj pikerisht kesaj gjalesie (ndryshe nga
    # test_drift_injection_raises_error_on_targeted_client me siper, qe
    # thjesht kontrollon "a u rrit gabimi mandej", pa kontrolluar KUR -- prandaj
    # kaloi edhe me bug-un prezent).
    #
    # Perdorim n_live_rounds te vogel (5) kunder nje fixture me shume rreshta
    # raw -- pra ratio rreshta/raund >> 1, keshtu qe nje gabim ne njesi (rresht
    # kunder raund) do te zbulohej qarte: nese start_round=3 aplikohet gabim
    # ne rreshtin raw #3 (brenda raundit 0), rrumbullaksimet 0-2 do te dilnin
    # te ndryshme nga nje skenar "none" -- ato DUHET te jene IDENTIKE.
    cfg = ExperimentConfig()
    per_device = load_clients(cfg, FIXTURE)
    target = sorted(per_device.keys())[0]
    n_rounds = 5

    scenario_none = DriftScenario(kind="none", start_round=10 ** 9)
    scenario_drift = DriftScenario(kind="gradual_sensor_bias", start_round=3, magnitude=8.0, ramp_rounds=2)

    clients_none = prepare_clients(per_device, cfg, n_live_rounds=n_rounds, round_minutes=5.0,
                                    drift_scenarios={target: scenario_none}, seed=1, n_warmup_epochs=150)
    logs_none = run_simulation(clients_none, policy="non_adaptive", thresholds=ControllerThresholds(), n_rounds=n_rounds)

    clients_drift = prepare_clients(per_device, cfg, n_live_rounds=n_rounds, round_minutes=5.0,
                                     drift_scenarios={target: scenario_drift}, seed=1, n_warmup_epochs=150)
    logs_drift = run_simulation(clients_drift, policy="non_adaptive", thresholds=ControllerThresholds(), n_rounds=n_rounds)

    rmse_none = {l.round: l.rmse for l in logs_none if l.client == target}
    rmse_drift = {l.round: l.rmse for l in logs_drift if l.client == target}
    for r in range(0, scenario_drift.start_round):
        assert rmse_none[r] == pytest.approx(rmse_drift[r], abs=1e-9), (
            f"Raundi {r} (< start_round={scenario_drift.start_round}) ndryshon mes skenarit 'none' dhe "
            f"skenarit me drift -- drift-i po 'rrjedh' prapa ne raunde qe duhet te jene te paprekura "
            f"(kontrollo perputhjen rresht-raw<->raund ne apply_drift_to_frame/prepare_clients)"
        )


def test_prepare_clients_rejects_too_many_rounds_for_sparse_live_data():
    # REGRESION per nje bug te gjetur duke testuar vete komanden e rekomanduar
    # me pare ne README_OVERLEAF.md §3.3 (--rounds 200, resample_freq=5min
    # parazgjedhje): me kaq pak rreshta 'live' per raund, axis-i (1) i
    # kontrolluesit ("new_samples < min_new_samples_to_train -> skip train")
    # kapercente trajnimin NE ÇDO RAUND, GJITHE XHIRIMIN, pa asnje gabim --
    # thjesht nje CSV me rounds_trained=0 kudo qe duket si rezultat normal po
    # s'e kontrollove shprehimisht ate kolone. Tani prepare_clients() e zbulon
    # vete kete dhe hedh ValueError, nese i jepet min_new_samples_to_train.
    cfg = ExperimentConfig()
    per_device = load_clients(cfg, FIXTURE)
    # Fixture-i ka 35 rreshta 'live' RAW/klient, por 23 pasi build_features()
    # heq 12 rreshta nga fundi (horizon_steps=12, y=shift(-12) del NaN atje)
    # -> me min_new_samples_to_train=6, max raunde te sigurta = 23//6 = 3.
    with pytest.raises(ValueError, match="rounds_trained=0"):
        prepare_clients(per_device, cfg, n_live_rounds=20, round_minutes=5.0,
                         seed=1, n_warmup_epochs=40, min_new_samples_to_train=6)
    # Nje numer "i sigurt" raundesh nuk duhet te hedhe gabim.
    prepare_clients(per_device, cfg, n_live_rounds=3, round_minutes=5.0,
                     seed=1, n_warmup_epochs=40, min_new_samples_to_train=6)


def test_adaptive_sends_fewer_or_equal_bytes_than_non_adaptive():
    cfg, clients_a = _load_tiny_clients(n_rounds=12)
    logs_a = run_simulation(clients_a, policy="adaptive", thresholds=ControllerThresholds(), n_rounds=12)
    _, clients_n = _load_tiny_clients(n_rounds=12)
    logs_n = run_simulation(clients_n, policy="non_adaptive", thresholds=ControllerThresholds(), n_rounds=12)
    bytes_a = sum(l.bytes_sent for l in logs_a)
    bytes_n = sum(l.bytes_sent for l in logs_n)
    assert bytes_a <= bytes_n, "Politika adaptive dergoi me shume bajte se baseline-i jo-adaptiv -- kontrollo controller.py"


def test_ablation_send_hold_always_sends_when_trained():
    # Bosht (3) i çaktivizuar -> çdo raund qe trajnon duhet te dergoje, pra
    # rounds_sent == rounds_trained (ndryshe nga politika plotesisht adaptive).
    cfg, clients = _load_tiny_clients(n_rounds=12)
    ablate = AblationFlags(disable_send_hold=True)
    logs = run_simulation(clients, policy="adaptive", thresholds=ControllerThresholds(), n_rounds=12, ablate=ablate)
    trained = [l for l in logs if l.should_train]
    assert all(l.should_send_update for l in trained), (
        "Ablacioni i boshtit (3) nuk ka funksionuar -- disa raunde trajnuan pa dërguar"
    )


def test_ablation_compression_never_compresses():
    # Bosht (4) i çaktivizuar -> update_mode duhet te jete GJITHMONE "full".
    cfg, clients = _load_tiny_clients(n_rounds=12)
    ablate = AblationFlags(disable_compression=True)
    logs = run_simulation(clients, policy="adaptive", thresholds=ControllerThresholds(), n_rounds=12, ablate=ablate)
    assert all(l.update_mode == "full" for l in logs), (
        "Ablacioni i boshtit (4) nuk ka funksionuar -- gjetën update te kompresuar"
    )


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"Running {t.__name__} ...")
        t()
        print("  OK")
    print(f"\n{len(tests)} teste kaluan.")
