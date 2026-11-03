#!/usr/bin/env python3
"""
Pike hyrjeje CLI per sistemin Federated Adaptive Forecasting.

Perdorim tipik (shih README.md per detaje te plota):

    python run_experiment.py --data data/iot_telemetry_data.csv --policy adaptive
    python run_experiment.py --data data/iot_telemetry_data.csv --policy non_adaptive
    python run_experiment.py --data data/iot_telemetry_data.csv --policy adaptive \
        --drift "b8:27:eb,gradual_sensor_bias,40,3.0"

    KUJDES: --drift perdor PRESJE (",") per te ndare fushat, JO ":" -- ID-te
    e pajisjeve Kaggle vete permbajne ":" (p.sh. "b8:27:eb:bf:9d:51"), prandaj
    ":" NUK mund te perdoret si ndares (shih parse_drift_arg() me poshte).

Secili run shkruan nje CSV te plote me log per-raund-per-klient ne
outputs/<run-name>.csv -- perdoret pastaj nga analyze_results.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import AblationFlags, ControllerThresholds, ExperimentConfig
from src.data_loader import device_report, load_clients
from src.drift import DriftScenario
from src.fl_simulation import prepare_clients, run_simulation


def parse_drift_arg(spec: str):
    # format: "<device_substring>,<kind>,<start_round>,<magnitude>"
    # (presje, JO ":", sepse ID-te e pajisjeve Kaggle vete permbajne ":", p.sh.
    # "b8:27:eb:bf:9d:51")
    parts = spec.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Format drift i pavlefshem. Prit: device,kind,start_round,magnitude "
            "(p.sh. 'b8:27:eb,gradual_sensor_bias,30,4.0' -- mjafton nje pjese "
            "e ID-se se pajisjes, s'duhet e plote)"
        )
    dev, kind, start, mag = parts
    return dev, DriftScenario(kind=kind, start_round=int(start), magnitude=float(mag))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Rruga drejt iot_telemetry_data.csv")
    ap.add_argument("--policy", choices=["adaptive", "non_adaptive"], default="adaptive")
    ap.add_argument("--rounds", type=int, default=60, help="Numri i raundeve 'live' te simulimit FL")
    ap.add_argument("--resample-freq", default="5min")
    ap.add_argument("--round-minutes", type=float, default=5.0,
                     help="Kohezgjatja reale (minuta) qe perfaqeson 1 raund -- perdoret per modelin e baterise")
    ap.add_argument("--non-adaptive-epochs", type=int, default=3)
    ap.add_argument("--warmup-epochs", type=int, default=150,
                     help="Epochs trajnimi mbi dritaren historike, para raundeve live (warm-start)")
    ap.add_argument("--drift", action="append", default=[],
                     help="Shtim skenari drift: device,kind,start_round,magnitude "
                          "(kind= gradual_sensor_bias|abrupt_relationship_shift). Mund te perseritet.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--ablate-train-skip", action="store_true",
                     help="Studim ablacioni, bosht (1): gjithmone trajno (ignoro battery/new_samples gating)")
    ap.add_argument("--ablate-epoch-adapt", action="store_true",
                     help="Studim ablacioni, bosht (2): epochs fikse (shih --ablate-fixed-epochs)")
    ap.add_argument("--ablate-send-hold", action="store_true",
                     help="Studim ablacioni, bosht (3): dergo gjithmone kur trajnon (mos mbaj update lokalisht)")
    ap.add_argument("--ablate-compression", action="store_true",
                     help="Studim ablacioni, bosht (4): gjithmone dergo 'full' (asnjehere i kompresuar)")
    ap.add_argument("--ablate-deactivation", action="store_true",
                     help="Studim ablacioni, bosht (5): modeli gjithmone aktiv (asnjehere fallback)")
    ap.add_argument("--ablate-lite-switch", action="store_true",
                     help="Studim ablacioni, bosht (6): gjithmone varianti 'full' (asnjehere 'lite')")
    ap.add_argument("--ablate-fixed-epochs", type=int, default=3,
                     help="Epochs fikse te perdorura vetem kur --ablate-epoch-adapt eshte aktiv")
    args = ap.parse_args()

    ablate = AblationFlags(
        disable_train_skip=args.ablate_train_skip,
        disable_epoch_adapt=args.ablate_epoch_adapt,
        disable_send_hold=args.ablate_send_hold,
        disable_compression=args.ablate_compression,
        disable_deactivation=args.ablate_deactivation,
        disable_lite_switch=args.ablate_lite_switch,
        fixed_epochs_for_ablation=args.ablate_fixed_epochs,
    )
    any_ablation = any([
        args.ablate_train_skip, args.ablate_epoch_adapt, args.ablate_send_hold,
        args.ablate_compression, args.ablate_deactivation, args.ablate_lite_switch,
    ])

    cfg = ExperimentConfig(resample_freq=args.resample_freq)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1/4] Duke ngarkuar dhe pastruar {args.data} ...")
    per_device = load_clients(cfg, args.data)
    report = device_report(per_device)
    print("Pajisjet e gjetura (klientet FL):")
    for row in report:
        print(f"  - {row['device']}: {row['n_rows']} rreshta, {row['start']} -> {row['end']}, "
              f"temp mean={row['temp_mean']:.2f} std={row['temp_std']:.2f}")
    if len(report) < 2:
        print("KUJDES: dataset-i ka vetem 1 pajisje te zbuluar -- kontrollo kolonen 'device' ne CSV.")

    drift_scenarios = {}
    for spec in args.drift:
        dev_substr, scenario = parse_drift_arg(spec)
        matches = [d for d in per_device if dev_substr in d]
        if not matches:
            print(f"KUJDES: asnje pajisje nuk permban '{dev_substr}', drift-i do te injorohet.")
        for m in matches:
            drift_scenarios[m] = scenario
            print(f"  Drift i caktuar per {m}: {scenario}")

    thresholds = ControllerThresholds()

    # Kontrolli i sigurise (shih fl_simulation.prepare_clients) vlen vetem
    # kur axis-i (1) i kontrolluesit vertet mund te kaperceje trajnimin per
    # new_samples te paket -- pra politika "adaptive" pa --ablate-train-skip.
    # Per non_adaptive (gjithmone trajno) ose per nje ablacion qe e ka
    # çaktivizuar vete kete bosht, numri i rreshtave/raund mund te jete i
    # vogel pa qene problem.
    min_samples_check = (
        thresholds.min_new_samples_to_train
        if (args.policy == "adaptive" and not args.ablate_train_skip)
        else None
    )

    print(f"[2/4] Duke pergatitur klientet ({args.rounds} raunde live, {args.resample_freq}/rresht) ...")
    clients = prepare_clients(
        per_device, cfg, n_live_rounds=args.rounds, round_minutes=args.round_minutes,
        drift_scenarios=drift_scenarios, seed=args.seed, n_warmup_epochs=args.warmup_epochs,
        min_new_samples_to_train=min_samples_check,
    )

    print(f"[3/4] Duke xhiruar simulimin FL (policy={args.policy}) ...")
    logs = run_simulation(clients, policy=args.policy, thresholds=thresholds, n_rounds=args.rounds,
                           non_adaptive_epochs=args.non_adaptive_epochs, ablate=ablate)

    default_name = f"{args.policy}_{args.rounds}r"
    if args.policy == "adaptive" and any_ablation:
        disabled = [name for name, flag in [
            ("notrainskip", args.ablate_train_skip), ("noepoch", args.ablate_epoch_adapt),
            ("nosendhold", args.ablate_send_hold), ("nocompress", args.ablate_compression),
            ("nodeactivate", args.ablate_deactivation), ("nolite", args.ablate_lite_switch),
        ] if flag]
        default_name += "_ablate-" + "-".join(disabled)
    run_name = args.run_name or default_name
    out_csv = os.path.join(args.output_dir, f"{run_name}.csv")
    df = pd.DataFrame([dataclasses.asdict(l) for l in logs])
    df.to_csv(out_csv, index=False)
    print(f"[4/4] U ruajt: {out_csv}  ({len(df)} rreshta)")

    run_config = {
        "policy": args.policy, "rounds": args.rounds, "resample_freq": args.resample_freq,
        "round_minutes": args.round_minutes, "devices": list(per_device.keys()),
        "drift": {k: dataclasses.asdict(v) for k, v in drift_scenarios.items()},
        "thresholds": dataclasses.asdict(thresholds),
        "ablation": dataclasses.asdict(ablate) if any_ablation else None,
    }
    with open(os.path.join(args.output_dir, f"{run_name}_config.json"), "w") as f:
        json.dump(run_config, f, indent=2, default=str)

    summary = df.groupby("client").agg(
        mean_rmse=("rmse", "mean"), total_bytes=("bytes_sent", "sum"),
        rounds_trained=("should_train", "sum"), rounds_sent=("should_send_update", "sum"),
    )
    print("\nPermbledhje per klient:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
