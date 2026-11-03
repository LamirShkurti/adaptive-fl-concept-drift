#!/usr/bin/env python3
"""
Gjeneron figuren e krahasimit te drift-it (adaptive vs. non-adaptive, RMSE e
klientit te targetuar nder raunde) -- SAKTESISHT figura qe eshte perdorur ne
Seksionin 4.2 te punimit (`figures/drift_rmse_comparison.png`).

Ky eshte skript i vecante nga `analyze_results.py` sepse ai fokusohet te
MESATARJA e te gjithe klienteve (e dobishme per krahasimin baseline, Seksioni
4.1), ndersa per drift duam RMSE-ne vetem te klientit te TARGETUAR (ndryshe
sinjali i drift-it "zhytet" ne mesataren e 3 klienteve, 2 prej te cileve
s'kane drift fare).

Perdorim (nga brenda `fed_adaptive_forecasting/`, pasi te keni xhiruar te 4
komandat e drift-it te README_OVERLEAF.md §3.1 dhe te keni skedaret CSV te
rinj ne `outputs/`):

    python plot_drift_comparison.py \
        --abrupt-nonadaptive outputs/nonadaptive_80r_drift_abrupt.csv \
        --abrupt-adaptive    outputs/adaptive_80r_drift_abrupt.csv \
        --gradual-nonadaptive outputs/nonadaptive_80r_drift_gradual.csv \
        --gradual-adaptive    outputs/adaptive_80r_drift_gradual.csv \
        --target b8:27:eb:bf:9d:51 \
        --start-round 40 --ramp-rounds 30 \
        --output-dir outputs/figures

(Parametrat --target/--start-round/--ramp-rounds kane tashme vlerat e
mesiperme si default -- te njejtat qe perdoren ne komandat e README_OVERLEAF.md
-- pra mund t'i leni te pa-specifikuara nese s'i keni ndryshuar vete.)

Prodhon:
  - <output-dir>/drift_rmse_comparison.png  (figura 2-panele e perdorur ne punim)
  - <output-dir>/drift_phase_summary.csv    (RMSE mesatare pre-drift/gjate/pas,
    per skenar+politike -- te njejtat numra qe shkuan ne Tabelen 4 te punimit)
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _load_target(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    sub = df[df["client"] == target].sort_values("round").reset_index(drop=True)
    if sub.empty:
        raise ValueError(
            f"S'u gjet asnje rresht per klientin '{target}' te {path} -- "
            f"kontrollo --target (klientet e pranishem: {sorted(df['client'].unique())})"
        )
    return sub


def phase_summary(df: pd.DataFrame, start_round: int, ramp_rounds: int) -> dict:
    pre = df[df["round"] < start_round]
    during = df[(df["round"] >= start_round) & (df["round"] < start_round + ramp_rounds)]
    post = df[df["round"] >= start_round + ramp_rounds]
    return {
        "rmse_pre": pre["rmse"].mean(),
        "rmse_during": during["rmse"].mean(),
        "rmse_post": post["rmse"].mean(),
        "drift_ratio_during_max": during["drift_ratio"].max() if "drift_ratio" in during else None,
    }


def plot_comparison(
    abrupt_na: pd.DataFrame, abrupt_ad: pd.DataFrame,
    gradual_na: pd.DataFrame, gradual_ad: pd.DataFrame,
    start_round: int, ramp_rounds: int, out_path: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    ax = axes[0]
    ax.plot(abrupt_na["round"], abrupt_na["rmse"], label="Non-adaptive", color="#d62728")
    ax.plot(abrupt_ad["round"], abrupt_ad["rmse"], label="Adaptive", color="#1f77b4")
    ax.axvline(start_round, color="gray", linestyle="--", linewidth=1, label=f"Drift onset (round {start_round})")
    ax.set_title(f"Abrupt relationship shift\n(targeted client)")
    ax.set_xlabel("Round")
    ax.set_ylabel("RMSE (targeted client)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(gradual_na["round"], gradual_na["rmse"], label="Non-adaptive", color="#d62728")
    ax.plot(gradual_ad["round"], gradual_ad["rmse"], label="Adaptive", color="#1f77b4")
    ax.axvspan(start_round, start_round + ramp_rounds, color="gray", alpha=0.12,
               label=f"Ramp (rounds {start_round}-{start_round + ramp_rounds})")
    ax.set_title(f"Gradual sensor bias\n(targeted client)")
    ax.set_xlabel("Round")
    ax.set_ylabel("RMSE (targeted client)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--abrupt-nonadaptive", required=True)
    ap.add_argument("--abrupt-adaptive", required=True)
    ap.add_argument("--gradual-nonadaptive", required=True)
    ap.add_argument("--gradual-adaptive", required=True)
    ap.add_argument("--target", default="b8:27:eb:bf:9d:51", help="ID (ose nen-ID) i klientit te targetuar nga --drift")
    ap.add_argument("--start-round", type=int, default=40)
    ap.add_argument("--ramp-rounds", type=int, default=30)
    ap.add_argument("--output-dir", default="outputs/figures")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    abrupt_na = _load_target(args.abrupt_nonadaptive, args.target)
    abrupt_ad = _load_target(args.abrupt_adaptive, args.target)
    gradual_na = _load_target(args.gradual_nonadaptive, args.target)
    gradual_ad = _load_target(args.gradual_adaptive, args.target)

    fig_path = os.path.join(args.output_dir, "drift_rmse_comparison.png")
    plot_comparison(abrupt_na, abrupt_ad, gradual_na, gradual_ad, args.start_round, args.ramp_rounds, fig_path)
    print(f"U ruajt figura: {fig_path}")

    rows = []
    for scenario, na, ad in [
        ("abrupt_relationship_shift", abrupt_na, abrupt_ad),
        ("gradual_sensor_bias", gradual_na, gradual_ad),
    ]:
        for policy, d in [("non_adaptive", na), ("adaptive", ad)]:
            s = phase_summary(d, args.start_round, args.ramp_rounds)
            s["scenario"] = scenario
            s["policy"] = policy
            rows.append(s)
    summ = pd.DataFrame(rows)[["scenario", "policy", "rmse_pre", "rmse_during", "rmse_post", "drift_ratio_during_max"]]
    summ_path = os.path.join(args.output_dir, "drift_phase_summary.csv")
    summ.to_csv(summ_path, index=False)
    print(summ.to_string(index=False))
    print(f"\nU ruajt tabela: {summ_path}")

    # Verifikim shtese: raundet PARA start_round duhet te jene identike mes
    # skenarit abrupt dhe gradual per te NJEJTIN klient (drift-i s'duhet te
    # 'rrjedhe' prapa) -- nese kjo dështon, diçka s'shkon (p.sh. nje seed
    # i ndryshuar mes xhirimeve).
    pre_a = abrupt_na[abrupt_na["round"] < args.start_round]["rmse"].reset_index(drop=True)
    pre_g = gradual_na[gradual_na["round"] < args.start_round]["rmse"].reset_index(drop=True)
    if len(pre_a) == len(pre_g):
        max_diff = (pre_a - pre_g).abs().max()
        print(f"\nVerifikim: diferenca max. e RMSE per raundet < {args.start_round} (abrupt vs. gradual, duhet ~0) = {max_diff:.6f}")
        if max_diff > 1e-6:
            print("KUJDES: diferenca s'eshte 0 -- a jane te dy xhirimet me te njejtin --seed (default 42)?")


if __name__ == "__main__":
    main()
