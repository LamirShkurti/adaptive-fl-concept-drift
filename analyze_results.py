#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_runs(paths, labels):
    frames = []
    for p, lab in zip(paths, labels):
        df = pd.read_csv(p)
        df["run_label"] = lab
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("run_label").agg(
        mean_rmse=("rmse", "mean"),
        median_rmse=("rmse", "median"),
        total_bytes=("bytes_sent", "sum"),
        pct_rounds_trained=("should_train", "mean"),
        pct_rounds_sent=("should_send_update", "mean"),
        pct_compressed=("update_mode", lambda s: (s == "compressed").mean()),
        pct_lite=("model_variant", lambda s: (s == "lite").mean()),
        pct_model_inactive=("model_active", lambda s: (~s).mean()),
    )
    g["pct_rounds_trained"] *= 100
    g["pct_rounds_sent"] *= 100
    g["pct_compressed"] *= 100
    g["pct_lite"] *= 100
    g["pct_model_inactive"] *= 100
    return g


def plot_rmse_over_rounds(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lab, g in df.groupby("run_label"):
        per_round = g.groupby("round")["rmse"].mean()
        ax.plot(per_round.index, per_round.values, label=lab)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean RMSE (across clients)")
    ax.set_title("Forecast error over FL rounds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bytes_cumulative(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lab, g in df.groupby("run_label"):
        per_round = g.groupby("round")["bytes_sent"].sum().cumsum()
        ax.plot(per_round.index, per_round.values / 1024.0, label=lab)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative bytes transmitted (KB)")
    ax.set_title("Communication cost over FL rounds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_decisions_per_client(df: pd.DataFrame, run_label: str, out_path: str):
    sub = df[df.run_label == run_label]
    clients = sorted(sub.client.unique())
    fig, axes = plt.subplots(len(clients), 1, figsize=(9, 2.2 * len(clients)), sharex=True)
    if len(clients) == 1:
        axes = [axes]
    for ax, cid in zip(axes, clients):
        g = sub[sub.client == cid].sort_values("round")
        ax.scatter(g["round"], g["should_train"].astype(int), s=10, label="train", marker="o")
        ax.scatter(g["round"], g["should_send_update"].astype(int) * 1.1, s=10, label="send", marker="x")
        ax.scatter(g["round"], (g["model_variant"] == "lite").astype(int) * 1.2, s=10, label="lite", marker="s")
        ax.set_ylabel(cid[:10], fontsize=8)
        ax.set_yticks([])
    axes[0].legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("Round")
    fig.suptitle(f"Per-client adaptive decisions ({run_label})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="CSV-te e prodhuara nga run_experiment.py")
    ap.add_argument("--labels", nargs="+", required=True, help="Etiketa per secilin run (e njejta renditje)")
    ap.add_argument("--output-dir", default="outputs/figures")
    args = ap.parse_args()
    assert len(args.runs) == len(args.labels), "--runs dhe --labels duhet te kene te njejtin numer elementesh"

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_runs(args.runs, args.labels)

    summ = summary_table(df)
    summ_path = os.path.join(args.output_dir, "summary_table.csv")
    summ.to_csv(summ_path)
    print(summ.to_string())
    print(f"\nU ruajt tabela: {summ_path}")

    plot_rmse_over_rounds(df, os.path.join(args.output_dir, "rmse_over_rounds.png"))
    plot_bytes_cumulative(df, os.path.join(args.output_dir, "bytes_cumulative.png"))
    for lab in args.labels:
        plot_decisions_per_client(df, lab, os.path.join(args.output_dir, f"decisions_{lab}.png"))
    print(f"U ruajten grafiket ne: {args.output_dir}")


if __name__ == "__main__":
    main()
