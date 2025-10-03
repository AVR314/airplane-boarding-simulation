#!/usr/bin/env python3
# Sensitivity: mean boarding time vs number of rows (and relative to RANDOM).
#sensitivity_analysis2

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model_boarding import Method, Plane, simulate_boarding

# -------- Settings --------
ROWS_GRID = [30, 40, 50, 60, 70]   # rows (6 seats per row)
N_RUNS = 100                       # simulations per (rows, method)
SEED = 42                          # reproducible results
OUT = Path("results"); OUT.mkdir(exist_ok=True)
# --------------------------

rng = np.random.default_rng(SEED)

METHODS = [Method.RANDOM, Method.BACK_TO_FRONT, Method.FRONT_TO_BACK, Method.BLOCKS_K2]
NAMES = {
    Method.RANDOM: "RANDOM",
    Method.BACK_TO_FRONT: "BACK_TO_FRONT",
    Method.FRONT_TO_BACK: "FRONT_TO_BACK",
    Method.BLOCKS_K2: "BLOCKS_K2",
}
COL = {"RANDOM":"#355C7D","BACK_TO_FRONT":"#F8B195","FRONT_TO_BACK":"#C06C84","BLOCKS_K2":"#6C5B7B"}
MRK = {"RANDOM":"o","BACK_TO_FRONT":"s","FRONT_TO_BACK":"D","BLOCKS_K2":"^"}

def ci95_half(a: np.ndarray) -> float:
    """Half-width of 95% CI for the mean (normal approx)."""
    return 1.96 * np.nanstd(a, ddof=1) / np.sqrt(len(a))

def run_setting(rows: int, method: Method, seed_seq: np.random.SeedSequence) -> np.ndarray:
    """Return vector of N_RUNS boarding times (minutes)."""
    out = np.empty(N_RUNS, float)
    for i, child in enumerate(seed_seq.spawn(N_RUNS)):
        out[i] = simulate_boarding(method, plane=Plane(rows=rows, cols=6), seed=np.random.default_rng(child))
    return out

def collect() -> pd.DataFrame:
    """Simulate all (rows × methods) and return tidy dataframe."""
    recs = []
    for r in ROWS_GRID:
        for m in METHODS:
            times = run_setting(r, m, np.random.SeedSequence(r * 1000 + int(m.value)))
            recs.append({
                "rows": r,
                "method": NAMES[m],
                "mean": float(times.mean()),
                "ci95": ci95_half(times),
                "n": N_RUNS,
            })
    return pd.DataFrame.from_records(recs)

def plot_mean_ci(df: pd.DataFrame) -> None:
    """Mean±CI95 vs rows (absolute minutes)."""
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.set_title(f"Sensitivity: Mean Boarding Time vs Rows (95% CI) — N={N_RUNS} per point", pad=8)
    for meth in df["method"].unique():
        d = df[df["method"] == meth].sort_values("rows")
        ax.errorbar(d["rows"], d["mean"], yerr=d["ci95"],
                    fmt=MRK[meth]+"-", color=COL[meth],
                    capsize=4, linewidth=2, markersize=7, label=meth)
    ax.set_xlabel("Rows (6 seats per row)")
    ax.set_ylabel("Mean boarding time (minutes)")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Method", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.savefig(OUT / "sensitivity_mean_ci.png", dpi=160)
    plt.show()

def plot_relative(df: pd.DataFrame) -> None:
    """Relative mean vs RANDOM (baseline = 1.0)."""
    piv = df.pivot(index="rows", columns="method", values="mean").sort_index()
    base = piv["RANDOM"]
    rel = piv.divide(base, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.set_title(f"Sensitivity: Relative Mean vs RANDOM baseline — N={N_RUNS} per point", pad=8)
    for meth in rel.columns:
        ax.plot(rel.index, rel[meth].values,
                marker=MRK[meth], color=COL[meth],
                linewidth=2, markersize=7, label=meth)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.2)
    ax.text(rel.index.max()+0.4, 1.0, "RANDOM = 1.0", va="center", ha="left", fontsize=9, color="grey")
    ax.set_xlabel("Rows (6 seats per row)")
    ax.set_ylabel("Relative mean (RANDOM = 1.0)")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Method", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.savefig(OUT / "sensitivity_relative.png", dpi=160)
    plt.show()

def main() -> None:
    df = collect()
    df.to_csv(OUT / "sensitivity_summary.csv", index=False)
    print("\nSensitivity summary (minutes):\n", df.to_string(index=False))
    plot_mean_ci(df)
    plot_relative(df)

if __name__ == "__main__":
    main()
