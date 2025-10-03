# sensitivity_analysis.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from model_boarding import Method, Plane, simulate_boarding

# ---------- Settings ----------
ROWS_GRID = [30, 40, 50, 60, 70]   # rows (6 seats per row)
N_RUNS = 100                       # simulations per (rows, method)
SEED = 42                          # global seed for reproducibility
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
# ------------------------------

rng = np.random.default_rng(SEED)

METHODS = [Method.RANDOM, Method.BACK_TO_FRONT, Method.FRONT_TO_BACK, Method.BLOCKS_K2]
NAMES = {
    Method.RANDOM: "RANDOM",
    Method.BACK_TO_FRONT: "BACK TO FRONT",
    Method.FRONT_TO_BACK: "FRONT TO BACK",
    Method.BLOCKS_K2: "BLOCKS K2",
}
# distinct styles
COL = {
    Method.RANDOM: "#355C7D",
    Method.BACK_TO_FRONT: "#F8B195",
    Method.FRONT_TO_BACK: "#C06C84",
    Method.BLOCKS_K2: "#6C5B7B",
}
MRK = {Method.RANDOM: "o", Method.BACK_TO_FRONT: "s", Method.FRONT_TO_BACK: "D", Method.BLOCKS_K2: "^"}

def ci95_half(a: np.ndarray) -> float:
    """Half-width of 95% CI for the mean."""
    return 1.96 * np.nanstd(a, ddof=1) / np.sqrt(len(a))

def run_setting(rows: int, method: Method, seed_seq: np.random.SeedSequence) -> np.ndarray:
    """Return vector of N_RUNS boarding times (minutes)."""
    out = np.empty(N_RUNS, float)
    for i, child in enumerate(seed_seq.spawn(N_RUNS)):
        out[i] = simulate_boarding(method, plane=Plane(rows=rows, cols=6), seed=np.random.default_rng(child))
    return out

def collect() -> pd.DataFrame:
    """Run simulations for all rows×methods and return tidy dataframe."""
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
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.set_title(f"Sensitivity: Mean Boarding Time vs Rows (CI95)\nN={N_RUNS} per point", fontsize=18, pad=10)
    for m in NAMES.values():
        d = df[df["method"] == m].sort_values("rows")
        ax.errorbar(
            d["rows"], d["mean"], yerr=d["ci95"],
            fmt=MRK[[k for k, v in NAMES.items() if v == m][0]] + "-",
            color=COL[[k for k, v in NAMES.items() if v == m][0]],
            capsize=4, linewidth=2, markersize=7, label=m
        )
    ax.set_xlabel("Rows (6 seats per row)", fontsize=12)
    ax.set_ylabel("Mean boarding time (minutes)", fontsize=12)
    ax.grid(True, alpha=0.25)
    leg = ax.legend(title=f"Method", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=10, title_fontsize=11)
    for txt in leg.get_texts(): txt.set_fontsize(10)
    fig.savefig(RESULTS_DIR / "sensitivity_mean_ci.png", dpi=160)
    plt.show()

def plot_relative(df: pd.DataFrame) -> None:
    """Relative mean vs RANDOM (RANDOM baseline = 1)."""
    # pivot to method × rows
    piv = df.pivot(index="rows", columns="method", values="mean").sort_index()
    base = piv["RANDOM"].values
    rel = piv.divide(base, axis=0)  # element-wise / RANDOM

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.set_title(f"Sensitivity: Relative Mean vs RANDOM baseline (index)\nN={N_RUNS} per point", fontsize=18, pad=10)

    for m in NAMES.values():
        y = rel[m].values
        ax.plot(
            ROWS_GRID, y,
            marker=MRK[[k for k, v in NAMES.items() if v == m][0]],
            linewidth=2, markersize=7,
            color=COL[[k for k, v in NAMES.items() if v == m][0]],
            label=m
        )

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.2)
    ax.text(ROWS_GRID[-1] + 0.4, 1.0, "RANDOM = 1.0", va="center", ha="left", fontsize=10, color="grey")

    ax.set_xlabel("Rows (6 seats per row)", fontsize=12)
    ax.set_ylabel("Relative mean (RANDOM = 1.0)", fontsize=12)
    ax.set_ylim(0.55, rel.max().max() * 1.05)
    ax.grid(True, alpha=0.25)
    leg = ax.legend(title="Method", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=10, title_fontsize=11)
    for txt in leg.get_texts(): txt.set_fontsize(10)
    fig.savefig(RESULTS_DIR / "sensitivity_relative.png", dpi=160)
    plt.show()

def main():
    df = collect()
    # optional: save table
    df.to_csv(RESULTS_DIR / "sensitivity_summary.csv", index=False)
    print("\nSensitivity summary (minutes):\n", df.to_string(index=False))
    plot_mean_ci(df)
    plot_relative(df)

if __name__ == "__main__":
    main()
