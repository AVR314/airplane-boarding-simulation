# run_boarding_study.py
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from model_boarding import Method, Plane, simulate_boarding

# ---------- Config ----------
N_RUNS = 100
SEED = 42                  # reproducible, not shown on plots
PLANE = Plane(rows=50, cols=6)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Clean, high-contrast colors
acecolors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]
method_order = [Method.RANDOM, Method.BACK_TO_FRONT, Method.FRONT_TO_BACK, Method.BLOCKS_K2]
method_labels = {
    Method.RANDOM: "RANDOM",
    Method.BACK_TO_FRONT: "BACK_TO_FRONT",
    Method.FRONT_TO_BACK: "FRONT_TO_BACK",
    Method.BLOCKS_K2: "BLOCKS_K2",
}

mpl.rcParams.update({
    "figure.dpi": 140,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.25,
})

# ---------- Helpers ----------
def ci95_half(x: np.ndarray) -> float:
    """95% CI half-width for mean (normal approx)."""
    if len(x) < 2:
        return float("nan")
    return 1.96 * (x.std(ddof=1) / math.sqrt(len(x)))

def simulate_all(n_runs: int, seed: int) -> pd.DataFrame:
    """Run all methods and return tidy DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in method_order:
        for i in range(n_runs):
            t = simulate_boarding(m, PLANE, seed=int(rng.integers(0, 2**31 - 1)))
            rows.append({"method": method_labels[m], "run": i + 1, "minutes": float(t)})
    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("method")["minutes"]
    out = pd.DataFrame({
        "count": g.size(),
        "mean": g.mean(),
        "median": g.median(),
        "std": g.std(ddof=1),
        "min": g.min(),
        "max": g.max(),
    }).reset_index()
    out["ci95_half"] = df.groupby("method")["minutes"].apply(lambda s: ci95_half(s.to_numpy())).values
    order_map = {method_labels[m]: i for i, m in enumerate(method_order)}
    return out.sort_values("method", key=lambda s: s.map(order_map))

# ---------- Plots ----------
def plot_cummean(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for i, m in enumerate(method_labels.values()):
        sub = df[df["method"] == m].sort_values("run")
        y = sub["minutes"].to_numpy()
        cum = np.cumsum(y) / np.arange(1, len(y) + 1)
        ax.plot(sub["run"], cum, lw=2.2, color=acecolors[i], label=m)
    ax.set_title("Cumulative Mean Boarding Time vs. Runs")
    ax.set_xlabel(f"Simulation index (1..{N_RUNS})")
    ax.set_ylabel("Boarding time (minutes)")
    ax.legend(title=f"Method (N={N_RUNS})", loc="upper right", frameon=True)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "cummean.png", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "cummean.pdf", bbox_inches="tight")
    plt.show()

def plot_ecdf(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for i, m in enumerate(method_labels.values()):
        x = np.sort(df[df["method"] == m]["minutes"].to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, lw=2.2, color=acecolors[i], label=m)
    ax.set_title("ECDF (Empirical Cumulative Distribution) of Boarding Time")
    ax.set_xlabel("Boarding time (minutes)")
    ax.set_ylabel("Empirical CDF  F(t)")
    ax.set_ylim(0, 1.02)
    ax.legend(title=f"Method (N={N_RUNS})", loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "ecdf.png", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "ecdf.pdf", bbox_inches="tight")
    plt.show()

def plot_box(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    data = [df[df["method"] == method_labels[m]]["minutes"].to_numpy() for m in method_order]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.6,
        tick_labels=[method_labels[m] for m in method_order],  # <- no deprecation
        whis=1.5,
        manage_ticks=True,
    )
    # style boxes
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(acecolors[i])
        patch.set_alpha(0.25)
        patch.set_edgecolor(acecolors[i])
        patch.set_linewidth(1.6)
    # whiskers (thin dashed gray) + caps (gray)
    for w in bp["whiskers"]:
        w.set_color("#666666")
        w.set_linewidth(1.4)
        w.set_linestyle("--")
    for c in bp["caps"]:
        c.set_color("#666666")
        c.set_linewidth(1.4)
    # medians (thick solid black)
    for m in bp["medians"]:
        m.set_color("#111111")
        m.set_linewidth(2.8)
    # outliers
    for f in bp["fliers"]:
        f.set_markerfacecolor("#222222")
        f.set_markeredgecolor("#222222")
        f.set_markersize(4.5)
        f.set_alpha(0.75)

    ax.set_title("Boarding Time by Method (Boxplot)")
    ax.set_ylabel("Minutes")

    # Legend: clearly distinguish elements
    lg_items = [
        mpatches.Patch(facecolor="#999999", edgecolor="#666666", alpha=0.25, label="IQR (25–75%)"),
        mlines.Line2D([], [], color="#111111", linewidth=2.8, label="Median"),
        mlines.Line2D([], [], color="#666666", linewidth=1.4, linestyle="--", label="Whiskers"),
        mlines.Line2D([], [], color="#222222", marker="o", linestyle="None", label="Outliers"),
        mlines.Line2D([], [], color="#666666", linewidth=1.4, label="Caps"),
    ]
    ax.legend(handles=lg_items, title="Boxplot guide", loc="upper right", frameon=True, ncol=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "boxplot.png", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "boxplot.pdf", bbox_inches="tight")
    plt.show()

def plot_bar_ci(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    x = np.arange(len(summary))
    means = summary["mean"].to_numpy()
    ci = summary["ci95_half"].to_numpy()
    bars = ax.bar(x, means, yerr=ci, capsize=5.0, color=acecolors, edgecolor="#222222", linewidth=1.2)
    ax.set_title("Average Boarding Time with 95% CI")
    ax.set_ylabel("Minutes")
    ax.set_xticks(x, summary["method"].tolist())
    # annotate N instead of legend (avoids warning and clutter)
    ax.text(0.99, 0.99, f"N={N_RUNS}", transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#222222")
    # numeric labels above bars
    y0, y1 = ax.get_ylim()
    offsets = np.where(np.isfinite(ci) & (ci > 0), ci * 0.12, (y1 - y0) * 0.015)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, means[i] + ci[i] + offsets[i],
                f"{means[i]:.1f} ± {ci[i]:.1f}", ha="center", va="bottom", fontsize=12, color="#111111")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "bar_ci95.png", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "bar_ci95.pdf", bbox_inches="tight")
    plt.show()

# ---------- Main ----------
def main() -> None:
    df = simulate_all(N_RUNS, SEED)
    summary = summarize(df)

    print("\nSummary by method (minutes):")
    print(summary.to_string(index=False))

    df.to_csv(RESULTS_DIR / "runs_raw.csv", index=False)
    summary.to_csv(RESULTS_DIR / "runs_summary.csv", index=False)

    plot_cummean(df)
    plot_ecdf(df)
    plot_box(df)
    plot_bar_ci(summary)

    print(f"\nSaved figures to '{RESULTS_DIR}/' and CSVs to the same folder.\n")

if __name__ == "__main__":
    main()
