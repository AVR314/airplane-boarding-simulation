# stat_tests.py
# Statistical analysis + publication-ready figures (shows and saves).

from __future__ import annotations
import math
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from statsmodels.graphics.gofplots import qqplot
from scipy import stats

# ------------------------ Config ------------------------
DATA_CSV = Path("results") / "runs_raw.csv"
OUT_DIR = Path("results") / "stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed method order and labels
ORDER = ["RANDOM", "BACK_TO_FRONT", "FRONT_TO_BACK", "BLOCKS_K2"]
LABELS = {
    "RANDOM": "RANDOM",
    "BACK_TO_FRONT": "BACK_TO_FRONT",
    "FRONT_TO_BACK": "FRONT_TO_BACK",
    "BLOCKS_K2": "BLOCKS_K2",
}

# High-contrast palette (color-blind friendly)
COLORS = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]

# Global style
mpl.rcParams.update({
    "figure.dpi": 140,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 10,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.25,
    "savefig.bbox": "tight",
})

# ------------------------ Utils ------------------------
def _ci95_half(x: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    return 1.96 * np.std(x, ddof=1) / math.sqrt(x.size)

def _load() -> pd.DataFrame:
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"Missing {DATA_CSV}. Run run_boarding_study.py first to create it."
        )
    df = pd.read_csv(DATA_CSV)

    # Normalize column names: expect 'method' and 'time_min'
    cols = {c.lower(): c for c in df.columns}
    if "method" not in cols:
        # Try case-insensitive
        mcol = [c for c in df.columns if c.lower() == "method"]
        if not mcol:
            raise ValueError("CSV must contain a 'method' column.")
        df.rename(columns={mcol[0]: "method"}, inplace=True)

    # Minutes column may be 'minutes' (your pipeline) or 'time_min' (older)
    if "minutes" in df.columns and "time_min" not in df.columns:
        df.rename(columns={"minutes": "time_min"}, inplace=True)
    elif "time_min" not in df.columns:
        tcol = [c for c in df.columns if c.lower() in ("time_min", "minutes")]
        if not tcol:
            raise ValueError("CSV must contain 'minutes' or 'time_min' column.")
        df.rename(columns={tcol[0]: "time_min"}, inplace=True)

    # Order methods as categorical for consistent plots
    df["Method"] = pd.Categorical(df["method"], categories=ORDER, ordered=True)
    # Keep only needed columns
    df = df[["Method", "time_min"]].copy()
    return df

def _summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Method", observed=False)["time_min"]
    out = pd.DataFrame({
        "mean": g.mean(),
        "median": g.median(),
        "std": g.std(ddof=1),
        "min": g.min(),
        "max": g.max(),
        "n": g.size(),
    })
    out["ci95_half"] = g.apply(lambda s: _ci95_half(s.to_numpy()))
    return out.reset_index()

def _pairwise(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return pairwise t-tests and Mann–Whitney results."""
    methods = ORDER
    rows_t, rows_mw = [], []
    for a, b in combinations(methods, 2):
        xa = df.loc[df["Method"] == a, "time_min"].to_numpy()
        xb = df.loc[df["Method"] == b, "time_min"].to_numpy()
        # Welch t-test (unequal variances)
        t = stats.ttest_ind(xa, xb, equal_var=False)
        # Cohen's d (Hedges' g small-sample correction)
        na, nb = xa.size, xb.size
        sa2, sb2 = xa.var(ddof=1), xb.var(ddof=1)
        sp = math.sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / (na + nb - 2))
        d = (xa.mean() - xb.mean()) / sp if sp > 0 else np.nan
        J = 1 - (3 / (4 * (na + nb) - 9))
        g_eff = J * d

        rows_t.append({
            "A": a, "B": b,
            "mean_A": xa.mean(), "mean_B": xb.mean(),
            "diff": xa.mean() - xb.mean(),
            "t_stat": t.statistic, "p_value": t.pvalue,
            "hedges_g": g_eff
        })

        # Mann–Whitney
        mw = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        rows_mw.append({"A": a, "B": b, "U": mw.statistic, "p_value": mw.pvalue})

    return pd.DataFrame(rows_t), pd.DataFrame(rows_mw)

# ------------------------ Figures ------------------------
def fig_box(df: pd.DataFrame) -> None:
    """Boxplot with legend that matches exactly what's drawn."""
    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    data = [df.loc[df["Method"] == m, "time_min"].to_numpy() for m in ORDER]
    bp = ax.boxplot(
        data,
        positions=np.arange(len(ORDER)) + 1,
        widths=0.6,
        patch_artist=True,
        whis=1.5,
        manage_ticks=True,
    )
    # Style elements
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=COLORS[i], edgecolor=COLORS[i], alpha=0.28, linewidth=1.6)
    for w in bp["whiskers"]:
        w.set(color="#666666", linewidth=1.4, linestyle="--")
    for c in bp["caps"]:
        c.set(color="#666666", linewidth=1.4)
    for m in bp["medians"]:
        m.set(color="#111111", linewidth=2.6)
    for f in bp["fliers"]:
        f.set(marker="o", markersize=4.2, markeredgecolor="#222222",
              markerfacecolor="#222222", alpha=0.75)

    ax.set_xticks(np.arange(len(ORDER)) + 1, [LABELS[m] for m in ORDER], rotation=0)
    ax.set_ylabel("Boarding time (minutes)")
    ax.set_title("Boarding time by method — boxplot", pad=8)

    # Legend: build from the same styles used above (accurate guide)
    guide = [
        mpatches.Patch(facecolor="#999999", edgecolor="#666666", alpha=0.28, label="IQR (box)"),
        mlines.Line2D([], [], color="#111111", linewidth=2.6, label="Median"),
        mlines.Line2D([], [], color="#666666", linewidth=1.4, linestyle="--", label="Whiskers"),
        mlines.Line2D([], [], color="#666666", linewidth=1.4, label="Caps"),
        mlines.Line2D([], [], color="#222222", marker="o", linestyle="None", label="Outliers"),
    ]
    ax.legend(handles=guide, title="Box elements", loc="upper right", frameon=True)

    fig.savefig(OUT_DIR / "violin_box.png")  # keep filename for continuity
    plt.show()

def fig_ecdf(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    for i, m in enumerate(ORDER):
        x = np.sort(df.loc[df["Method"] == m, "time_min"].to_numpy())
        y = np.arange(1, x.size + 1) / x.size
        ax.plot(x, y, color=COLORS[i], linewidth=2.2, label=LABELS[m])
    ax.set_xlabel("Boarding time (minutes)")
    ax.set_ylabel("Empirical CDF  F(t)")
    ax.set_ylim(0, 1.02)
    ax.set_title("ECDF of boarding time", pad=8)
    ax.legend(title="Method", loc="lower right", frameon=True)
    fig.savefig(OUT_DIR / "ecdf.png")
    plt.show()

def fig_means_ci(df: pd.DataFrame) -> None:
    summ = _summary(df)
    summ["Method"] = pd.Categorical(summ["Method"], categories=ORDER, ordered=True)
    summ = summ.sort_values("Method")
    x = np.arange(len(summ))
    means = summ["mean"].to_numpy()
    ci = summ["ci95_half"].to_numpy()

    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    bars = ax.bar(x, means, yerr=ci, capsize=5, color=COLORS, edgecolor="#222222", linewidth=1.1)
    ax.set_xticks(x, [LABELS[m] for m in ORDER])
    ax.set_ylabel("Mean boarding time (minutes)")
    ax.set_title("Mean ± 95% CI by method", pad=8)

    # Numeric labels above bars
    y0, y1 = ax.get_ylim()
    offs = np.where(np.isfinite(ci) & (ci > 0), ci * 0.12, (y1 - y0) * 0.015)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2,
                means[i] + ci[i] + offs[i],
                f"{means[i]:.1f} ± {ci[i]:.1f}",
                ha="center", va="bottom", fontsize=10, color="#111111")
    fig.savefig(OUT_DIR / "means_ci.png")
    # Also save the table
    summ.to_csv(OUT_DIR / "means_ci.csv", index=False)
    plt.show()

def fig_qq_grid(df: pd.DataFrame) -> None:
    """2×2 QQ plots with minimal overlap and small axis labels close to axes."""
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.8))
    axes = axes.ravel()
    for i, m in enumerate(ORDER):
        ax = axes[i]
        x = df.loc[df["Method"] == m, "time_min"].to_numpy()
        qqplot(x, line="s", ax=ax, marker=".", markerfacecolor="#333333",
               markeredgecolor="#333333", markersize=3)
        ax.set_title(f"QQ — {LABELS[m]}", pad=4, fontsize=11)
        ax.set_xlabel("Theoretical quantiles", fontsize=9, labelpad=2)
        ax.set_ylabel("Sample quantiles", fontsize=9, labelpad=2)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Normal QQ plots by method", y=0.98, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "qq_grid.png")
    plt.show()

def fig_forest_pairwise(df: pd.DataFrame) -> None:
    """Forest plot of pairwise mean differences with 95% CI (Welch)."""
    ttab, _ = _pairwise(df)

    # Compute CI for mean difference via Welch SE
    rows = []
    for _, r in ttab.iterrows():
        a, b = r["A"], r["B"]
        xa = df.loc[df["Method"] == a, "time_min"].to_numpy()
        xb = df.loc[df["Method"] == b, "time_min"].to_numpy()
        na, nb = xa.size, xb.size
        va, vb = xa.var(ddof=1), xb.var(ddof=1)
        se = math.sqrt(va / na + vb / nb)
        diff = float(r["diff"])
        lo, hi = diff - 1.96 * se, diff + 1.96 * se
        rows.append({
            "pair": f"{a} − {b}",
            "diff": diff,
            "lo": lo,
            "hi": hi,
            "p_value": r["p_value"],
            "hedges_g": r["hedges_g"],
        })
    dfp = pd.DataFrame(rows)
    # Order pairs by absolute effect size (largest first)
    dfp = dfp.sort_values("diff", key=lambda s: s.abs(), ascending=True)

    fig, ax = plt.subplots(figsize=(9.6, 6.2), constrained_layout=True)
    y = np.arange(len(dfp))
    ax.hlines(y, dfp["lo"], dfp["hi"], color="#555555", linewidth=1.6)
    ax.plot(dfp["diff"], y, "o", color="#111111", markersize=4.5)
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1.0)

    ax.set_yticks(y, dfp["pair"])
    ax.set_xlabel("Mean difference (minutes)")
    ax.set_title("Pairwise differences (mean ± 95% CI)", pad=8)

    # Text columns on the right (p-value, g)
    for yi, (p, g) in enumerate(zip(dfp["p_value"], dfp["hedges_g"])):
        ax.text(ax.get_xlim()[1], yi, f"p={p:.3g}   g={g:.2f}",
                va="center", ha="right", fontsize=9, color="#333333")

    fig.savefig(OUT_DIR / "forest_pairwise.png")
    plt.show()

# ------------------------ Main ------------------------
def main() -> None:
    df = _load()

    # Save assumption diagnostics summary (normality via Shapiro; variance via Levene)
    # Note: large-sample Shapiro can be overly sensitive; this is illustrative.
    rows = []
    for m in ORDER:
        x = df.loc[df["Method"] == m, "time_min"].to_numpy()
        sh = stats.shapiro(x) if len(x) <= 5000 else stats.normaltest(x)
        rows.append({"Method": m, "test": "normality", "stat": float(sh.statistic), "p_value": float(sh.pvalue)})
    lev = stats.levene(*[df.loc[df["Method"] == m, "time_min"].to_numpy() for m in ORDER])
    rows.append({"Method": "ALL", "test": "equal_var (Levene)", "stat": float(lev.statistic), "p_value": float(lev.pvalue)})
    pd.DataFrame(rows).to_csv(OUT_DIR / "assumptions.csv", index=False)

    # ANOVA (Welch) for omnibus difference in means
    # Using one-way ANOVA (equal_var False via Welch alternative: use statsmodels if needed; here classic ANOVA shown)
    # We still report it as omnibus; pairwise follow below.
    f_stat, p_val = stats.f_oneway(*[df.loc[df["Method"] == m, "time_min"] for m in ORDER])
    pd.DataFrame([{"F": float(f_stat), "p_value": float(p_val)}]).to_csv(OUT_DIR / "omnibus.csv", index=False)

    # Pairwise tests
    ttab, mwtab = _pairwise(df)
    ttab.to_csv(OUT_DIR / "pairwise_ttests.csv", index=False)
    mwtab.to_csv(OUT_DIR / "pairwise_mw.csv", index=False)

    # Figures (each both saved and shown)
    fig_box(df)
    fig_means_ci(df)
    fig_ecdf(df)
    fig_qq_grid(df)
    fig_forest_pairwise(df)

    print("Saved outputs to:", OUT_DIR)
    for p in sorted(OUT_DIR.glob("*")):
        print(" -", p)

if __name__ == "__main__":
    main()
