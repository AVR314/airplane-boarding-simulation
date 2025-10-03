# extra_tests.py
# Additional statistical tests: Kruskal–Wallis and Permutation test

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# Load the same data file produced by run_boarding_study.py
DATA_CSV = Path("results") / "runs_raw.csv"
ORDER = ["RANDOM", "BACK_TO_FRONT", "FRONT_TO_BACK", "BLOCKS_K2"]

def main():
    if not DATA_CSV.exists():
        raise FileNotFoundError("Missing results/runs_raw.csv. Run run_boarding_study.py first.")

    # Load dataset
    df = pd.read_csv(DATA_CSV)

    # Normalize column names
    if "minutes" in df.columns:
        df.rename(columns={"minutes": "time_min"}, inplace=True)
    elif "time_min" not in df.columns:
        raise ValueError("CSV must contain a 'minutes' or 'time_min' column.")

    # --- Kruskal–Wallis test ---
    # Non-parametric alternative to one-way ANOVA
    samples = [df.loc[df["method"] == m, "time_min"].to_numpy() for m in ORDER]
    kw_stat, kw_p = stats.kruskal(*samples)
    print("Kruskal–Wallis test:")
    print(f"  H = {kw_stat:.4f}, p = {kw_p:.4g}")

    # --- Permutation test example ---
    # Compare RANDOM vs BLOCKS_K2 as illustration
    x = df.loc[df["method"] == "RANDOM", "time_min"].to_numpy()
    y = df.loc[df["method"] == "BLOCKS_K2", "time_min"].to_numpy()

    # Define statistic: difference in means
    def mean_diff(a, b):
        return np.mean(a) - np.mean(b)

    observed = mean_diff(x, y)
    res = stats.permutation_test(
        (x, y),
        statistic=mean_diff,
        vectorized=False,
        n_resamples=10000,  # number of permutations
        alternative="two-sided",
        random_state=42
    )

    print("\nPermutation test (RANDOM vs BLOCKS_K2):")
    print(f"  Observed mean diff = {observed:.3f}")
    print(f"  p-value = {res.pvalue:.4g}")

if __name__ == "__main__":
    main()
