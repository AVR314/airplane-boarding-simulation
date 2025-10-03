#!/usr/bin/env python3
# Sanity checks: (A) theoretical targets 150/172.5, (B) quick model means.

from __future__ import annotations
import math
import sys
from typing import Dict, List

import numpy as np

# Import ONLY from your model
from model_boarding import simulate_boarding, Method, Plane

# -------- small utils --------

def ci95(x: np.ndarray) -> float:
    """95% CI half-width (normal approx)."""
    if x.size < 2:
        return float("nan")
    return 1.96 * np.std(x, ddof=1) / math.sqrt(x.size)

def print_line(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))

# -------- A) theoretical-only sanity (no model calls) --------

def check_theoretical_targets(seed: int = 7, reps: int = 4000) -> None:
    """Reproduce 150 and 172.5 using the spec’s exponentials."""
    rng = np.random.default_rng(seed)

    # Target 1: sum of 300 Exp(mean=0.5 min) -> mean ~ 150 min
    sums1 = np.sum(rng.exponential(scale=0.5, size=(reps, 300)), axis=1)

    # Target 2: add 50 row-block exponentials Exp(mean=0.45) -> +22.5 min
    sums2 = sums1 + np.sum(rng.exponential(scale=0.45, size=(reps, 50)), axis=1)

    m1, h1 = float(np.mean(sums1)), ci95(sums1)
    m2, h2 = float(np.mean(sums2)), ci95(sums2)

    print_line("Sanity targets (means in minutes)")
    print(f"  Target 1 (luggage only)   ~ 150.0 -> {m1:.2f}  ±{h1:.2f}")
    print(f"  Target 2 (+row blocking)  ~ 172.5 -> {m2:.2f}  ±{h2:.2f}")

    # Tight assertions
    assert abs(m1 - 150.0) < 2.0, "Target 1 should be ~150 min"
    assert abs(m2 - 172.5) < 2.0, "Target 2 should be ~172.5 min"

# -------- B) quick model sanity (imports your model) --------

def run_model_sanity(n_runs: int = 20, seed: int = 123) -> None:
    """Run a few model calls and print mean times (minutes)."""
    rng = np.random.default_rng(seed)
    plane = Plane(rows=50, cols=6)

    methods: List[Method] = [
        Method.RANDOM,
        Method.BACK_TO_FRONT,
        Method.FRONT_TO_BACK,
        Method.BLOCKS_K2,
    ]

    results: Dict[str, np.ndarray] = {}
    for m in methods:
        vals = []
        for _ in range(n_runs):
            # simulate_boarding returns MINUTES per your model’s docstring
            vals.append(simulate_boarding(m, plane=plane, seed=int(rng.integers(0, 1e9))))
        results[m.name] = np.asarray(vals, dtype=float)

    print_line(f"Quick model sanity (n={n_runs} per method) — minutes")
    for name in results:
        mean = float(np.mean(results[name]))
        half = ci95(results[name])
        print(f"{name:<14} mean = {mean:7.2f}  ±{half:5.2f}")

    # Soft qualitative checks (won’t crash on tiny overlaps)
    if all(k in results for k in ("FRONT_TO_BACK", "RANDOM", "BACK_TO_FRONT", "BLOCKS_K2")):
        m_ftb = float(np.mean(results["FRONT_TO_BACK"]))
        m_rnd = float(np.mean(results["RANDOM"]))
        m_btf = float(np.mean(results["BACK_TO_FRONT"]))
        m_blk = float(np.mean(results["BLOCKS_K2"]))
        eps = 1.0  # tolerance in minutes
        if not (m_ftb > m_rnd - eps):
            print("Note: FRONT_TO_BACK not clearly slower than RANDOM (within eps).")
        if not (m_rnd > m_btf - eps):
            print("Note: BACK_TO_FRONT not clearly faster than RANDOM (within eps).")
        if not (m_blk + eps < min(m_btf, m_rnd, m_ftb)):
            print("Note: BLOCKS_K2 not clearly fastest (within eps).")

def main() -> None:
    check_theoretical_targets()   # 150 / 172.5
    run_model_sanity()            # quick means from your model

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}", file=sys.stderr)
        sys.exit(1)
