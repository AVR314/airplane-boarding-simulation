# ✈️ Airplane Boarding Simulation Study

This repository contains a **full simulation and statistical analysis** of airplane boarding methods.  
It was developed as part of an Operations Research / Statistics course project.

---

## 📂 Contents

### Core Simulation
- `model_boarding.py` — Implements the airplane boarding model (single-aisle, 50 rows, 6 seats per row).
- `run_boarding_study.py` — Runs repeated simulations and generates main figures (ECDF, boxplots, cumulative means).
- `sanity_checks.py` — Verifies theoretical expectations (150 / 172.5 minutes targets) and quick validation runs.

### Statistical Analysis
- `stat_tests.py` — Welch-ANOVA, pairwise tests, QQ-plots, ECDF, boxplots, forest plots.
- `extra_tests.py` — Non-parametric alternatives (Kruskal–Wallis, Mann–Whitney, permutation test).

### Sensitivity
- `sensitivity_analysis.py`, `sensitivity_analysis2.py` — Boarding time vs. number of rows, both absolute and relative.

### Utilities
- `export_for_overleaf.py` — Collects all generated figures into a `for_overleaf/` folder (for report writing).
- Other helpers: `fix_images.py`, `convert_clean.py`.

### Reports
- `Simulation_Boarding_OR (35).pdf` — Final submitted academic report (Overleaf).
- `סימולציה הוראות.pdf` — Assignment instructions.

---

## 🎯 Project Goal
To empirically evaluate the efficiency of four boarding policies:
- **RANDOM**
- **BACK_TO_FRONT**
- **FRONT_TO_BACK**
- **BLOCKS_K2** (our designed method: seat-order + two-row blocks)

---

## 🔬 Methodology
- **Model:** event-driven simulation of 300 passengers in a single-aisle cabin.  
- **Service times:** exponential (luggage storage, seat blocking).  
- **Repetitions:** 100 independent runs per method (fixed seeds for reproducibility).  
- **Analysis:** mean boarding time, 95% confidence intervals, effect sizes, robustness checks, and sensitivity to plane size.

---

## 📊 Key Results
- `BLOCKS_K2` consistently outperformed others:  
  ~40% faster than RANDOM, ~60–70% faster than directional methods.  
- Distribution tails narrower under `BLOCKS_K2` (lower risk of extreme delays).  
- Robustness confirmed across different plane sizes and alternative tests.

---

## 🚀 How to Run
```bash
# Run main simulation
python run_boarding_study.py

# Statistical analysis
python stat_tests.py

# Sensitivity analysis
python sensitivity_analysis.py
python sensitivity_analysis2.py

# Sanity checks
python sanity_checks.py
