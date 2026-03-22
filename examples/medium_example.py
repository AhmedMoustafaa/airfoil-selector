"""
Medium Example
==============
Multi-operating point, weighted distributions, constraints via apply_constraints.
Demonstrates:
  - distribute() for peaked AOA weighting
  - apply_constraints() pipeline: thickness + camber + cl_max + cm (no reflex)
  - preset scoring ('max_cl32_cd' — propeller endurance)
  - summary_report with score column (now shows all five metrics)
  - standard 2-D aerodynamic plots including the new efficiency curves
  - Excel export
"""
import numpy as np
from airfoil_selector import AirfoilSelector


def main():
    selector = AirfoilSelector(db_path='../databases/full_data.pkl')

    # ── Operating conditions ───────────────────────────────────────────────────
    Re_range   = [1e5, 3e5, 5e5]
    Re_weights = [3, 5, 2]             # 3e5 is the primary cruise Re

    aoa_range   = np.linspace(-5, 20, 26)
    # Peak at 5° (typical cruise), tapering to baseline 1
    aoa_weights = selector.distribute(center=5, w1=5, w2=1, width=5, arr=aoa_range)

    # ── Objective: best propeller endurance ───────────────────────────────────
    cl32_cd_dict = selector.max_cl32_cd(Re_range, Re_weights, aoa_weights)

    # ── Constraint pipeline (single call) ─────────────────────────────────────
    # cl_max constraint uses ×1.10 factor to convert 3-D requirement → 2-D minimum
    constrained = selector.apply_constraints(cl32_cd_dict, [
        {'type': 'thickness', 'min': 0.10, 'max': 0.15},
        {'type': 'camber',    'min': 0.02, 'max': 0.05},
        {'type': 'cl_max',    'min': 1.2 * 1.10, 're': 3e5},   # 3-D CLmax ≥ 1.2
        {'type': 'cm',        'max': 0.00, 're':  3e5},         # exclude reflexed
    ])

    # ── Top 5 ──────────────────────────────────────────────────────────────────
    top_5      = selector.get_top_n(constrained, n=5, ascending=False)
    best_names = list(top_5.keys())

    # ── Scoring: best endurance preset ────────────────────────────────────────
    scores = selector.score_airfoils(best_names, re_target=3e5, criteria='max_cl32_cd')

    # ── Report ─────────────────────────────────────────────────────────────────
    selector.summary_report(best_names, re_target=3e5, scores_dict=scores)

    # ── Plots: CL-alpha, endurance curve, drag polar ──────────────────────────
    selector.plot_airfoils(best_names, re_target=3e5, plot_type='cl_alpha')
    selector.plot_airfoils(best_names, re_target=3e5, plot_type='cl32_cd')   # new
    selector.plot_airfoils(best_names, re_target=3e5, plot_type='drag_polar')

    # ── Export ─────────────────────────────────────────────────────────────────
    selector.export_to_excel(best_names, re_target=3e5,
                             filepath='results_medium.xlsx',
                             scores_dict=scores)


if __name__ == "__main__":
    main()