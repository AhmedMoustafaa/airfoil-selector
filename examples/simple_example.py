"""
Simple Example
==============
Single operating point, uniform weighting, no constraints.
Demonstrates:
  - basic max_cl search at one Re
  - preset scoring ('max_cl')
  - summary_report with score column
  - geometry plot
  - Excel export
"""
import numpy as np
from airfoil_selector import AirfoilSelector


def main():
    selector = AirfoilSelector(db_path='../databases/full_data.pkl')

    # ── Single operating point ─────────────────────────────────────────────────
    Re_range    = [3e5]
    Re_weights  = [1]
    aoa_weights = np.ones(26)   # uniform across all 26 AOA points

    # ── Objective ──────────────────────────────────────────────────────────────
    max_cl_dict = selector.max_cl(Re_range, Re_weights, aoa_weights)

    # ── Top 5 ──────────────────────────────────────────────────────────────────
    top_5      = selector.get_top_n(max_cl_dict, n=5, ascending=False)
    best_names = list(top_5.keys())

    # ── Preset scoring ─────────────────────────────────────────────────────────
    scores = selector.score_airfoils(best_names, re_target=3e5, criteria='max_cl')

    # ── Report & visualise ─────────────────────────────────────────────────────
    selector.summary_report(best_names, re_target=3e5, scores_dict=scores)
    selector.plot_shapes(best_names, db_dir='airfoilsdb')

    # ── Export ─────────────────────────────────────────────────────────────────
    selector.export_to_excel(best_names, re_target=3e5,
                             filepath='results_simple.xlsx',
                             scores_dict=scores)


if __name__ == "__main__":
    main()