"""
Advanced Example
================
Full multi-objective workflow: parallel objectives, apply_constraints pipeline,
compute_objective dispatcher, custom scoring, and the complete visualisation suite.

Demonstrates:
  - compute_objective() dispatcher for all five metrics
  - apply_constraints() with all four constraint types (incl. cl_max & cm)
  - Custom score_weights replicating MATLAB-style weighted scoring, extended
    with the two new propeller efficiency metrics
  - Pareto scatter across two objectives
  - Radar chart (normalised, now with 7 axes)
  - New plot types: cl32_cd, cl12_cd, operating envelopes for both new metrics
  - Parallel coordinates (now includes CL^1.5/CD and CL^0.5/CD axes)
  - Excel export sorted by custom score
"""
import numpy as np
from airfoil_selector import AirfoilSelector


def main():
    selector = AirfoilSelector(db_path='../databases/small_data.pkl')

    # ── Operating conditions ───────────────────────────────────────────────────
    Re_range   = [1e5, 3e5, 5e5]
    Re_weights = [3, 5, 2]

    aoa_range   = np.linspace(-5, 20, 26)
    aoa_weights = selector.distribute(center=2, w1=5, w2=1, width=5, arr=aoa_range)

    cl_range   = np.linspace(0.1, 1.2, 20)
    cl_weights = selector.distribute(center=0.6, w1=5, w2=1, width=5, arr=cl_range)

    # ── Objectives via unified dispatcher ─────────────────────────────────────
    # compute_objective() replaces the individual named methods when you want
    # to select the metric programmatically (e.g., from a config file or UI).
    cl_dict     = selector.compute_objective('cl',      Re_range, Re_weights, aoa_weights=aoa_weights)
    cl_cd_dict  = selector.compute_objective('cl_cd',   Re_range, Re_weights, aoa_weights=aoa_weights)
    cl32_dict   = selector.compute_objective('cl32_cd', Re_range, Re_weights, aoa_weights=aoa_weights)
    cl12_dict   = selector.compute_objective('cl12_cd', Re_range, Re_weights, aoa_weights=aoa_weights)
    min_cd_dict = selector.compute_objective(
        'min_cd', Re_range, Re_weights,
        cl_range=cl_range, cl_weights=cl_weights, aoa_range=aoa_range)

    # ── Constraint pipeline ────────────────────────────────────────────────────
    constraints = [
        {'type': 'thickness', 'min': 0.10, 'max': 0.25},
        {'type': 'camber',    'min': 0.00, 'max': 0.05},
        {'type': 'cl_max',    'min': 1.3 * 1.10, 're': 3e5},   # 3-D CLmax ≥ 1.3
        {'type': 'cm',        'max': 0.00, 're':  3e5},         # no reflexed sections
    ]

    constrained_cl    = selector.apply_constraints(cl_dict,    constraints)
    constrained_cl32  = selector.apply_constraints(cl32_dict,  constraints)
    constrained_min_cd = selector.apply_constraints(min_cd_dict, constraints)

    # ── Shortlist from primary objective (max CL) ──────────────────────────────
    top_5       = selector.get_top_n(constrained_cl, n=5, ascending=False)
    best_names  = list(top_5.keys())
    top_airfoil = best_names[0]

    # ── Custom scoring (MATLAB-equivalent, extended with new metrics) ──────────
    # Weights are applied after per-metric min-max normalisation → physically
    # comparable regardless of scale.
    custom_weights = {
        'cl':          0.10,    # takeoff performance
        'cd':          0.10,    # cruise drag
        'cm':          0.00,    # pitching moment (not a priority here)
        'cl_cd':       0.30,    # glide ratio / jet-range efficiency
        'cl32_cd':     0.35,    # prop endurance  ← primary mission metric
        'cl12_cd':     0.10,    # prop range / min-power
        'alpha_stall': 0.05,    # stall margin
    }
    scores = selector.score_airfoils(
        best_names, re_target=3e5,
        criteria='custom', score_weights=custom_weights)

    # ── Terminal report ────────────────────────────────────────────────────────
    selector.summary_report(best_names, re_target=3e5, scores_dict=scores)

    # ── Advanced visualisations ────────────────────────────────────────────────

    # 1. Parallel coordinates — full database, now with all five metrics
    selector.plot_parallel_coordinates(re_target=3e5)

    # 2. Pareto: endurance (CL^1.5/CD) vs drag — two competing objectives
    selector.plot_pareto_scatter(
        constrained_cl32, constrained_min_cd,
        x_label="Weighted CL^1.5/CD (Endurance)",
        y_label="Weighted Min CD (Drag)")

    # 3. Radar — 7-axis normalised comparison (includes CL^1.5/CD, CL^0.5/CD)
    selector.plot_radar_chart(best_names, re_target=3e5)

    # 4. New efficiency curves vs AOA for the shortlist
    selector.plot_airfoils(best_names, re_target=3e5, plot_type='cl32_cd')
    selector.plot_airfoils(best_names, re_target=3e5, plot_type='cl12_cd')

    # 5. Operating envelopes — endurance and range/power factors over Re × α
    selector.plot_operating_envelope(top_airfoil, metric='cl32_cd')
    selector.plot_operating_envelope(top_airfoil, metric='cl12_cd')

    # 6. Drag polar across all Re for the winner
    selector.plot_multi_re_drag_polar(top_airfoil)

    # ── Export (sorted by custom score) ───────────────────────────────────────
    selector.export_to_excel(best_names, re_target=3e5,
                             filepath='results_advanced.xlsx',
                             scores_dict=scores)


if __name__ == "__main__":
    main()