"""
XFoil Hybrid Database Generation Script
=========================================
Builds full_data_xfoil.pkl using the same schema as full_data.pkl so
AirfoilSelector works with either database without any changes.

Strategy
--------
  Re ≤ RE_XFOIL_MAX  -> XFoil (viscous panel method, accurate at low Re)
  Re > RE_XFOIL_MAX  -> NeuralFoil (neural surrogate, better above ~2e6)

  For each XFoil polar that partially fails to converge:
    - Non-converged alpha points are filled individually from NeuralFoil.
    - If XFoil fails entirely for a (airfoil, Re) pair, the full row falls
      back to NeuralFoil.

  analysis_confidence column:
    1.0  -> converged XFoil result
    0.7  -> XFoil result with some NeuralFoil fill-in (< 30% of alphas)
    0.5  -> majority NeuralFoil fill-in (≥ 30% of alphas)
    0.3  -> full NeuralFoil fallback for a XFoil Re point
    0.1  -> NeuralFoil high-Re range
    0.0  -> complete failure, row left as zeros

Requirements
------------
  pip install aerosandbox tqdm
  XFoil binary must be on PATH (or set XFOIL_CMD below).
  On Linux/macOS:  sudo apt install xfoil  /  brew install xfoil
  On Windows:      add xfoil.exe to PATH or set XFOIL_CMD = r"C:/path/to/xfoil.exe"
"""

import os
import warnings
import multiprocessing as mp
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import aerosandbox as asb

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Configuration

DB_DIR     = Path('../airfoilsdb')
OUT_PKL    = Path('../databases/full_data_xfoil.pkl')
XFOIL_CMD  = 'xfoil'   # set to full path if xfoil is not on PATH

ALPHA = np.linspace(-5, 20, 26)

# XFoil Re range  (1e5 – 1e6, where it is most reliable)
RE_XFOIL = np.array([1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 1e6])

# NeuralFoil Re range  (high Re where XFoil degrades)
RE_NEURAL = np.array([2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7])

# Combined Re grid stored in the database (XFoil first, then NeuralFoil)
RE = np.concatenate([RE_XFOIL, RE_NEURAL])

RE_XFOIL_MAX  = RE_XFOIL[-1]   # 1e6

# XFoil settings
N_CRIT         = 9       # turbulence intensity (9 = clean wind tunnel / free flight)
XTR_UPPER      = 1.0     # free transition
XTR_LOWER      = 1.0
MAX_ITER       = 40      # iteration limit per alpha point
MAX_ITER_RETRY = 100     # relaxed limit used on retry
XFOIL_TIMEOUT  = 30      # seconds per polar — 30s is generous for a 26-alpha polar

# NeuralFoil settings
NF_MODEL_SIZE  = 'large'
MACH           = 0.0

# Fallback threshold: if this fraction or more of alpha points are NaN,
# treat the whole polar as failed and run NeuralFoil instead.
NAN_FALLBACK_THRESHOLD = 0.5

# Repaneling (XFoil only — NeuralFoil uses original coordinates)
N_PANELS = 150    # total points (75 per side) — well below XFoil's 365-node limit

# Multiprocessing
N_WORKERS = max(1, mp.cpu_count() // 2)  # XFoil is single-threaded; too many workers causes timeouts

def _load_airfoil(dat_path: Path):
    """Three-strategy loader: file path -> ASB stem name -> None."""
    for attempt in (
        lambda: asb.Airfoil(dat_path.name, coordinates=str(dat_path)),
        lambda: asb.Airfoil(dat_path.stem),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                af = attempt()
                if af.coordinates is not None and len(af.coordinates) >= 10:
                    return af
            except Exception:
                pass
    return None

def _prepare_airfoil(af: asb.Airfoil, n_panels: int = 150) -> asb.Airfoil:
    """
    Repanel to n_panels cosine-spaced points and run geometry sanity checks.

    We let XFoil do the actual repaneling via its PPAR command (ASB passes
    n_panels through to XFoil internally), so this function only validates
    that the repaneled geometry is non-degenerate.  If repaneling fails or
    produces bad geometry the original airfoil is returned unchanged so we
    never silently drop an airfoil due to a repaneling edge case.

    Checks performed
    ----------------
    - Minimum coordinate count (< 20 -> degenerate, reject)
    - Duplicate / near-duplicate points (distance < 1e-6 chord) -> XFoil hangs
    - Leading-edge index sanity: LE must not be at index 0 or -1 (open loop)
    """
    try:
        repaneled = af.repanel(n_points_per_side=n_panels // 2)
        coords = repaneled.coordinates

        if coords is None or len(coords) < 20:
            return af

        # Check for near-duplicate points anywhere in the contour
        dists = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        if np.any(dists < 1e-6):
            return af

        # LE should be somewhere in the middle of the coordinate array
        le_idx = int(np.argmin(coords[:, 0]))
        if le_idx == 0 or le_idx == len(coords) - 1:
            return af

        return repaneled

    except Exception:
        return af

print("Scanning airfoil database directory …")
name_map: dict[str, asb.Airfoil] = {}
skipped: list[str] = []

for path in sorted(DB_DIR.glob("*")):
    af = _load_airfoil(path)
    if af is None:
        skipped.append(path.name)
    else:
        name_map[path.name] = af

if skipped:
    print(f"  Skipped {len(skipped)} unparseable files.")

print(f"  {len(name_map)} valid airfoils found.\n")

if not name_map:
    raise RuntimeError(f"No valid airfoils found in '{DB_DIR}'.")

valid_names = list(name_map.keys())
airfoils    = np.array(valid_names)
n_af        = len(airfoils)
n_re_xf     = len(RE_XFOIL)
n_re_nf     = len(RE_NEURAL)
n_re        = len(RE)
n_alpha     = len(ALPHA)

# NeuralFoil uses the original coordinates from name_map (variable panel
# counts are fine for the neural surrogate and repaneling adds no accuracy).
# XFoil gets cosine-spaced coordinates so its panel solver has consistent
# LE/TE resolution across all airfoils.

print(f"Repaneling for XFoil ({N_PANELS} cosine-spaced points) …")
xfoil_coord_map: dict[str, np.ndarray] = {}
n_repanel_ok      = 0
n_repanel_fallback = 0

for name, af in name_map.items():
    repaneled = _prepare_airfoil(af, n_panels=N_PANELS)
    xfoil_coord_map[name] = repaneled.coordinates
    if repaneled is af:
        n_repanel_fallback += 1
    else:
        n_repanel_ok += 1

print(f"  Repaneled OK : {n_repanel_ok}")
if n_repanel_fallback:
    print(f"  Used original: {n_repanel_fallback}  (repanel failed geometry check)")
print()

# Task tuple layout: (name, coords_array, re_val, alpha, cfg_dict)
# We pass coords as a plain numpy array instead of an ASB object because
# ASB objects are not guaranteed to serialise cleanly across processes.

def _run_xfoil_polar(task):
    """
    Run one XFoil polar for a single (airfoil, Re) pair.

    Returns
    -------
    dict with keys: cl, cd, cm, top_xtr, bot_xtr, nan_mask, confidence
        Each aerodynamic array has length n_alpha.
        nan_mask[i] is True where XFoil did not converge.
        confidence is a scalar in [0, 1].
    """
    name, coords, re_val, alpha, cfg = task

    n_alpha = len(alpha)
    def _null(reason=''):
        return {
            'cl':          np.zeros(n_alpha),
            'cd':          np.zeros(n_alpha),
            'cm':          np.zeros(n_alpha),
            'top_xtr':     np.zeros(n_alpha),
            'bot_xtr':     np.zeros(n_alpha),
            'nan_mask':    np.ones(n_alpha, dtype=bool),
            'confidence':  0.0,
            'error_reason': reason,
        }

    # Reconstruct airfoil inside the worker process
    try:
        af = asb.Airfoil(name=name, coordinates=coords)
    except Exception as e:
        return _null(f'airfoil_reconstruct: {e}')

    last_error = ['']   # mutable so the nested function can write to it

    def _attempt(max_iter):
        try:
            xf = asb.XFoil(
                airfoil       = af,
                Re            = re_val,
                mach          = cfg['mach'],
                n_crit        = cfg['n_crit'],
                xtr_upper     = cfg['xtr_upper'],
                xtr_lower     = cfg['xtr_lower'],
                max_iter      = max_iter,
                timeout       = cfg['timeout'],
                xfoil_command = cfg['xfoil_cmd'],
                xfoil_repanel = False,  # already pre-repaneled to 150 pts
                # hinge_point_x defaults to 0.75 — ASB crashes if set to None
            )
            result = xf.alpha(alpha)
            return result
        except Exception as e:
            last_error[0] = f'{type(e).__name__}: {e}'
            return None

    # First attempt with default max_iter
    result = _attempt(cfg['max_iter'])

    # Check how many points converged
    if result is None:
        nan_mask = np.ones(n_alpha, dtype=bool)
    else:
        cl_raw   = np.asarray(result.get('CL', result.get('cl', np.full(n_alpha, np.nan))))
        nan_mask = ~np.isfinite(cl_raw)

    nan_frac = nan_mask.mean()

    # Retry with relaxed iteration limit if many points failed
    if nan_frac > 0.2 and result is not None:
        result2 = _attempt(cfg['max_iter_retry'])
        if result2 is not None:
            cl2      = np.asarray(result2.get('CL', result2.get('cl', np.full(n_alpha, np.nan))))
            nan2     = ~np.isfinite(cl2)
            # Accept retry only if it converged more points
            if nan2.sum() < nan_mask.sum():
                result   = result2
                nan_mask = nan2
                nan_frac = nan_mask.mean()

    if result is None or nan_frac == 1.0:
        return _null(last_error[0])

    # Build a full-length nan_mask by matching which alpha values XFoil
    # actually returned against the full input alpha array.
    # XFoil only returns rows for converged points, so the result arrays
    # may be shorter than n_alpha — we cannot use the raw nan_mask built
    # from the short CL array to index into n_alpha-length arrays.
    alpha_converged = np.asarray(
        result.get('alpha', result.get('Alpha', np.array([]))), dtype=float)
    nan_mask_full = np.ones(n_alpha, dtype=bool)   # start: all failed
    for i, a in enumerate(alpha):
        if np.any(np.isclose(alpha_converged, a, atol=0.01)):
            nan_mask_full[i] = False               # this alpha converged
    nan_mask = nan_mask_full
    nan_frac = float(nan_mask.mean()) if len(nan_mask) > 0 else 1.0

    if nan_frac == 1.0 or np.isnan(nan_frac):
        return _null('all alpha points non-converged after remapping')

    # Extract fields and map converged values to the correct alpha indices.
    # XFoil returns results sorted by alpha; we match them positionally.
    def _get_mapped(key_caps, key_lower, fallback):
        v = result.get(key_caps, result.get(key_lower, None))
        out = np.full(n_alpha, fallback, dtype=float)
        if v is None:
            return out
        arr = np.asarray(v, dtype=float)
        converged_indices = np.where(~nan_mask)[0]
        n_fill = min(len(arr), len(converged_indices))
        out[converged_indices[:n_fill]] = arr[:n_fill]
        return out

    cl      = _get_mapped('CL',      'cl',      0.0)
    cd      = _get_mapped('CD',      'cd',      0.0)
    cm      = _get_mapped('CM',      'cm',      0.0)
    top_xtr = _get_mapped('Top_Xtr', 'top_xtr', 1.0)
    bot_xtr = _get_mapped('Bot_Xtr', 'bot_xtr', 1.0)

    # Non-converged slots are already at fallback values (0.0 / 1.0)
    # and nan_mask marks them for NeuralFoil fill-in downstream.

    # Confidence score
    if nan_frac == 0.0:
        confidence = 1.0
    elif nan_frac < 0.30:
        confidence = 0.7
    else:
        confidence = 0.5

    return {
        'cl':           cl,
        'cd':           cd,
        'cm':           cm,
        'top_xtr':      top_xtr,
        'bot_xtr':      bot_xtr,
        'nan_mask':     nan_mask,
        'confidence':   confidence,
        'error_reason': '',
    }

shape = (n_af, n_re, n_alpha)

cl         = np.zeros(shape)
cd         = np.zeros(shape)
cm         = np.zeros(shape)
cl_cd      = np.zeros(shape)
top_xtr    = np.zeros(shape)
bot_xtr    = np.zeros(shape)
mach_crit  = np.zeros(shape)    # XFoil doesn't provide this; filled from NeuralFoil

max_cl = np.zeros((n_af, n_re))
min_cd = np.zeros((n_af, n_re))

max_thickness       = np.zeros(n_af)
camber_arr          = np.zeros(n_af)
analysis_confidence = np.zeros((n_af, n_re))   # per (airfoil, Re) for diagnostics

Alpha_nf, Re_nf_mesh = np.meshgrid(ALPHA, RE_NEURAL)

print("Running NeuralFoil sweep for high-Re range …")
nf_iterator = (tqdm(enumerate(airfoils), total=n_af, desc="NeuralFoil")
               if HAS_TQDM else enumerate(airfoils))

# Also build a per-(airfoil, alpha) NeuralFoil cache for the XFoil Re range
# so individual NaN fill-ins don't require a separate ASB call.
# Cache shape: (n_af, n_re_xf, n_alpha) — allocated lazily below.
nf_cache_xfoil = np.zeros((n_af, n_re_xf, n_alpha))
nf_cache_cd_xf = np.zeros((n_af, n_re_xf, n_alpha))
nf_cache_cm_xf = np.zeros((n_af, n_re_xf, n_alpha))

Alpha_xf, Re_xf_mesh = np.meshgrid(ALPHA, RE_XFOIL)

for i, name in nf_iterator:
    if not HAS_TQDM:
        print(f"  [{i+1}/{n_af}]  {name}")
    try:
        af = name_map[name]

        # High-Re NeuralFoil sweep
        aero_hi = af.get_aero_from_neuralfoil(
            alpha      = Alpha_nf.flatten(),
            Re         = Re_nf_mesh.flatten(),
            mach       = MACH,
            model_size = NF_MODEL_SIZE,
        )
        Aero_hi = {k: v.reshape(Alpha_nf.shape) for k, v in aero_hi.items()}

        re_start = n_re_xf   # high-Re block starts after XFoil block
        cl[i, re_start:]        = Aero_hi['CL']
        cd[i, re_start:]        = Aero_hi['CD']
        cm[i, re_start:]        = Aero_hi['CM']
        top_xtr[i, re_start:]   = Aero_hi['Top_Xtr']
        bot_xtr[i, re_start:]   = Aero_hi['Bot_Xtr']
        mach_crit[i, re_start:] = Aero_hi['mach_crit']
        analysis_confidence[i, re_start:] = 0.1

        # XFoil-range NeuralFoil cache (used to fill NaN gaps below)
        aero_lo = af.get_aero_from_neuralfoil(
            alpha      = Alpha_xf.flatten(),
            Re         = Re_xf_mesh.flatten(),
            mach       = MACH,
            model_size = NF_MODEL_SIZE,
        )
        Aero_lo = {k: v.reshape(Alpha_xf.shape) for k, v in aero_lo.items()}
        nf_cache_xfoil[i] = Aero_lo['CL']
        nf_cache_cd_xf[i] = Aero_lo['CD']
        nf_cache_cm_xf[i] = Aero_lo['CM']
        mach_crit[i, :n_re_xf] = Aero_lo['mach_crit']   # store for XFoil Re rows

        max_thickness[i] = af.max_thickness()
        camber_arr[i]    = af.max_camber()

    except Exception as exc:
        print(f"\n  WARNING: NeuralFoil failed for {name}: {exc}")

print()

xfoil_cfg = {
    'mach':          MACH,
    'n_crit':        N_CRIT,
    'xtr_upper':     XTR_UPPER,
    'xtr_lower':     XTR_LOWER,
    'max_iter':      MAX_ITER,
    'max_iter_retry':MAX_ITER_RETRY,
    'timeout':       XFOIL_TIMEOUT,
    'xfoil_cmd':     XFOIL_CMD,
}

# Build flat task list: one entry per (airfoil_idx, re_idx) pair
# Shape: n_af × n_re_xf tasks total
tasks = []
task_index = []   # (airfoil_idx, re_idx) for writing results back

for i, name in enumerate(airfoils):
    coords = xfoil_coord_map[name]   # repaneled coords for XFoil
    for j, re_val in enumerate(RE_XFOIL):
        tasks.append((name, coords, re_val, ALPHA, xfoil_cfg))
        task_index.append((i, j))

n_tasks   = len(tasks)
n_failed  = 0
n_partial = 0
n_full_fallback = 0

# Preflight: run one polar in the main process before launching the pool.
# This surfaces configuration errors (bad binary path, wrong ASB API, etc.)
# with a readable message instead of 22,000 silent failures.
print("Preflight XFoil check (1 polar in main process) …")
_test_task = tasks[0]
_test      = _run_xfoil_polar(_test_task)

if _test['confidence'] == 0.0:
    reason = _test.get('error_reason', 'unknown')
    print(f"\n{'='*70}")
    print(f"  PREFLIGHT FAILED — XFoil is not working correctly.")
    print(f"  Error: {reason}")
    print()
    print(f"  Likely causes:")
    print(f"    1. XFoil binary not found: make sure '{XFOIL_CMD}' is on PATH")
    print(f"       (Linux: sudo apt install xfoil  |  macOS: brew install xfoil)")
    print(f"       (Windows: set XFOIL_CMD to the full path of xfoil.exe)")
    print(f"    2. Wrong asb.XFoil() argument names for your AeroSandbox version.")
    print("       Run:  python -c 'import aerosandbox as asb; help(asb.XFoil)'")
    print(f"       and compare the constructor signature to the call in _attempt().")
    print(f"    3. XFoil binary found but crashes immediately (test it manually):")
    print(f"       echo '' | xfoil")
    print(f"{'='*70}\n")
    _user_input = input("Continue anyway with 100% NeuralFoil fallback? [y/N] ").strip().lower()
    if _user_input != 'y':
        raise SystemExit("Aborted. Fix XFoil configuration and re-run.")
    print("Continuing with NeuralFoil fallback for all XFoil Re points.\n")
else:
    nan_pct = 100 * _test['nan_mask'].mean()
    print(f"  OK — confidence={_test['confidence']:.1f}, "
          f"NaN alpha points: {nan_pct:.0f}%\n")

print(f"Running XFoil sweep: {n_tasks} polars across {N_WORKERS} workers …")
print(f"  ({n_af} airfoils × {n_re_xf} Re points)\n")

error_counts: dict[str, int] = {}

with mp.Pool(processes=N_WORKERS) as pool:
    iterator = pool.imap(_run_xfoil_polar, tasks, chunksize=4)
    if HAS_TQDM:
        iterator = tqdm(iterator, total=n_tasks, desc="XFoil",
                        unit="polar", dynamic_ncols=True)

    for task_num, xf_result in enumerate(iterator):
        i, j = task_index[task_num]
        nan_mask   = xf_result['nan_mask']
        confidence = xf_result['confidence']

        if confidence == 0.0:
            # Full failure -> use NeuralFoil cache for this whole row
            cl[i, j]       = nf_cache_xfoil[i, j]
            cd[i, j]       = nf_cache_cd_xf[i, j]
            cm[i, j]       = nf_cache_cm_xf[i, j]
            top_xtr[i, j]  = 1.0   # assume free transition
            bot_xtr[i, j]  = 1.0
            analysis_confidence[i, j] = 0.3
            n_full_fallback += 1
            reason = xf_result.get('error_reason', '')
            if reason:
                error_counts[reason] = error_counts.get(reason, 0) + 1

        else:
            # Write XFoil results
            cl[i, j]      = xf_result['cl']
            cd[i, j]      = xf_result['cd']
            cm[i, j]      = xf_result['cm']
            top_xtr[i, j] = xf_result['top_xtr']
            bot_xtr[i, j] = xf_result['bot_xtr']
            analysis_confidence[i, j] = confidence

            # Fill individual NaN alpha points with NeuralFoil
            if nan_mask.any():
                cl[i, j, nan_mask] = nf_cache_xfoil[i, j, nan_mask]
                cd[i, j, nan_mask] = nf_cache_cd_xf[i, j, nan_mask]
                cm[i, j, nan_mask] = nf_cache_cm_xf[i, j, nan_mask]
                if confidence < 1.0:
                    n_partial += 1
            elif confidence == 1.0:
                pass   # clean convergence, nothing to do
            else:
                n_partial += 1

with np.errstate(divide='ignore', invalid='ignore'):
    cl_cd[:] = np.where(cd > 0, cl / cd, 0.0)

max_cl[:] = np.max(cl,  axis=2)
min_cd[:] = np.min(cd,  axis=2)

# Mean confidence per airfoil (scalar, for backward compat with NeuralFoil schema)
mean_confidence = analysis_confidence.mean(axis=1)

total_xfoil_polars = n_af * n_re_xf
n_clean = total_xfoil_polars - n_full_fallback - n_partial

print(f"\nXFoil convergence summary:")
print(f"  Total polars      : {total_xfoil_polars:>7,}")
print(f"  Clean convergence : {n_clean:>7,}  ({100*n_clean/total_xfoil_polars:.1f}%)")
print(f"  Partial (NaN fill): {n_partial:>7,}  ({100*n_partial/total_xfoil_polars:.1f}%)")
print(f"  Full NF fallback  : {n_full_fallback:>7,}  ({100*n_full_fallback/total_xfoil_polars:.1f}%)")

if error_counts:
    print(f"\n  Top failure reasons (full-fallback polars):")
    for reason, count in sorted(error_counts.items(), key=lambda x: -x[1])[:10]:
        short = reason[:80] + ('…' if len(reason) > 80 else '')
        print(f"    {count:>6,}×  {short}")

df = pd.DataFrame({
    'name':                airfoils.tolist(),
    're':                  [RE.tolist()]     * n_af,
    'alpha':               [ALPHA.tolist()]  * n_af,
    'cl':                  cl.tolist(),
    'cd':                  cd.tolist(),
    'cl_cd':               cl_cd.tolist(),
    'cm':                  cm.tolist(),
    'top_xtr':             top_xtr.tolist(),
    'bot_xtr':             bot_xtr.tolist(),
    'mach_crit':           mach_crit.tolist(),
    'max_cl':              max_cl.tolist(),
    'min_cd':              min_cd.tolist(),
    'thickness':           max_thickness.tolist(),
    'camber':              camber_arr.tolist(),
    'analysis_confidence': mean_confidence.tolist(),
})

df.to_pickle(OUT_PKL)
print(f"\nDatabase saved -> {OUT_PKL}")
print(f"  Airfoils : {n_af}")
print(f"  Re grid  : {n_re} points  ({RE[0]:.0e} – {RE[-1]:.0e})")
print(f"    XFoil  : {n_re_xf} points  ({RE_XFOIL[0]:.0e} – {RE_XFOIL[-1]:.0e})")
print(f"    NeuralFoil: {n_re_nf} points  ({RE_NEURAL[0]:.0e} – {RE_NEURAL[-1]:.0e})")
print(f"  Alpha    : {n_alpha} points  ({ALPHA[0]:.1f}° – {ALPHA[-1]:.1f}°)")
print(f"  File size: {OUT_PKL.stat().st_size / 1e6:.1f} MB")