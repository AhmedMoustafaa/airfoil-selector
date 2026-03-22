"""
Database Generation Script
===========================
Builds full_data.pkl — the airfoil performance database consumed by AirfoilSelector.
"""

import warnings
import pandas as pd
import numpy as np
import aerosandbox as asb
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Configuration — edit these to change the sweep

DB_DIR   = Path('../airfoilsdb')
OUT_PKL  = Path('../databases/full_data.pkl')

ALPHA = np.linspace(-5, 20, 26)          # AOA grid  (degrees)
RE    = np.array([                                  # Reynolds number grid
    1e5, 2e5, 3e5, 4e5, 5e5,
    6e5, 7e5, 8e5, 9e5, 1e6,
    2e6, 3e6, 4e6, 5e6, 6e6,
    7e6, 8e6, 9e6, 1e7,
])

MODEL_SIZE = 'large'    # NeuralFoil model: 'xxsmall' … 'large'
MACH       = 0.0

# collect valid airfoil filenames
def _load_airfoil(dat_path: Path) -> asb.Airfoil | None:
    """
    Try to return a usable asb.Airfoil for a given .dat file, using three
    strategies in order:

    1. Load directly from the .dat file path (most reliable when the file is
       well-formed — coordinates are read from disk, no name matching needed).
    2. Try ASB's internal database with the bare stem name (e.g. "e374" for
       "e374.dat"). Catches files whose content ASB cannot parse but whose name
       matches a built-in entry.
    3. Give up and return None.

    The coordinates-count guard (< 10 points) rejects degenerate files that
    load without error but contain too few points for NeuralFoil to use.
    """
    name = dat_path.name
    stem = dat_path.stem          # filename without extension

    # Strategy 1 — load from file path
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            af = asb.Airfoil(name, coordinates=str(dat_path))
            if af.coordinates is not None and len(af.coordinates) >= 10:
                return af
        except Exception:
            pass

    # Strategy 2 — fallback to ASB built-in database using the stem name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            af = asb.Airfoil(stem)
            if af.coordinates is not None and len(af.coordinates) >= 10:
                return af
        except Exception:
            pass

    return None

print("Scanning airfoil database directory …")

# name_map: filename (as stored in the DataFrame) -> asb.Airfoil object
# We keep the Airfoil objects here so the sweep loop reuses them without
# re-loading from disk a second time.
name_map: dict[str, asb.Airfoil] = {}
skipped: list[str] = []

for path in sorted(DB_DIR.glob("*")):
    af = _load_airfoil(path)
    if af is None:
        skipped.append(path.name)
    else:
        name_map[path.name] = af

if skipped:
    print(f"  Skipped {len(skipped)} unparseable files:")
    for s in skipped:
        print(f"    {s}")

print(f"  {len(name_map)} valid airfoils found.\n")

if not name_map:
    raise RuntimeError(f"No valid airfoils found in '{DB_DIR}'. "
                       "Check the directory path.")

valid_names = list(name_map.keys())
airfoils    = np.array(valid_names)

Alpha, Re_mesh = np.meshgrid(ALPHA, RE)

n_af    = len(airfoils)
n_re    = len(RE)
n_alpha = len(ALPHA)
shape   = (n_af, n_re, n_alpha)

cl         = np.zeros(shape)
cd         = np.zeros(shape)
cm         = np.zeros(shape)
cl_cd      = np.zeros(shape)
top_xtr    = np.zeros(shape)
bot_xtr    = np.zeros(shape)
mach_crit  = np.zeros(shape)

max_cl  = np.zeros((n_af, n_re))
min_cd  = np.zeros((n_af, n_re))

max_thickness       = np.zeros(n_af)
camber_arr          = np.zeros(n_af)
analysis_confidence = np.zeros(n_af)

print("Running NeuralFoil sweep …")
iterator = tqdm(enumerate(airfoils), total=n_af) if HAS_TQDM else enumerate(airfoils)

for i, name in iterator:
    if not HAS_TQDM:
        print(f"  [{i+1}/{n_af}]  {name}")

    try:
        af = name_map[name]   # reuse the object validated during scanning

        aero = af.get_aero_from_neuralfoil(
            alpha = Alpha.flatten(),
            Re    = Re_mesh.flatten(),
            mach  = MACH,
            model_size = MODEL_SIZE,
        )
        # Reshape back to (n_re, n_alpha)
        Aero = {k: v.reshape(Alpha.shape) for k, v in aero.items()}

        cl[i]  = Aero['CL']
        cd[i]  = Aero['CD']
        cm[i]  = Aero['CM']

        # Safe CL/CD: avoid division by zero or negative CD from the model
        with np.errstate(divide='ignore', invalid='ignore'):
            cl_cd[i] = np.where(Aero['CD'] > 0, Aero['CL'] / Aero['CD'], 0.0)

        top_xtr[i]   = Aero['Top_Xtr']
        bot_xtr[i]   = Aero['Bot_Xtr']
        mach_crit[i] = Aero['mach_crit']

        # Per-Re summary stats (axis=1 -> over alpha)
        max_cl[i] = np.max(cl[i],  axis=1)
        min_cd[i] = np.min(cd[i],  axis=1)

        max_thickness[i]       = af.max_thickness()
        camber_arr[i]          = af.max_camber()
        analysis_confidence[i] = float(np.mean(Aero['analysis_confidence']))

    except Exception as exc:
        print(f"\n  WARNING: failed for {name}: {exc}. Arrays left as zero.")

# Store re and alpha as scalar columns (same value in every row) so the
# AirfoilSelector can read the grid without any hardcoded values.
df = pd.DataFrame({
    'name':                 airfoils.tolist(),
    # Grid metadata — one list per row (identical across rows)
    're':                   [RE.tolist()]    * n_af,
    'alpha':                [ALPHA.tolist()] * n_af,
    # 3-D aerodynamic data — stored as nested lists (shape: n_re × n_alpha)
    'cl':                   cl.tolist(),
    'cd':                   cd.tolist(),
    'cl_cd':                cl_cd.tolist(),
    'cm':                   cm.tolist(),
    'top_xtr':              top_xtr.tolist(),
    'bot_xtr':              bot_xtr.tolist(),
    'mach_crit':            mach_crit.tolist(),
    # Per-Re summary stats — stored as 1-D lists (shape: n_re)
    'max_cl':               max_cl.tolist(),
    'min_cd':               min_cd.tolist(),
    # Scalar geometry
    'thickness':            max_thickness.tolist(),
    'camber':               camber_arr.tolist(),
    'analysis_confidence':  analysis_confidence.tolist(),
})

df.to_pickle(OUT_PKL)
#df.to_excel('full_data.xlsx')
print(f"\nDatabase saved -> {OUT_PKL}")
print(f"  Airfoils : {n_af}")
print(f"  Re grid  : {n_re} points  ({RE[0]:.0e} – {RE[-1]:.0e})")
print(f"  Alpha    : {n_alpha} points  ({ALPHA[0]:.1f}° – {ALPHA[-1]:.1f}°)")
print(f"  File size: {OUT_PKL.stat().st_size / 1e6:.1f} MB")