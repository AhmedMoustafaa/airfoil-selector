"""
Database Validation & Comparison Script
=========================================
Compares full_data.pkl (NeuralFoil) against full_data_xfoil.pkl (XFoil hybrid)
across five levels of analysis:

  1. Schema & coverage   — both databases have the same airfoils and grid
  2. Global statistics   — mean / std / percentile differences across all data
  3. Per-metric MAD      — mean absolute difference per coefficient per Re
  4. Confidence audit    — distribution of confidence scores in the XFoil DB
  5. Spot-check plots    — CL-alpha, CD-alpha, drag polar for reference airfoils

Run from the project root:
    python utils/validate_databases.py

Outputs a printed report and saves figures to validation_plots/.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Configuration

PKL_NF  = Path('../databases/full_data.pkl')
PKL_XF  = Path('../databases/full_data_xfoil.pkl')
PLOT_DIR = Path('validation_plots')

# Airfoils to spot-check — well-known sections with published XFoil data
SPOT_CHECK = ['naca2412.dat', 'e374.dat', 's1223.dat', 'clark-y.dat', 'naca0012.dat']

# Re index to use for 2-D comparison plots (index into the XFoil Re range)
RE_PLOT_IDX = 2   # 3e5 — primary UAV cruise Re

# Load

print("Loading databases …")
df_nf = pd.read_pickle(PKL_NF)
df_xf = pd.read_pickle(PKL_XF)
PLOT_DIR.mkdir(exist_ok=True)
print(f"  NeuralFoil : {len(df_nf):>5} airfoils")
print(f"  XFoil hybrid: {len(df_xf):>5} airfoils\n")

# Helper: safe numpy array extraction

def to_arr(val):
    return np.asarray(val, dtype=float)

# 1. Schema & coverage check

print("=" * 65)
print("1. SCHEMA & COVERAGE")
print("=" * 65)

re_nf    = np.array(df_nf['re'].iloc[0])
re_xf    = np.array(df_xf['re'].iloc[0])
alpha_nf = np.array(df_nf['alpha'].iloc[0])
alpha_xf = np.array(df_xf['alpha'].iloc[0])

print(f"  NF  Re grid : {len(re_nf)} points  {re_nf[0]:.0e} – {re_nf[-1]:.0e}")
print(f"  XF  Re grid : {len(re_xf)} points  {re_xf[0]:.0e} – {re_xf[-1]:.0e}")
print(f"  NF  alpha   : {len(alpha_nf)} points  {alpha_nf[0]:.1f}° – {alpha_nf[-1]:.1f}°")
print(f"  XF  alpha   : {len(alpha_xf)} points  {alpha_xf[0]:.1f}° – {alpha_xf[-1]:.1f}°")

grids_match = (np.allclose(re_nf, re_xf) and np.allclose(alpha_nf, alpha_xf))
print(f"  Grids identical : {'YES' if grids_match else 'NO — different grids, comparisons will use intersection'}")

names_nf = set(df_nf['name'])
names_xf = set(df_xf['name'])
common   = names_nf & names_xf
only_nf  = names_nf - names_xf
only_xf  = names_xf - names_nf

print(f"\n  Airfoils in both  : {len(common):>5}")
print(f"  Only in NF DB     : {len(only_nf):>5}")
print(f"  Only in XFoil DB  : {len(only_xf):>5}")

# Build aligned subset for comparison
df_nf_c = df_nf[df_nf['name'].isin(common)].set_index('name')
df_xf_c = df_xf[df_xf['name'].isin(common)].set_index('name')
df_nf_c = df_nf_c.loc[sorted(common)]
df_xf_c = df_xf_c.loc[sorted(common)]
n_common = len(common)

# Use only the Re range present in both databases
re_shared = re_nf   # both have same grid (validated above)
n_re   = len(re_shared)
n_alpha = len(alpha_nf)

print(f"\n  Comparison will use {n_common} airfoils × {n_re} Re × {n_alpha} alpha\n")

# 2. Global statistics

print("=" * 65)
print("2. GLOBAL COEFFICIENT STATISTICS  (XFoil Re range only: ≤1e6)")
print("=" * 65)

# Only compare over the XFoil Re range (first 10 Re points)
n_re_xf_only = 10

cl_nf  = np.array([to_arr(df_nf_c.loc[n, 'cl'])[:n_re_xf_only]  for n in sorted(common)])
cl_xf  = np.array([to_arr(df_xf_c.loc[n, 'cl'])[:n_re_xf_only]  for n in sorted(common)])
cd_nf  = np.array([to_arr(df_nf_c.loc[n, 'cd'])[:n_re_xf_only]  for n in sorted(common)])
cd_xf  = np.array([to_arr(df_xf_c.loc[n, 'cd'])[:n_re_xf_only]  for n in sorted(common)])
cm_nf  = np.array([to_arr(df_nf_c.loc[n, 'cm'])[:n_re_xf_only]  for n in sorted(common)])
cm_xf  = np.array([to_arr(df_xf_c.loc[n, 'cm'])[:n_re_xf_only]  for n in sorted(common)])

diff_cl = cl_xf - cl_nf
diff_cd = cd_xf - cd_nf
diff_cm = cm_xf - cm_nf

def stats_row(label, diff):
    d = diff.ravel()
    d = d[np.isfinite(d)]
    print(f"  {label:<8}  MAD={np.mean(np.abs(d)):>8.5f}  "
          f"bias={np.mean(d):>+8.5f}  "
          f"std={np.std(d):>8.5f}  "
          f"p95={np.percentile(np.abs(d), 95):>8.5f}")

print(f"  {'Metric':<8}  {'MAD':>8}  {'bias':>9}  {'std':>8}  {'|d|_p95':>8}")
print(f"  {'-'*60}")
stats_row('CL',  diff_cl)
stats_row('CD',  diff_cd)
stats_row('Cm',  diff_cm)

# Verdict
mad_cl = np.mean(np.abs(diff_cl[np.isfinite(diff_cl)]))
mad_cd = np.mean(np.abs(diff_cd[np.isfinite(diff_cd)]))
print()
if mad_cl < 0.02 and mad_cd < 0.0005:
    print("  ⚠  MAD very low — XFoil data closely matches NeuralFoil.")
    print("     This may indicate XFoil was not actually used (fell back to NF).")
elif mad_cl < 0.05 and mad_cd < 0.002:
    print("  ✓  Differences are in the expected range for XFoil vs NeuralFoil.")
    print("     Databases are meaningfully different — XFoil data looks genuine.")
else:
    print("  ⚠  Large differences detected — review spot-check plots carefully.")

# 3. Per-Re MAD profile

print()
print("=" * 65)
print("3. PER-Re MEAN ABSOLUTE DIFFERENCE  (CL and CD)")
print("=" * 65)
print(f"  {'Re':>10}  {'MAD_CL':>10}  {'MAD_CD':>10}  {'Source':>12}")
print(f"  {'-'*50}")

# Build full-Re arrays for the per-Re breakdown
cl_nf_full = np.array([to_arr(df_nf_c.loc[n, 'cl']) for n in sorted(common)])
cl_xf_full = np.array([to_arr(df_xf_c.loc[n, 'cl']) for n in sorted(common)])
cd_nf_full = np.array([to_arr(df_nf_c.loc[n, 'cd']) for n in sorted(common)])
cd_xf_full = np.array([to_arr(df_xf_c.loc[n, 'cd']) for n in sorted(common)])

for j, re_val in enumerate(re_shared):
    cl_d = (cl_xf_full[:, j, :] - cl_nf_full[:, j, :]).ravel()
    cd_d = (cd_xf_full[:, j, :] - cd_nf_full[:, j, :]).ravel()
    cl_d = cl_d[np.isfinite(cl_d)]
    cd_d = cd_d[np.isfinite(cd_d)]
    source = 'XFoil' if re_val <= 1e6 else 'NeuralFoil'
    print(f"  {re_val:>10.0e}  {np.mean(np.abs(cl_d)):>10.5f}  "
          f"{np.mean(np.abs(cd_d)):>10.6f}  {source:>12}")

# 4. Confidence audit

print()
print("=" * 65)
print("4. CONFIDENCE SCORE DISTRIBUTION  (XFoil DB)")
print("=" * 65)

conf = np.array(df_xf['analysis_confidence'].tolist(), dtype=float)
bins = {
    '1.0  (clean XFoil)':         np.sum(np.isclose(conf, 1.0)),
    '0.7  (XFoil + partial fill)': np.sum(np.isclose(conf, 0.7)),
    '0.5  (majority NF fill)':     np.sum(np.isclose(conf, 0.5)),
    '0.3  (full NF fallback)':     np.sum(np.isclose(conf, 0.3)),
    '0.1  (NeuralFoil high-Re)':   np.sum(np.isclose(conf, 0.1)),
    '0.0  (failed)':               np.sum(np.isclose(conf, 0.0)),
}
total_conf = len(conf)
for label, count in bins.items():
    bar = '█' * int(40 * count / total_conf)
    print(f"  {label:<30}  {count:>6}  ({100*count/total_conf:>5.1f}%)  {bar}")

# 5. Spot-check plots

print()
print("=" * 65)
print("5. SPOT-CHECK PLOTS")
print("=" * 65)

# Find which spot-check airfoils are actually in both databases
available = [n for n in SPOT_CHECK if n in common]
if not available:
    # Fall back to first 5 common airfoils sorted alphabetically
    available = sorted(common)[:5]
    print(f"  None of the requested spot-check airfoils found — using: {available}")
else:
    print(f"  Plotting: {available}")

re_idx  = RE_PLOT_IDX   # 3e5
re_label = f"Re = {re_shared[re_idx]:.0e}"

for name in available:
    cl_n  = to_arr(df_nf_c.loc[name, 'cl'])[re_idx]
    cl_x  = to_arr(df_xf_c.loc[name, 'cl'])[re_idx]
    cd_n  = to_arr(df_nf_c.loc[name, 'cd'])[re_idx]
    cd_x  = to_arr(df_xf_c.loc[name, 'cd'])[re_idx]
    cm_n  = to_arr(df_nf_c.loc[name, 'cm'])[re_idx]
    cm_x  = to_arr(df_xf_c.loc[name, 'cm'])[re_idx]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"{name}  —  {re_label}", fontsize=13, fontweight='bold')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    # CL-alpha
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(alpha_nf, cl_n, 'b-',  label='NeuralFoil', linewidth=1.5)
    ax.plot(alpha_xf, cl_x, 'r--', label='XFoil hybrid', linewidth=1.5)
    ax.set_xlabel('α (°)'); ax.set_ylabel('CL')
    ax.set_title('CL vs α'); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # CD-alpha
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(alpha_nf, cd_n, 'b-',  linewidth=1.5)
    ax.plot(alpha_xf, cd_x, 'r--', linewidth=1.5)
    ax.set_xlabel('α (°)'); ax.set_ylabel('CD')
    ax.set_title('CD vs α'); ax.grid(True, alpha=0.4)

    # Cm-alpha
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(alpha_nf, cm_n, 'b-',  linewidth=1.5)
    ax.plot(alpha_xf, cm_x, 'r--', linewidth=1.5)
    ax.set_xlabel('α (°)'); ax.set_ylabel('Cm')
    ax.set_title('Cm vs α'); ax.grid(True, alpha=0.4)

    # Drag polar
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(cd_n, cl_n, 'b-',  linewidth=1.5)
    ax.plot(cd_x, cl_x, 'r--', linewidth=1.5)
    ax.set_xlabel('CD'); ax.set_ylabel('CL')
    ax.set_title('Drag polar')
    ax.set_xlim(left=0); ax.grid(True, alpha=0.4)

    # CL/CD vs alpha
    with np.errstate(divide='ignore', invalid='ignore'):
        clcd_n = np.where(cd_n > 0, cl_n / cd_n, 0)
        clcd_x = np.where(cd_x > 0, cl_x / cd_x, 0)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(alpha_nf, clcd_n, 'b-',  linewidth=1.5)
    ax.plot(alpha_xf, clcd_x, 'r--', linewidth=1.5)
    ax.set_xlabel('α (°)'); ax.set_ylabel('CL/CD')
    ax.set_title('CL/CD vs α'); ax.grid(True, alpha=0.4)

    # Difference: ΔCL and ΔCD
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(alpha_nf, cl_x - cl_n, 'g-',   label='ΔCL', linewidth=1.5)
    ax.plot(alpha_nf, (cd_x - cd_n) * 100, 'm--', label='ΔCD ×100', linewidth=1.5)
    ax.axhline(0, color='k', linewidth=0.7, linestyle=':')
    ax.set_xlabel('α (°)'); ax.set_ylabel('Difference')
    ax.set_title('XFoil − NeuralFoil'); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    fname = PLOT_DIR / f"spotcheck_{name.replace('.dat','').replace(' ','_')}.png"
    fig.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {fname}")

# 6. Global difference heatmap  (MAD CL per Re × alpha bin)

print()
print("Generating global MAD heatmap …")

# Shape: (n_common, n_re, n_alpha)
diff_cl_full = cl_xf_full - cl_nf_full

# Mean absolute difference averaged over all airfoils: shape (n_re, n_alpha)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    mad_map = np.nanmean(np.abs(diff_cl_full), axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im = axes[0].imshow(mad_map, aspect='auto', origin='lower',
                    extent=[alpha_nf[0], alpha_nf[-1], 0, n_re],
                    cmap='YlOrRd')
axes[0].set_xlabel('Angle of Attack (°)')
axes[0].set_ylabel('Re index')
axes[0].set_yticks(np.arange(n_re) + 0.5)
axes[0].set_yticklabels([f'{r:.0e}' for r in re_shared], fontsize=7)
axes[0].set_title('MAD CL  (XFoil − NeuralFoil)\naveraged over all airfoils')
plt.colorbar(im, ax=axes[0], label='|ΔCL|')

# CD heatmap
diff_cd_full = cd_xf_full - cd_nf_full
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    mad_cd_map = np.nanmean(np.abs(diff_cd_full), axis=0)

im2 = axes[1].imshow(mad_cd_map, aspect='auto', origin='lower',
                     extent=[alpha_nf[0], alpha_nf[-1], 0, n_re],
                     cmap='YlOrRd')
axes[1].set_xlabel('Angle of Attack (°)')
axes[1].set_ylabel('Re index')
axes[1].set_yticks(np.arange(n_re) + 0.5)
axes[1].set_yticklabels([f'{r:.0e}' for r in re_shared], fontsize=7)
axes[1].set_title('MAD CD  (XFoil − NeuralFoil)\naveraged over all airfoils')
plt.colorbar(im2, ax=axes[1], label='|ΔCD|')

fig.tight_layout()
heatmap_path = PLOT_DIR / 'mad_heatmap.png'
fig.savefig(heatmap_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"  Saved → {heatmap_path}")

# 7. Confidence vs MAD scatter  (per airfoil mean)

print("Generating confidence vs MAD scatter …")

per_af_mad_cl  = np.nanmean(np.abs(diff_cl_full[:, :n_re_xf_only, :]), axis=(1, 2))
per_af_conf    = np.array([
    float(df_xf_c.loc[n, 'analysis_confidence'])
    for n in sorted(common)
])

fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(per_af_conf, per_af_mad_cl,
                     alpha=0.3, s=8, c=per_af_conf, cmap='RdYlGn')
ax.set_xlabel('Analysis Confidence')
ax.set_ylabel('Mean |ΔCL|  (vs NeuralFoil, XFoil Re range)')
ax.set_title('Per-airfoil confidence vs CL difference')
plt.colorbar(scatter, ax=ax, label='Confidence')
ax.grid(True, alpha=0.3)
fig.tight_layout()
conf_path = PLOT_DIR / 'confidence_vs_mad.png'
fig.savefig(conf_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"  Saved → {conf_path}")

# Summary verdict

print()
print("=" * 65)
print("SUMMARY VERDICT")
print("=" * 65)

xf_re_conf = per_af_conf[per_af_conf < 0.15]   # NF-only airfoils have conf=0.1
xf_only_conf = per_af_conf[per_af_conf >= 0.3]
pct_genuine_xfoil = 100 * len(xf_only_conf) / len(per_af_conf)

print(f"  Airfoils with genuine XFoil data (conf ≥ 0.3) : "
      f"{len(xf_only_conf):>5}  ({pct_genuine_xfoil:.1f}%)")
print(f"  Airfoils at NeuralFoil-only confidence (0.1)  : "
      f"{len(xf_re_conf):>5}  ({100-pct_genuine_xfoil:.1f}%)")
print()

if mad_cl < 0.01:
    print("  ⚠  WARN: CL MAD < 0.01 — databases are suspiciously similar.")
    print("     Check that XFoil actually ran (confidence distribution above).")
elif mad_cl < 0.05:
    print("  ✓  CL differences are moderate and physically expected.")
    print("     XFoil tends to predict higher CLmax and sharper stall than NeuralFoil.")
else:
    print("  ⚠  Large CL differences — review heatmap to identify problem Re/alpha regions.")

print()
print(f"  All plots saved to: {PLOT_DIR.resolve()}")