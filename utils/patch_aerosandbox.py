"""
Patches AeroSandbox's XFoil wrapper for compatibility with XFoil 6.97 on Ubuntu.

Run once after installing AeroSandbox:
    python utils/patch_aerosandbox.py

See README_patches.md for a full explanation of what is changed and why.
These patches were developed against AeroSandbox as installed in early 2025.
They may not be needed with other AeroSandbox versions, and may conflict with
future ones. The script will warn if the installed version doesn't match the
tested one and ask before proceeding.
"""

import sys
import importlib
from pathlib import Path

TESTED_ASB_VERSION = "0.9"   # major.minor prefix we tested against

PATCH_A_OLD = '        run_file_contents += ["cinc"]  # include minimum Cp'
PATCH_A_NEW = '        # run_file_contents += ["cinc"]  # disabled: XFoil 6.97 adds Cpmin to data but not header'

PATCH_B_OLD = '        output = {k: v[sort_order] for k, v in output.items()}'
PATCH_B_NEW = ('        output = {k: (v[sort_order] if len(v) == len(sort_order) else v)\n'
               '                  for k, v in output.items()}')


def find_xfoil_py():
    try:
        import aerosandbox
    except ImportError:
        print("ERROR: aerosandbox is not installed. Run: pip install aerosandbox")
        sys.exit(1)

    path = Path(aerosandbox.__file__).parent / "aerodynamics" / "aero_2D" / "xfoil.py"
    if not path.exists():
        print(f"ERROR: Could not find xfoil.py at expected path:\n  {path}")
        sys.exit(1)
    return path


def check_version():
    try:
        import aerosandbox
        version = getattr(aerosandbox, "__version__", "unknown")
    except ImportError:
        version = "unknown"

    if not version.startswith(TESTED_ASB_VERSION):
        print(f"WARNING: These patches were tested against AeroSandbox {TESTED_ASB_VERSION}.x")
        print(f"         Installed version: {version}")
        print(f"         The patches may not apply cleanly or may not be needed.")
        answer = input("Proceed anyway? [y/N] ").strip().lower()
        if answer != 'y':
            print("Aborted.")
            sys.exit(0)
    return version


def apply_patch(src, old, new, label):
    if old not in src:
        if new.strip().split('\n')[0].strip() in src or new in src:
            return src, 'already_applied'
        return src, 'not_found'
    return src.replace(old, new, 1), 'applied'


def main():
    print("AeroSandbox XFoil patcher for XFoil 6.97 / Ubuntu\n")

    version = check_version()
    xfoil_py = find_xfoil_py()
    print(f"Target file: {xfoil_py}")

    with open(xfoil_py) as f:
        src = f.read()

    original_src = src
    results = {}

    src, results['A'] = apply_patch(src, PATCH_A_OLD, PATCH_A_NEW,
                                     'A: disable cinc command')
    # Patch B appears twice (in alpha() and cl()) — apply both
    src, r1 = apply_patch(src, PATCH_B_OLD, PATCH_B_NEW, 'B1: sort step in alpha()')
    src, r2 = apply_patch(src, PATCH_B_OLD, PATCH_B_NEW, 'B2: sort step in cl()')
    results['B'] = 'applied' if 'applied' in (r1, r2) else r1

    if src == original_src:
        print("\nNo changes made — all patches already applied or patterns not found.")
    else:
        # Back up original
        backup = xfoil_py.with_suffix('.py.pre_patch')
        if not backup.exists():
            backup.write_text(original_src)
            print(f"Backup saved to: {backup}")

        with open(xfoil_py, 'w') as f:
            f.write(src)

    print()
    for label, status in results.items():
        icon = {'applied': 'APPLIED', 'already_applied': 'already applied', 'not_found': 'NOT FOUND'}.get(status, status)
        print(f"  Patch {label}: {icon}")

    if 'not_found' in results.values():
        print()
        print("WARNING: One or more patches could not be applied — the source file")
        print("  structure may have changed. Check README_patches.md and apply manually.")
        sys.exit(1)

    print()
    print("Done. Run a quick test to confirm:")
    print("  python -c \"")
    print("  import aerosandbox as asb, numpy as np")
    print("  af = asb.Airfoil('naca2412').repanel(n_points_per_side=100)")
    print("  xf = asb.XFoil(af, Re=3e5, xfoil_command='xfoil/xfoil_uav')")
    print("  print(xf.alpha(np.linspace(-5, 15, 5)))")
    print("  \"")


if __name__ == "__main__":
    main()