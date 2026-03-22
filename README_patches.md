# Patches & Modifications

This file documents every change made to third-party code to get the XFoil
hybrid database generation working on Ubuntu with gfortran.

---

## 1. XFoil 6.97  (`xfoil/xfoil_uav`)

The apt-packaged XFoil binary on Ubuntu (`sudo apt install xfoil`) is compiled
with `-ftrapuv -fpe0`, which enables strict Fortran FPE trapping. This causes
XFoil to raise `SIGFPE` and crash on floating-point edge cases that the solver
is designed to handle gracefully — specifically inside the boundary layer
marching routines at high AOA or low Re.

The binary included in `xfoil/xfoil_uav` is XFoil 6.97 compiled from source
with gfortran and the following Makefile flags added to `bin/Makefile` and
`plotlib/Makefile`:

```makefile
FC     = gfortran
FFLAGS = -O2 -ffpe-trap=none -fdefault-real-8 -fno-range-check -fallow-argument-mismatch
FFLOPT = -O2 -ffpe-trap=none -fdefault-real-8 -fno-range-check -fallow-argument-mismatch
```

Flag rationale:
- `-ffpe-trap=none` — disables FPE trapping, the main fix
- `-fdefault-real-8` — matches ifort's default double precision behaviour
- `-fno-range-check` — suppresses `2**31` integer overflow in `src/xoper.f`
- `-fallow-argument-mismatch` — suppresses rank mismatch errors in `src/xgdes.f`

### Source patch: `src/iopol.f`

gfortran requires an explicit file reposition before writing after EOF.
ifort handled this silently. Without this patch XFoil writes only the first
converged alpha point and crashes with a Fortran runtime error on the second.

Around line 649, immediately before the `DO 40` loop:

```fortran
! --- PATCH: gfortran requires BACKSPACE before writing after EOF ---
      BACKSPACE(LU)
! -------------------------------------------------------------------
      DO 40 IA = IA1, IA2
        WRITE(LU,LINEF)
     &         (CPOL(IA,IPOL(KP)), KP=1, NIPOL), ...
   40 CONTINUE
```

### Platform note

The compiled binary targets Linux x86_64. It will not run on macOS, Windows,
or ARM. On those platforms either build from source using the flags above, or
use the NeuralFoil-only database (`full_data.pkl`) instead.

---

## 2. AeroSandbox XFoil wrapper (`aerosandbox/aerodynamics/aero_2D/xfoil.py`)

Two patches are needed for XFoil 6.97 compatibility on Ubuntu. They are
applied by running `utils/patch_aerosandbox.py` once after installing
AeroSandbox.

These patches were tested against AeroSandbox as installed in early 2025.
They may not be needed with other versions — `patch_aerosandbox.py` will warn
if the installed version differs and lets you decide whether to proceed.

### Patch A — disable `cinc` command

XFoil 6.97 writes `Cpmin` to polar data rows when `cinc` is active, but does
not add it to the header line. This causes AeroSandbox's polar parser to raise
a column count mismatch error.

In `_default_keystrokes`, comment out the `cinc` line:

```python
# run_file_contents += ["cinc"]  # disabled: XFoil 6.97 adds Cpmin to data but not header
```

`Cpmin` and `Xcpmin` will be empty arrays in the result dict. They are not
used by `AirfoilSelector`.

### Patch B — fix sort step for empty arrays

When some output arrays are empty (e.g. `Cpmin` after disabling `cinc`),
the sort step at the end of `alpha()` and `cl()` raises `IndexError` because
it tries to index an empty array with a full-length sort order.

In both `alpha()` and `cl()`, change:

```python
# before
output = {k: v[sort_order] for k, v in output.items()}

# after
output = {k: (v[sort_order] if len(v) == len(sort_order) else v)
          for k, v in output.items()}
```

This is a general correctness fix independent of XFoil version. An upstream
PR for this patch has been submitted to the AeroSandbox repository.