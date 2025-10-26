#!/usr/bin/env python3
# Transition state optimization with Psi4 using ωB97X-D3
# Save as: ts_opt_psi4.py
# Run:     psi4 ts_opt_psi4.py -n 8    (example with 8 threads)
#
# Notes:
# - Replace the placeholder geometry with your guess for the TS.
# - Psi4 is CPU-based; there is no GPU acceleration in Psi4 as of now.
# - After the TS search, we do a frequency calc to verify a single imaginary mode.

import psi4
import sys
from pathlib import Path

# ---------- User settings ----------
# Memory & I/O
psi4.set_memory("8 GB")
psi4.core.set_output_file("psi4_ts.out", False)

# Threads (can also be passed with -n on the command line)
psi4.set_num_threads(max(1, psi4.get_num_threads()))

# Method & basis
method = "wb97x-d3"     # ωB97X-D3
basis  = "def2-tzvp"

# Convergence / optimizer controls
psi4.set_options({
    # SCF / DFT
    "basis": basis,
    "scf_type": "df",            # density fitting for speed (recommended with def2- basis)
    "guess": "sad",              # superposition of atomic densities
    "d_convergence": 4e-3,
    "e_convergence": 5e-6,
    "maxiter": 200,

    # Geometry optimization (OptKing)
    "opt_type": "ts",            # <-- transition-state optimization
    "geom_maxiter": 200,
    "g_convergence": "gau_loose",
    "opt_coordinates": "internal",
    
})

# ---------- Molecule ----------

def read_xyz(path: Path):
    txt = path.read_text().strip().splitlines()
    if len(txt) < 2:
        raise ValueError("XYZ file too short")
    try:
        n = int(txt[0].strip())
    except Exception:
        raise ValueError("First line of XYZ must be atom count")
    
    # skip comment line (txt[1])
    atoms = []
    for line in txt[2+n+2:2+n+2+n]:
    # for line in txt[2:2+n]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Bad XYZ atom line: {line}")
        sym, x, y, z = parts[0], parts[1], parts[2], parts[3]
        atoms.append((sym, float(x), float(y), float(z)))
    return atoms

def atoms_to_psi4_geom(atoms, charge, multiplicity):
    lines = [f"{charge} {multiplicity}"]
    for sym, x, y, z in atoms:
        lines.append(f"{sym:2s}  {x: .10f}  {y: .10f}  {z: .10f}")
    lines += ["symmetry c1", "units angstrom"]
    return "\n".join(lines)

# Put your best TS *guess* below. Keep key forming/breaking bonds ~50% progressed.

import sys
inpath = Path(sys.argv[1])
atoms = read_xyz(inpath)
geom_str = atoms_to_psi4_geom(atoms, 0, 1)
mol = psi4.geometry(geom_str)
# Optional: force UHF for open-shell; for this example (singlet anion), RHF is fine.
# psi4.set_options({"reference": "uhf"})

# ---------- Run TS optimization ----------
print(">> Running TS optimization at {} / {}".format(method, basis))
ts_energy = psi4.optimize(f"{method}/{basis}", molecule=mol)
print("TS energy (Eh):", ts_energy)

# Save TS geometry
ts_xyz = mol.save_string_xyz()
Path("ts_optimized.xyz").write_text(ts_xyz)
print("Saved optimized TS geometry to ts_optimized.xyz")

# ---------- Verify with frequency analysis ----------
print(">> Running frequency analysis to verify exactly one imaginary mode")
freq_result = psi4.frequency(f"{method}/{basis}", molecule=mol, return_wfn=True)
wfn = freq_result[1]

# Extract vibrational frequencies (cm^-1)
# vibfreqs = list(wfn.frequency_analysis["frequencies"].data)
vibfreqs = wfn.frequencies().to_array()
n_imag = sum(1 for f in vibfreqs if f < 0.0)
print("Frequencies (cm^-1):", vibfreqs)
print("Number of imaginary modes:", n_imag)

# Write a small report
report = []
report.append(f"Method: {method} / {basis}")
report.append(f"TS Energy (Eh): {ts_energy:.12f}")
report.append(f"Number of imaginary frequencies: {n_imag}")
report.append("First 10 frequencies (cm^-1): " + ", ".join(f"{f:.1f}" for f in vibfreqs[:10]))
Path("ts_report.txt").write_text("\n".join(report))

print("Wrote summary to ts_report.txt")
print("Done.")
