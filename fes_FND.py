
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.special import logsumexp, expit
import matplotlib.pyplot as plt

# Folder containing FND-inference.py outputs:
#   Logp_<idx>.txt
#   Uzs_<idx>.txt
#   optionally generated structures / target energies
out_dir = Path("./data/")

# Temperature conversion for reduced free energy -> eV/atom.
# If your u = U/kBT, then ΔF[eV] = Δf_reduced * T_in_eV.
temperature_K = 1600
T_in_eV = 8.617333262145e-5 * temperature_K
crystal_type = 'coesite'
num_atoms = 864

# Provide target reduced energies u_B(x_1) for the generated final structures.
# Supported formats:
# 1. None: notebook will load forward logp and prior energy, then stop before ΔF.
# 2. One text file with one u_B per rollout/sample.
# 3. A directory pattern handled in load_target_uB below.

def _idx_from_name(path: Path) -> int:
    m = re.search(r"_(\d+)\.(?:txt|dat)$", path.name)
    if not m:
        raise ValueError(f"Cannot parse integer index from {path.name}")
    return int(m.group(1))


def as_1d(x):
    return np.asarray(x, dtype=float).reshape(-1)


def logmeanexp(x):
    x = as_1d(x)
    return logsumexp(x) - np.log(x.size)


def ess_from_logw(logw):
    logw = as_1d(logw)
    return np.exp(2.0 * logsumexp(logw) - logsumexp(2.0 * logw))


def split_saved_laststep_logp(path: Path):
    """
    Current FND-inference.py saves:
        np.savetxt(filename_logp, logp.view(1, -1).numpy())

    Current fed_wrapper.py returns:
        logp = concatenate([samples_logp, _samples_logp], dim=-1)

    This loader handles either a saved 2-column array or a flattened one-row array.

    Returns
    -------
    logq_plus, logq_minus : 1D arrays
        logq_plus is the forward sampling transition log-probability.
        logq_minus is the reverse/opposite transition log-probability evaluated on the same forward path.
    """
    arr = np.loadtxt(path)
    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 0:
        raise ValueError(f"{path} contains only one scalar; expected two FND logp parts.")

    # Case A: saved as N x 2 or 1 x 2
    if arr.ndim == 2 and arr.shape[-1] == 2:
        return arr[..., 0].reshape(-1), arr[..., 1].reshape(-1)

    flat = arr.reshape(-1)
    if flat.size % 2 != 0:
        raise ValueError(f"{path} has {flat.size} values; cannot split into two equal FND logp parts.")

    half = flat.size // 2
    return flat[:half], flat[half:]

def split_saved_fnd_logp(path: Path):
    """
    Load three FND log-probability components.

    Handles either:
        - an N x 3 saved array, or
        - a flattened one-row array containing three concatenated parts.

    Returns
    -------
    logq_plus, logq_minus, logq_third : 1D arrays
        The three saved FND log-probability components.
    """
    arr = np.loadtxt(path)
    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 0:
        raise ValueError(
            f"{path} contains only one scalar; expected three FND logp parts."
        )

    # Case A: saved as N x 3 or 1 x 3
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return (
            arr[..., 0].reshape(-1),
            arr[..., 1].reshape(-1),
            arr[..., 2].reshape(-1),
        )
    elif arr.ndim == 1 and arr.shape[-1] == 3:
        return arr[0:1], arr[1:2], arr[2:3]
    else:
        raise ValueError(
            f"{path} has shape {arr.shape}; expected either N x 3 or 1 x 3 array."
        )

def load_uA_from_uzs(path: Path):
    """
    Current FND-inference.py writes one prior reduced energy in Uzs_<idx>.txt:
        [[ ((zs @ cell)**2 / (2*x0std**2)).sum() ]]

    Older EJE notebooks sometimes had more columns. Here we take the first column as u_A(x0).
    """
    arr = np.loadtxt(path)
    arr = np.asarray(arr, dtype=float)
    flat = arr.reshape(-1)
    if flat.size < 1:
        raise ValueError(f"{path} is empty")
    return float(flat[0])


def load_target_uB(target_uB_file):
    if target_uB_file is None:
        return None
    target_uB_file = Path(target_uB_file)
    vals = np.loadtxt(target_uB_file)/T_in_eV

    return vals


logp_files = sorted(
    [
        p for p in out_dir.glob("Logp_*.txt")
        if not p.name.startswith("Logp_grad_last_step_")
    ],
    key=_idx_from_name
)
laststep_logp_files = sorted(out_dir.glob("Logp_grad_last_step_*.txt"), key=_idx_from_name)
uzs_files = {_idx_from_name(p): p for p in out_dir.glob("Uzs_*.txt")}
uB_files = {_idx_from_name(p): p for p in out_dir.glob("all_energy_atoms_*.dat")}

if len(logp_files) == 0:
    raise FileNotFoundError(f"No Logp_*.txt files found in {out_dir.resolve()}")

import sys
rows = []
rows_eV = []
for p in logp_files:
    idx = _idx_from_name(p)
    if idx not in uzs_files:
        raise FileNotFoundError(f"Missing Uzs_{idx}.txt for {p.name}")
    logq_plus, logq_minus, logq0 = split_saved_fnd_logp(p)
    if len(laststep_logp_files) > 0:
        laststep_logq, logstep_logq_corr = split_saved_laststep_logp(laststep_logp_files[idx]) 
    # uA = load_uA_from_uzs(uzs_files[idx])
    uB = load_target_uB(uB_files[idx])
    if logq_plus.size != logq_minus.size:
        raise RuntimeError("Internal split error")

    # Usually one rollout per file. If not, keep sub-index j.
    for j, (lp, lm, l0) in enumerate(zip(logq_plus, logq_minus, logq0)):
        rows.append({
            "idx": idx,
            "subidx": j,
            "logq_plus": lp,
            "logq_minus": lm,
            "logq0": l0,
            "laststep_logq": laststep_logq[j] if len(laststep_logp_files) > 0 else None,
            "logstep_logq_corr": logstep_logq_corr[j] if len(laststep_logp_files) > 0 else None,
            "uB_x1": uB,
        })

        rows_eV.append({
            "idx": idx,
            "subidx": j,
            "logq_plus": lp * T_in_eV/num_atoms,
            "logq_minus": lm * T_in_eV/num_atoms,
            "logq0": l0 * T_in_eV/num_atoms,
            "laststep_logq": laststep_logq[j] * T_in_eV/num_atoms if len(laststep_logp_files) > 0 else None,
            "logstep_logq_corr": logstep_logq_corr[j] * T_in_eV/num_atoms if len(laststep_logp_files) > 0 else None,
            "uB_x1": uB * T_in_eV/num_atoms,
        })
    if len(rows) >= int(sys.argv[1]):
        break
idx_min_uB = np.argmin(np.array([rows[i]['uB_x1'] for i in range(len(rows))]))
rows.pop(idx_min_uB)
rows_eV.pop(idx_min_uB)
df = pd.DataFrame(rows).sort_values(["idx", "subidx"]).reset_index(drop=True)
# print("Loaded rows:", len(df))

df.head()

df_eV = pd.DataFrame(rows_eV).sort_values(["idx", "subidx"]).reset_index(drop=True)

ref_ex = np.loadtxt(f"/home/tuoping/odefed_mdgen/workdir_odefed_mdgen/data/SiO2/npt_1600K_1GPa/npt_quartz_dense/npt/thermo-lammps.dat")[100:,2]
ref_ex_c = np.loadtxt(f"/home/tuoping/odefed_mdgen/workdir_odefed_mdgen/data/SiO2/npt_1600K_1GPa/npt_coesite_dense/npt/thermo-lammps.dat")[100:,2]

'''
plt.figure(figsize=(5, 3.5))
plt.hist(df_eV["uB_x1"].to_numpy(), bins=100, alpha=0.8, density=True, label=f"{temperature_K} K (FM)")
plt.axvline(df_eV["uB_x1"].to_numpy().max(), ls='--', c='k')
plt.axvline(df_eV["uB_x1"].to_numpy().min(), ls='--', c='k')
plt.axvline(np.median(df_eV["uB_x1"].to_numpy()), ls='dotted', c='k')

_ = plt.hist(ref_ex/1125, bins=100, alpha=0.5, color='r', density=True, label="Quartz 1600 K")
plt.axvline(ref_ex.max()/1125, ls='--', c='r')
plt.axvline(ref_ex.min()/1125, ls='--', c='r')


_ = plt.hist(ref_ex_c/864, bins=100, alpha=0.5, color='green', density=True, label="Coesite 1600 K")
plt.axvline(ref_ex_c.max()/864, ls='--', c='green')
plt.axvline(ref_ex_c.min()/864, ls='--', c='green')

plt.xlabel(r"$E_{x_1}$")
plt.ylabel("Fraction")
plt.legend()
plt.xlim(-326.9, -326.48)
plt.tight_layout()
plt.savefig("x_1.png")
'''
if len(laststep_logp_files) > 0:
    df["W_fwd"] = (
        df["uB_x1"].to_numpy()
        + df["logq0"].to_numpy()
        + df["logq_plus"].to_numpy()
        - df["logq_minus"].to_numpy()
        + df["laststep_logq"].to_numpy() - df["logstep_logq_corr"].to_numpy()/2
    )
else:
    df["W_fwd"] = (
        df["uB_x1"].to_numpy()
        + df["logq0"].to_numpy()
        + df["logq_plus"].to_numpy()
        - df["logq_minus"].to_numpy()
    )

sigma = np.sqrt(T_in_eV)
df["F_fwd"] = (
    df["W_fwd"].to_numpy()
)


df_eV["W_fwd"] = (
    df["W_fwd"] * T_in_eV/num_atoms
)

df_eV["F_fwd"] = (
    df["F_fwd"] * T_in_eV/num_atoms
)


if "W_fwd" not in df:
    raise RuntimeError("Cannot compute ΔF: set target_uB_file or fill df['uB_x1'] first.")

W_fwd = df["W_fwd"].to_numpy()
logw_fwd = -W_fwd

Delta_f_fwd = -logmeanexp(-W_fwd) 
ESS_fwd = ess_from_logw(logw_fwd)

summary = {
    "n_samples": len(W_fwd),
    "DeltaF_forward_eV_per_atom": Delta_f_fwd * T_in_eV / num_atoms,
    "F_forward_eV_per_atom": (Delta_f_fwd) * T_in_eV / num_atoms,
    "ESS_forward": ESS_fwd,
    "ESS_forward_fraction": ESS_fwd / len(W_fwd),
    "W_mean": float(np.mean(W_fwd)* T_in_eV / num_atoms),
    "W_std": float(np.std(W_fwd)* T_in_eV / num_atoms),
    "W_min": float(np.min(W_fwd)* T_in_eV / num_atoms),
    "W_max": float(np.max(W_fwd)* T_in_eV / num_atoms),
}

pd.Series(summary)

print("%.5f"%(summary['F_forward_eV_per_atom']))
# print(summary)
