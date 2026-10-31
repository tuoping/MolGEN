"""Utilities for selecting generated trajectories from evaluation reports."""

from pathlib import Path
import re


_ENERGY_PER_ATOM_RE = re.compile(
    r"^Potential energy per atom\s*=\s*([-+0-9.eE]+)\s*$",
    re.MULTILINE,
)


def read_report_energy_per_atom(report_filename):
    """Read the final potential energy per atom from an evaluation report."""

    report_filename = Path(report_filename)
    match = _ENERGY_PER_ATOM_RE.search(report_filename.read_text())
    if match is None:
        raise ValueError(
            f"No 'Potential energy per atom' entry in {report_filename}"
        )
    return float(match.group(1))


def filter_trajectories_by_end_energy(
    trajectory_filenames,
    max_energy_per_atom=-5.0,
):
    """Split trajectories using their ``report_<name>_0.eval`` end energy.

    The threshold is strict, matching the requested ``energy < threshold``
    criterion.  Files are only selected or rejected; nothing is deleted.

    Returns
    -------
    accepted, rejected : list
        ``accepted`` contains trajectory paths. ``rejected`` contains
        ``(trajectory_path, energy_per_atom)`` pairs.
    """

    accepted = []
    rejected = []
    for trajectory_filename in trajectory_filenames:
        trajectory_filename = Path(trajectory_filename)
        report_filename = trajectory_filename.with_name(
            f"report_{trajectory_filename.name}_0.eval"
        )
        if not report_filename.is_file():
            raise FileNotFoundError(
                f"Missing evaluation report for {trajectory_filename}: "
                f"{report_filename}"
            )

        energy_per_atom = read_report_energy_per_atom(report_filename)
        if energy_per_atom < max_energy_per_atom:
            accepted.append(str(trajectory_filename))
        else:
            rejected.append((str(trajectory_filename), energy_per_atom))

    return accepted, rejected
