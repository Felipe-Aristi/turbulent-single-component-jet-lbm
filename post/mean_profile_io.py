import argparse
import os
import re
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def parse_scalar_constant(text, name):
    pattern = rf"{name}\s*=\s*(?:static_cast<[^>]+>\()?([0-9.+\-eE]+)"
    match = re.search(pattern, text)
    if match is None:
        return None
    value = match.group(1)
    return float(value) if any(ch in value for ch in ".eE") else int(value)


def read_constants(project_root=PROJECT_ROOT):
    constants_path = Path(project_root) / "constants.cuh"
    if not constants_path.exists():
        return {}

    text = constants_path.read_text()
    constants = {}
    for name in ("NOUTPUT", "NSTATS_SAMPLE", "NPROFILE_OUTPUT", "NX", "NY", "NZ"):
        value = parse_scalar_constant(text, name)
        if value is not None:
            constants[name] = int(value)

    jet_radius = parse_scalar_constant(text, "jet_radius")
    if jet_radius is not None:
        constants["jet_radius"] = float(jet_radius)
        constants["D"] = 2.0 * float(jet_radius)

    jet_velocity = parse_scalar_constant(text, "jet_velocity")
    if jet_velocity is not None:
        constants["U_MAX"] = float(jet_velocity)

    re_value = parse_scalar_constant(text, "Re")
    if re_value is not None:
        constants["Re"] = int(round(float(re_value)))

    return constants


def read_metadata(mean_dir):
    metadata_path = Path(mean_dir) / "radial_profile_metadata.txt"
    metadata = {}
    if not metadata_path.exists():
        return metadata

    for line in metadata_path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        key, value = parts
        metadata[key] = value

    for key in ("NX", "NY", "NZ", "NR_BINS", "NradialProfileCells", "NSTATS_SAMPLE", "NPROFILE_OUTPUT", "start_step", "radial_sample_count"):
        if key in metadata:
            metadata[key] = int(metadata[key])

    return metadata


def find_mean_dir(project_root=PROJECT_ROOT, case=None):
    vtk_root = Path(project_root) / "JET_VTK"
    if case is not None:
        case_name = case if str(case).startswith("Re") else f"Re{case}"
        mean_dir = vtk_root / case_name / "mean_profiles"
        if not mean_dir.exists():
            raise FileNotFoundError(f"Mean profile folder not found: {mean_dir}")
        return mean_dir

    candidates = sorted(vtk_root.glob("Re*/mean_profiles"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No mean profile folders found under: {vtk_root}")

    return candidates[0]


def read_radial_moments(mean_dir):
    mean_dir = Path(mean_dir)
    metadata = read_metadata(mean_dir)
    constants = read_constants()

    ny = int(metadata.get("NY", constants.get("NY", 0)))
    nr = int(metadata.get("NR_BINS", constants.get("NX", 0) // 2))
    if ny <= 0 or nr <= 0:
        raise ValueError("Could not determine NY and NR_BINS from metadata/constants.")

    shape = (ny, nr)

    def read_array(name, dtype):
        path = mean_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing mean-profile file: {path}")
        data = np.fromfile(path, dtype=dtype)
        expected = ny * nr
        if data.size == 0:
            raise ValueError(
                f"{path} is empty. The old solver writes radial profiles only at normal shutdown; "
                "recompile with the periodic profile-output change, rerun, or wait for a normal completed run."
            )
        if data.size != expected:
            raise ValueError(f"{path} has {data.size} values, expected {expected}.")
        return data.reshape(shape)

    moments = {
        "sum_uy": read_array("radial_sum_uy.bin", np.float64),
        "sum_uy2": read_array("radial_sum_uy2.bin", np.float64),
        "sum_ur": read_array("radial_sum_ur.bin", np.float64),
        "sum_ur2": read_array("radial_sum_ur2.bin", np.float64),
        "sum_uruy": read_array("radial_sum_uruy.bin", np.float64),
        "count": read_array("radial_count.bin", np.uint64),
        "metadata": metadata,
        "mean_dir": mean_dir,
    }
    return moments


def averaged_radial_fields(moments):
    count = moments["count"].astype(np.float64)
    valid = count > 0

    def avg(name):
        out = np.full(count.shape, np.nan, dtype=np.float64)
        np.divide(moments[name], count, out=out, where=valid)
        return out

    uy = avg("sum_uy")
    uy2 = avg("sum_uy2")
    ur = avg("sum_ur")
    ur2 = avg("sum_ur2")
    uruy = avg("sum_uruy")

    uy_var = np.maximum(uy2 - uy * uy, 0.0)
    ur_var = np.maximum(ur2 - ur * ur, 0.0)

    return {
        "uy": uy,
        "uy2": uy2,
        "ur": ur,
        "ur2": ur2,
        "uruy": uruy,
        "uy_rms": np.sqrt(uy_var),
        "ur_rms": np.sqrt(ur_var),
        "uruy_reynolds": uruy - ur * uy,
        "count": moments["count"],
    }


def require_samples(moments):
    total = int(np.sum(moments["count"]))
    if total <= 0:
        raise ValueError(
            "The radial_count.bin file has zero samples. Run the solver until the energy criterion starts averaging."
        )


def fit_virtual_origin(y, uc, diameter, u_jet, yi_over_d=10.0, yf_over_d=15.0):
    y_over_d = y / diameter
    fit_mask = (
        np.isfinite(uc)
        & (uc > 0.0)
        & (y_over_d >= yi_over_d)
        & (y_over_d <= yf_over_d)
    )

    if np.count_nonzero(fit_mask) < 2:
        raise ValueError("Not enough valid centerline samples in the selected fit range.")

    x_fit = y_over_d[fit_mask]
    y_fit = u_jet / uc[fit_mask]

    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    spreading_rate = 1.0 / slope
    y0 = -intercept * spreading_rate * diameter
    return spreading_rate, y0, x_fit, y_fit, slope, intercept


def parse_common_args(description):
    constants = read_constants()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--case", default=f"Re{constants.get('Re', 5000)}", help="Case folder, e.g. Re5000.")
    parser.add_argument("--diameter", type=float, default=constants.get("D", 16.0), help="Jet diameter in lattice units.")
    parser.add_argument("--u-jet", type=float, default=constants.get("U_MAX", 0.05), help="Inlet jet velocity.")
    parser.add_argument("--fit-start", type=float, default=10.0, help="Start of centerline fit in y/D.")
    parser.add_argument("--fit-end", type=float, default=15.0, help="End of centerline fit in y/D.")
    parser.add_argument("--slices", type=float, nargs="+", default=[10, 11, 12, 13, 14, 15], help="Axial slices in y/D.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "resultsJET"), help="Output folder.")
    return parser
