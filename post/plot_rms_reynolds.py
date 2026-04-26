from pathlib import Path
import sys

import numpy as np

from exportAndCrop import export_and_crop
from mean_profile_io import (
    PROJECT_ROOT,
    averaged_radial_fields,
    find_mean_dir,
    fit_virtual_origin,
    parse_common_args,
    read_radial_moments,
    require_samples,
)
from prettyPlot import pretty_plot


green = "#61BB46"
red = "#E03A3E"
blue = "#009DDC"
purple = "#963D97"
yellow = "#FDB827"
orange = "#F5821F"
black = "#000000"
colors = [red, blue, green, purple, orange, yellow]
markers = ["o", "s", "^", "D", "v", "<", ">"]


# Digitized reference points from Hussein et al. (1994), used as a benchmark
# overlay for the self-similar region. Replace these arrays with your own
# high-resolution digitization if you need publication-grade reference data.
HUSSEIN_UY_RMS_X = np.array([
    0.000, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070,
    0.080, 0.090, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150,
    0.160, 0.170, 0.180, 0.190, 0.200, 0.210, 0.220, 0.235,
])
HUSSEIN_UY_RMS = np.array([
    0.278, 0.282, 0.286, 0.286, 0.283, 0.276, 0.265, 0.252,
    0.235, 0.216, 0.196, 0.176, 0.155, 0.136, 0.118, 0.101,
    0.084, 0.069, 0.056, 0.045, 0.036, 0.029, 0.023, 0.018,
])

HUSSEIN_UR_RMS_X = np.array([
    0.000, 0.015, 0.030, 0.045, 0.060, 0.075, 0.090, 0.105,
    0.120, 0.135, 0.150, 0.165, 0.180, 0.195, 0.210, 0.225,
    0.240, 0.260, 0.280, 0.300,
])
HUSSEIN_UR_RMS = np.array([
    0.235, 0.224, 0.216, 0.207, 0.197, 0.185, 0.174, 0.162,
    0.148, 0.132, 0.114, 0.096, 0.079, 0.063, 0.049, 0.037,
    0.027, 0.017, 0.011, 0.007,
])

HUSSEIN_URUY_X = np.array([
    0.000, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070,
    0.080, 0.090, 0.105, 0.120, 0.135, 0.150, 0.170, 0.190,
    0.210, 0.230,
])
HUSSEIN_URUY = np.array([
    0.000, 0.006, 0.011, 0.016, 0.020, 0.022, 0.022, 0.021,
    0.019, 0.017, 0.014, 0.011, 0.009, 0.007, 0.0045, 0.0025,
    0.0010, 0.0002,
])


def make_pretty_rms_axis(xlim, ylim, ylabel):
    fig, ax, _ = pretty_plot(
        xLim=xlim,
        yLim=ylim,
        cLim=(-1, 1),
        plotAspectRatio=(1, 1, 1),
        xLabel=r"$r/(y-y_0)$",
        yLabel=ylabel,
        yScientificNotation=False,
        xTickFormat=2,
        yTickFormat=2,
        nxTicks=5,
        nyTicks=7,
        useColorBar=False,
        paperPoints=612,
        marginPoints=54,
        textWidth=1,
        boxMarginScale=0.085,
        yLabelAngle=90,
        useGrid=False,
        fontSize=14,
        dpi=300,
    )
    ax.yaxis.set_label_coords(-0.125, 0.5)
    return fig, ax


def plot_lbm_slices(ax, radius, values, count, uc, y0, diameter, slices):
    for j, slice_over_d in enumerate(slices):
        y_idx = int(round(slice_over_d * diameter))
        if y_idx <= 0 or y_idx >= values.shape[0] or not np.isfinite(uc[y_idx]) or uc[y_idx] <= 0.0:
            print(f"Skipping y/D={slice_over_d}: no valid data.")
            continue

        denom = y_idx - y0
        if denom <= 0.0:
            print(f"Skipping y/D={slice_over_d}: y-y0 <= 0.")
            continue

        valid = (count[y_idx, :] > 0) & np.isfinite(values[y_idx, :])
        color = colors[j % len(colors)]
        marker = markers[j % len(markers)]
        label = rf"$y={slice_over_d:g}D$"

        ax.plot(
            radius[valid] / denom,
            values[y_idx, valid],
            ls="none",
            marker=marker,
            ms=3,
            color=color,
            label=label,
        )


def plot_reference(ax, x, y):
    ax.plot(
        x,
        y,
        ls="none",
        marker="+",
        ms=6,
        mew=2,
        color=black,
        label="Hussein et al. (1994)",
    )


def plot_rms_reynolds(mean_dir, output_dir, diameter, u_jet, slices, fit_start, fit_end, stress_sign):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    moments = read_radial_moments(mean_dir)
    require_samples(moments)
    fields = averaged_radial_fields(moments)

    uy = fields["uy"]
    count = fields["count"]
    ny, nr = uy.shape
    y = np.arange(ny, dtype=np.float64)
    radius = np.arange(nr, dtype=np.float64)
    uc = uy[:, 0]

    _, y0, _, _, _, _ = fit_virtual_origin(y, uc, diameter, u_jet, fit_start, fit_end)
    print(f"Using mean profile folder: {mean_dir}")
    print(f"y0 = {y0}")

    uc2 = uc[:, None] * uc[:, None]
    uy_rms_norm = fields["uy_rms"] / uc[:, None]
    ur_rms_norm = fields["ur_rms"] / uc[:, None]
    uruy_norm = stress_sign * fields["uruy_reynolds"] / uc2

    fig_uy, ax_uy = make_pretty_rms_axis(
        xlim=(0, 0.4),
        ylim=(0, 0.3),
        ylabel=r"$u_{y,rms}/U_{y,c}$",
    )
    plot_lbm_slices(ax_uy, radius, uy_rms_norm, count, uc, y0, diameter, slices)
    plot_reference(ax_uy, HUSSEIN_UY_RMS_X, HUSSEIN_UY_RMS)
    ax_uy.legend(fontsize=10)
    export_and_crop(fig_uy, output_dir / "uy_rms_profile.png")

    fig_ur, ax_ur = make_pretty_rms_axis(
        xlim=(0, 0.4),
        ylim=(0, 0.3),
        ylabel=r"$u_{r,rms}/U_{y,c}$",
    )
    plot_lbm_slices(ax_ur, radius, ur_rms_norm, count, uc, y0, diameter, slices)
    plot_reference(ax_ur, HUSSEIN_UR_RMS_X, HUSSEIN_UR_RMS)
    ax_ur.legend(fontsize=10)
    export_and_crop(fig_ur, output_dir / "ur_rms_profile.png")

    fig_stress, ax_stress = make_pretty_rms_axis(
        xlim=(0, 0.3),
        ylim=(0, 0.04),
        ylabel=r"$\langle u_r'u_y'\rangle/U_{y,c}^{2}$",
    )
    plot_lbm_slices(ax_stress, radius, uruy_norm, count, uc, y0, diameter, slices)
    plot_reference(ax_stress, HUSSEIN_URUY_X, HUSSEIN_URUY)
    ax_stress.legend(fontsize=10)
    export_and_crop(fig_stress, output_dir / "uruy_reynolds_profile.png")


if __name__ == "__main__":
    parser = parse_common_args("Plot RMS velocity and Reynolds-stress profiles from radial moment files.")
    parser.add_argument(
        "--stress-sign",
        type=float,
        default=1.0,
        help="Use -1 if your Reynolds shear stress has the opposite sign from the desired convention.",
    )
    args = parser.parse_args()

    mean_dir = find_mean_dir(PROJECT_ROOT, args.case)
    try:
        plot_rms_reynolds(
            mean_dir=mean_dir,
            output_dir=args.output,
            diameter=args.diameter,
            u_jet=args.u_jet,
            slices=args.slices,
            fit_start=args.fit_start,
            fit_end=args.fit_end,
            stress_sign=args.stress_sign,
        )
    except Exception as exc:
        print(f"plot_rms_reynolds.py: {exc}", file=sys.stderr)
        sys.exit(1)
