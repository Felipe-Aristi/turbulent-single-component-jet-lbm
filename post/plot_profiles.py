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


HUSSEIN_X = np.array([
    0, 0.010599057494345044, 0.019815647572123837, 0.030875562697158625,
    0.04009215277493742, 0.04838709790833872, 0.056221185411050346,
    0.06405527291376191, 0.07281104051935197, 0.08479260058876421,
    0.09400919066654304, 0.10737325331102249, 0.11658984338880131,
    0.12949308356109202, 0.14147464363050433, 0.15345620369991658,
    0.16866359139165193, 0.18248847650832015, 0.19723500656936588,
    0.21290321673329005, 0.22672810184995826, 0.2447004595333271,
])

HUSSEIN_U = np.array([
    0.9986970883174464, 0.988273596045439, 0.9622149647712097,
    0.9205211944947582, 0.8657980588782979, 0.8136807963298389,
    0.7563517876453765, 0.7042345250969178, 0.6469055164124551,
    0.5635178764537632, 0.514006536676201, 0.4332247200826162,
    0.38371338030505414, 0.32899019498569865, 0.2690553629361291,
    0.2247556698887811, 0.16742676061010808, 0.11791512261517773,
    0.08143319788051967, 0.05276854442960405, 0.03452768146806465,
    0.024104189196056974,
])


def first_half_radius(profile, radius, center_value):
    if not np.isfinite(center_value) or center_value <= 0.0:
        return np.nan

    ratio = profile / center_value
    finite = np.isfinite(ratio)
    below = np.where(finite & (ratio <= 0.5))[0]
    below = below[below > 0]
    if below.size == 0:
        return np.nan

    idx = int(below[0])
    r1, r2 = radius[idx - 1], radius[idx]
    u1, u2 = ratio[idx - 1], ratio[idx]
    if not np.isfinite(u1) or not np.isfinite(u2) or np.isclose(u1, u2):
        return np.nan

    return r1 + (0.5 - u1) * (r2 - r1) / (u2 - u1)


def plot_mean_profiles(mean_dir, output_dir, diameter, u_jet, slices, fit_start, fit_end):
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
    valid_y = np.isfinite(uc) & (uc > 0.0) & (count[:, 0] > 0)
    if np.count_nonzero(valid_y) < 2:
        raise ValueError("Not enough valid centerline samples to plot mean profiles.")

    spreading_rate, y0, x_fit, y_fit, slope, intercept = fit_virtual_origin(
        y, uc, diameter, u_jet, fit_start, fit_end
    )

    print(f"Using mean profile folder: {mean_dir}")
    print(f"B = {spreading_rate}")
    print(f"y0 = {y0}")

    paper_width = 612
    margin_points = 54

    inv_centerline = np.full_like(uc, np.nan)
    np.divide(u_jet, uc, out=inv_centerline, where=valid_y)
    inv_valid = inv_centerline[valid_y]

    fig1, ax1, _ = pretty_plot(
        xLim=(0, ny / diameter),
        yLim=(0, np.ceil(np.nanmax(inv_valid))),
        cLim=(-1, 1),
        plotAspectRatio=(1, 1, 1),
        xLabel=r"$y/D$",
        yLabel=r"$U_{jet}/U_{y,c}$",
        yScientificNotation=False,
        yTickFormat=2,
        nxTicks=9,
        nyTicks=9,
        useColorBar=False,
        paperPoints=paper_width,
        marginPoints=margin_points,
        textWidth=1,
        boxMarginScale=0.085,
        yLabelAngle=90,
        useGrid=False,
        fontSize=14,
        dpi=300,
    )
    ax1.yaxis.set_label_coords(-0.125, 0.5)
    ax1.plot(y[valid_y] / diameter, inv_centerline[valid_y], ls="-", lw=3, color=black)
    export_and_crop(fig1, output_dir / "uy_centerline.png")

    fig2, ax2, _ = pretty_plot(
        xLim=(fit_start, fit_end),
        yLim=(np.nanmin(y_fit) * 0.95, np.nanmax(y_fit) * 1.05),
        cLim=(-1, 1),
        plotAspectRatio=(1, 1, 1),
        xLabel=r"$y/D$",
        yLabel=r"$U_{jet}/U_{y,c}$",
        yScientificNotation=False,
        yTickFormat=2,
        nxTicks=6,
        nyTicks=9,
        useColorBar=False,
        paperPoints=paper_width,
        marginPoints=margin_points,
        textWidth=1,
        boxMarginScale=0.085,
        yLabelAngle=90,
        useGrid=False,
        fontSize=14,
        dpi=300,
    )
    ax2.yaxis.set_label_coords(-0.125, 0.5)
    ax2.plot(x_fit, y_fit, marker="o", mfc="none", ms=2, ls="none", lw=3, color=black, label="LBM")
    ax2.plot(x_fit, slope * x_fit + intercept, ls="--", lw=3, color=red, label=r"$(y-y_0)/(BD)$")
    ax2.legend(fontsize=14)
    export_and_crop(fig2, output_dir / "uy_centerline_fit.png")

    fig3, ax3, _ = pretty_plot(
        xLim=(0, 3),
        yLim=(0, 1.1),
        cLim=(-1, 1),
        plotAspectRatio=(1, 1, 1),
        xLabel=r"$r/D$",
        yLabel=r"$\langle u_y\rangle/U_{jet}$",
        yScientificNotation=False,
        xTickFormat=1,
        yTickFormat=1,
        nxTicks=7,
        nyTicks=6,
        useColorBar=False,
        paperPoints=paper_width,
        marginPoints=margin_points,
        textWidth=1,
        boxMarginScale=0.085,
        yLabelAngle=90,
        useGrid=False,
        fontSize=14,
        dpi=300,
    )

    fig4, ax4, _ = pretty_plot(
        xLim=(0, 0.3),
        yLim=(0, 1.05),
        cLim=(-1, 1),
        plotAspectRatio=(1, 1, 1),
        xLabel=r"$r/(y-y_0)$",
        yLabel=r"$\langle u_y\rangle/U_{y,c}$",
        yScientificNotation=False,
        xTickFormat=2,
        yTickFormat=1,
        nxTicks=7,
        nyTicks=6,
        useColorBar=False,
        paperPoints=paper_width,
        marginPoints=margin_points,
        textWidth=1,
        boxMarginScale=0.085,
        yLabelAngle=90,
        useGrid=False,
        fontSize=14,
        dpi=300,
    )

    fig5, ax5, _ = pretty_plot(
        xLim=(0, 3),
        yLim=(0, 1.05),
        cLim=(-1, 1),
        plotAspectRatio=(1, 1, 1),
        xLabel=r"$r/r_{1/2}$",
        yLabel=r"$\langle u_y\rangle/U_{y,c}$",
        yScientificNotation=False,
        xTickFormat=2,
        yTickFormat=1,
        nxTicks=7,
        nyTicks=6,
        useColorBar=False,
        paperPoints=paper_width,
        marginPoints=margin_points,
        textWidth=1,
        boxMarginScale=0.085,
        yLabelAngle=90,
        useGrid=False,
        fontSize=14,
        dpi=300,
    )

    for ax in (ax3, ax4, ax5):
        ax.yaxis.set_label_coords(-0.125, 0.585)

    for j, slice_over_d in enumerate(slices):
        y_idx = int(round(slice_over_d * diameter))
        if y_idx <= 0 or y_idx >= ny or not np.isfinite(uc[y_idx]) or uc[y_idx] <= 0.0:
            print(f"Skipping y/D={slice_over_d}: no valid data.")
            continue

        profile = uy[y_idx, :]
        valid_r = np.isfinite(profile) & (count[y_idx, :] > 0)
        color = colors[j % len(colors)]
        marker = markers[j % len(markers)]
        label = rf"$y={slice_over_d:g}D$"

        r_half = first_half_radius(profile, radius, uc[y_idx])
        print(f"Slice {slice_over_d:g}D: r_half = {r_half:.4f}")

        ax3.plot(
            radius[valid_r] / diameter,
            profile[valid_r] / u_jet,
            lw=3,
            ls="none",
            marker=marker,
            ms=3,
            color=color,
            label=label,
        )

        denom = y_idx - y0
        if denom > 0.0:
            ax4.plot(
                radius[valid_r] / denom,
                profile[valid_r] / uc[y_idx],
                lw=3,
                ls="none",
                marker=marker,
                ms=3,
                color=color,
                label=label,
            )

        if np.isfinite(r_half) and r_half > 0.0:
            ax5.plot(
                radius[valid_r] / r_half,
                profile[valid_r] / uc[y_idx],
                lw=3,
                ls="none",
                marker=marker,
                ms=3,
                color=color,
                label=label,
            )

    ax4.plot(HUSSEIN_X, HUSSEIN_U, ls="none", marker="x", ms=4, color=red, label="Hussein et al. (1994)")

    ax3.legend(fontsize=10)
    ax4.legend(fontsize=10)
    ax5.legend(fontsize=10)

    export_and_crop(fig3, output_dir / "uy_radial.png")
    export_and_crop(fig4, output_dir / "uy_selfsimilar.png")
    export_and_crop(fig5, output_dir / "uy_rhalf.png")


if __name__ == "__main__":
    parser = parse_common_args("Plot mean jet profiles from radial moment files.")
    args = parser.parse_args()

    mean_dir = find_mean_dir(PROJECT_ROOT, args.case)
    try:
        plot_mean_profiles(
            mean_dir=mean_dir,
            output_dir=args.output,
            diameter=args.diameter,
            u_jet=args.u_jet,
            slices=args.slices,
            fit_start=args.fit_start,
            fit_end=args.fit_end,
        )
    except Exception as exc:
        print(f"plot_profiles.py: {exc}", file=sys.stderr)
        sys.exit(1)
