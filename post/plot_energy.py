from pathlib import Path
import sys

import numpy as np

from exportAndCrop import export_and_crop
from mean_profile_io import PROJECT_ROOT, find_mean_dir, parse_common_args, read_constants, read_metadata
from prettyPlot import pretty_plot


blue = "#009DDC"


def total_kinetic_energy(field, input_path, output_path, diameter, u_jet):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = read_metadata(input_path)
    constants = read_constants()

    nx = int(metadata.get("NX", constants.get("NX", 1)))
    ny = int(metadata.get("NY", constants.get("NY", 1)))
    nz = int(metadata.get("NZ", constants.get("NZ", 1)))
    noutput = int(constants.get("NOUTPUT", 2000))

    filename = input_path / f"tke_{field}.bin"
    if not filename.exists():
        raise FileNotFoundError(f"Missing energy file: {filename}")

    tke = np.fromfile(filename, dtype=np.float32)
    if tke.size == 0:
        raise ValueError(f"Energy file is empty: {filename}")

    t_star = np.arange(tke.size, dtype=np.float64) * noutput * u_jet / diameter

    if field == "diff":
        y_values = np.maximum(tke.astype(np.float64), 1e-30)
        y_lim = (max(np.nanmin(y_values) * 0.5, 1e-8), max(np.nanmax(y_values) * 2.0, 1e-6))
        y_label = r"$\Delta E_K/E_K$"
    else:
        norm = 0.5 * u_jet * u_jet * float(nx * ny * nz)
        y_values = tke.astype(np.float64) / norm
        y_lim = (np.nanmin(y_values) * 0.95, np.nanmax(y_values) * 1.05)
        y_label = r"$E_K^*$"

    fig, ax, _ = pretty_plot(
        xLim=(0, max(t_star[-1], 1e-12)),
        yLim=y_lim,
        cLim=(-1, 1),
        plotAspectRatio=(1, 1, 1),
        xLabel=r"$t^*$",
        yLabel=y_label,
        yScientificNotation=(field != "diff"),
        yTickFormat=2,
        nxTicks=5,
        nyTicks=9,
        useColorBar=False,
        paperPoints=612,
        marginPoints=54,
        textWidth=1,
        boxMarginScale=0.085,
        yLabelAngle=90,
        useGrid=True,
        fontSize=14,
        dpi=300,
    )

    ax.yaxis.set_label_coords(-0.125, 0.5)
    ax.plot(t_star, y_values, ls="-", lw=3, color=blue)
    if field == "diff":
        ax.set_yscale("log")

    export_and_crop(fig, output_path / f"kinetic_energy_{field}.png")


if __name__ == "__main__":
    parser = parse_common_args("Plot kinetic-energy history from JET_VTK/ReXXXX/mean_profiles.")
    parser.add_argument("field", choices=("total", "avg", "diff"), help="Energy file suffix to plot.")
    args = parser.parse_args()

    mean_dir = find_mean_dir(PROJECT_ROOT, args.case)
    try:
        total_kinetic_energy(
            field=args.field,
            input_path=mean_dir,
            output_path=args.output,
            diameter=args.diameter,
            u_jet=args.u_jet,
        )
    except Exception as exc:
        print(f"plot_energy.py: {exc}", file=sys.stderr)
        sys.exit(1)
