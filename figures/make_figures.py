"""Generate paper figures from the fitted and reconstructed products."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from common import set_plot_style
from coverage import draw_coverage
from diagnostics import draw_predictor_dependence, draw_regional_attribution
from results import draw_reconstruction, draw_regional_contrasts, draw_annual_aggregation
from validation import draw_window_validation, draw_model_performance
from workflow import draw_workflow


DRAW = {
    1: draw_workflow, 2: draw_coverage, 3: draw_window_validation,
    4: draw_model_performance, 5: draw_reconstruction, 6: draw_regional_contrasts,
    7: draw_annual_aggregation, 8: draw_predictor_dependence, 9: draw_regional_attribution,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=DRAW, type=int, action="append",
                        help="Generate just these manuscript figures. Repeat for multiple figures.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Project directory containing data, models and results.")
    parser.add_argument("--output-dir", type=Path, help="Figure output directory. Default: ROOT/figures/exports.")
    parser.add_argument("--dpi", type=int, default=400,
                        help="Raster export DPI for figures 2–9. Figure 1 uses its fixed 300 dpi.")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive.")
    root = args.root.resolve()
    output = args.output_dir.resolve() if args.output_dir is not None else root / "figures/exports"
    output.mkdir(parents=True, exist_ok=True)
    for number in args.figure or DRAW:
        with matplotlib.rc_context():
            set_plot_style(args.dpi)
            DRAW[number](root, output)
        print(f"Figure {number}: {output / f'figure{number:02d}.pdf'}")


if __name__ == "__main__":
    main()
