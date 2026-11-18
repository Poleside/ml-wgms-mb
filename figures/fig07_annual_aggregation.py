"""fig07: Annual mass balance under two area-weighting schemes."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from aggregation import annual_aggregation

from common import figure_args, panel_label, save_figure, set_plot_style


def draw_annual_aggregation(root: Path, output: Path) -> None:
    data = annual_aggregation(root)
    data.to_csv(output / "figure07_source.csv", index=False)
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(7.2, 5), sharex=True)
    for series, label, color in [("region_area_reweighted", "Region-area-weighted mean", "#286fb4"),
                                 ("completed_series", "Study-sample mean", "#4d4d4d")]:
        centre, sigma = data[f"{series}_mean"], data[f"{series}_sigma"]
        ax_a.fill_between(data["year"], centre - sigma, centre + sigma, color=color, alpha=.12, linewidth=0)
        ax_a.plot(data["year"], centre, color=color, linewidth=1.7, label=label)
        total = data[f"{series}_cumulative"]
        ax_b.plot(data["year"], total, color=color, linewidth=1.8, label=label)
        ax_b.annotate(f"{total.iloc[-1]:.2f} m w.e.", (data["year"].iloc[-1], total.iloc[-1]),
                      xytext=((-118, 5) if series == "region_area_reweighted" else (-94, 12)),
                      textcoords="offset points", color=color, fontsize=7.5,
                      arrowprops={"arrowstyle": "-", "color": color, "lw": .7})
    for ax, label in zip((ax_a, ax_b), "ab"):
        ax.axhline(0, color="#222222", linewidth=.8)
        ax.set(xlim=(1978.4, 2025.6), xlabel="Year")
        ax.grid(axis="y", color="#d9d9d9", linewidth=.35, alpha=.75)
        ax.legend(loc="lower left", ncol=2)
        panel_label(ax, label, x=-.08)
    ax_a.set_ylabel("Annual balance\n(m w.e.)")
    ax_a.set_ylim(-1.3, 1.1)
    ax_a.yaxis.set_major_locator(MultipleLocator(.5))
    ax_a.tick_params(labelbottom=True)
    ax_b.set_ylabel("Cumulative annual\nbalance (m w.e.)")
    fig.subplots_adjust(left=.11, right=.98, top=.95, bottom=.10, hspace=.32)
    save_figure(fig, output, "figure07")


if __name__ == "__main__":
    root, output, dpi = figure_args(__doc__)
    set_plot_style(dpi=dpi)
    draw_annual_aggregation(root, output)
