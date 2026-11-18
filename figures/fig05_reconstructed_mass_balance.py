"""fig05: Reconstructed glacier mass balance."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

from aggregation import sample_series

from common import COLORS, TARGETS, figure_args, panel_label, save_figure, set_plot_style


def draw_reconstruction(root: Path, output: Path) -> None:
    data = sample_series(root, "annual")
    for target in ("winter", "summer"):
        data = data.merge(sample_series(root, target), on="year", validate="one_to_one")
    data["annual_network_cumulative"] = data["annual_network_mean"].cumsum()
    years = data["year"]
    for target in TARGETS:
        centre = data[f"{target}_network_mean"]
        slope, intercept = np.polyfit(years, centre, 1)
        data[f"{target}_linear_trend"] = slope * years + intercept
    data.to_csv(output / "figure05_source.csv", index=False)

    fig = plt.figure(figsize=(7.2, 6.5))
    top, middle, bottom = fig.add_gridspec(3, 1, height_ratios=[1.15, .78, 1.05], hspace=.44)
    ax_a, ax_b = fig.add_subplot(top), fig.add_subplot(middle)
    halves = bottom.subgridspec(1, 2, wspace=.37)
    ax_c, ax_d = fig.add_subplot(halves[0]), fig.add_subplot(halves[1])
    trend_colors = {"annual": "#174a7e", "winter": "#12624d", "summer": "#994600"}
    for target, ax in [("annual", ax_a), ("winter", ax_d), ("summer", ax_d)]:
        mean, sigma = data[f"{target}_network_mean"], data[f"{target}_network_sigma"]
        ax.fill_between(years, mean - sigma, mean + sigma, color=COLORS[target], alpha=.12, linewidth=0)
        ax.plot(years, mean, color=COLORS[target], linewidth=1.65,
                label="Study-sample mean" if target == "annual" else target.capitalize())
        ax.plot(years, data[f"{target}_linear_trend"], "--", color=trend_colors[target], linewidth=1,
                label="Linear trend" if target == "annual" else None)
    ax_a.plot(years, data["annual_observed_sample_mean"], color=COLORS["observed"], linewidth=1,
              marker="o", markersize=2.2, label="Observation-only mean")
    handles, labels = ax_a.get_legend_handles_labels()
    ax_a.legend([handles[i] for i in (0, 2, 1)], [labels[i] for i in (0, 2, 1)], loc="lower left", ncol=3)
    ax_a.set_ylabel("Annual balance (m w.e.)")
    ax_b.bar(years, data["annual_network_n"], width=.84, color="#d8e6f2", label="316-glacier sample")
    ax_b.bar(years, data["annual_observed_n"], width=.84, color=COLORS["observed"], label="Observed glaciers")
    ax_b.set(ylabel="Glaciers", ylim=(0, 350))
    ax_b.legend(loc="upper left", bbox_to_anchor=(0, 1.10), ncol=2)
    ax_c.plot(years, data["annual_network_cumulative"], color=COLORS["annual"], linewidth=1.8)
    annual_total = data["annual_network_cumulative"].iloc[-1]
    ax_c.annotate(f"{annual_total:.2f} m w.e.", (years.iloc[-1], annual_total), xytext=(-73, 8),
                  textcoords="offset points", color=COLORS["annual"], fontsize=7.5,
                  arrowprops={"arrowstyle": "-", "color": COLORS["annual"], "lw": .7})
    ax_c.set_ylabel("Cumulative annual balance (m w.e.)")
    ax_c.yaxis.set_major_locator(MultipleLocator(2.5))
    ax_d.set_ylabel("Seasonal balance (m w.e.)")
    ax_d.legend(loc="center right")
    for ax in (ax_a, ax_c, ax_d):
        ax.axhline(0, color="#333333", linewidth=.65, zorder=0)
    fig.subplots_adjust(left=.11, right=.98, top=.97, bottom=.08)
    for ax, label in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        ax.set(xlim=(1978.4, 2025.6), xlabel="Year")
        panel_label(ax, label, x=-.054, y=1.10, reference=ax_a)
    save_figure(fig, output, "figure05")


if __name__ == "__main__":
    root, output, dpi = figure_args(__doc__)
    set_plot_style(dpi=dpi)
    draw_reconstruction(root, output)
