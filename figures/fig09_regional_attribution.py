"""fig09: Signed regional contributions to seasonal mass-balance change."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from common import GROUP_COLORS, REGION_NAMES, figure_args, panel_label, save_figure, set_plot_style


FOCUS_REGIONS = (8, 1, 2, 11, 13)


ATTRIBUTION_GROUPS = (*tuple(GROUP_COLORS)[:3], "Geometry / interactions")


REGIONAL_ATTRIBUTION_PATH = Path("results/analysis/regional_change_attribution.csv")


def regional_vector(frame: pd.DataFrame, term: str, column: str = "delta") -> np.ndarray:
    series = frame.loc[frame["term"].eq(term)].set_index("O1Region")[column].reindex(FOCUS_REGIONS)
    values = series.to_numpy(float)
    invalid = ~np.isfinite(values)
    if invalid.any():
        raise ValueError(f"Missing or non-finite {column} for {term} in RGI regions {series.index[invalid].tolist()}")
    return values


def draw_regional_attribution(root: Path, output: Path) -> None:
    data = pd.read_csv(root / REGIONAL_ATTRIBUTION_PATH)
    terms = [*ATTRIBUTION_GROUPS, "prediction", "winter_recon", "summer_recon"]
    subset = data.loc[data["O1Region"].isin(FOCUS_REGIONS) & data["term"].isin(terms)].copy()
    subset.to_csv(output / "figure09_source.csv", index=False)
    plt.rcParams.update({"font.size": 10, "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.titlesize": 11})
    fig = plt.figure(figsize=(7.2, 3.8))
    left = fig.add_axes([.285, .32, .315, .535])
    right = fig.add_axes([.668, .32, .315, .535], sharex=left)
    left.set_xticks([-.6, -.4, -.2, 0, .2])
    left.set_xlim(-.64, .22)
    y = np.arange(len(FOCUS_REGIONS))
    for panel, (season, ax) in enumerate(zip(("winter", "summer"), (left, right))):
        frame = subset.loc[subset["season"].eq(season)]
        groups = np.array([regional_vector(frame, group) for group in ATTRIBUTION_GROUPS])
        fitted = regional_vector(frame, "prediction")
        reconstructed = regional_vector(frame, f"{season}_recon")
        np.testing.assert_allclose(groups.sum(axis=0), fitted, rtol=0, atol=1e-6, equal_nan=False)
        positive, negative = np.zeros(len(y)), np.zeros(len(y))
        for values, color in zip(groups, GROUP_COLORS.values()):
            starts = np.where(values >= 0, positive, negative)
            ax.barh(y, values, left=starts, height=.52, color=color, edgecolor="white", linewidth=.45, zorder=2)
            positive += np.maximum(values, 0)
            negative += np.minimum(values, 0)
        ax.scatter(fitted, y - .12, marker="D", s=25, facecolor="#202020", edgecolor="white", linewidth=.45, zorder=5)
        ax.scatter(reconstructed, y + .12, marker="o", s=28, facecolor="white", edgecolor="#202020", linewidth=.9, zorder=5)
        ax.set_ylim(4.6, -.6)
        ax.set_yticks(y, [REGION_NAMES[region] for region in FOCUS_REGIONS] if panel == 0 else [""] * len(y))
        ax.tick_params(axis="y", length=0, pad=8)
        ax.tick_params(axis="x", length=2.2, width=.65, pad=5)
        ax.spines["left"].set_visible(False)
        ax.axvline(0, color="#b3b3b3", linewidth=.75, zorder=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#ededed", linewidth=.6)
        ax.set_title(season.capitalize(), pad=9)
        panel_label(ax, "ab"[panel], x=-.075, y=1.11)
    fig.text(.634, .226, r"Change in mean seasonal balance (m w.e. yr$^{-1}$)", ha="center", va="center")
    group_keys = [Patch(facecolor=color, label=group) for group, color in GROUP_COLORS.items()]
    fig.legend(handles=group_keys, loc="center", bbox_to_anchor=(.634, .13), ncol=2, fontsize=9.5,
               columnspacing=1.7, handlelength=1.15, handletextpad=.5)
    marker_keys = [
        Line2D([], [], marker="D", markersize=5, color="none", markerfacecolor="#202020",
               markeredgecolor="white", label="Fitted change"),
        Line2D([], [], marker="o", markersize=5, color="none", markerfacecolor="white",
               markeredgecolor="#202020", label="Reconstructed change"),
    ]
    fig.legend(handles=marker_keys, loc="center", bbox_to_anchor=(.634, .043), ncol=2, fontsize=9.5,
               columnspacing=1.5, handlelength=1.35, handletextpad=.5)
    save_figure(fig, output, "figure09")


if __name__ == "__main__":
    root, output, dpi = figure_args(__doc__)
    set_plot_style(dpi=dpi)
    draw_regional_attribution(root, output)
