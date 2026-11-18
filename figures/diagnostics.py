"""Plot predictor dependence and signed regional attributions."""

import calendar
from pathlib import Path

import dill
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from common import COLORS, REGION_NAMES, TARGETS, panel_label, save_figure


GROUPS = {
    "Snow / precipitation": ("tp", "sf", "rsn", "sde"),
    "Temperature / melt": ("t2m", "smlt", "tsn"),
    "Radiation / flux / pressure": ("fal", "slhf", "ssr", "strd", "sshf", "ssrd", "sp"),
    "Topography / geometry": ("Area", "CenLon", "CenLat", "Zmin", "Zmax", "Zmed", "Slope", "Aspect", "Lmax"),
}
GROUP_COLORS = dict(zip(GROUPS, ("#3A94BF", "#E66B00", "#8278B8", "#969696")))
FIELD_GROUP = {feature: group for group, names in GROUPS.items() for feature in names}
FEATURE_LABELS = {
    "tp": "total\nprecipitation", "sf": "snowfall", "smlt": "snowmelt", "rsn": "snow\ndensity",
    "sde": "snow water\nequivalent", "t2m": "2 m air\ntemperature", "tsn": "snow-layer\ntemperature",
    "fal": "forecast\nalbedo", "slhf": "surface latent\nheat flux", "sshf": "surface sensible\nheat flux",
    "ssr": "net shortwave\nradiation", "ssrd": "downward shortwave\nradiation",
    "strd": "downward\nlongwave\nradiation", "sp": "surface\npressure", "Area": "glacier area",
    "CenLon": "centroid\nlongitude", "CenLat": "centroid\nlatitude", "Zmin": "minimum\nelevation",
    "Zmed": "median\nelevation", "Zmax": "maximum\nelevation", "Slope": "mean glacier-\nsurface slope",
    "Aspect": "mean glacier-\nsurface aspect", "Lmax": "longest glacier\nsurface flowline",
}
FOCUS_REGIONS = (8, 1, 2, 11, 13)
ATTRIBUTION_GROUPS = (*list(GROUPS)[:3], "Geometry / interactions")
REGIONAL_ATTRIBUTION_PATH = Path("results/analysis/regional_change_attribution.csv")


def feature_label(feature: str) -> str:
    month, sep, field = feature.partition("_")
    if not sep:
        label = FEATURE_LABELS[feature]
        return label[0].upper() + label[1:]
    return f"{calendar.month_name[int(month)]} {FEATURE_LABELS[field]}"


def fitted_predictors(root: Path, target: str) -> pd.DataFrame:
    folder = root / "models" / ("annual" if target == "annual" else "annual_seasonal")
    payload_name = "preprocessed_data.pkl" if target == "annual" else f"preprocessed_{target}_data.pkl"
    with (folder / payload_name).open("rb") as handle:
        payload = dill.load(handle)
    features = pd.DataFrame(payload["X_all"], columns=payload["feature_columns"]).reset_index(drop=True)
    balances = pd.Series(payload["y_all"]).reset_index(drop=True)
    correlations = features.corrwith(balances)
    # Match the original cutoff for sufficient pairs when calculating Pearson r.
    correlations = correlations.where(features.loc[balances.notna()].count().ge(10))
    gains = pd.read_csv(folder / f"xgboost_{target}_feature_importance.csv", usecols=["feature", "gain"])
    gains["science_group"] = gains["feature"].str.rsplit("_", n=1).str[-1].map(FIELD_GROUP)
    if gains["science_group"].isna().any():
        raise ValueError(f"Unknown predictor group for {target}")
    gains["direction_corr"] = gains["feature"].map(correlations)
    gains["importance_norm"] = gains["gain"] / gains["gain"].max()
    gains["feature_label"] = gains["feature"].map(feature_label)
    return gains.assign(experiment=target)


def draw_predictor_dependence(root: Path, output: Path) -> None:
    permutations = pd.read_csv(root / "results/analysis/group_permutation_importance.csv")
    permutations = permutations.loc[permutations["target"].isin(TARGETS) & permutations["feature_group"].isin(GROUPS)]
    features = pd.concat([fitted_predictors(root, target) for target in TARGETS], ignore_index=True)
    shares = features.groupby(["experiment", "science_group"], as_index=False)["gain"].sum()
    shares["share"] = shares["gain"] / shares.groupby("experiment")["gain"].transform("sum")
    rankings = features.sort_values(["experiment", "gain"], ascending=[True, False]).groupby("experiment").head(10)
    rankings = rankings.copy()
    rankings["rank"] = rankings.groupby("experiment").cumcount() + 1
    permutations.to_csv(output / "figure08_group_permutation.csv", index=False)
    rankings.to_csv(output / "figure08_predictor_rankings.csv", index=False)
    shares.to_csv(output / "figure08_gain_share.csv", index=False)

    fig = plt.figure(figsize=(7.85, 7.8))
    outer = fig.add_gridspec(2, 1, height_ratios=[.73, 1.75], hspace=.27)
    ax_a = fig.add_subplot(outer[0])
    grid = outer[1].subgridspec(2, 3, height_ratios=[1.55, .11], hspace=.10, wspace=.58)
    x = np.arange(4)
    width = .24
    for index, target in enumerate(TARGETS):
        values = permutations.loc[permutations["target"].eq(target)].set_index("feature_group").loc[list(GROUPS)]
        ax_a.bar(x + (index - 1) * width, values["delta_RMSE_mean"], width, yerr=values["delta_RMSE_sd"],
                 color=COLORS[target], error_kw={"elinewidth": .7, "capsize": 2}, label=target.capitalize())
    ax_a.set_xticks(x, ["Snow /\nprecipitation", "Temperature /\nmelt", "Radiation /\nflux / pressure", "Topography /\ngeometry"])
    ax_a.set(ylabel="Increase in fitted RMSE (m w.e.)", ylim=(0, None))
    ax_a.legend(loc="upper right", ncol=3)

    cmap = mpl.colormaps["RdBu"]
    norm = TwoSlopeNorm(vmin=-.7, vcenter=0, vmax=.7)
    share_order = ["Topography / geometry", *list(GROUPS)[:3]]
    for col, (target, label) in enumerate(zip(TARGETS, "bcd")):
        ax = fig.add_subplot(grid[0, col])
        share_ax = fig.add_subplot(grid[1, col])
        selected = rankings.loc[rankings["experiment"].eq(target)].sort_values("gain")
        y = np.arange(len(selected))
        ax.barh(y, selected["importance_norm"], color=selected["science_group"].map(GROUP_COLORS), alpha=.90)
        r = selected["direction_corr"].to_numpy(float)
        ax.scatter(np.full(len(y), 1.075), y, s=22 + 32 * np.abs(r), c=r, cmap=cmap, norm=norm,
                   edgecolor="white", linewidth=.4, zorder=3)
        ax.axvline(1.0, color="#8f8f8f", linewidth=.6)
        ax.set(yticks=y, yticklabels=selected["feature_label"], xlim=(0, 1.16))
        ax.set_xticks([0, .25, .5, .75, 1.0])
        ax.tick_params(axis="y", labelsize=7)
        ax.text(1.075, len(y) - .1, "corr.", ha="center", va="bottom", color="#555555")
        panel_label(ax, label, x=-.17, y=1.06)
        proportions = shares.loc[shares["experiment"].eq(target)].set_index("science_group")["share"]
        left = 0.0
        for group in share_order:
            value = proportions[group]
            share_ax.barh([0], [value], left=left, height=.50, color=GROUP_COLORS[group], edgecolor="white", linewidth=.5)
            if value >= .075:
                share_ax.text(left + value / 2, 0, f"{value * 100:.0f}%", ha="center", va="center",
                              fontsize=6.3, color="white", weight="bold")
            left += value
        share_ax.set(xlim=(0, 1), ylim=(-.42, .42))
        share_ax.axis("off")
        if col == 0:
            annual_rank_ax = ax
            share_ax.text(-.055, .5, "Summed gain\nshare (%)", ha="right", va="center", transform=share_ax.transAxes)

    group_keys = [Patch(color=GROUP_COLORS[group], label=group) for group in share_order]
    fig.legend(handles=group_keys, loc="lower center", bbox_to_anchor=(.27, .011), ncol=2, fontsize=6.7)
    size_keys = [Line2D([], [], marker="o", linestyle="none", markersize=np.sqrt(22 + 32 * val),
                        markerfacecolor="#777777", markeredgecolor="white", label=f"{val:.1f}") for val in (.2, .4, .6)]
    fig.legend(handles=size_keys, loc="lower center", bbox_to_anchor=(.55, .032), ncol=3, fontsize=6.5)
    fig.text(.55, .015, r"$|r|$", ha="center", va="bottom")
    scale = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([.77, .048, .18, .017]),
                         orientation="horizontal")
    scale.set_label("Predictor–target correlation")
    fig.subplots_adjust(left=.15, right=.985, top=.96, bottom=.105)
    panel_label(ax_a, "a", x=-.17, y=1.06, reference=annual_rank_ax)
    save_figure(fig, output, "figure08")


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
