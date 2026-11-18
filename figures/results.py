"""Plot the annual and seasonal reconstruction results and regional contrasts."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from aggregation import annual_aggregation, sample_series
from common import (COLORS, REGION_NAMES, TARGETS, configure_cartopy, observation_support,
                    observation_tables, panel_label, save_figure)


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


def regional_contrasts(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    recon = root / "results/reconstruction"
    annual = pd.read_csv(recon / "annual_reconstruction_product_1979_2025_uncertainty.csv")
    glaciers = annual.loc[annual["O1Region"].ne(19)].groupby(["RGIId", "O1Region", "glacier_name"], as_index=False,
                                                            dropna=False).agg(
        mean_annual_balance=("annual_recon", "mean"), observed_years=("annual_obs", "count"),
        reconstructed_years=("annual_recon", "count"), Area=("Area", "first"),
    )
    tables = observation_tables(root)
    meta = tables["annual"][["RGIId", "CenLon", "CenLat"]].drop_duplicates("RGIId")
    glaciers = glaciers.merge(meta, on="RGIId", validate="one_to_one")
    region_summaries = []
    for target in TARGETS:
        region_year = pd.read_csv(recon / f"regional_{target}_balance_summary_uncertainty.csv")
        region_year = region_year.loc[region_year["O1Region"].ne(19)]
        means = region_year.groupby("O1Region")["area_weighted_mean_pred"].mean().rename(f"{target}_mean")
        early = region_year.loc[region_year["year"].between(1979, 1999)].groupby("O1Region")["area_weighted_mean_pred"].mean()
        late = region_year.loc[region_year["year"].between(2000, 2025)].groupby("O1Region")["area_weighted_mean_pred"].mean()
        region_summaries.extend([means, early.rename(f"{target}_early"), late.rename(f"{target}_late"),
                                 (late - early).rename(f"{target}_change")])
    regional = pd.concat(region_summaries, axis=1).rename_axis("O1Region").reset_index()
    regional["region_name"] = regional["O1Region"].map(REGION_NAMES)
    support = observation_support(tables).pivot(index="O1Region", columns="experiment", values="observations")
    regional = regional.merge(support.add_prefix("observations_"), on="O1Region", validate="one_to_one")
    return glaciers, regional


def draw_regional_contrasts(root: Path, output: Path) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    configure_cartopy(root)
    glaciers, regional = regional_contrasts(root)
    glaciers.to_csv(output / "figure06_glacier_source.csv", index=False)
    regional.to_csv(output / "figure06_regional_source.csv", index=False)

    fig = plt.figure(figsize=(7.2, 6.42))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.12, 1], hspace=.34, wspace=.65)
    ax_a = fig.add_subplot(grid[0, :], projection=ccrs.PlateCarree())
    ax_b, ax_c = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    geographic = ccrs.PlateCarree()
    ax_a.set_extent([-180, 180, -70, 85], crs=geographic)
    ax_a.set_aspect("auto")
    ax_a.spines["geo"].set_visible(False)
    ax_a.spines["left"].set_visible(True)
    ax_a.spines["bottom"].set_visible(True)
    ax_a.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#e5e5e5", edgecolor="#8f8f8f", linewidth=.35)
    ax_a.set_xticks([-120, -60, 0, 60, 120], crs=geographic)
    ax_a.set_yticks([-60, -30, 0, 30, 60], crs=geographic)
    ax_a.set(xlabel="Longitude", ylabel="Latitude")
    norm = TwoSlopeNorm(vmin=min(-1.5, glaciers["mean_annual_balance"].quantile(.02)), vcenter=0,
                        vmax=max(.5, glaciers["mean_annual_balance"].quantile(.98)))
    maximum = glaciers["observed_years"].max()
    scatter = ax_a.scatter(glaciers["CenLon"], glaciers["CenLat"], c=glaciers["mean_annual_balance"],
                           s=10 + 36 * np.sqrt(glaciers["observed_years"].clip(lower=1) / maximum),
                           cmap="RdBu", norm=norm, edgecolors="white", linewidths=.25, alpha=.92, zorder=3)
    scale = fig.colorbar(scatter, ax=ax_a, orientation="horizontal", fraction=.045, pad=.22,
                         ticks=[-1.5, -1, -.5, 0, .5], extend="min", shrink=.40, anchor=(.85, 1))
    scale.set_label(r"Mean annual balance (m w.e. yr$^{-1}$)")
    map_handles = [Line2D([], [], marker="o", linestyle="none", markersize=np.sqrt(10 + 36 * np.sqrt(n / maximum)),
                          markerfacecolor="#777777", markeredgecolor="white", label=str(n)) for n in (5, 20, 40)]
    ax_a.legend(handles=map_handles, loc="upper center", bbox_to_anchor=(.33, -.30), ncol=3,
                title="Observed annual balances", columnspacing=.7, handletextpad=.3, fontsize=6)

    annual = regional.dropna(subset=["annual_early", "annual_late"]).sort_values("annual_late")
    y = np.arange(len(annual))
    ax_b.hlines(y, annual["annual_early"], annual["annual_late"], color="#bdbdbd", linewidth=1)
    ax_b.scatter(annual["annual_early"], y, color=COLORS["early"], s=20, label="1979–1999", zorder=3)
    ax_b.scatter(annual["annual_late"], y, color=COLORS["late"], s=20, label="2000–2025", zorder=3)
    ax_b.set_yticks(y, annual["region_name"], fontsize=6)
    ax_b.xaxis.set_major_locator(MultipleLocator(.2))
    ax_b.set_xlabel(r"Annual balance (m w.e. yr$^{-1}$)")
    ax_b.legend(loc="lower right", fontsize=6)

    seasonal = regional.dropna(subset=["winter_mean", "summer_mean"]).copy()
    seasonal["amplitude"] = seasonal["winter_mean"] - seasonal["summer_mean"]
    seasonal = seasonal.sort_values("amplitude")
    y = np.arange(len(seasonal))
    ax_c.hlines(y, seasonal["summer_mean"], seasonal["winter_mean"], color="#c9c9c9", linewidth=.9)
    for target in ("winter", "summer"):
        size = 12 + 1.3 * np.sqrt(seasonal[f"observations_{target}"].fillna(0))
        ax_c.scatter(seasonal[f"{target}_mean"], y, s=size, color=COLORS[target], label=target.capitalize(), zorder=3)
    ax_c.set_yticks(y, seasonal["region_name"], fontsize=6)
    ax_c.set_xlabel(r"Seasonal balance (m w.e. yr$^{-1}$)")
    legend = ax_c.legend(loc="upper left", bbox_to_anchor=(.02, -.20), fontsize=6)
    ax_c.add_artist(legend)
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=np.sqrt(12 + 1.3 * np.sqrt(n)),
                      markerfacecolor="#888888", markeredgecolor="none", label=str(n)) for n in (25, 200, 800)]
    ax_c.legend(handles=handles, loc="upper right", bbox_to_anchor=(1, -.20), ncol=3, fontsize=6,
                title="Observed seasonal balances", handletextpad=.2, columnspacing=.6)
    for ax in (ax_b, ax_c):
        ax.axvline(0, color="#333333", linewidth=.65, zorder=0)
    fig.subplots_adjust(left=.17, right=.98, top=.97, bottom=.13)
    panel_label(ax_a, "a", x=-.10)
    for ax, label in zip((ax_b, ax_c), "bc"):
        panel_label(ax, label, x=-.10, y=1.10, reference=ax_a)
    save_figure(fig, output, "figure06")


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
