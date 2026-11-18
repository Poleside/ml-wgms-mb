"""fig06: Spatial and regional contrasts in glacier mass balance."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from common import (COLORS, REGION_NAMES, TARGETS, configure_cartopy, figure_args, observation_support,
                    observation_tables, panel_label, save_figure, set_plot_style)


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


if __name__ == "__main__":
    root, output, dpi = figure_args(__doc__)
    set_plot_style(dpi=dpi)
    draw_regional_contrasts(root, output)
