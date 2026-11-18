"""fig02: Glacier inventory and annual observation coverage."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd

from common import (configure_cartopy, figure_args, observation_support, observation_tables,
                    save_figure, set_plot_style)


def draw_coverage(root: Path, output: Path) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import shapefile

    configure_cartopy(root)
    inventory = root / "data/RGI6"
    regional_files = [path for path in sorted(inventory.glob("[0-9][0-9]_rgi60_*.csv"))
                      if not path.name.startswith("00_") and not path.stem.endswith("_hypso")]
    if len(regional_files) != 19:
        raise FileNotFoundError(f"Expected the 19 RGI 6.0 regional attribute CSVs in {inventory}")
    rgi_centroids = pd.concat([
        pd.read_csv(path, encoding="latin1", usecols=["CenLon", "CenLat"]) for path in regional_files
    ], ignore_index=True).dropna()
    tables = observation_tables(root)
    observation_support(tables).to_csv(output / "figure02_observation_support.csv", index=False)
    annual = tables["annual"]
    samples = annual[["RGIId", "CenLon", "CenLat", "O1Region"]].dropna().drop_duplicates("RGIId")
    duration = annual.groupby("RGIId")["year"].nunique().rename("obs_years")
    samples = samples.join(duration, on="RGIId", validate="one_to_one")
    samples.to_csv(output / "figure02_source.csv", index=False)

    with plt.rc_context({
        "font.family": "Arial", "font.sans-serif": ["Arial"], "font.size": 6,
        "axes.linewidth": .5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
    }):
        fig = plt.figure(figsize=(7.2, 3.55), dpi=600)
        ax = fig.add_axes([.055, .19, .89, .73], projection=ccrs.Robinson())
        ax.set_global()
        geographic = ccrs.PlateCarree()
        ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#eef3f6", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#d8d8d8", edgecolor="none", zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=.34, edgecolor="#6f6f6f", zorder=2)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=.16, edgecolor="#b9b9b9", zorder=2)
        ax.scatter(rgi_centroids["CenLon"], rgi_centroids["CenLat"], transform=geographic,
                   s=.26, color="#3f7394", alpha=.46, edgecolors="none", rasterized=True, zorder=2.5)
        with shapefile.Reader(str(inventory / "00_rgi60_O1Regions.shp")) as reader:
            for shape in reader.iterShapes():
                points = np.asarray(shape.points)
                bounds = [*shape.parts, len(points)]
                for start, end in zip(bounds[:-1], bounds[1:], strict=True):
                    if end - start >= 2:
                        ax.plot(points[start:end, 0], points[start:end, 1], transform=geographic,
                                color="#4f4f4f", linewidth=.62, alpha=.96, zorder=3)
        maximum = int(np.ceil(samples["obs_years"].max()))
        scatter = ax.scatter(samples["CenLon"], samples["CenLat"], transform=geographic, s=19,
                             c=samples["obs_years"], cmap="YlOrRd", norm=Normalize(0, maximum),
                             alpha=.94, edgecolors="white", linewidths=.17, zorder=4)

        longitudes = [-120, -60, 0, 60, 120]
        latitudes = [-60, -30, 0, 30, 60]
        grid = ax.gridlines(crs=geographic, linewidth=.22, color="#9ea7ad", alpha=.58)
        grid.xlocator = FixedLocator(longitudes)
        grid.ylocator = FixedLocator(latitudes)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        for longitude in longitudes:
            x, _ = ax.projection.transform_point(longitude, 85, geographic)
            suffix = "W" if longitude < 0 else "E" if longitude > 0 else ""
            ax.text(x, ymax + .018 * (ymax - ymin), fr"{abs(longitude)}$^\circ${suffix}", ha="center",
                    va="bottom", fontsize=5, color="#222222", clip_on=False)
        for latitude in reversed(latitudes):
            x, y = ax.projection.transform_point(-180, latitude, geographic)
            suffix = "S" if latitude < 0 else "N" if latitude > 0 else ""
            ax.text(x - .012 * (xmax - xmin), y, fr"{abs(latitude)}$^\circ${suffix}", ha="right",
                    va="center", fontsize=5, color="#222222", clip_on=False)

        legend = fig.add_axes([.30, .125, .22, .05])
        legend.axis("off")
        legend.scatter([.23], [.72], s=14, color="#3f7394", alpha=.72, edgecolors="none")
        legend.text(.23, -.1, "RGI glaciers", ha="center", va="bottom", fontsize=5)
        legend.plot([.50, .78], [.72, .72], color="#4f4f4f", linewidth=1)
        legend.text(.64, -.1, "RGI regions", ha="center", va="bottom", fontsize=5)
        legend.set(xlim=(0, 1), ylim=(0, 1))
        scale = fig.colorbar(scatter, cax=fig.add_axes([.51, .15, .175, .02]),
                             orientation="horizontal", ticks=[0, maximum])
        scale.set_label("Annual observation years per glacier", fontsize=5, labelpad=-1)
        scale.ax.tick_params(labelsize=5, direction="in", width=.45, length=2.2, pad=1.8)
        save_figure(fig, output, "figure02", min_render_dpi=600)


if __name__ == "__main__":
    root, output, dpi = figure_args(__doc__)
    set_plot_style(dpi=dpi)
    draw_coverage(root, output)
