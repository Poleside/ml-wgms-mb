"""Shared plotting defaults and source paths for the manuscript figures."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


TARGETS = ("annual", "winter", "summer")
COLORS = {
    "annual": "#286FB4", "observed": "#8C8C8C", "winter": "#2A9D7E",
    "summer": "#E16A00", "early": "#77AADD", "late": "#CC6677",
}
REGION_NAMES = dict(enumerate((
    "Alaska", "Western Canada and USA", "Arctic Canada North", "Arctic Canada South",
    "Greenland Periphery", "Iceland", "Svalbard and Jan Mayen", "Scandinavia", "Russian Arctic",
    "North Asia", "Central Europe", "Caucasus and Middle East", "Central Asia", "South Asia West",
    "South Asia East", "Low Latitudes", "Southern Andes", "New Zealand", "Antarctic and Subantarctic",
), start=1))


def set_plot_style(dpi: int = 400) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "savefig.dpi": dpi, "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7.2, "axes.labelsize": 7.4,
        "axes.titlesize": 8.4, "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 2.0,
        "ytick.major.size": 2.0, "legend.fontsize": 7.0, "legend.frameon": False,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.7,
    })


def save_figure(fig, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, label: str, x: float = -0.10, y: float = 1.04, *, reference=None) -> None:
    if reference is not None:
        bounds, reference_bounds = ax.get_position(), reference.get_position()
        x = x * reference_bounds.width / bounds.width
        y = 1 + (y - 1) * reference_bounds.height / bounds.height
    ax.text(x, y, f"({label})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def observation_tables(root: Path) -> dict[str, pd.DataFrame]:
    columns = ["RGIId", "year", "O1Region", "CenLon", "CenLat"]
    annual = pd.read_csv(root / "data/interim/wgms_era5land_merged/wgms_global_rgi_era5land_1979_2025.csv",
                         usecols=[*columns, "mwe"])
    folder = root / "data/processed/seasonal_training"
    tables = {"annual": annual.dropna(subset=["mwe"])}
    for season in ("winter", "summer"):
        tables[season] = pd.read_csv(folder / f"wgms_global_{season}_era5land_1979_2025.csv",
                                     usecols=["RGIId", "year", "O1Region"])
    return tables


def observation_support(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summaries = []
    for target, table in tables.items():
        grouped = table.dropna(subset=["year", "RGIId"]).groupby("O1Region", as_index=False)
        summary = grouped.agg(observations=("year", "size"), glaciers=("RGIId", "nunique"),
                              year_min=("year", "min"), year_max=("year", "max"))
        summaries.append(summary.assign(experiment=target))
    return pd.concat(summaries, ignore_index=True)


def configure_cartopy(root: Path) -> None:
    """Use a locally available Natural Earth cache or let Cartopy retrieve it."""
    import cartopy

    local_cache = root / "figures/cartopy_data"
    if local_cache.is_dir():
        cartopy.config["pre_existing_data_dir"] = str(local_cache)
