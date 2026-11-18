"""Area-weighted time series used by Figures 5 and 7."""

from pathlib import Path

import numpy as np
import pandas as pd


def weighted_mean(group: pd.DataFrame, value: str) -> float:
    mask = group[value].notna() & group["Area"].gt(0)
    if not mask.any():
        return float("nan")
    return float(np.average(group.loc[mask, value], weights=group.loc[mask, "Area"]))


def weighted_sigma(group: pd.DataFrame, field: str) -> float:
    mask = group[field].notna() & group["Area"].gt(0)
    if not mask.any():
        return float("nan")
    weights = group.loc[mask, "Area"].to_numpy(float)
    sigmas = group.loc[mask, field].to_numpy(float)
    return float(np.sqrt(np.sum((weights * sigmas) ** 2)) / weights.sum())


def sample_series(root: Path, target: str) -> pd.DataFrame:
    name = f"{target}_reconstruction_product_1979_2025_uncertainty.csv"
    values = pd.read_csv(root / "results/reconstruction" / name, low_memory=False)
    field = f"{target}_recon"
    values = values.loc[values["O1Region"].ne(19) & values[field].notna()]
    rows = []
    for year, yearly in values.groupby("year", sort=True):
        row = {
            "year": int(year),
            f"{target}_network_mean": weighted_mean(yearly, field),
            f"{target}_network_sigma": weighted_sigma(yearly, f"{target}_total_sigma"),
            f"{target}_network_n": yearly["RGIId"].nunique(),
        }
        if target == "annual":
            row["annual_observed_sample_mean"] = weighted_mean(yearly, "annual_obs")
            row["annual_observed_n"] = yearly.loc[yearly["annual_obs"].notna(), "RGIId"].nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def annual_aggregation(root: Path) -> pd.DataFrame:
    """Return both annual summary series, with one-standard-deviation uncertainty."""
    by_region = pd.read_csv(root / "results/reconstruction/global_timeseries_uncertainty.csv")
    by_region = by_region[["year", "annual_mean", "annual_sigma"]].rename(columns={
        "annual_mean": "region_area_reweighted_mean",
        "annual_sigma": "region_area_reweighted_sigma",
    })
    sample = sample_series(root, "annual").loc[:, ["year", "annual_network_mean", "annual_network_sigma"]]
    sample = sample.rename(columns={
        "annual_network_mean": "completed_series_mean",
        "annual_network_sigma": "completed_series_sigma",
    })
    result = by_region.merge(sample, on="year", validate="one_to_one").sort_values("year")
    if result["year"].tolist() != list(range(1979, 2026)):
        raise ValueError("Annual aggregation requires every year from 1979 to 2025.")
    for series in ("region_area_reweighted", "completed_series"):
        result[f"{series}_cumulative"] = result[f"{series}_mean"].cumsum()
    return result
