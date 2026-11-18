"""Plot the seasonal-window and fold-held-out validation results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import COLORS, TARGETS, panel_label, save_figure


WINDOW_LABELS = {
    "observed_month_window": "Reported window", "glacier_history": "Glacier history",
    "nearest_neighbor": "Nearby glaciers", "o1region_ensemble": "Regional ensemble",
}
WINDOW_COLORS = dict(zip(WINDOW_LABELS.values(), ("#3F3F3F", "#4C9F70", "#2B8CBE", "#756BB1")))
WINDOW_TESTS = {
    "glacier_year_withheld": "Single-window-withheld\ntest",
    "glacier_held_out": "Glacier-history-withheld\ntest",
}
SPLITS = {"random_kfold": "10-fold random CV", "logo_rgiid": "LOGO", "loyo_year": "LOYO"}


def window_sources(root: Path) -> pd.DataFrame:
    frames = []
    for season in ("winter", "summer"):
        filename = f"{season}_reconstruction_product_1979_2025_uncertainty.csv"
        product = pd.read_csv(root / "results/reconstruction" / filename, usecols=["season_window_source"])
        counts = product["season_window_source"].value_counts().reindex(WINDOW_LABELS, fill_value=0)
        if counts.sum() != len(product):
            raise ValueError(f"Unrecognised seasonal-window source in {filename}")
        frames.append(counts.rename_axis("source_flag").reset_index(name="n").assign(season=season))
    sources = pd.concat(frames, ignore_index=True)
    sources["source"] = sources["source_flag"].map(WINDOW_LABELS)
    return sources


def window_recovery(root: Path) -> pd.DataFrame:
    records = pd.read_csv(root / "results/analysis/seasonal_window_withheld_validation_records.csv")
    records = records.loc[records["validation_regime"].isin(WINDOW_TESTS)].copy()
    records["boundary_near"] = records["start_month_abs_error"].le(1) & records["end_month_abs_error"].le(1)
    grouped = records.groupby(["season", "validation_regime"], as_index=False)
    return grouped.agg(exact=("exact_key_match", "mean"), within_one_month=("boundary_near", "mean"),
                       mean_absolute_prediction_difference=("abs_delta_pred", "mean"),
                       n_prediction_pairs=("abs_delta_pred", "count"),
                       n=("record_id", "size"))


def draw_window_validation(root: Path, output: Path) -> None:
    sources = window_sources(root)
    recovery = window_recovery(root)
    sources.to_csv(output / "figure03_window_sources.csv", index=False)
    recovery.to_csv(output / "figure03_window_recovery.csv", index=False)

    fig, panels = plt.subplots(2, 2, figsize=(7.2, 5.3))
    ax_a, ax_b, ax_c, ax_d = panels.flat
    for y, season in [(1, "winter"), (0, "summer")]:
        left = 0
        for source, count in sources.loc[sources["season"].eq(season), ["source", "n"]].itertuples(index=False):
            ax_a.barh(y, count, left=left, height=.72, color=WINDOW_COLORS[source],
                      edgecolor="white", linewidth=.5, label=source if season == "winter" else None)
            label_size = 5.4 if source == "Regional ensemble" else 6.3
            ax_a.text(left + count / 2, y, f"{count:,}", ha="center", va="center", color="white", fontsize=label_size)
            left += count
        ax_a.text(left + 370, y, f"{left:,}", va="center", fontsize=7)
    ax_a.set(yticks=[1, 0], yticklabels=["Winter", "Summer"], xlabel="Number of seasonal windows",
             xlim=(0, 15500), ylim=(-.6, 2.05))
    ax_a.legend(loc="upper left", ncol=2, columnspacing=.8, handlelength=1.4)

    x = np.arange(3)
    offsets = {"winter": -.18, "summer": .18}
    gap = sources.loc[sources["source_flag"].ne("observed_month_window")]
    for season, offset in offsets.items():
        subset = gap.loc[gap["season"].eq(season)]
        shares = 100 * subset["n"] / subset["n"].sum()
        bars = ax_b.bar(x + offset, shares, width=.34, color=subset["source"].map(WINDOW_COLORS),
                        edgecolor="#333333", hatch="///" if season == "winter" else None)
        ax_b.bar_label(bars, labels=[f"{s:.1f}%" for s in shares], padding=2, fontsize=6.6)
    ax_b.set(xticks=x, xticklabels=["Glacier\nhistory", "Nearby\nglaciers", "Regional\nensemble"],
             ylabel="Share of assigned windows (%)", ylim=(0, 55))
    ax_b.legend(handles=[Patch(facecolor="white", edgecolor="#333333", hatch="///", label="Winter"),
                         Patch(facecolor="white", edgecolor="#333333", label="Summer")], loc="upper right")

    x = np.arange(2)
    for season, offset in offsets.items():
        subset = recovery.loc[recovery["season"].eq(season)].set_index("validation_regime").loc[list(WINDOW_TESTS)]
        for metric, marker, label, linestyle in [("exact", "o", "exact", ":"),
                                                  ("within_one_month", "s", "within one month", "-")]:
            ax_c.plot(x + offset, 100 * subset[metric], marker=marker, color=COLORS[season],
                      alpha=.55 if metric == "exact" else 1, linestyle=linestyle, linewidth=1.2,
                      markersize=4, label=f"{season.capitalize()}, {label}")
        bars = ax_d.bar(x + offset, subset["mean_absolute_prediction_difference"], width=.34,
                        color=COLORS[season], label=season.capitalize())
        ax_d.bar_label(bars, fmt="%.3f", padding=2, fontsize=6.5)
    for ax in (ax_c, ax_d):
        ax.set_xticks(x, list(WINDOW_TESTS.values()), fontsize=6.4)
    ax_c.set(ylabel="Window recovery (%)", ylim=(20, 101))
    ax_c.legend(loc="lower left", ncol=2, fontsize=6.7)
    ax_d.set(ylabel="Mean absolute prediction difference (m w.e.)", ylim=(0, .215))
    ax_d.legend(loc="upper left")
    for ax, label in zip(panels.flat, "abcd"):
        panel_label(ax, label, y=1.13)
    fig.subplots_adjust(hspace=.44, wspace=.34, left=.10, right=.98, top=.94, bottom=.12)
    save_figure(fig, output, "figure03")


def model_validation_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, metrics = [], []
    for split in SPLITS:
        for target in TARGETS:
            folder = root / "models" / ("annual" if target == "annual" else "annual_seasonal")
            suffix = "oof_predictions" if target == "annual" else "by_fold"
            oof = pd.read_csv(folder / f"xgboost_{target}_{split}_{suffix}.csv").dropna(subset=["obs", "pred"])
            records.append(oof.assign(target=target, split=split))
            obs, pred = oof["obs"], oof["pred"]
            slope, intercept = np.polyfit(obs, pred, 1)
            metrics.append({"split": split, "target": target, "n": len(oof), "r2": r2_score(obs, pred),
                            "rmse": np.sqrt(mean_squared_error(obs, pred)), "mae": mean_absolute_error(obs, pred),
                            "slope": slope, "intercept": intercept})
    return pd.concat(records, ignore_index=True), pd.DataFrame(metrics)


def draw_model_performance(root: Path, output: Path) -> None:
    observations, metrics = model_validation_data(root)
    observations.to_csv(output / "figure04_source.csv", index=False)
    metrics.to_csv(output / "figure04_metrics.csv", index=False)
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.0), sharex="col", sharey="col")
    for col, target in enumerate(TARGETS):
        sample = observations.loc[observations["target"].eq(target)]
        values = sample[["obs", "pred"]].to_numpy().ravel()
        pad = .04 * np.ptp(values)
        limits = np.array([values.min() - pad, values.max() + pad])
        for row, (split, split_label) in enumerate(SPLITS.items()):
            ax = axes[row, col]
            data = sample.loc[sample["split"].eq(split)]
            stats = metrics.loc[metrics["split"].eq(split) & metrics["target"].eq(target)].iloc[0]
            ax.scatter(data["obs"], data["pred"], s=5.5, color=COLORS[target], alpha=.18,
                       linewidths=0, rasterized=True)
            ax.plot(limits, limits, color="black", linewidth=.9)
            ax.plot(limits, stats["intercept"] + stats["slope"] * limits, "--", color="#555555", linewidth=1.1)
            ax.set(xlim=limits, ylim=limits, aspect="equal")
            if target != "annual":
                ax.yaxis.set_major_locator(MultipleLocator(2))
            text = "\n".join(f"{name} = {stats[key]:.3f}" for name, key in
                             [("R²", "r2"), ("RMSE", "rmse"), ("MAE", "mae"), ("Slope", "slope")])
            ax.text(.04, .96, text, transform=ax.transAxes, va="top", fontsize=7.4,
                    bbox={"facecolor": "white", "alpha": .82, "edgecolor": "none", "pad": 1.8})
            panel_label(ax, "abcdefghi"[row * 3 + col], x=-.13, y=1.12)
            if row == 0:
                ax.set_title(f"{target.capitalize()} balance", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{split_label}\nPredicted (m w.e.)", fontsize=8.2)
            if row == 2:
                ax.set_xlabel("Observed (m w.e.)", fontsize=8.2)
    fig.subplots_adjust(left=.12, right=.98, top=.965, bottom=.07, hspace=.28, wspace=.22)
    save_figure(fig, output, "figure04")
