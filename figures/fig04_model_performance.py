"""fig04: Fold-held-out model performance."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import COLORS, TARGETS, figure_args, panel_label, save_figure, set_plot_style


SPLITS = {"random_kfold": "10-fold random CV", "logo_rgiid": "LOGO", "loyo_year": "LOYO"}


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


if __name__ == "__main__":
    root, output, dpi = figure_args(__doc__)
    set_plot_style(dpi=dpi)
    draw_model_performance(root, output)
