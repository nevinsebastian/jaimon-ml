#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config
from src.data_prep import prepare_all
from src.forecast import forecast_horizon, product_level_rollups, top_products_by_week
from src.train import evaluate_and_train, save_artifacts


def plot_backtest_sample(test_out: pd.DataFrame, n_skus: int = 6):
    top = test_out.groupby("sku")["qty"].sum().sort_values(ascending=False).head(n_skus).index
    fig, axes = plt.subplots(n_skus, 1, figsize=(11, 2.2 * n_skus), sharex=True)
    if n_skus == 1:
        axes = [axes]
    for ax, sku in zip(axes, top):
        sub = test_out[test_out["sku"] == sku].sort_values("week_start")
        ax.plot(sub["week_start"], sub["qty"], label="actual", marker="o", ms=3)
        ax.plot(sub["week_start"], sub["pred_model"], label="forecast", marker="o", ms=3)
        ax.set_title(sku[:80], fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylabel("units")
    fig.tight_layout()
    out = config.REPORT_DIR / "backtest_sample.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main():
    tagged, channel_summary, panel = prepare_all()
    print(f"Rows: {len(panel):,} | SKUs: {panel['sku'].nunique()} | Weeks: {panel['week_start'].nunique()}")

    artifacts, test_out, sku_eval, importance, panel = evaluate_and_train(panel)
    model_path = save_artifacts(artifacts, test_out, sku_eval, importance, channel_summary, panel)

    forecast = forecast_horizon(panel, model_path=model_path, horizon=config.HORIZON_WEEKS)
    top_weekly = top_products_by_week(forecast, top_n=25)
    by_product = product_level_rollups(forecast)

    forecast.to_csv(config.OUTPUT_DIR / "forecast_sku_weekly.csv", index=False)
    top_weekly.to_csv(config.OUTPUT_DIR / "forecast_top_skus_by_week.csv", index=False)
    by_product.to_csv(config.OUTPUT_DIR / "forecast_product_weekly.csv", index=False)

    plot_backtest_sample(test_out)
    print(f"Model saved: {model_path}")
    print(f"Forecasts saved in: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
