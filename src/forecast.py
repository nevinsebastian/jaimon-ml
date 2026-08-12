from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import FEATURE_COLS, LAGS, ROLL_WINDOWS


def recursive_forecast(
    train_panel: pd.DataFrame,
    bundle: dict,
    future_weeks: list | pd.DatetimeIndex,
) -> pd.DataFrame:
    model = bundle["model"]
    cats = bundle["sku_categories"]
    blend_model = float(bundle.get("blend_model", 0.55))
    blend_lag1 = float(bundle.get("blend_lag1", 0.45))
    use_log = bundle.get("target") == "log1p"

    hist = train_panel.sort_values(["sku", "week_start"]).copy()
    qty_hist = {sku: g["qty"].tolist() for sku, g in hist.groupby("sku")}
    meta = (
        hist.groupby("sku", as_index=False)
        .agg(
            Brand=("Brand", "first"),
            Product=("Product", "first"),
            Packet_Size=("Packet_Size", "first"),
            Packet_Type=("Packet_Type", "first"),
            Main_Category=("Main_Category", "first"),
            Final_Category=("Final_Category", "first"),
        )
    )
    meta_map = meta.set_index("sku").to_dict("index")

    rows = []
    for week in future_weeks:
        week_ts = pd.Timestamp(week)
        for sku, series in qty_hist.items():
            m = meta_map[sku]
            feats = {}
            for lag in LAGS:
                feats[f"lag_{lag}"] = series[-lag] if len(series) >= lag else 0.0
            for w in ROLL_WINDOWS:
                window = series[-w:] if series else [0.0]
                feats[f"roll_mean_{w}"] = float(np.mean(window))
                feats[f"roll_std_{w}"] = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
            feats["lag_diff_1"] = feats["lag_1"] - feats.get("lag_2", 0.0)
            streak = 0
            for v in reversed(series):
                if v == 0:
                    streak += 1
                else:
                    break
            feats["zero_streak_lag1"] = streak
            feats["week_of_year"] = int(week_ts.isocalendar().week)
            feats["month"] = int(week_ts.month)
            feats["quarter"] = int(week_ts.quarter)
            feats["is_festival_month"] = int(week_ts.month in {4, 8, 9, 12, 1})
            feats["is_monsoon"] = int(week_ts.month in {6, 7, 8, 9})
            feats["Brand_code"] = cats["Brand"].index(m["Brand"]) if m["Brand"] in cats["Brand"] else -1
            feats["Product_code"] = cats["Product"].index(m["Product"]) if m["Product"] in cats["Product"] else -1
            feats["Packet_Size_code"] = (
                cats["Packet_Size"].index(m["Packet_Size"]) if m["Packet_Size"] in cats["Packet_Size"] else -1
            )
            feats["Main_Category_code"] = (
                cats["Main_Category"].index(m["Main_Category"]) if m["Main_Category"] in cats["Main_Category"] else -1
            )
            feats["Final_Category_code"] = (
                cats["Final_Category"].index(m["Final_Category"]) if m["Final_Category"] in cats["Final_Category"] else -1
            )
            feats["sku_code"] = cats["sku"].index(sku) if sku in cats["sku"] else -1

            x = pd.DataFrame([{c: feats.get(c, 0) for c in FEATURE_COLS}])
            raw = float(model.predict(x)[0])
            model_pred = float(np.expm1(max(raw, 0.0))) if use_log else float(max(raw, 0.0))
            lag1 = float(feats["lag_1"])
            pred = blend_model * model_pred + blend_lag1 * lag1
            pred_units = float(np.round(max(pred, 0.0), 2))
            series.append(pred_units)
            qty_hist[sku] = series
            rows.append(
                {
                    "week_start": week_ts,
                    "sku": sku,
                    "Brand": m["Brand"],
                    "Product": m["Product"],
                    "Packet_Size": m["Packet_Size"],
                    "Packet_Type": m["Packet_Type"],
                    "Main_Category": m["Main_Category"],
                    "Final_Category": m["Final_Category"],
                    "forecast_qty_units": pred_units,
                }
            )

    forecast = pd.DataFrame(rows)
    if forecast.empty:
        return forecast
    return forecast.sort_values(["week_start", "forecast_qty_units"], ascending=[True, False])


def forecast_horizon(panel: pd.DataFrame, model_path: Path | None = None, horizon: int | None = None) -> pd.DataFrame:
    horizon = horizon or config.HORIZON_WEEKS
    model_path = model_path or (config.MODEL_DIR / "weekly_demand_model.joblib")
    bundle = joblib.load(model_path)
    hist = panel.sort_values(["sku", "week_start"]).copy()
    last_week = hist["week_start"].max()
    future_weeks = pd.date_range(last_week + pd.Timedelta(days=7), periods=horizon, freq="7D")
    return recursive_forecast(hist, bundle, future_weeks)


def top_products_by_week(forecast: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    return (
        forecast.sort_values(["week_start", "forecast_qty_units"], ascending=[True, False])
        .groupby("week_start", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def product_level_rollups(forecast: pd.DataFrame) -> pd.DataFrame:
    return (
        forecast.groupby(["week_start", "Product", "Main_Category"], as_index=False)["forecast_qty_units"]
        .sum()
        .sort_values(["week_start", "forecast_qty_units"], ascending=[True, False])
    )
