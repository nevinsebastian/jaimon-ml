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

    # Build a lookup: (sku, week_of_year) -> median kg from real history.
    # This is used as a stable seasonal anchor for long-horizon forecasts so
    # predictions don't decay when fed back as lags.
    hist["_woy"] = hist["week_start"].dt.isocalendar().week.astype(int)
    seasonal_anchor: dict[tuple, float] = (
        hist.groupby(["sku", "_woy"])["qty"]
        .median()
        .to_dict()
    )
    # Per-SKU overall median as fallback
    sku_median: dict[str, float] = hist.groupby("sku")["qty"].median().to_dict()

    # Real weekly qty indexed by (sku, week_start) for exact lookups
    real_qty: dict[tuple, float] = {
        (r["sku"], r["week_start"]): r["qty"]
        for _, r in hist.iterrows()
    }
    hist_weeks = sorted(hist["week_start"].unique())

    qty_hist = {sku: g["qty"].tolist() for sku, g in hist.groupby("sku")}
    # Keep a parallel "anchor series" that always uses real or seasonal values
    # for lag/rolling features so we don't amplify compounding errors.
    anchor_hist = {sku: g["qty"].tolist() for sku, g in hist.groupby("sku")}

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
    for week_idx, week in enumerate(future_weeks):
        week_ts = pd.Timestamp(week)
        woy = int(week_ts.isocalendar().week)

        for sku, series in qty_hist.items():
            m = meta_map[sku]
            # Use anchor series (real history + seasonal fill) for lag features
            anc = anchor_hist[sku]

            feats = {}
            for lag in LAGS:
                feats[f"lag_{lag}"] = anc[-lag] if len(anc) >= lag else 0.0
            for w in ROLL_WINDOWS:
                window = anc[-w:] if anc else [0.0]
                feats[f"roll_mean_{w}"] = float(np.mean(window))
                feats[f"roll_std_{w}"] = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
            feats["lag_diff_1"] = feats["lag_1"] - feats.get("lag_2", 0.0)
            streak = 0
            for v in reversed(anc):
                if v == 0:
                    streak += 1
                else:
                    break
            feats["zero_streak_lag1"] = streak
            feats["week_of_year"] = woy
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

            # Seasonal anchor: same week-of-year median from real history
            seasonal = float(seasonal_anchor.get((sku, woy), sku_median.get(sku, 0.0)))

            # For near-term (first 8 weeks) blend model + lag-1.
            # Beyond that, blend model with seasonal anchor to keep predictions stable.
            lag1 = float(feats["lag_1"])
            if week_idx < 8:
                pred = blend_model * model_pred + blend_lag1 * lag1
            else:
                # Gradually increase seasonal anchor weight as horizon grows
                seasonal_weight = min(0.6, 0.3 + (week_idx - 8) * 0.01)
                model_weight = 1.0 - seasonal_weight
                pred = model_weight * model_pred + seasonal_weight * seasonal

            pred_units = float(np.round(max(pred, 0.0), 2))

            # Append prediction to qty_hist (used to track what we predicted)
            series.append(pred_units)
            qty_hist[sku] = series

            # Anchor hist: use prediction but clamp it to [50%, 200%] of seasonal
            # so compounding errors don't spiral out of control
            if seasonal > 0:
                clamped = float(np.clip(pred_units, seasonal * 0.5, seasonal * 2.0))
            else:
                clamped = pred_units
            anc.append(clamped)
            anchor_hist[sku] = anc

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
                    "forecast_qty_kg": pred_units,
                }
            )

    forecast = pd.DataFrame(rows)
    if forecast.empty:
        return forecast
    return forecast.sort_values(["week_start", "forecast_qty_kg"], ascending=[True, False])


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
        forecast.sort_values(["week_start", "forecast_qty_kg"], ascending=[True, False])
        .groupby("week_start", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def product_level_rollups(forecast: pd.DataFrame) -> pd.DataFrame:
    return (
        forecast.groupby(["week_start", "Product", "Main_Category"], as_index=False)["forecast_qty_kg"]
        .sum()
        .sort_values(["week_start", "forecast_qty_kg"], ascending=[True, False])
    )
