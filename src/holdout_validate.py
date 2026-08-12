from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import config
from .data_prep import prepare_all
from .features import FEATURE_COLS, build_feature_matrix, make_xy
from .forecast import recursive_forecast
from .train import metrics, predict_qty, recency_weights, train_model, wape

DEFAULT_CUTOFF = pd.Timestamp("2024-10-01")


def _build_bundle(train_panel: pd.DataFrame):
    feats = build_feature_matrix(train_panel)
    train_df, X_tr, y_tr = make_xy(feats, drop_incomplete=True)
    model = train_model(X_tr, np.log1p(y_tr), sample_weight=recency_weights(train_df["week_start"]))
    bundle = {
        "model": model,
        "feature_cols": list(X_tr.columns),
        "blend_model": 0.55,
        "blend_lag1": 0.45,
        "target": "log1p",
        "sku_categories": {
            "sku": sorted(train_panel["sku"].astype(str).unique().tolist()),
            "Brand": sorted(train_panel["Brand"].astype(str).unique().tolist()),
            "Product": sorted(train_panel["Product"].astype(str).unique().tolist()),
            "Packet_Size": sorted(train_panel["Packet_Size"].astype(str).unique().tolist()),
            "Main_Category": sorted(train_panel["Main_Category"].astype(str).unique().tolist()),
            "Final_Category": sorted(train_panel["Final_Category"].astype(str).unique().tolist()),
        },
    }
    return bundle, train_df


def walkforward_one_step(full_panel: pd.DataFrame, train_weeks, test_weeks, bundle) -> pd.DataFrame:
    model = bundle["model"]
    rows = []
    history_weeks = list(train_weeks)

    for tw in test_weeks:
        visible = full_panel[full_panel["week_start"].isin(history_weeks + [tw])].copy()
        feats = build_feature_matrix(visible)
        week_feats = feats[feats["week_start"] == tw].copy()
        if week_feats.empty:
            history_weeks.append(tw)
            continue
        X = week_feats[FEATURE_COLS].fillna(0)
        pred = predict_qty(model, X, week_feats["lag_1"].fillna(0).to_numpy())
        tmp = week_feats[["week_start", "sku", "Product", "Brand", "Main_Category", "Packet_Size", "qty"]].copy()
        tmp = tmp.rename(columns={"qty": "actual"})
        tmp["predicted"] = pred
        rows.append(tmp)
        history_weeks.append(tw)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["abs_err"] = (out["predicted"] - out["actual"]).abs()
    out["err"] = out["predicted"] - out["actual"]
    out["month"] = out["week_start"].dt.to_period("M").astype(str)
    out["method"] = "walkforward_1step"
    return out


def run_holdout(cutoff: pd.Timestamp | None = None):
    cutoff = pd.Timestamp(cutoff or DEFAULT_CUTOFF)
    _, _, panel = prepare_all()

    weeks = np.sort(panel["week_start"].unique())
    train_weeks = weeks[weeks < cutoff]
    test_weeks = weeks[weeks >= cutoff]
    if len(train_weeks) < 12 or len(test_weeks) < 4:
        raise ValueError(f"Bad cutoff {cutoff}: train={len(train_weeks)} test={len(test_weeks)} weeks")

    train_panel = panel[panel["week_start"].isin(train_weeks)].copy()
    test_panel = panel[panel["week_start"].isin(test_weeks)].copy()

    train_skus = train_panel.loc[train_panel["qty"] > 0, "sku"].unique()
    train_panel = train_panel[train_panel["sku"].isin(train_skus)].copy()
    test_panel = test_panel[test_panel["sku"].isin(train_skus)].copy()
    panel_f = panel[panel["sku"].isin(train_skus)].copy()

    bundle, _ = _build_bundle(train_panel)

    preds = recursive_forecast(train_panel, bundle, test_weeks).rename(
        columns={"forecast_qty_kg": "predicted"}
    )
    actuals = test_panel[
        ["week_start", "sku", "qty", "Product", "Brand", "Main_Category", "Packet_Size"]
    ].rename(columns={"qty": "actual"})
    blind = actuals.merge(preds[["week_start", "sku", "predicted"]], on=["week_start", "sku"], how="left")
    blind["predicted"] = blind["predicted"].fillna(0.0)
    blind["abs_err"] = (blind["predicted"] - blind["actual"]).abs()
    blind["err"] = blind["predicted"] - blind["actual"]
    blind["month"] = blind["week_start"].dt.to_period("M").astype(str)
    blind["method"] = "blind_multistep"

    last_train = (
        train_panel.sort_values("week_start")
        .groupby("sku")
        .tail(1)[["sku", "qty"]]
        .rename(columns={"qty": "naive_last"})
    )
    blind = blind.merge(last_train, on="sku", how="left")
    blind["naive_last"] = blind["naive_last"].fillna(0.0)

    walk = walkforward_one_step(panel_f, list(train_weeks), list(test_weeks), bundle)
    walk = walk.merge(last_train, on="sku", how="left")
    walk["naive_last"] = walk["naive_last"].fillna(0.0)

    overall = {
        "cutoff": str(cutoff.date()),
        "train_start": str(pd.Timestamp(train_weeks.min()).date()),
        "train_end": str(pd.Timestamp(train_weeks.max()).date()),
        "test_start": str(pd.Timestamp(test_weeks.min()).date()),
        "test_end": str(pd.Timestamp(test_weeks.max()).date()),
        "n_train_weeks": int(len(train_weeks)),
        "n_test_weeks": int(len(test_weeks)),
        "n_skus": int(blind["sku"].nunique()),
        "blind_multistep": {
            "model": metrics(blind["actual"], blind["predicted"]),
            "naive_last_train_week": metrics(blind["actual"], blind["naive_last"]),
        },
        "walkforward_1step": {
            "model": metrics(walk["actual"], walk["predicted"]),
        },
    }

    def weekly_table(df: pd.DataFrame) -> pd.DataFrame:
        w = (
            df.groupby("week_start", as_index=False)
            .agg(actual=("actual", "sum"), predicted=("predicted", "sum"))
            .sort_values("week_start")
        )
        w["abs_err"] = (w["predicted"] - w["actual"]).abs()
        cum_a = cum_e = 0.0
        run = []
        for _, r in w.iterrows():
            cum_a += abs(r["actual"])
            cum_e += abs(r["predicted"] - r["actual"])
            run.append(cum_e / cum_a if cum_a else np.nan)
        w["WAPE_cumulative"] = run
        return w

    def monthly_table(df: pd.DataFrame) -> pd.DataFrame:
        m = (
            df.groupby("month", as_index=False)
            .agg(actual=("actual", "sum"), predicted=("predicted", "sum"))
            .sort_values("month")
        )
        m["WAPE"] = m.apply(
            lambda r: abs(r["predicted"] - r["actual"]) / r["actual"] if r["actual"] else np.nan,
            axis=1,
        )
        return m

    def sku_table(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for sku, g in df.groupby("sku"):
            rows.append(
                {
                    "sku": sku,
                    "Product": g["Product"].iloc[0],
                    "Brand": g["Brand"].iloc[0],
                    "Main_Category": g["Main_Category"].iloc[0],
                    "actual": float(g["actual"].sum()),
                    "predicted": float(g["predicted"].sum()),
                    "WAPE": wape(g["actual"].to_numpy(), g["predicted"].to_numpy()),
                }
            )
        return pd.DataFrame(rows).sort_values("actual", ascending=False)

    weekly_blind = weekly_table(blind).assign(method="blind_multistep")
    weekly_walk = weekly_table(walk).assign(method="walkforward_1step")
    monthly_blind = monthly_table(blind).assign(method="blind_multistep")
    monthly_walk = monthly_table(walk).assign(method="walkforward_1step")
    sku_blind = sku_table(blind).assign(method="blind_multistep")
    sku_walk = sku_table(walk).assign(method="walkforward_1step")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    blind.to_csv(config.OUTPUT_DIR / "holdout_blind_sku_week.csv", index=False)
    walk.to_csv(config.OUTPUT_DIR / "holdout_walk_sku_week.csv", index=False)
    pd.concat([weekly_blind, weekly_walk], ignore_index=True).to_csv(
        config.OUTPUT_DIR / "holdout_weekly_compare.csv", index=False
    )
    pd.concat([monthly_blind, monthly_walk], ignore_index=True).to_csv(
        config.OUTPUT_DIR / "holdout_monthly_compare.csv", index=False
    )
    pd.concat([sku_blind, sku_walk], ignore_index=True).to_csv(
        config.OUTPUT_DIR / "holdout_sku_wape.csv", index=False
    )
    walk.to_csv(config.OUTPUT_DIR / "holdout_sku_week_compare.csv", index=False)

    with open(config.REPORT_DIR / "holdout_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)
    joblib.dump(bundle, config.MODEL_DIR / "holdout_train_model.joblib")

    return overall, blind, walk, weekly_walk, monthly_walk, sku_walk
