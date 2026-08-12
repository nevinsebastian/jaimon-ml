from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config
from .features import FEATURE_COLS, build_feature_matrix, make_xy


def time_split(feats: pd.DataFrame, test_weeks: int | None = None):
    test_weeks = test_weeks or config.TEST_WEEKS
    weeks = np.sort(feats["week_start"].unique())
    if len(weeks) <= test_weeks + 8:
        raise ValueError("Not enough weeks for train/test split")
    cutoff = weeks[-test_weeks]
    train = feats[feats["week_start"] < cutoff].copy()
    test = feats[feats["week_start"] >= cutoff].copy()
    return train, test, cutoff


def wape(y_true, y_pred) -> float:
    denom = np.abs(y_true).sum()
    if denom == 0:
        return np.nan
    return float(np.abs(y_true - y_pred).sum() / denom)


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "WAPE": wape(y_true, y_pred),
        "bias": float((y_pred - y_true).mean()),
    }


def recency_weights(weeks: pd.Series) -> np.ndarray:
    ranked = weeks.rank(method="dense")
    max_r = ranked.max()
    return np.exp((ranked - max_r) / 8.0).to_numpy()


def train_model(X_train, y_train, sample_weight=None) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=500,
        min_samples_leaf=15,
        l2_regularization=2.0,
        random_state=config.RANDOM_SEED,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def predict_qty(model, X, lag1: np.ndarray) -> np.ndarray:
    model_pred = np.expm1(np.clip(model.predict(X), 0, None))
    model_pred = np.clip(model_pred, 0, None)
    lag1 = np.nan_to_num(np.asarray(lag1, dtype=float), nan=0.0)
    blended = 0.55 * model_pred + 0.45 * lag1
    return np.clip(blended, 0, None)


def naive_forecast(test_frame: pd.DataFrame) -> np.ndarray:
    return test_frame["lag_1"].fillna(0).to_numpy()


def seasonal_naive(test_frame: pd.DataFrame) -> np.ndarray:
    pred = test_frame["lag_8"].copy()
    pred = pred.fillna(test_frame["lag_4"]).fillna(test_frame["lag_1"]).fillna(0)
    return pred.to_numpy()


def evaluate_and_train(panel: pd.DataFrame):
    feats = build_feature_matrix(panel)
    train_raw, test_raw, cutoff = time_split(feats)
    train_df, X_tr, y_tr = make_xy(train_raw, drop_incomplete=True)
    test_df, X_te, y_te = make_xy(test_raw, drop_incomplete=False)

    y_tr_log = np.log1p(y_tr)
    sw = recency_weights(train_df["week_start"])
    model = train_model(X_tr, y_tr_log, sample_weight=sw)

    pred = predict_qty(model, X_te, test_df["lag_1"].fillna(0).to_numpy())
    naive = naive_forecast(test_df)
    seas = seasonal_naive(test_df)
    model_only = np.expm1(np.clip(model.predict(X_te), 0, None))
    model_only = np.clip(model_only, 0, None)

    report = {
        "cutoff_week": str(pd.Timestamp(cutoff).date()),
        "n_train_rows": int(len(X_tr)),
        "n_test_rows": int(len(X_te)),
        "n_skus": int(panel["sku"].nunique()),
        "model_blend": metrics(y_te, pred),
        "model_only": metrics(y_te, model_only),
        "naive_lag1": metrics(y_te, naive),
        "seasonal_naive": metrics(y_te, seas),
    }

    test_out = test_df[["week_start", "sku", "Product", "Brand", "Packet_Size", "qty"]].copy()
    test_out["pred_model"] = pred
    test_out["pred_model_only"] = model_only
    test_out["pred_naive"] = naive
    test_out["abs_err"] = (test_out["pred_model"] - test_out["qty"]).abs()

    rows = []
    for sku, g in test_out.groupby("sku"):
        rows.append(
            {
                "sku": sku,
                "actual": float(g["qty"].sum()),
                "pred": float(g["pred_model"].sum()),
                "WAPE": wape(g["qty"].to_numpy(), g["pred_model"].to_numpy()),
            }
        )
    sku_eval = pd.DataFrame(rows).sort_values("actual", ascending=False)

    imp_vals = []
    for col in FEATURE_COLS:
        c = np.corrcoef(X_tr[col].fillna(0), y_tr)[0, 1]
        imp_vals.append(0.0 if np.isnan(c) else abs(c))
    importance = pd.DataFrame({"feature": FEATURE_COLS, "importance": imp_vals}).sort_values(
        "importance", ascending=False
    )

    full_df, X_full, y_full = make_xy(feats, drop_incomplete=True)
    final_model = train_model(X_full, np.log1p(y_full), sample_weight=recency_weights(full_df["week_start"]))

    artifacts = {
        "eval_model": model,
        "final_model": final_model,
        "feature_cols": FEATURE_COLS,
        "blend_model": 0.55,
        "blend_lag1": 0.45,
        "sku_categories": {
            "sku": sorted(panel["sku"].astype(str).unique().tolist()),
            "Brand": sorted(panel["Brand"].astype(str).unique().tolist()),
            "Product": sorted(panel["Product"].astype(str).unique().tolist()),
            "Packet_Size": sorted(panel["Packet_Size"].astype(str).unique().tolist()),
            "Main_Category": sorted(panel["Main_Category"].astype(str).unique().tolist()),
            "Final_Category": sorted(panel["Final_Category"].astype(str).unique().tolist()),
        },
        "cutoff": cutoff,
        "report": report,
    }
    return artifacts, test_out, sku_eval, importance, panel


def save_artifacts(artifacts, test_out, sku_eval, importance, channel_summary, panel):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = config.MODEL_DIR / "weekly_demand_model.joblib"
    joblib.dump(
        {
            "model": artifacts["final_model"],
            "feature_cols": artifacts["feature_cols"],
            "sku_categories": artifacts["sku_categories"],
            "blend_model": artifacts["blend_model"],
            "blend_lag1": artifacts["blend_lag1"],
            "target": "log1p",
        },
        model_path,
    )

    test_out.to_csv(config.OUTPUT_DIR / "backtest_predictions.csv", index=False)
    sku_eval.to_csv(config.OUTPUT_DIR / "backtest_sku_wape.csv", index=False)
    importance.to_csv(config.OUTPUT_DIR / "feature_importance.csv", index=False)
    channel_summary.to_csv(config.OUTPUT_DIR / "channel_summary.csv", index=False)
    panel.to_csv(config.OUTPUT_DIR / "weekly_demand_panel.csv", index=False)

    with open(config.REPORT_DIR / "metrics.json", "w") as f:
        json.dump(artifacts["report"], f, indent=2)

    return model_path
