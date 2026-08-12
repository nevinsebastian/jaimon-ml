from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [1, 2, 4, 8]
ROLL_WINDOWS = [4, 8, 12]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["week_of_year"] = out["week_start"].dt.isocalendar().week.astype(int)
    out["month"] = out["week_start"].dt.month.astype(int)
    out["quarter"] = out["week_start"].dt.quarter.astype(int)
    out["is_festival_month"] = out["month"].isin([4, 8, 9, 12, 1]).astype(int)
    out["is_monsoon"] = out["month"].isin([6, 7, 8, 9]).astype(int)
    return out


def add_lag_roll_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["sku", "week_start"]).copy()
    g = out.groupby("sku", group_keys=False)
    for lag in LAGS:
        out[f"lag_{lag}"] = g["qty"].shift(lag)
    for w in ROLL_WINDOWS:
        out[f"roll_mean_{w}"] = g["qty"].shift(1).rolling(w, min_periods=1).mean()
        out[f"roll_std_{w}"] = g["qty"].shift(1).rolling(w, min_periods=1).std()
    out["lag_diff_1"] = out["lag_1"] - out["lag_2"]

    def _zero_streak(s: pd.Series) -> pd.Series:
        streak = []
        cur = 0
        for v in s:
            cur = cur + 1 if v == 0 else 0
            streak.append(cur)
        return pd.Series(streak, index=s.index)

    out["zero_streak"] = g["qty"].transform(_zero_streak)
    out["zero_streak_lag1"] = g["zero_streak"].shift(1)
    return out


def add_category_codes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Brand", "Product", "Packet_Size", "Main_Category", "Final_Category"]:
        out[f"{col}_code"] = out[col].astype("category").cat.codes
    out["sku_code"] = out["sku"].astype("category").cat.codes
    return out


def build_feature_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    feats = add_calendar_features(panel)
    feats = add_lag_roll_features(feats)
    feats = add_category_codes(feats)
    return feats


FEATURE_COLS = (
    [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}" for w in ROLL_WINDOWS]
    + [
        "lag_diff_1",
        "zero_streak_lag1",
        "week_of_year",
        "month",
        "quarter",
        "is_festival_month",
        "is_monsoon",
        "Brand_code",
        "Product_code",
        "Packet_Size_code",
        "Main_Category_code",
        "Final_Category_code",
        "sku_code",
    ]
)


def make_xy(feats: pd.DataFrame, drop_incomplete: bool = True):
    data = feats.copy()
    if drop_incomplete:
        data = data[data["lag_8"].notna()].copy()
    X = data[FEATURE_COLS].fillna(0)
    y = data["qty"].astype(float)
    return data, X, y
