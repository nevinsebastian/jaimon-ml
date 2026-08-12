from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def classify_channel(voucher_type: str) -> str:
    v = str(voucher_type).strip()
    vl = v.lower()
    if v in config.EXPORT_VOUCHERS:
        return "export"
    if v in config.BULK_SPECIAL_VOUCHERS:
        return "bulk_special"
    if any(k in vl for k in config.COUNTER_KEYWORDS):
        return "counter"
    if any(k in vl for k in config.TAB_KEYWORDS):
        return "tab"
    if any(k in vl for k in config.ROUTE_KEYWORDS):
        return "route"
    if any(k in vl for k in config.VANSALES_KEYWORDS):
        return "vansales"
    if "gst" in vl or v == "Sales":
        return "wholesale"
    return "other"


def load_transactions(path=None) -> pd.DataFrame:
    path = path or config.DATA_PATH
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%y")
    df[config.QTY_COL] = pd.to_numeric(df["quantity_in_kg"], errors="coerce").fillna(0)
    df["sku"] = (
        df["Cleaned"]
        .fillna(df["item_name"])
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    df["channel"] = df["voucher_type"].map(classify_channel)
    df["week_start"] = df["date"].dt.to_period("W-SUN").dt.start_time
    for col in ["Brand", "Product", "Packet_Size", "Packet_Type", "Main_Category", "Final_Category", "season"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    return df


def flag_bulk_lines(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_bulk_channel"] = out["channel"].isin(["export", "bulk_special"])

    thresholds = (
        out.groupby(["sku", "channel"])[config.QTY_COL]
        .transform(lambda s: np.nanpercentile(s, config.BULK_PERCENTILE * 100) if len(s) else np.nan)
    )
    abs_floor = out["channel"].map(config.MIN_BULK_ABS).fillna(200)
    out["bulk_threshold"] = np.maximum(thresholds.fillna(abs_floor), abs_floor)
    out["is_bulk_outlier"] = out[config.QTY_COL] > out["bulk_threshold"]
    out["is_bulk"] = out["is_bulk_channel"] | out["is_bulk_outlier"]

    out["qty_capped"] = np.where(
        out["is_bulk_channel"],
        0.0,
        np.minimum(out[config.QTY_COL], out["bulk_threshold"]),
    )
    return out


def channel_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("channel", as_index=False)
        .agg(
            lines=(config.QTY_COL, "count"),
            qty_sum=(config.QTY_COL, "sum"),
            qty_mean=(config.QTY_COL, "mean"),
            qty_median=(config.QTY_COL, "median"),
            qty_max=(config.QTY_COL, "max"),
            bulk_lines=("is_bulk", "sum"),
        )
        .sort_values("qty_sum", ascending=False)
    )


def build_weekly_demand(df: pd.DataFrame, mode: str | None = None) -> pd.DataFrame:
    mode = mode or config.DEMAND_MODE
    work = df.copy()

    if mode == "regular_capped":
        work = work[~work["is_bulk_channel"]].copy()
        qty_col = "qty_capped"
    elif mode == "counter_only":
        work = work[work["channel"] == "counter"].copy()
        qty_col = "qty_capped"
    elif mode == "all_raw":
        qty_col = config.QTY_COL
    else:
        raise ValueError(f"Unknown demand mode: {mode}")

    meta = (
        work.groupby("sku", as_index=False)
        .agg(
            Brand=("Brand", "first"),
            Product=("Product", "first"),
            Packet_Size=("Packet_Size", "first"),
            Packet_Type=("Packet_Type", "first"),
            Main_Category=("Main_Category", "first"),
            Final_Category=("Final_Category", "first"),
        )
    )

    weekly = (
        work.groupby(["week_start", "sku"], as_index=False)
        .agg(
            qty=(qty_col, "sum"),
            n_lines=(qty_col, "count"),
            n_bulk=("is_bulk", "sum"),
            amount=("amount", "sum"),
        )
    )

    all_weeks = pd.Index(sorted(weekly["week_start"].unique()))
    skus = weekly["sku"].unique()
    panel = pd.MultiIndex.from_product([all_weeks, skus], names=["week_start", "sku"]).to_frame(index=False)
    panel = panel.merge(weekly, on=["week_start", "sku"], how="left")
    panel["qty"] = panel["qty"].fillna(0.0)
    panel["n_lines"] = panel["n_lines"].fillna(0).astype(int)
    panel["n_bulk"] = panel["n_bulk"].fillna(0).astype(int)
    panel["amount"] = panel["amount"].fillna(0.0)
    panel = panel.merge(meta, on="sku", how="left")
    panel = panel.sort_values(["sku", "week_start"]).reset_index(drop=True)
    panel["demand_mode"] = mode
    return panel


def sku_activity_filter(panel: pd.DataFrame, min_weeks: int | None = None) -> pd.DataFrame:
    min_weeks = min_weeks or config.MIN_WEEKS_HISTORY
    active = panel.loc[panel["qty"] > 0].groupby("sku")["week_start"].nunique()
    keep = active[active >= min_weeks].index
    return panel[panel["sku"].isin(keep)].copy()


def prepare_all(path=None):
    raw = load_transactions(path)
    tagged = flag_bulk_lines(raw)
    summary = channel_summary(tagged)
    panel = build_weekly_demand(tagged, mode=config.DEMAND_MODE)
    panel = sku_activity_filter(panel)
    return tagged, summary, panel
