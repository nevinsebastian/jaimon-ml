#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"

LABEL_PAST = "Past sales"
LABEL_FORECAST = "Expected sales"
LABEL_PREDICTED = "What we predicted"

st.set_page_config(
    page_title="Sales Forecast",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading…")
def load_all():
    hist = pd.read_csv(OUT / "weekly_demand_panel.csv", parse_dates=["week_start"])
    forecast = pd.read_csv(OUT / "forecast_sku_weekly.csv", parse_dates=["week_start"])
    channel = pd.read_csv(OUT / "channel_summary.csv")

    holdout_metrics = None
    holdout_weekly = holdout_monthly = holdout_sku = None
    hm_path = REPORTS / "holdout_metrics.json"
    if hm_path.exists():
        holdout_metrics = json.loads(hm_path.read_text())
        holdout_weekly = pd.read_csv(OUT / "holdout_weekly_compare.csv", parse_dates=["week_start"])
        holdout_monthly = pd.read_csv(OUT / "holdout_monthly_compare.csv")
        holdout_sku = pd.read_csv(OUT / "holdout_sku_wape.csv")

    hist_slim = hist.rename(columns={"qty": "units"})[
        ["week_start", "sku", "Brand", "Product", "Packet_Size", "Main_Category", "Final_Category", "units"]
    ]
    hist_slim["series"] = LABEL_PAST

    fc = forecast.rename(columns={"forecast_qty_units": "units"}).copy()
    fc["series"] = LABEL_FORECAST

    combined = pd.concat([hist_slim, fc], ignore_index=True)
    combined["month"] = combined["week_start"].dt.to_period("M").astype(str)
    fc["month"] = fc["week_start"].dt.to_period("M").astype(str)
    hist_slim["month"] = hist_slim["week_start"].dt.to_period("M").astype(str)

    return {
        "hist": hist_slim,
        "forecast": fc,
        "combined": combined,
        "channel": channel,
        "holdout_metrics": holdout_metrics,
        "holdout_weekly": holdout_weekly,
        "holdout_monthly": holdout_monthly,
        "holdout_sku": holdout_sku,
    }


def apply_filters(df, brands, categories, products, skus):
    out = df
    if brands:
        out = out[out["Brand"].isin(brands)]
    if categories:
        out = out[out["Main_Category"].isin(categories)]
    if products:
        out = out[out["Product"].isin(products)]
    if skus:
        out = out[out["sku"].isin(skus)]
    return out


def accuracy_label(wape: float) -> str:
    pct = wape * 100
    if pct <= 15:
        return "Excellent"
    if pct <= 25:
        return "Good"
    if pct <= 35:
        return "Fair"
    return "Needs improvement"


def style_fig(fig, title: str, y_label: str = "Number of packets"):
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=16)),
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
        template="plotly_white",
        height=420,
        font=dict(size=13),
    )
    fig.update_yaxes(title_text=y_label)
    return fig


def friendly_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "week_start" in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out["week_start"]):
            out["week_start"] = out["week_start"].dt.strftime("%d %b %Y")
    rename = {
        "week_start": "Week starting",
        "month": "Month",
        "sku": "Product name (full)",
        "Brand": "Brand",
        "Product": "Product",
        "Packet_Size": "Pack size",
        "Main_Category": "Category",
        "units": "Expected packets",
        "forecast_units": "Expected packets",
        "forecast_qty_units": "Expected packets",
        "actual": "Actual packets sold",
        "predicted": "Predicted packets",
        "abs_err": "Difference",
        "WAPE": "Error rate",
        "WAPE_pct": "Error rate (%)",
        "skus": "Number of items",
        "products": "Number of products",
    }
    return out.rename(columns={k: v for k, v in rename.items() if k in out.columns})


data = load_all()
combined_all = data["combined"]

st.sidebar.title("Filters")
st.sidebar.markdown(
    "Past sales = what already happened. Expected sales = forecast.\n\n"
    "Leave filters empty to show everything."
)

view_mode = st.sidebar.radio("Show by", ["Week", "Month"])

st.sidebar.markdown("---")
st.sidebar.subheader("Products")

brands = sorted(combined_all["Brand"].dropna().unique())
categories = sorted(combined_all["Main_Category"].dropna().unique())

sel_brand = st.sidebar.multiselect("Brand", brands, default=[])
sel_cat = st.sidebar.multiselect("Category", categories, default=[])

prod_pool = combined_all
if sel_brand:
    prod_pool = prod_pool[prod_pool["Brand"].isin(sel_brand)]
if sel_cat:
    prod_pool = prod_pool[prod_pool["Main_Category"].isin(sel_cat)]
products_f = sorted(prod_pool["Product"].dropna().unique())
sel_prod = st.sidebar.multiselect("Product", products_f, default=[])

sku_pool = prod_pool
if sel_prod:
    sku_pool = sku_pool[sku_pool["Product"].isin(sel_prod)]
skus_f = sorted(sku_pool["sku"].dropna().unique())
sel_sku = st.sidebar.multiselect("Pack / item", skus_f, default=[])

show_past = st.sidebar.checkbox("Past sales", value=True)
show_future = st.sidebar.checkbox("Expected sales", value=True)
top_n = st.sidebar.slider("Top products in lists", 5, 25, 10)

hist_f = apply_filters(data["hist"], sel_brand, sel_cat, sel_prod, sel_sku)
fc_f = apply_filters(data["forecast"], sel_brand, sel_cat, sel_prod, sel_sku)
comb_f = apply_filters(data["combined"], sel_brand, sel_cat, sel_prod, sel_sku)

keep = []
if show_past:
    keep.append(LABEL_PAST)
if show_future:
    keep.append(LABEL_FORECAST)
comb_f = comb_f[comb_f["series"].isin(keep)]

st.title("Sales Forecast Dashboard")
st.markdown(
    f"""
See **what sold before** and **what is expected to sell next** — by week or by month.

| | Dates |
|---|---|
| **Past sales data** | {data['hist']['week_start'].min().strftime('%d %b %Y')} → {data['hist']['week_start'].max().strftime('%d %b %Y')} |
| **Future forecast** | {data['forecast']['week_start'].min().strftime('%d %b %Y')} → {data['forecast']['week_start'].max().strftime('%d %b %Y')} |
"""
)

holdout_wape = None
if data["holdout_metrics"]:
    holdout_wape = data["holdout_metrics"]["walkforward_1step"]["model"]["WAPE"]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Expected total (forecast)", f"{fc_f['units'].sum():,.0f} packets")
with c2:
    st.metric("Products", f"{comb_f['Product'].nunique():,}")
with c3:
    if holdout_wape is not None:
        st.metric("Tested error rate", f"{holdout_wape * 100:.0f}%")
        st.caption(f"Rating: {accuracy_label(holdout_wape)}")
    else:
        st.metric("Tested error rate", "Not run yet")
with c4:
    st.metric("Items tracked", f"{comb_f['sku'].nunique():,}")

tab_home, tab_week, tab_month, tab_products, tab_test, tab_download = st.tabs(
    [
        "Home — summary",
        "Week by week",
        "Month by month",
        "Product details",
        "How accurate is this?",
        "Download data",
    ]
)

# Home
with tab_home:
    st.subheader("At a glance")
    st.markdown("Blue line = **past sales**. Orange line = **expected future sales**.")

    if view_mode == "Week":
        ts = comb_f.groupby(["week_start", "series"], as_index=False)["units"].sum()
        xcol, xlab = "week_start", "Week"
    else:
        ts = comb_f.groupby(["month", "series"], as_index=False)["units"].sum()
        xcol, xlab = "month", "Month"

    fig = px.line(
        ts, x=xcol, y="units", color="series", markers=True,
        color_discrete_map={LABEL_PAST: "#1f4e79", LABEL_FORECAST: "#e67e22"},
        labels={"units": "Packets", "series": ""},
    )
    style_fig(fig, f"Total sales over time (by {xlab.lower()})")
    fig.update_xaxes(title_text=xlab)
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([1, 1])
    with left:
        cat_mix = fc_f.groupby("Main_Category", as_index=False)["units"].sum().sort_values("units", ascending=False)
        fig2 = px.pie(cat_mix, names="Main_Category", values="units", hole=0.4,
                      labels={"units": "Packets", "Main_Category": "Category"})
        style_fig(fig2, "Expected sales — share by category", y_label="")
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, width="stretch")

    with right:
        st.markdown("### What to focus on next week")
        if not fc_f.empty:
            first_week = fc_f["week_start"].min()
            top_items = fc_f[fc_f["week_start"] == first_week].nlargest(top_n, "units")
            for _, row in top_items.iterrows():
                st.markdown(f"- **{row['Product']}** ({row['Packet_Size']}) — **{row['units']:,.0f}** packets expected")
            st.caption(f"Week starting {first_week.strftime('%d %b %Y')}")

    st.markdown("---")
    st.markdown("### Sales trend by category")
    grain = "week_start" if view_mode == "Week" else "month"
    cat_ts = comb_f.groupby([grain, "Main_Category"], as_index=False)["units"].sum()
    fig3 = px.area(cat_ts, x=grain, y="units", color="Main_Category",
                   labels={"units": "Packets", "Main_Category": "Category"})
    style_fig(fig3, "How each category is selling over time", y_label="Packets")
    fig3.update_layout(height=450)
    st.plotly_chart(fig3, width="stretch")

# Week by week
with tab_week:
    st.subheader("Expected sales — week by week")
    st.markdown("Use this for **short-term planning**: stock, production, and dispatch.")

    weekly = (
        fc_f.groupby("week_start", as_index=False)
        .agg(**{"Expected packets": ("units", "sum"), "Products": ("Product", "nunique")})
        .sort_values("week_start")
    )
    fig = px.bar(weekly, x="week_start", y="Expected packets", text_auto=".2s")
    style_fig(fig, "Total expected packets each week")
    fig.update_traces(marker_color="#e67e22")
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Week totals")
    st.dataframe(
        friendly_table(weekly),
        width="stretch",
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("#### Pick a week — see every product")
    week_options = sorted(fc_f["week_start"].unique())
    if week_options:
        pick = st.selectbox(
            "Which week?",
            week_options,
            format_func=lambda x: f"Week of {pd.Timestamp(x).strftime('%d %b %Y')}",
        )
        detail = fc_f[fc_f["week_start"] == pick].sort_values("units", ascending=False)
        st.dataframe(
            friendly_table(detail[["sku", "Product", "Packet_Size", "Main_Category", "units"]]),
            width="stretch",
            hide_index=True,
            height=400,
        )

    if len(weekly) >= 2:
        st.markdown("---")
        st.markdown("#### Week-to-week change")
        st.caption("Green = expected sales going up. Red = going down.")
        w2w = fc_f.groupby("week_start", as_index=False)["units"].sum().sort_values("week_start")
        w2w["change_pct"] = w2w["units"].pct_change() * 100
        fig = px.bar(
            w2w.dropna(), x="week_start", y="change_pct", color="change_pct",
            color_continuous_scale=["#c0392b", "#f5f5f5", "#27ae60"],
            color_continuous_midpoint=0,
            labels={"change_pct": "Change (%)", "week_start": "Week"},
        )
        style_fig(fig, "How much expected sales change from one week to the next", y_label="Change (%)")
        st.plotly_chart(fig, width="stretch")

# Month by month
with tab_month:
    st.subheader("Expected sales — month by month")
    st.markdown("Use this for **monthly targets**, purchase planning, and management reports.")

    monthly = (
        fc_f.groupby("month", as_index=False)
        .agg(**{"Expected packets": ("units", "sum"), "Products": ("Product", "nunique")})
        .sort_values("month")
    )
    fig = px.bar(monthly, x="month", y="Expected packets", text_auto=".2s", color="Expected packets",
                 color_continuous_scale="Oranges", labels={"month": "Month"})
    style_fig(fig, "Total expected packets each month")
    st.plotly_chart(fig, width="stretch")

    hist_m = hist_f.groupby("month", as_index=False)["units"].sum()
    hist_m["Type"] = LABEL_PAST
    fc_m = fc_f.groupby("month", as_index=False)["units"].sum()
    fc_m["Type"] = LABEL_FORECAST
    both_m = pd.concat([
        hist_m.rename(columns={"units": "Packets"}),
        fc_m.rename(columns={"units": "Packets"}),
    ])
    fig2 = px.bar(both_m, x="month", y="Packets", color="Type", barmode="group",
                  color_discrete_map={LABEL_PAST: "#1f4e79", LABEL_FORECAST: "#e67e22"},
                  labels={"month": "Month"})
    style_fig(fig2, "Past months vs forecast months")
    st.plotly_chart(fig2, width="stretch")

    st.markdown("#### Top products each month")
    mprod = fc_f.groupby(["month", "Product"], as_index=False)["units"].sum()
    top_products = mprod.groupby("Product")["units"].sum().nlargest(top_n).index
    heat = mprod[mprod["Product"].isin(top_products)]
    pivot = heat.pivot(index="Product", columns="month", values="units").fillna(0)
    fig3 = px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrRd",
                     labels=dict(color="Expected packets", x="Month", y="Product"))
    style_fig(fig3, f"Expected packets — top {top_n} products (darker = more)", y_label="Product")
    fig3.update_layout(height=max(400, top_n * 28))
    st.plotly_chart(fig3, width="stretch")

    st.dataframe(friendly_table(monthly), width="stretch", hide_index=True)

# Products
with tab_products:
    st.subheader("Which products will sell the most?")
    st.markdown("Ranked list for the **upcoming forecast period** (based on your filters).")

    prod_rank = (
        fc_f.groupby(["Product", "Main_Category", "Brand"], as_index=False)["units"]
        .sum()
        .sort_values("units", ascending=False)
        .head(top_n)
    )
    fig = px.bar(
        prod_rank.sort_values("units"), x="units", y="Product", color="Main_Category",
        orientation="h", labels={"units": "Expected packets", "Product": "Product", "Main_Category": "Category"},
    )
    style_fig(fig, f"Top {top_n} products — expected total sales")
    fig.update_layout(height=max(380, top_n * 28))
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.markdown("#### Look at one product in detail")
    sku_choices = sorted(fc_f["sku"].unique()) if not fc_f.empty else []
    if sku_choices:
        top_sku = fc_f.groupby("sku")["units"].sum().sort_values(ascending=False).index[0]
        default_ix = sku_choices.index(top_sku) if top_sku in sku_choices else 0
        pick_sku = st.selectbox(
            "Choose product (full name with pack size)",
            sku_choices,
            index=default_ix,
        )
        sku_hist = data["hist"][data["hist"]["sku"] == pick_sku]
        sku_fc = data["forecast"][data["forecast"]["sku"] == pick_sku]

        fig = go.Figure()
        if not sku_hist.empty:
            fig.add_trace(go.Scatter(
                x=sku_hist["week_start"], y=sku_hist["units"], name=LABEL_PAST,
                mode="lines+markers", line=dict(color="#1f4e79"),
            ))
        if not sku_fc.empty:
            fig.add_trace(go.Scatter(
                x=sku_fc["week_start"], y=sku_fc["units"], name=LABEL_FORECAST,
                mode="lines+markers", line=dict(color="#e67e22", dash="dash"),
            ))
        style_fig(fig, f"Past vs expected — {pick_sku}")
        st.plotly_chart(fig, width="stretch")

        c1, c2, c3 = st.columns(3)
        c1.metric("Sold in past period", f"{sku_hist['units'].sum():,.0f} packets")
        c2.metric("Expected in forecast", f"{sku_fc['units'].sum():,.0f} packets")
        c3.metric("Average per week (forecast)", f"{sku_fc['units'].mean():,.0f} packets" if not sku_fc.empty else "—")

# Accuracy
with tab_test:
    st.subheader("Can we trust these numbers?")
    st.markdown("We trained on older months, predicted newer months, and compared to real sales.")

    hm = data.get("holdout_metrics")
    if hm is None:
        st.warning("Run `python run_holdout_validation.py` first (see README).")
    else:
        st.success(
            f"**Test setup:** Learn from **{hm['train_start']} to {hm['train_end']}** → "
            f"Predict **{hm['test_start']} to {hm['test_end']}** → Compare to real sales"
        )

        mm = hm["walkforward_1step"]["model"]
        err_pct = mm["WAPE"] * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Error rate", f"{err_pct:.0f}%")
        c2.metric("Rating", accuracy_label(mm["WAPE"]))
        c3.metric("Typical gap", f"{mm['MAE']:.0f} packets/week")

        st.markdown(
            f"On average, predictions were about **{err_pct:.0f}%** off compared to actual sales."
        )

        weekly_h = data["holdout_weekly"]
        weekly_h = weekly_h[weekly_h["method"] == "walkforward_1step"]
        monthly_h = data["holdout_monthly"]
        monthly_h = monthly_h[monthly_h["method"] == "walkforward_1step"]

        left, right = st.columns(2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weekly_h["week_start"], y=weekly_h["actual"], name="What actually sold",
            line=dict(color="#1f4e79"),
        ))
        fig.add_trace(go.Scatter(
            x=weekly_h["week_start"], y=weekly_h["predicted"], name=LABEL_PREDICTED,
            line=dict(color="#e67e22", dash="dash"),
        ))
        style_fig(fig, "Test period — real sales vs our prediction (by week)")
        left.plotly_chart(fig, width="stretch")

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=monthly_h["month"], y=monthly_h["actual"], name="What actually sold", marker_color="#1f4e79"))
        fig2.add_trace(go.Bar(x=monthly_h["month"], y=monthly_h["predicted"], name=LABEL_PREDICTED, marker_color="#e67e22"))
        fig2.update_layout(barmode="group")
        style_fig(fig2, "Test period — real sales vs our prediction (by month)")
        right.plotly_chart(fig2, width="stretch")

        st.markdown("#### Month-by-month — how close were we?")
        show_m = monthly_h.copy()
        show_m["Error rate (%)"] = (show_m["WAPE"] * 100).round(1)
        show_m["Actual sold"] = show_m["actual"].round(0).astype(int)
        show_m["We predicted"] = show_m["predicted"].round(0).astype(int)
        st.dataframe(
            show_m[["month", "Actual sold", "We predicted", "Error rate (%)"]].rename(columns={"month": "Month"}),
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### Products with largest prediction gaps")
        st.caption("Higher error rate = harder to predict (often very high or very low volume items).")
        sku_h = data["holdout_sku"]
        sku_h = sku_h[sku_h["method"] == "walkforward_1step"].head(top_n)
        sku_show = sku_h.copy()
        sku_show["Error rate (%)"] = (sku_show["WAPE"] * 100).round(1)
        fig3 = px.bar(
            sku_show.sort_values("WAPE", ascending=True),
            x="Error rate (%)", y="sku", orientation="h",
            labels={"sku": "Product"},
        )
        style_fig(fig3, f"Prediction error by product (top {top_n} by sales volume)", y_label="Product")
        fig3.update_layout(height=max(380, top_n * 28))
        st.plotly_chart(fig3, width="stretch")

# Download
with tab_download:
    st.subheader("Download reports")
    st.markdown("Export tables to Excel or share with your team.")

    kind = st.radio(
        "What do you want to download?",
        [
            "Expected sales — week by week (each product)",
            "Expected sales — by product & week",
            "Past sales — week by week",
        ],
        horizontal=False,
    )

    if kind.startswith("Expected sales — week"):
        table = fc_f.sort_values(["week_start", "units"], ascending=[True, False])
    elif kind.startswith("Expected sales — by product"):
        table = (
            fc_f.groupby(["week_start", "Product", "Main_Category"], as_index=False)["units"]
            .sum()
            .sort_values(["week_start", "units"], ascending=[True, False])
        )
    else:
        table = hist_f.sort_values(["week_start", "units"], ascending=[True, False])

    st.dataframe(friendly_table(table), width="stretch", hide_index=True, height=400)
    st.download_button(
        label="Download as Excel/CSV file",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="sales_forecast_export.csv",
        mime="text/csv",
        help="Open in Excel or Google Sheets",
    )

    st.markdown("---")
    st.markdown("#### Why bulk orders are handled separately")
    ch = data["channel"].sort_values("qty_sum", ascending=False)
    ch_show = ch.rename(columns={
        "channel": "Sales type",
        "qty_sum": "Total packets",
        "qty_mean": "Avg per bill line",
        "bulk_lines": "Large orders flagged",
    })
    fig = px.bar(ch_show, x="Sales type", y="Total packets", text_auto=".2s")
    style_fig(fig, "Where sales come from (all transaction types)")
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig, width="stretch")

st.caption("Refresh browser after re-running the forecast.")
