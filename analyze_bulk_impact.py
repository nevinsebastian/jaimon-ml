#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config
from src.data_prep import build_weekly_demand, flag_bulk_lines, load_transactions


def main():
    raw = load_transactions()
    tagged = flag_bulk_lines(raw)

    rows = []
    for mode in ["all_raw", "regular_capped", "counter_only"]:
        panel = build_weekly_demand(tagged, mode=mode)
        weekly_total = panel.groupby("week_start")["qty"].sum()
        rows.append(
            {
                "mode": mode,
                "sku_count": panel["sku"].nunique(),
                "total_qty": panel["qty"].sum(),
                "avg_weekly_qty": weekly_total.mean(),
                "max_week_qty": weekly_total.max(),
            }
        )

    comparison = pd.DataFrame(rows)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(config.OUTPUT_DIR / "bulk_impact_comparison.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
