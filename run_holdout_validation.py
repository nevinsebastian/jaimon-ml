#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.holdout_validate import run_holdout


def main():
    overall, _, _, _, monthly, sku_eval = run_holdout()

    print(f"Train: {overall['train_start']} to {overall['train_end']}")
    print(f"Test:  {overall['test_start']} to {overall['test_end']}")
    wm = overall["walkforward_1step"]["model"]
    print(f"Error rate: {wm['WAPE']*100:.1f}%")
    print("\nMonthly results:")
    print(monthly.to_string(index=False))
    print(f"\nSaved to {ROOT / 'outputs'}")


if __name__ == "__main__":
    main()
