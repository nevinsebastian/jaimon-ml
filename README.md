# Sales Forecast

Weekly and monthly sales forecast for Pavizham products. Reads sales CSV, trains a model, and opens a dashboard.

Numbers are in **packets** (the `quantity_in_kg` column is packet count, not kg).

---

## Requirements

- Python 3.10 or newer
- macOS, Linux, or Windows

---

## First-time setup

```bash
# 1. Clone or download this folder, then open terminal inside it
cd path/to/jaimon-ml-model

# 2. Create virtual environment
python3 -m venv .venv

# 3. Activate it
source .venv/bin/activate          # Mac / Linux
# .venv\Scripts\activate           # Windows

# 4. Install packages
pip install -r requirements.txt
```

---

## Sales data file

Put your sales CSV in the project root folder.

Default filename (already set):

```
24_25_data(removedfeatures).csv
```

**Required columns** (same format as the current file):

`date`, `voucher_type`, `item_name`, `quantity_in_kg`, `amount`, `Cleaned`, `Brand`, `Product`, `Packet_Size`, `Packet_Type`, `Main_Category`, `Final_Category`

Date format: `10-Sep-24` (day-month-year)

---

## Run forecast (every time you update data)

With the virtual environment active:

```bash
python run_pipeline.py
```

This creates:

- `outputs/forecast_sku_weekly.csv` — expected sales per product per week
- `outputs/forecast_product_weekly.csv` — totals by product
- `outputs/weekly_demand_panel.csv` — past weekly sales used for training
- `models/weekly_demand_model.joblib` — saved model

Optional — check how accurate predictions are against known past sales:

```bash
python run_holdout_validation.py
```

---

## Open dashboard

```bash
streamlit run dashboard.py
```

Browser opens at `http://localhost:8501`

**Dashboard tabs:**

| Tab | Use for |
|-----|---------|
| Home | Overview, charts, top products next week |
| Week by week | Stock and dispatch planning |
| Month by month | Monthly targets and reports |
| Product details | One product at a time |
| How accurate is this? | Prediction vs actual (after running holdout test) |
| Download data | Export CSV for Excel |

Use the sidebar to filter by brand, category, or product.

---

## When you get new sales data

1. Export sales from your system as CSV (same columns as above)
2. Replace the old file in the project folder, **or** save with a new name
3. If the filename changed, edit `src/config.py`:

```python
DATA_FILE = "your_new_file.csv"
```

4. Re-run:

```bash
source .venv/bin/activate
python run_pipeline.py
python run_holdout_validation.py    # optional
streamlit run dashboard.py
```

5. Refresh the browser tab

---

## Settings (`src/config.py`)

| Setting | Default | Meaning |
|---------|---------|---------|
| `DATA_FILE` | `24_25_data(removedfeatures).csv` | Input CSV filename |
| `HORIZON_WEEKS` | `8` | How many future weeks to forecast |
| `DEMAND_MODE` | `regular_capped` | Ignores export/bulk orders when training |

`DEMAND_MODE` options:

- `regular_capped` — normal use (recommended)
- `counter_only` — shop counter sales only
- `all_raw` — includes everything (not recommended)

To change the accuracy test split date, edit `DEFAULT_CUTOFF` in `src/holdout_validate.py`.

---

## Project layout

```
.
├── 24_25_data(removedfeatures).csv   # sales input
├── dashboard.py                      # Streamlit app
├── run_pipeline.py                   # train + forecast
├── run_holdout_validation.py         # accuracy test
├── requirements.txt
├── src/
│   ├── config.py                     # paths and settings
│   ├── data_prep.py                  # load CSV, handle bulk sales
│   ├── features.py
│   ├── train.py
│   ├── forecast.py
│   └── holdout_validate.py
├── outputs/                          # generated CSVs (not in git)
├── models/                           # saved model (not in git)
└── reports/                          # metrics JSON (not in git)
```

---

## Troubleshooting

**`FileNotFoundError` for CSV**  
Check the file is in the project root and `DATA_FILE` in `src/config.py` matches the name.

**Dashboard shows empty or old data**  
Run `python run_pipeline.py` first, then refresh the browser.

**`streamlit: command not found`**  
Activate the virtual environment: `source .venv/bin/activate`

**Bulk / export orders**  
Large one-time orders are excluded from the forecast so they don't skew normal weekly demand. Plan those separately.

---

## Git note

Generated files in `outputs/`, `models/`, and `reports/` are gitignored. After cloning, run `python run_pipeline.py` to generate them. Include the sales CSV in the repo only if your team agrees (it may be large or sensitive).
