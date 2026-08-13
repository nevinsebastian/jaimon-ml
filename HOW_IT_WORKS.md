# How This Sales Forecast Works — A Full Walkthrough

This guide explains the project end to end so you can understand every piece: the data, why bulk sales matter, how the model learns, how future weeks are predicted, how accuracy is tested, and how the dashboard is used.

Read it in order the first time. After that, jump to any section you need.

---

## 1. What this project does (in one sentence)

It takes **past sales in kilograms**, learns normal weekly demand for each product pack, and predicts **how many kg of each product will sell in future weeks** — then shows that in a simple dashboard.

---

## 2. The business problem

A company sells many grocery products (salt, rice, spices, vinegar, etc.) in different pack sizes.

Managers need answers like:

- Next week, how much Salt (1 kg pack) should we expect?
- Over the next 3 months, which products will sell the most?
- Can we trust these numbers?

Raw sales data is messy because it mixes:

1. **Regular demand** — daily/weekly shop, route, van, and wholesale sales
2. **Bulk / export / one-time big orders** — huge quantities that are not “normal weekly demand”

If you train a model on raw totals without separating those, one export order of 750 kg can make the model think that product always sells that much. So the first design rule is:

> Forecast **regular weekly demand in kg**, and treat bulk/export separately.

---

## 3. What the numbers mean

### Important: unit is kilograms (kg)

The CSV column is named `quantity_in_kg`.

That column means **kg sold**, not “number of packets.”

Examples:

| Product | `quantity_in_kg` | Meaning |
|---------|------------------|---------|
| 01kg Salt | `6` | 6 kg sold (6 × 1 kg packs) |
| 250gm Turmeric | `0.25` | 0.25 kg sold (one 250 g pack) |
| 500ml Coconut Oil | `9.24` | 9.24 kg sold (about 20 bottles) |

So every forecast number you see in outputs and the dashboard is in **kg**.

### What is a SKU?

A **SKU** is one unique product + pack size, for example:

- `01kg Pavizham Salt(Packet)`
- `500gm Pavizham Beaten Rice Matta Spl(Packet)`

The model forecasts **per SKU, per week**.

---

## 4. Project map (files and roles)

```
jaimon ml model/
│
├── 24_25_data(removedfeatures).csv   ← raw sales (input)
├── requirements.txt                  ← Python packages
├── README.md                         ← install & run instructions
├── HOW_IT_WORKS.md                   ← this learning guide
│
├── run_pipeline.py                   ← main: prepare → train → forecast
├── run_holdout_validation.py         ← accuracy test (half history vs later months)
├── analyze_bulk_impact.py            ← optional: compare bulk vs regular demand
├── dashboard.py                      ← Streamlit app for non-technical users
│
├── src/
│   ├── config.py                     ← settings (file name, horizon, bulk rules)
│   ├── data_prep.py                  ← load CSV, tag channels, build weekly panel
│   ├── features.py                   ← create lag / rolling / calendar features
│   ├── train.py                      ← train model + backtest metrics
│   ├── forecast.py                   ← predict future weeks
│   └── holdout_validate.py           ← train on first half, test on second half
│
├── outputs/                          ← generated CSVs (forecasts, panels)
├── models/                           ← saved trained model (.joblib)
└── reports/                          ← accuracy metrics JSON + charts
```

### Mental model of the flow

```
CSV sales lines
      │
      ▼
data_prep.py     → weekly kg per SKU (bulk cleaned)
      │
      ▼
features.py      → lags, rolling averages, month/festival flags
      │
      ▼
train.py         → learn patterns from history
      │
      ▼
forecast.py      → predict future weeks (up to ~2 years)
      │
      ▼
dashboard.py     → pick dates and see charts / tables
```

Accuracy check is separate:

```
holdout_validate.py → hide later months → predict them → compare to real sales
```

That check lives in the dashboard tab **“How accurate is this?”** only. The live forecast uses the **full dataset**.

---

## 5. Step by step: data preparation (`src/data_prep.py`)

### 5.1 Load transactions

The loader:

1. Reads the CSV
2. Parses `date` (format like `10-Sep-24`)
3. Copies `quantity_in_kg` into an internal column `qty_kg`
4. Builds `sku` from the cleaned product name
5. Assigns a **channel** from `voucher_type`
6. Buckets each row into a week starting Monday (`week_start`)

### 5.2 Channel classification

Each bill type is mapped into a sales channel, roughly:

| Channel | Meaning |
|---------|---------|
| `counter` | Shop counter sales |
| `tab` | Tab sales |
| `route` | Route sales |
| `vansales` | Van sales |
| `wholesale` | Normal GST wholesale |
| `export` | Export sales |
| `bulk_special` | Special bulk voucher types |
| `other` | Anything else |

Why? Because counter sales and export sales behave very differently.

### 5.3 Bulk handling (very important)

Two kinds of “bulk”:

1. **Bulk by channel** — export / special bulk vouchers  
   → these are excluded from regular demand (`qty_capped = 0` for those channels)

2. **Bulk by size** — unusually large lines inside normal channels  
   → detected with:
   - 95th percentile of qty for that SKU + channel
   - plus a minimum absolute floor (e.g. counter ≥ 15 kg, wholesale ≥ 500 kg)

Those large lines are **capped**, not deleted, so one giant order does not pull the weekly total unrealistically high.

### 5.4 Weekly demand panel

All line items are summed into:

> one row per **SKU × week**

Columns include:

- `qty` — weekly kg used for modeling (after bulk rules)
- product metadata (Brand, Product, Packet_Size, category, …)
- `amount` — money total (for reference)

Quiet SKUs (sold in fewer than 12 weeks) are dropped so the model focuses on products with enough history.

Default demand mode (`config.DEMAND_MODE`):

- `regular_capped` ← recommended (used in production)
- `counter_only`
- `all_raw` (includes everything; usually worse for forecasting)

---

## 6. Step by step: features (`src/features.py`)

The model does not look at the raw CSV directly. It looks at engineered features.

### 6.1 Calendar features

For each week:

- week of year
- month
- quarter
- festival-month flag (Apr, Aug, Sep, Dec, Jan)
- monsoon flag (Jun–Sep)

These help the model learn seasonality (festivals, rainy season, etc.).

### 6.2 Lag features

A **lag** means “what sold in a previous week.”

| Feature | Meaning |
|---------|---------|
| `lag_1` | kg sold last week |
| `lag_2` | kg sold 2 weeks ago |
| `lag_4` | kg sold 4 weeks ago |
| `lag_8` | kg sold 8 weeks ago |

Idea: recent sales are a strong clue for next week.

### 6.3 Rolling features

Rolling = average / variation over recent weeks:

- `roll_mean_4`, `roll_mean_8`, `roll_mean_12`
- `roll_std_4`, `roll_std_8`, `roll_std_12`

Idea: is this product usually steady, or jumpy?

### 6.4 Extra demand signals

- `lag_diff_1` — change from 2 weeks ago to last week (up or down?)
- `zero_streak_lag1` — how many recent weeks had zero sales

### 6.5 Category codes

Brand / Product / Packet size / Category / SKU are converted to numeric codes so the model can treat similar products similarly.

---

## 7. Step by step: training (`src/train.py`)

### 7.1 Model choice

We use scikit-learn’s **HistGradientBoostingRegressor**.

Plain English:

> It builds many small decision trees and combines them. Each tree corrects mistakes of previous trees. Good for tabular sales data.

### 7.2 Target transformation

We train on `log1p(qty)` instead of raw kg.

Why?

- Sales are skewed (some SKUs sell tiny amounts, some sell thousands of kg)
- Log scale makes big and small products easier to learn together
- At prediction time we convert back with `expm1`

### 7.3 Recency weights

Recent weeks get higher training weight than old weeks.

Why? Demand patterns drift. Last few months usually matter more than a year ago.

### 7.4 Blending with last week

Final short-term prediction is not “model only.” It is:

```
0.55 × model prediction + 0.45 × last week’s sales (lag_1)
```

Why? Pure ML can overreact. Blending with last week keeps forecasts more stable and realistic.

### 7.5 What “train on full data” means for the live forecast

When you run `python run_pipeline.py`:

1. Features are built from **all historical weeks**
2. Model is trained on that history
3. Future weeks are forecast from the end of history forward

That is the model the dashboard uses for **future sales**.

---

## 8. Step by step: forecasting future weeks (`src/forecast.py`)

### 8.1 Horizon

`HORIZON_WEEKS` in `src/config.py` controls how far ahead we generate predictions.

Currently it is set to **104 weeks** (~2 years), so the dashboard date picker can go far into the future.

### 8.2 Why naive recursive forecasting fails

A simple recursive approach does this:

1. Predict week 1
2. Feed that prediction as “last week” into week 2
3. Feed week 2 into week 3
4. … repeat for 100+ weeks

Problem: small errors compound. After many months, forecasts can collapse toward near-zero. That is why an earlier graph looked “false” (past ~130k kg/week, future suddenly ~20–40k and dying).

### 8.3 What we do instead (stable long-horizon forecast)

For each future week and SKU:

1. Build features using an **anchor series** (real history + carefully controlled fill)
2. Get a model prediction
3. Blend differently by horizon:
   - **First 8 weeks:** blend model + lag-1 (short-term behavior)
   - **After 8 weeks:** blend model + **seasonal anchor**
4. Seasonal anchor = median real kg for that SKU in the **same week-of-year** historically
5. Clamp the values fed into future lags so one bad week cannot pull everything down forever

Result: future weekly totals stay in a realistic range (similar order of magnitude to history), and seasonal patterns (festival dips/peaks) continue to appear.

---

## 9. Accuracy testing (half-and-half) — only for trust, not for live forecast

This is the part in the dashboard tab **“How accurate is this?”**

### 9.1 Idea

We already have about 1 year of data. So we can simulate the future:

1. Train only on earlier months (example: Apr–Sep 2024)
2. Hide later months (example: Oct 2024–Mar 2025)
3. Predict those hidden months
4. Compare prediction vs what actually sold

If the model is decent there, we can trust it more for unknown future dates.

### 9.2 Two evaluation styles

1. **Blind multi-step**  
   Predict many weeks ahead without updating with new actuals. Harder. Closer to long-range planning.

2. **Walk-forward 1-step** (main score shown in dashboard)  
   Each week: use everything known up to that week, predict next week, then move forward. Closer to “we refresh weekly with new sales.”

### 9.3 Error metric: WAPE

**WAPE** = Weighted Absolute Percentage Error

Plain English:

> Total absolute mistake ÷ total actual sales

Example:

- Actual total = 100,000 kg
- Absolute mistakes add up to 27,000 kg
- WAPE = 27%

Lower is better.

Rough guide used in the dashboard:

| Error rate | Rating |
|------------|--------|
| ≤ 15% | Excellent |
| ≤ 25% | Good |
| ≤ 35% | Fair |
| > 35% | Needs improvement |

Run it with:

```bash
python run_holdout_validation.py
```

---

## 10. Dashboard (`dashboard.py`) — for non-technical users

Open with:

```bash
streamlit run dashboard.py
```

### Sidebar controls

1. **From / To dates** — pick any forecast window inside the generated future range
2. Brand / Category / Product / Pack filters
3. Week vs Month view
4. Top-N products slider

Important:

- Date picker filters the **future forecast**
- Past sales still appear on overview charts for context
- Changing dates updates totals, charts, and download tables

### Tabs

| Tab | Purpose |
|-----|---------|
| Home | Past + future overview, category mix, top products |
| Week by week | Weekly kg totals for planning |
| Month by month | Monthly planning / targets |
| Product details | One SKU: history + long forecast line |
| How accurate is this? | Holdout test only (trust check) |
| Download data | Export CSV for Excel |

Language is kept simple (“Expected sales”, “Error rate”, “kg”) so non-technical people can use it without ML jargon.

---

## 11. How to run everything (practical recipe)

### First time

```bash
cd "path/to/jaimon ml model"
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Every time data changes

```bash
source .venv/bin/activate
python run_pipeline.py             # train on full data + forecast future
python run_holdout_validation.py   # optional trust check
streamlit run dashboard.py
```

Then refresh the browser.

### If CSV file name changes

Edit `src/config.py`:

```python
DATA_FILE = "your_new_file.csv"
```

---

## 12. Key settings (`src/config.py`)

| Setting | What it controls |
|---------|------------------|
| `DATA_FILE` | Which CSV to read |
| `QTY_COL` | Internal name for kg (`qty_kg`) |
| `HORIZON_WEEKS` | How many future weeks to generate |
| `DEMAND_MODE` | How bulk is treated (`regular_capped` recommended) |
| `MIN_WEEKS_HISTORY` | Drop SKUs with too little history |
| `BULK_PERCENTILE` / `MIN_BULK_ABS` | Bulk outlier rules |

---

## 13. Output files (what each one is)

After `run_pipeline.py`:

| File | Meaning |
|------|---------|
| `outputs/weekly_demand_panel.csv` | Historical weekly kg by SKU (training panel) |
| `outputs/forecast_sku_weekly.csv` | Future predicted kg by SKU (`forecast_qty_kg`) |
| `outputs/forecast_product_weekly.csv` | Future kg rolled up by product |
| `outputs/forecast_top_skus_by_week.csv` | Top SKUs each week |
| `outputs/channel_summary.csv` | Sales mix by channel |
| `models/weekly_demand_model.joblib` | Saved trained model |

After holdout validation:

| File | Meaning |
|------|---------|
| `reports/holdout_metrics.json` | Overall accuracy numbers |
| `outputs/holdout_weekly_compare.csv` | Week-level predicted vs actual |
| `outputs/holdout_monthly_compare.csv` | Month-level predicted vs actual |
| `outputs/holdout_sku_wape.csv` | Per-SKU error rates |

---

## 14. Common misunderstandings

### “Is `quantity_in_kg` packets?”
No. It is kilograms sold.

### “Why exclude bulk/export?”
Because forecasting is for **repeatable weekly demand**. One-time export spikes are planned separately.

### “Why does the accuracy tab use half the data?”
Only to **test** the model on known history. The live future forecast uses the **full** history.

### “Why can long forecasts go wrong if built poorly?”
Because recursive models feed predictions into themselves. Without seasonal anchors and clamping, errors compound and the chart collapses.

### “Can the user pick any future date?”
Yes, within the generated horizon (currently ~2 years ahead of the last history week). Change `HORIZON_WEEKS` and re-run the pipeline if you need longer.

---

## 15. How to explain this to someone in 60 seconds

1. We clean sales so bulk/export does not distort normal demand.
2. We convert bill lines into weekly kg per product pack.
3. We teach a model using recent history, seasonality, and product identity.
4. We forecast future weeks with safeguards so long-range numbers stay realistic.
5. We separately hide part of history to measure accuracy.
6. A dashboard lets anyone pick dates and see expected kg by week/month/product.

---

## 16. Suggested learning path for your friend

1. Open `README.md` and run setup once.
2. Look at a few rows of the CSV and confirm kg interpretation.
3. Read `src/data_prep.py` (bulk + weekly panel).
4. Read `src/features.py` (lags / rolling / calendar).
5. Read `src/train.py` (model + blend).
6. Read `src/forecast.py` (why long-horizon needs seasonal anchors).
7. Run `run_pipeline.py`, then open the dashboard and play with dates.
8. Run `run_holdout_validation.py` and open **How accurate is this?**

If you understand those eight steps, you understand the whole system.

---

## 17. Quick glossary

| Term | Meaning |
|------|---------|
| SKU | One product + pack size |
| kg | Kilograms sold |
| Lag | Past week’s sales used as a clue |
| Rolling mean | Average of recent weeks |
| Horizon | How far into the future we predict |
| Holdout | Accuracy test on hidden past months |
| WAPE | Overall % error (lower is better) |
| Seasonal anchor | Typical historical kg for same week-of-year |
| Blend | Mix model output with a simple baseline for stability |

---

If something in the dashboard looks wrong, first re-run `python run_pipeline.py`, then refresh the browser. Most “stale chart” issues are just old output files.
