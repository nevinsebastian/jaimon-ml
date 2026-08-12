from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = "24_25_data(removedfeatures).csv"
DATA_PATH = ROOT / DATA_FILE

OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

SKU_COL = "sku"
QTY_COL = "qty_kg"

EXPORT_VOUCHERS = {"EXPORT SALES"}
BULK_SPECIAL_VOUCHERS = {
    "GST-Sales(NT)",
    "GST-Sales(Un Reg.)(NT)",
    "GST SALES(IF)",
    "GST Sales (IFCO) (NT)",
}
COUNTER_KEYWORDS = ("counter",)
TAB_KEYWORDS = ("tab",)
ROUTE_KEYWORDS = ("route",)
VANSALES_KEYWORDS = ("vansales",)

BULK_PERCENTILE = 0.95
MIN_BULK_ABS = {
    "counter": 15,
    "tab": 100,
    "wholesale": 500,
    "route": 500,
    "vansales": 200,
    "other": 200,
}

MIN_WEEKS_HISTORY = 12
TEST_WEEKS = 8
HORIZON_WEEKS = 104
RANDOM_SEED = 42

DEMAND_MODE = "regular_capped"
