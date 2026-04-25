from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
# DATA_DIR = BASE_DIR / "data"
DATA_DIR = Path(r"C:\Users\akash\OneDrive\Documents\Time_Series\TS_project\src\dataset")
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
PLOTS_DIR = OUTPUT_DIR / "plots"
ARTIFACT_DIR = OUTPUT_DIR / "artifacts"

RANDOM_STATE = 42
N_TEST_WEEKS = 20
FORECAST_CONFIDENCE = 0.95
RETRAIN_THRESHOLD_WMAPE = 8.0

TARGET_COL = "y"
DATE_COL = "Date"
STORE_COL = "Store"

RAW_FILES = {
    "train": DATA_DIR / "train.csv",
    "features": DATA_DIR / "features.csv",
    "stores": DATA_DIR / "stores.csv",
}

HOLIDAY_WEEKS = {1, 6, 36, 47, 52}

MARKDOWN_COLS = [f"MarkDown{i}" for i in range(1, 6)]
EXOG_COLS = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "IsHoliday"] + MARKDOWN_COLS

FEATURE_COLS = [
    "Store",
    "TypeCode",
    "Size",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_52",
    "roll_4_mean",
    "roll_4_std",
    "roll_13_mean",
    "roll_26_mean",
    "month",
    "week",
    "quarter",
    "is_q4",
    "year_trend",
    "sin_w1",
    "cos_w1",
    "sin_w2",
    "cos_w2",
    "sin_m1",
    "cos_m1",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "IsHoliday",
    "markdown_total",
    "markdown_active",
]

STAT_EXOG_COLS = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "IsHoliday"] + MARKDOWN_COLS