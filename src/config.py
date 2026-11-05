"""Central configuration: paths and constants used across the pipeline."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Raw CSV files
RAW_FILES = {
    "customers": DATA_RAW / "customers.csv",
    "loans": DATA_RAW / "loan_accounts.csv",
    "bureau": DATA_RAW / "bureau_credit.csv",
    "repayment": DATA_RAW / "repayment_history.csv",
    "collections": DATA_RAW / "collections_events.csv",
}

# Processed artefacts
ACCOUNT_DATASET = DATA_PROCESSED / "account_master.parquet"
CUSTOMER_DATASET = DATA_PROCESSED / "customer_master.parquet"
DQ_REPORT = DATA_PROCESSED / "data_quality_report.json"

# Model artefacts
MODEL_FILE = MODELS_DIR / "xgb_model.joblib"
LOGREG_FILE = MODELS_DIR / "logreg_model.joblib"
FEATURE_LIST_FILE = MODELS_DIR / "feature_list.json"
METRICS_FILE = MODELS_DIR / "metrics.json"
SHAP_VALUES_FILE = MODELS_DIR / "shap_summary.json"
SCORED_ACCOUNTS_FILE = DATA_PROCESSED / "scored_accounts.parquet"
ANOMALIES_FILE = DATA_PROCESSED / "anomalies.parquet"

# ---------------------------------------------------------------------------
# Modelling constants
# ---------------------------------------------------------------------------
TARGET = "target_default"
RANDOM_STATE = 42
TEST_SIZE = 0.25

# Reference "today" for recency features. The data runs through ~2026, so we
# anchor relative-time features to a fixed date for reproducibility.
REFERENCE_DATE = "2026-10-27"
