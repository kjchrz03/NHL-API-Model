# ============================================================
# config.py
# ============================================================
from pathlib import Path

# ------------------------------------------------------------
# Base Directory (project root)
# ------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

# DuckDB database (moved under /data)
DB_PATH = ROOT_DIR / "data" / "duckdb" / "nhl_data.duckdb"

# ------------------------------------------------------------
# Tables
# ------------------------------------------------------------

SOURCE_TABLE = "team_daily_stats"
TARGET_TABLE = "playoff_daily_predictions"

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

#Change this anytime a new model has to be trained
MODEL_VERSION = "v5"

MODEL_FILENAME = f"playoff_models_{MODEL_VERSION}.joblib"
MODEL_PATH = ROOT_DIR / "models" / MODEL_FILENAME

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

#Change this annually
EXCLUDE_SEASON_ID = 20252026



# ------------------------------------------------------------
# Notes
# ------------------------------------------------------------
# - season_id is NOT defined here
# - it is passed dynamically from the pipeline
# - this keeps config static and pipeline-driven