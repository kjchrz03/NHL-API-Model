# ============================================================
# config.py
# ============================================================

from pathlib import Path

# ------------------------------------------------------------
# Base Directory (project root)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

# DuckDB database
DB_PATH = BASE_DIR / "duckdb" / "nhl_data.duckdb"

# Model file
MODEL_PATH = BASE_DIR / "prediction_model" / "models" / "playoff_models_wc_field_pressure_v4.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

# ------------------------------------------------------------
# Tables
# ------------------------------------------------------------

SOURCE_TABLE = "team_daily_stats"
TARGET_TABLE = "playoff_daily_predictions"

# ------------------------------------------------------------
# Notes
# ------------------------------------------------------------
# - season_id is NOT defined here
# - it is passed dynamically from the pipeline
# - this keeps config static and pipeline-driven