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

DB_PATH = ROOT_DIR / "data" / "duckdb" / "nhl_data.duckdb"

# ------------------------------------------------------------
# Tables
# ------------------------------------------------------------

SOURCE_TABLE = "team_daily_stats"
TARGET_TABLE = "playoff_daily_predictions"

# ------------------------------------------------------------
# Current NHL Season
# ------------------------------------------------------------

# SINGLE SOURCE OF TRUTH
# Update this value once at the start of each new NHL season.

CURRENT_SEASON_ID = 20262027

# Automatically generate the readable season label.
# Example: 20262027 -> "2026-27"

SEASON_START_YEAR = CURRENT_SEASON_ID // 10000
SEASON_END_YEAR = CURRENT_SEASON_ID % 10000

CURRENT_SEASON_LABEL = (
    f"{SEASON_START_YEAR}-{str(SEASON_END_YEAR)[-2:]}"
)

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

# Production model version.
# v6 remains the default while v7 runs in parallel.
# Change only when officially promoting a new model.

MODEL_VERSION = "v6"

MODEL_FILENAME = f"playoff_models_{MODEL_VERSION}.joblib"
MODEL_PATH = ROOT_DIR / "models" / MODEL_FILENAME

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

# Exclude the current/incomplete season from model training.
# Automatically updates when CURRENT_SEASON_ID changes.

EXCLUDE_SEASON_ID = CURRENT_SEASON_ID

# ------------------------------------------------------------
# Notes
# ------------------------------------------------------------

# - Update CURRENT_SEASON_ID once at the start of each season.
# - EXCLUDE_SEASON_ID automatically follows CURRENT_SEASON_ID.
# - CURRENT_SEASON_LABEL is generated automatically.
# - MODEL_VERSION controls the default production model.
# - v7 can still be selected explicitly for parallel scoring.
# - MODEL_PATH may not exist yet when training a new model.
# - Scripts requiring an existing model should validate
#   MODEL_PATH themselves.
# - Historical season IDs and season-specific rules should
#   remain explicit where appropriate.
# - Scripts must import CURRENT_SEASON_ID to use this setting.