# ============================================================
# run_daily_playoff_predictions.py
# ============================================================

"""
Daily execution script for NHL playoff probability predictions.

Responsibilities:
- Load team-day snapshots from DuckDB
- Apply feature engineering
- Run inference
- Attach metadata + presentation fields
- Store DAILY (time-series) probabilities
"""

# ============================================================
# Core imports
# ============================================================

import duckdb
import pandas as pd
from datetime import datetime, timezone
import sys
from pathlib import Path

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
# ============================================================
# Project imports
# ============================================================

from config import DB_PATH, SOURCE_TABLE, TARGET_TABLE, CURRENT_SEASON_ID
from feature_engineering import build_features
from inference import score_dataframe
from metadata import (
    MODEL_METADATA,
    PROBABILITY_TIERS,
    EARLY_SEASON_NOTE,
)

# ============================================================
# 1. Connect + load data
# ============================================================

print("🔌 Connecting to DuckDB")
con = duckdb.connect(DB_PATH)

query = f"""
WITH
teams AS (
  SELECT DISTINCT team_id, tri_code
  FROM {SOURCE_TABLE}
  WHERE season_id = {CURRENT_SEASON_ID}
),

dates AS (
  SELECT DISTINCT game_date
  FROM {SOURCE_TABLE}
  WHERE season_id = {CURRENT_SEASON_ID}
),

spine AS (
  SELECT
    {CURRENT_SEASON_ID}::BIGINT AS season_id,
    d.game_date,
    t.team_id,
    t.tri_code
  FROM dates d
  CROSS JOIN teams t
),

snap AS (
  SELECT
    season_id,
    game_date,
    team_id,
    tri_code,
    conference,
    division,
    conference_rank,
    division_rank,
    wildcard_rank,
    points,
    points_pct,
    projected_points,
    games_played
  FROM {SOURCE_TABLE}
  WHERE season_id = {CURRENT_SEASON_ID}
),

filled AS (
  SELECT
    s.season_id,
    s.game_date,
    s.team_id,
    s.tri_code,

    -- forward fill per team across dates
    LAST_VALUE(snap.conference       IGNORE NULLS) OVER w AS conference,
    LAST_VALUE(snap.division         IGNORE NULLS) OVER w AS division,
    LAST_VALUE(snap.conference_rank  IGNORE NULLS) OVER w AS conference_rank,
    LAST_VALUE(snap.division_rank    IGNORE NULLS) OVER w AS division_rank,
    LAST_VALUE(snap.wildcard_rank    IGNORE NULLS) OVER w AS wildcard_rank,
    LAST_VALUE(snap.points           IGNORE NULLS) OVER w AS points,
    LAST_VALUE(snap.points_pct        IGNORE NULLS) OVER w AS points_pct,
    LAST_VALUE(snap.projected_points IGNORE NULLS) OVER w AS projected_points,
    LAST_VALUE(snap.games_played     IGNORE NULLS) OVER w AS games_played

  FROM spine s
  LEFT JOIN snap
    ON snap.season_id = s.season_id
   AND snap.team_id   = s.team_id
   AND snap.game_date = s.game_date

  WINDOW w AS (
    PARTITION BY s.season_id, s.team_id
    ORDER BY s.game_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  )
)

SELECT *
FROM filled
ORDER BY game_date, team_id
"""
df_raw = con.execute(query).df()

if df_raw.empty:
    raise RuntimeError("❌ No rows loaded — aborting run")

print(f"📥 Loaded {len(df_raw)} rows from {SOURCE_TABLE}")

# ============================================================
# 2. Feature engineering
# ============================================================

print("🧠 Applying feature engineering")
df_features = build_features(df_raw)

# ============================================================
# SHAP FEATURE MATRIX (MODEL INPUT)
# ============================================================

from joblib import load


bundle = load(r"C:\Users\kjchr\Desktop\hockey\new_hockey_env\API data\prediction_model\models\playoff_models_wc_field_pressure_v4.joblib")
xgb_pipeline = bundle["xgb_pipeline"]
FEATURE_COLS = bundle["features"]

# grab steps (if your pipeline has these names)
xgb_model = xgb_pipeline.named_steps["model"]
scaler    = xgb_pipeline.named_steps.get("scaler", None) 


LEAK_COLS = [
    "game_id", "game_date", "date",
    "clinch_type", "made_playoffs",
    "team_name", "opponent",
    "season_id"
]

ID_COLS = ["team_id", "opponent_id", "tri_code"]

X_shap = (
    df_features
    .drop(columns=LEAK_COLS + ID_COLS, errors="ignore")
    .select_dtypes(include="number")
    .fillna(0)
    .reindex(columns=FEATURE_COLS, fill_value=0)
)

# what the model actually sees
X_shap_scaled = scaler.transform(X_shap)

X_shap_scaled = pd.DataFrame(
    X_shap_scaled,
    columns=X_shap.columns,
    index=X_shap.index
)
# ============================================================
# 3. Model inference
# ============================================================

print("🎯 Scoring playoff probabilities")
df_scored = score_dataframe(df_features)

df_present = df_scored.copy()

# Expected columns after inference:
# - playoff_prob_xgb
# - playoff_prob_rf
# - playoff_prob (primary / ensemble)

# ============================================================
# 4. Attach model identity + run metadata
# ============================================================

run_ts = datetime.now(timezone.utc)

df_present["model_name"] = MODEL_METADATA["model_name"]
df_present["model_version"] = MODEL_METADATA["version"]
df_present["model_run_ts"] = run_ts

# ============================================================
# 5. Presentation-only adjustments
# ============================================================

def season_bucket(gp: int) -> str:
    if gp <= 20:
        return "early"
    elif gp <= 50:
        return "mid"
    else:
        return "late"


def adjust_for_uncertainty(p: float, bucket: str) -> float:
    if bucket == "early":
        return 0.45 * p + 0.275
    return p


if "season_bucket" not in df_present.columns:
    df_present["season_bucket"] = df_present["games_played"].apply(season_bucket)

df_present["playoff_prob_display"] = df_present.apply(
    lambda r: adjust_for_uncertainty(
        r["playoff_prob"],
        r["season_bucket"],
    ),
    axis=1,
)

# ============================================================
# 6. Tier assignment
# ============================================================

def assign_tier(p: float) -> str:
    for tier, (low, high) in PROBABILITY_TIERS.items():
        if low <= p < high:
            return tier
    return "Unknown"


df_present["playoff_tier"] = df_present["playoff_prob_display"].apply(assign_tier)

# ============================================================
# 7. Select + rename output fields
# ============================================================

output_cols = [
    # identity
    "season_id",
    "game_date",
    "team_id",
    "tri_code",
    "conference",
    "division",

    # standings context
    "conference_rank",
    "division_rank",
    "wildcard_rank",
    "points",
    "projected_points",

    # probabilities
    "playoff_prob",
    "playoff_prob_xgb",
    "playoff_prob_rf",
    "playoff_prob_display",
    "playoff_tier",
    "season_bucket",

    # metadata
    "model_name",
    "model_version",
    "model_run_ts",
]

df_out = df_present[output_cols].copy()

# ============================================================
# 8. Write results back to DuckDB (UPSERT — DAILY GRAIN)
# ============================================================

print("🧪 Writing predictions to DuckDB")

con.execute(f"""
CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
    season_id BIGINT,
    game_date DATE,
    team_id BIGINT,
    tri_code VARCHAR,
    conference VARCHAR,
    division VARCHAR,

    conference_rank INTEGER,
    division_rank INTEGER,
    wildcard_rank INTEGER,
    points INTEGER,
    projected_points DOUBLE,

    playoff_prob DOUBLE,
    playoff_prob_xgb DOUBLE,
    playoff_prob_rf DOUBLE,
    playoff_prob_display DOUBLE,
    playoff_tier VARCHAR,
    season_bucket VARCHAR,

    model_name VARCHAR,
    model_version VARCHAR,
    model_run_ts TIMESTAMP,

    PRIMARY KEY (season_id, game_date, team_id, model_version)
)
""")

con.register("pred_df", df_out)

con.execute(f"""
DELETE FROM {TARGET_TABLE}
WHERE (season_id, game_date, team_id, model_version) IN (
    SELECT
        season_id,
        game_date,
        team_id,
        model_version
    FROM pred_df
)
""")

con.execute(f"""
INSERT INTO {TARGET_TABLE}
SELECT
    season_id,
    game_date,
    team_id,
    tri_code,
    conference,
    division,

    conference_rank,
    division_rank,
    wildcard_rank,
    points,
    projected_points,

    playoff_prob,
    playoff_prob_xgb,
    playoff_prob_rf,
    playoff_prob_display,
    playoff_tier,
    season_bucket,

    model_name,
    model_version,
    model_run_ts
FROM pred_df
""")

con.unregister("pred_df")

print("✅ Predictions upserted successfully")

# ============================================================
# 9. Summary
# ============================================================

print("\n✅ DAILY PLAYOFF PROBABILITIES GENERATED")
print(f"Teams scored: {df_out['tri_code'].nunique()}")
print(f"Season context: {CURRENT_SEASON_ID}")
print(f"Date range: {df_out['game_date'].min()} → {df_out['game_date'].max()}")
print(f"Model version: {MODEL_METADATA['version']}")
print(EARLY_SEASON_NOTE)
