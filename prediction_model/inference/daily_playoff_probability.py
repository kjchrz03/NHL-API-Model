# ============================================================
# run_daily_playoff_predictions.py
# ============================================================

"""
Daily execution script for NHL playoff probability predictions.
"""

# ============================================================
# Core imports
# ============================================================

import duckdb
import pandas as pd
from datetime import datetime, timezone

from config import SOURCE_TABLE, TARGET_TABLE
from feature_engineering import build_features
from prediction_model.inference import score_dataframe
from prediction_model.metadata import (
    MODEL_METADATA,
    PROBABILITY_TIERS,
    EARLY_SEASON_NOTE,
)

# ============================================================
# Prediction Data Runner
# ============================================================

def run_probability_scores(
    *,
    con: duckdb.DuckDBPyConnection,
    season_id: int,
) -> None:
    print(f"▶ Running prediction data for season {season_id}")

    # ============================================================
    # Ensure target table exists
    # ============================================================
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

    # ============================================================
    # Load source data
    # ============================================================
    print("🔌 Using DuckDB connection")

    query = f"""
    WITH
    teams AS (
        SELECT DISTINCT team_id, tri_code
        FROM {SOURCE_TABLE}
        WHERE season_id = {season_id}
    ),

    dates AS (
        SELECT DISTINCT game_date
        FROM {SOURCE_TABLE}
        WHERE season_id = {season_id}
    ),

    spine AS (
        SELECT
            {season_id}::BIGINT AS season_id,
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
        WHERE season_id = {season_id}
    ),

    filled AS (
        SELECT
            s.season_id,
            s.game_date,
            s.team_id,
            s.tri_code,

            LAST_VALUE(snap.conference IGNORE NULLS) OVER w AS conference,
            LAST_VALUE(snap.division IGNORE NULLS) OVER w AS division,
            LAST_VALUE(snap.conference_rank IGNORE NULLS) OVER w AS conference_rank,
            LAST_VALUE(snap.division_rank IGNORE NULLS) OVER w AS division_rank,
            LAST_VALUE(snap.wildcard_rank IGNORE NULLS) OVER w AS wildcard_rank,
            LAST_VALUE(snap.points IGNORE NULLS) OVER w AS points,
            LAST_VALUE(snap.points_pct IGNORE NULLS) OVER w AS points_pct,
            LAST_VALUE(snap.projected_points IGNORE NULLS) OVER w AS projected_points,
            LAST_VALUE(snap.games_played IGNORE NULLS) OVER w AS games_played

        FROM spine s
        LEFT JOIN snap
            ON snap.season_id = s.season_id
            AND snap.team_id = s.team_id
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
    # Feature engineering
    # ============================================================
    print("🧠 Applying feature engineering")
    df_features = build_features(df_raw)

    # ============================================================
    # Model inference
    # ============================================================
    print("🎯 Scoring playoff probabilities")
    df_scored = score_dataframe(df_features)

    # ============================================================
    # BACKFILL-SAFE FILTERING
    # ============================================================
    print("🔍 Checking for missing/backfill dates")

    existing_dates = con.execute(f"""
        SELECT DISTINCT game_date
        FROM {TARGET_TABLE}
        WHERE season_id = {season_id}
        AND model_version = '{MODEL_METADATA["version"]}'
    """).df()["game_date"]

    existing_dates = set(pd.to_datetime(existing_dates).dt.date)

    df_scored["game_date"] = pd.to_datetime(df_scored["game_date"]).dt.date

    df_present = (
        df_scored[
            ~df_scored["game_date"].isin(existing_dates)
        ]
        .sort_values(["game_date", "team_id"])
        .copy()
    )

    if df_present.empty:
        print("ℹ️ No new or missing dates to score")
        return

    # ============================================================
    # Metadata
    # ============================================================
    run_ts = datetime.now(timezone.utc)

    df_present["model_name"] = MODEL_METADATA["model_name"]
    df_present["model_version"] = MODEL_METADATA["version"]
    df_present["model_run_ts"] = run_ts

    # ============================================================
    # Presentation adjustments
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
        lambda r: adjust_for_uncertainty(r["playoff_prob"], r["season_bucket"]),
        axis=1,
    )

    # ============================================================
    # Tier assignment
    # ============================================================
    def assign_tier(p: float) -> str:
        for tier, (low, high) in PROBABILITY_TIERS.items():
            if low <= p < high:
                return tier
        return "Unknown"

    df_present["playoff_tier"] = df_present["playoff_prob_display"].apply(assign_tier)

    # ============================================================
    # Output columns
    # ============================================================
    output_cols = [
        "season_id",
        "game_date",
        "team_id",
        "tri_code",
        "conference",
        "division",
        "conference_rank",
        "division_rank",
        "wildcard_rank",
        "points",
        "projected_points",
        "playoff_prob",
        "playoff_prob_xgb",
        "playoff_prob_rf",
        "playoff_prob_display",
        "playoff_tier",
        "season_bucket",
        "model_name",
        "model_version",
        "model_run_ts",
    ]

    df_out = df_present[output_cols].copy()

    # ============================================================
    # UPSERT
    # ============================================================
    print("🧪 Writing predictions to DuckDB")

    df_out = (
        df_out
        .sort_values(["season_id", "game_date", "team_id", "model_version"])
        .drop_duplicates(
            subset=["season_id", "game_date", "team_id", "model_version"],
            keep="last",
        )
        .copy()
    )

    con.register("pred_df", df_out)

    con.execute(f"""
    DELETE FROM {TARGET_TABLE}
    WHERE (season_id, game_date, team_id, model_version) IN (
        SELECT season_id, game_date, team_id, model_version
        FROM pred_df
    )
    """)

    con.execute(f"""
    INSERT INTO {TARGET_TABLE}
    SELECT * FROM pred_df
    """)

    con.unregister("pred_df")

    # ============================================================
    # Summary
    # ============================================================
    print("✅ Predictions upserted successfully")
    print(f"Teams scored: {df_out['tri_code'].nunique()}")
    print(f"Date range: {df_out['game_date'].min()} → {df_out['game_date'].max()}")
    print(EARLY_SEASON_NOTE)