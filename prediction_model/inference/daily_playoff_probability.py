# ============================================================
# run_daily_playoff_predictions.py
# ============================================================

"""
Daily execution script for NHL playoff probability predictions.

Prediction grain:
    one row per team per NHL game date

For dates when a team does not play, the team's most recent
available team_daily_stats snapshot is carried forward.
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
from prediction_model.inference.versioned_scoring import score_version
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
    refresh_existing: bool = False,
    model_version: str = "v6",
) -> None:

    if model_version not in {"v6", "v7"}:
        raise ValueError(f"Unsupported model version: {model_version}")

    print(f"▶ Running playoff probabilities for season {season_id} ({model_version})")

    # ========================================================
    # Ensure target table exists
    # ========================================================

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

            PRIMARY KEY (
                season_id,
                game_date,
                team_id,
                model_version
            )
        )
    """)

    # ========================================================
    # Load canonical team-day snapshots
    #
    # SOURCE_TABLE is team-game grain.
    #
    # We want team-day grain:
    #     every team × every NHL game date.
    #
    # On dates when a team did not play, use that team's most
    # recent team_daily_stats row as of that date.
    # ========================================================

    print("🔌 Using DuckDB connection")

    query = f"""
        WITH teams AS (
            SELECT DISTINCT
                team_id
            FROM {SOURCE_TABLE}
            WHERE season_id = ?
        ),

        dates AS (
            SELECT DISTINCT
                game_date
            FROM {SOURCE_TABLE}
            WHERE season_id = ?
        ),

        spine AS (
            SELECT
                ?::BIGINT AS season_id,
                d.game_date,
                t.team_id
            FROM dates d
            CROSS JOIN teams t
        )

        SELECT
            snap.* EXCLUDE (
                season_id,
                game_date,
                team_id
            ),

            spine.season_id,
            spine.game_date,
            spine.team_id

        FROM spine

        LEFT JOIN LATERAL (
            SELECT *
            FROM {SOURCE_TABLE} AS candidate
            WHERE candidate.team_id = spine.team_id
              AND candidate.season_id = spine.season_id
              AND candidate.game_date <= spine.game_date
            ORDER BY candidate.game_date DESC, candidate.game_id DESC
            LIMIT 1
        ) AS snap ON TRUE

        ORDER BY
            spine.game_date,
            spine.team_id
    """

    df_raw = con.execute(
        query,
        [
            season_id,
            season_id,
            season_id,
        ],
    ).df()

    if df_raw.empty:
        raise RuntimeError(
            f"❌ No rows loaded from {SOURCE_TABLE} "
            f"for season {season_id}"
        )

    print(
        f"📥 Loaded {len(df_raw):,} team-day rows "
        f"from {SOURCE_TABLE}"
    )

    # ========================================================
    # Remove pre-first-game snapshots
    # ========================================================

    missing_snapshot = (
        df_raw["games_played"].isna()
        | df_raw["tri_code"].isna()
    )

    missing_count = int(missing_snapshot.sum())

    if missing_count:
        print(
            f"ℹ️ Removing {missing_count:,} team-day rows "
            "without an available team snapshot"
        )

        df_raw = (
            df_raw.loc[~missing_snapshot]
            .copy()
            .reset_index(drop=True)
        )

    if df_raw.empty:
        raise RuntimeError(
            "❌ No usable team-day snapshots remain"
        )

    # ========================================================
    # Validate team-day uniqueness
    # ========================================================

    duplicate_mask = df_raw.duplicated(
        subset=[
            "season_id",
            "game_date",
            "team_id",
        ],
        keep=False,
    )

    if duplicate_mask.any():

        duplicates = (
            df_raw.loc[
                duplicate_mask,
                [
                    "season_id",
                    "game_date",
                    "team_id",
                ],
            ]
            .sort_values(
                [
                    "season_id",
                    "game_date",
                    "team_id",
                ]
            )
        )

        raise RuntimeError(
            "❌ Duplicate team-day snapshots detected:\n"
            + duplicates.head(20).to_string(index=False)
        )

    # ========================================================
    # Feature engineering
    # ========================================================

    print("🧠 Applying feature engineering")

    df_features = build_features(
        df_raw.copy()
    )

    # ========================================================
    # Model inference
    #
    # inference.py enforces the exact saved model feature
    # contract.
    # ========================================================

    print("🎯 Scoring playoff probabilities")

    if model_version == "v6":
        df_scored = score_dataframe(df_features)
    else:
        df_scored = score_version(df_features, version=model_version)

    # ========================================================
    # Backfill and refresh filtering
    # ========================================================

    print("🔍 Checking existing prediction coverage")

    df_scored["game_date"] = pd.to_datetime(
        df_scored["game_date"]
    ).dt.date

    # Compare exact team identities, not just team counts.
    expected_teams = df_scored.groupby("game_date")["team_id"].agg(lambda x: set(x))
    stored_teams_df = con.execute(
        f"""SELECT game_date, team_id FROM {TARGET_TABLE}
            WHERE season_id = ? AND model_version = ?""",
        [season_id, model_version],
    ).df()
    stored_teams_df["game_date"] = pd.to_datetime(stored_teams_df["game_date"]).dt.date
    stored_teams = stored_teams_df.groupby("game_date")["team_id"].agg(lambda x: set(x))
    incomplete_dates = {
        date for date, expected in expected_teams.items()
        if stored_teams.get(date, set()) != expected
    }

    if refresh_existing:
        print("🔄 Refresh mode: replacing existing predictions")
        df_present = df_scored.copy()
    else:
        print("➕ Incremental mode: inserting missing/incomplete dates")
        df_present = df_scored.loc[
            df_scored["game_date"].isin(incomplete_dates)
        ].copy()

    df_present = (
        df_present
        .sort_values(["game_date", "team_id"])
        .reset_index(drop=True)
    )

    if df_present.empty:
        print("ℹ️ No predictions need writing")
        return

    print(f"Rows selected for writing: {len(df_present):,}")
    # ========================================================
    # Metadata
    # ========================================================

    run_ts = datetime.now(timezone.utc)

    df_present["model_name"] = (
        MODEL_METADATA["model_name"]
    )

    df_present["model_version"] = (
        model_version
    )

    df_present["model_run_ts"] = run_ts

    # ========================================================
    # Season bucket
    # ========================================================

    def season_bucket(gp: int) -> str:

        if pd.isna(gp):
            return "early"

        if gp <= 20:
            return "early"

        if gp <= 50:
            return "mid"

        return "late"

    df_present["season_bucket"] = (
        df_present["games_played"]
        .apply(season_bucket)
    )

    # ========================================================
    # Presentation uncertainty adjustment
    #
    # playoff_prob remains the raw model probability.
    #
    # Only playoff_prob_display receives the early-season
    # compression toward 50%.
    # ========================================================

    def adjust_for_uncertainty(
        p: float,
        bucket: str,
    ) -> float:

        if model_version == "v6" and bucket == "early":
            return 0.45 * p + 0.275

        return p

    df_present["playoff_prob_display"] = (
        df_present.apply(
            lambda r: adjust_for_uncertainty(
                r["playoff_prob"],
                r["season_bucket"],
            ),
            axis=1,
        )
    )

    # ========================================================
    # Tier assignment
    # ========================================================

    def assign_tier(p: float) -> str:

        for tier, (low, high) in PROBABILITY_TIERS.items():

            if low <= p < high:
                return tier

        return "Unknown"

    df_present["playoff_tier"] = (
        df_present["playoff_prob_display"]
        .apply(assign_tier)
    )

    # ========================================================
    # Output columns
    # ========================================================

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

    missing_output_cols = [
        col
        for col in output_cols
        if col not in df_present.columns
    ]

    if missing_output_cols:

        raise RuntimeError(
            "❌ Missing prediction output columns: "
            + ", ".join(missing_output_cols)
        )

    df_out = (
        df_present[output_cols]
        .copy()
    )

    # ========================================================
    # Transactional UPSERT
    # ========================================================

    print("🧪 Writing predictions to DuckDB")

    key_cols = ["season_id", "game_date", "team_id", "model_version"]
    duplicate_mask = df_out.duplicated(subset=key_cols, keep=False)
    if duplicate_mask.any():
        duplicates = df_out.loc[duplicate_mask, key_cols].head(20)
        raise RuntimeError(
            "Duplicate scored prediction keys detected:\n"
            + duplicates.to_string(index=False)
        )
    df_out = df_out.sort_values(key_cols).copy()

    con.register("pred_df", df_out)

    try:
        con.execute("BEGIN TRANSACTION")

        # Remove previous predictions for the dates being refreshed.
        # This also removes stale team rows on those dates.
        con.execute(f"""
            DELETE FROM {TARGET_TABLE}
            WHERE season_id = ?
            AND model_version = ?
            AND game_date IN (
                SELECT DISTINCT game_date
                FROM pred_df
            )
        """, [season_id, model_version])

        con.execute(f"""
            INSERT INTO {TARGET_TABLE} (
                {", ".join(output_cols)}
            )
            SELECT
                {", ".join(output_cols)}
            FROM pred_df
        """)

        con.execute("COMMIT")

    except Exception:
        con.execute("ROLLBACK")
        raise

    finally:
        con.unregister("pred_df")

    # ========================================================
    # Summary
    # ========================================================

    print("✅ Predictions upserted successfully")

    print(
        f"Rows scored: {len(df_out):,}"
    )

    print(
        f"Teams scored: "
        f"{df_out['team_id'].nunique()}"
    )

    print(
        f"Date range: "
        f"{df_out['game_date'].min()} "
        f"→ {df_out['game_date'].max()}"
    )

    print(EARLY_SEASON_NOTE)
    
# if __name__ == "__main__":
#     from config import DB_PATH

#     con = duckdb.connect(DB_PATH)

#     try:
#         run_probability_scores(
#             con=con,
#             season_id=20262027,
#             refresh_existing=True
#         )
#     finally:
#         con.close()