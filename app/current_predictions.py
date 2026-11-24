import requests
import numpy as np
import pandas as pd
from datetime import date
import os
import time
import random
import duckdb
import html
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prediction_model.explainability.feature_metadata import (
    FEATURE_LABELS,
    FEATURE_DESCRIPTIONS,
    get_feature_label,
    get_feature_description,
)
logger = logging.getLogger(__name__)


# ============================================================
# SAFE REQUEST WITH BACKOFF
# ============================================================


def safe_request(url, params=None, max_retries=5):
    delay = 1

    for _ in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=10)

            if r.status_code == 200:
                return r

            if r.status_code == 429:
                time.sleep(delay)
                delay = min(delay * 2, 5)
                continue

            return None

        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 5)

    return None


# ============================================================
# DATABASE CONFIGURATION
# ============================================================
from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCAL_DB_PATH = PROJECT_ROOT / "data" / "duckdb" / "nhl_data.duckdb"
DEPLOYMENT_DIR = PROJECT_ROOT / "data" / "deployment"


def get_connection():
    if LOCAL_DB_PATH.exists():
        return duckdb.connect(str(LOCAL_DB_PATH), read_only=True)

    # Streamlit Cloud: use an in-memory DuckDB database
    # backed by the deployed Parquet files.
    con = duckdb.connect(":memory:")

    if not DEPLOYMENT_DIR.is_dir():
        raise FileNotFoundError(
            f"Deployment data directory not found: {DEPLOYMENT_DIR}"
        )

    for parquet_file in DEPLOYMENT_DIR.glob("*.parquet"):
        table_name = parquet_file.stem
        escaped_path = parquet_file.as_posix().replace("'", "''")

        con.execute(f'''
            CREATE VIEW "{table_name}" AS
            SELECT * FROM read_parquet('{escaped_path}')
        ''')

    return con
pd.options.mode.chained_assignment = None

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


# ============================================================
# FETCH CURRENT PLAYOFF PREDICTIONS
# ============================================================


def fetch_current_predictions(season_id, model_version="v6"):

    con = None

    try:
        logger.info(
            "Fetching playoff predictions | Season: %s | Model: %s",
            season_id,
            model_version
        )

        con = get_connection()

        # ====================================================
        # 1. FETCH GAME SUMMARY
        # ====================================================
        daily_summary = con.execute("""
            SELECT *
            FROM all_game_summary
            WHERE season_id = ?
        """, [season_id]).df()

        logger.info(
            "Loaded %s game summary records",
            len(daily_summary)
        )

        # ====================================================
        # 2. CREATE HOME TEAM PERSPECTIVE
        # ====================================================
        home_games = daily_summary[
            [
                "game_id",
                "season_id",
                "game_date",
                "home_tri_code",
                "away_tri_code",
                "homeTeam.score",
                "awayTeam.score",
                "gameState"
            ]
        ].copy()

        home_games = home_games.rename(columns={
            "home_tri_code": "tri_code",
            "away_tri_code": "opponent",
            "homeTeam.score": "team_score",
            "awayTeam.score": "opponent_score"
        })

        home_games["home_away"] = "Home"

        # ====================================================
        # 3. CREATE AWAY TEAM PERSPECTIVE
        # ====================================================
        away_games = daily_summary[
            [
                "game_id",
                "season_id",
                "game_date",
                "away_tri_code",
                "home_tri_code",
                "awayTeam.score",
                "homeTeam.score",
                "gameState"
            ]
        ].copy()

        away_games = away_games.rename(columns={
            "away_tri_code": "tri_code",
            "home_tri_code": "opponent",
            "awayTeam.score": "team_score",
            "homeTeam.score": "opponent_score"
        })

        away_games["home_away"] = "Away"

        # ====================================================
        # 4. COMBINE HOME AND AWAY GAMES
        # ====================================================
        team_games = pd.concat(
            [home_games, away_games],
            ignore_index=True
        )

        team_games["game_date"] = pd.to_datetime(
            team_games["game_date"]
        ).dt.normalize()

        # Remove duplicate game records, if any
        team_games = team_games.drop_duplicates(
            subset=["season_id", "game_id", "tri_code"]
        )

        logger.info(
            "Prepared %s team-game records",
            len(team_games)
        )

        # Check for multiple games per team per day
        duplicate_game_keys = team_games.duplicated(
            subset=["season_id", "game_date", "tri_code"]
        ).sum()

        if duplicate_game_keys:
            logger.warning(
                "Found %s duplicate team/date game keys",
                duplicate_game_keys
            )

        # ====================================================
        # 5. FETCH TEAM INFORMATION
        # ====================================================
        teams = con.execute("""
            SELECT *
            FROM all_teams
        """).df()

        teams = teams[
            ["team_id","tri_code", "team_name"]
            ].drop_duplicates(subset=["team_id"])

        # ====================================================
        # ADD OPPONENT NAMES
        # ====================================================

        opponents = (
            teams[["tri_code", "team_name"]]
            .drop_duplicates(subset=["tri_code"])
            .rename(columns={
                "tri_code": "opponent",
                "team_name": "opponent_name"
            })
        )

        team_games = team_games.merge(
            opponents,
            on="opponent",
            how="left",
            validate="many_to_one"
        )

        # ====================================================
        # 6. FETCH PLAYOFF PREDICTIONS
        # ====================================================
        predictions = con.execute("""
            SELECT *
            FROM playoff_daily_predictions
            WHERE season_id = ?
              AND model_version = ?
        """, [season_id, model_version]).df()

        logger.info(
            "Loaded %s prediction records",
            len(predictions)
        )

        # Avoid introducing duplicate team_name columns
        if "team_name" in predictions.columns:
            predictions = predictions.drop(
                columns=["team_name"]
            )

        predictions = predictions.merge(
            teams[["team_id", "team_name"]],
            on="team_id",
            how="left",
            validate="many_to_one"
        )

        predictions = predictions.sort_values(
            by="game_date"
        ).reset_index(drop=True)

        # ====================================================
        # 7. NORMALIZE DATES
        # ====================================================
        predictions["game_date"] = pd.to_datetime(
            predictions["game_date"]
        ).dt.normalize()

        # ====================================================
        # 8. CHECK PREDICTION KEYS
        # ====================================================
        duplicate_prediction_keys = predictions.duplicated(
            subset=["season_id", "game_date", "tri_code"]
        ).sum()

        if duplicate_prediction_keys:
            logger.warning(
                "Found %s duplicate prediction team/date keys",
                duplicate_prediction_keys
            )

        # ====================================================
        # 9. MERGE PREDICTIONS WITH GAMES
        # ====================================================
        predictions_df = predictions.merge(
            team_games,
            on=["season_id", "game_date", "tri_code"],
            how="left",
            validate="many_to_many"
        )

        logger.info(
            "Merge completed | Predictions: %s | Merged rows: %s",
            len(predictions),
            len(predictions_df)
        )

        if len(predictions_df) > len(predictions):
            logger.warning(
                "Game merge increased row count by %s",
                len(predictions_df) - len(predictions)
            )

        # ====================================================
        # 10. IDENTIFY COMPLETED GAMES
        # ====================================================
        predictions_df["played_game"] = (
            predictions_df["game_id"].notna()
            & predictions_df["gameState"].eq("OFF")
            & predictions_df["team_score"].notna()
            & predictions_df["opponent_score"].notna()
        )

        # ====================================================
        # 11. FORMAT GAME IDS
        # ====================================================
        predictions_df["game_id"] = (
            pd.to_numeric(
                predictions_df["game_id"],
                errors="coerce"
            )
            .astype("Int64")
            .astype("string")
        )

        logger.info(
            "Completed successfully | Total rows: %s | Completed-game rows: %s",
            len(predictions_df),
            predictions_df["played_game"].sum()
        )

        return predictions_df

    except Exception:
        logger.exception(
            "Failed to fetch playoff predictions | Season: %s | Model: %s",
            season_id,
            model_version
        )
        raise

    finally:
        if con is not None:
            con.close()


# ============================================================
# FETCH SHAP VALUES
# ============================================================


def fetch_shap_values(season_id, model_version="v6"):

    con = get_connection()

    try:
        shap_df = con.execute("""
            SELECT *
            FROM playoff_shap_values_v6_balanced
            WHERE season_id = ?
              AND model_version = ?
            ORDER BY game_date, tri_code
        """, [season_id, model_version]).df()

        return shap_df

    finally:
        con.close()

def fetch_shap_context(season_id, model_version="v6"):
    if model_version != "v6":
        raise ValueError("Postgame SHAP context currently supports v6 only")

    con = get_connection()
    try:
        return con.execute("""
            SELECT *
            FROM playoff_shap_context_v6
            WHERE season_id = ?
              AND model_version = ?
            ORDER BY snapshot_date, team_id
        """, [season_id, model_version]).df()
    finally:
        con.close()
# ============================================================
# FETCH SHAP SUMMARY
# ============================================================


def fetch_shap_summary(season_id, model_version="v6"):

    # Match the version-specific tables in shap_summary.py
    if model_version == "v6":
        table_name = "playoff_shap_summary_v6_balanced"
    else:
        import re

        if not re.fullmatch(r"v[0-9]+", model_version):
            raise ValueError(
                f"Invalid model version: {model_version}"
            )

        table_name = f"playoff_shap_summary_{model_version}"

    con = None

    try:
        logger.info(
            "Fetching SHAP summary | Season: %s | Model: %s",
            season_id,
            model_version
        )

        con = get_connection()

        shap_summary_df = con.execute(f"""
            SELECT
                season_id,
                snapshot_date,
                game_date,
                team_id,
                tri_code,
                model_version,
                playoff_prob,
                feature,
                shap_value,
                direction,
                rank
            FROM "{table_name}"
            WHERE season_id = ?
              AND model_version = ?
            ORDER BY
                snapshot_date,
                tri_code,
                direction,
                rank
        """, [season_id, model_version]).df()

        # Normalize dates
        for col in ["snapshot_date", "game_date"]:
            shap_summary_df[col] = pd.to_datetime(
                shap_summary_df[col]
            ).dt.normalize()

        # Check for duplicate rankings
        duplicate_keys = shap_summary_df.duplicated(
            subset=[
                "season_id",
                "snapshot_date",
                "team_id",
                "model_version",
                "direction",
                "rank"
            ]
        ).sum()

        if duplicate_keys:
            raise ValueError(
                f"Found {duplicate_keys} duplicate "
                "SHAP summary rankings"
            )

        logger.info(
            "Loaded %s SHAP summary records across %s snapshots",
            len(shap_summary_df),
            shap_summary_df[
                ["snapshot_date", "team_id"]
            ].drop_duplicates().shape[0]
        )

        return shap_summary_df

    except Exception:

        logger.exception(
            "Failed to fetch SHAP summary | Season: %s | Model: %s",
            season_id,
            model_version
        )

        raise

    finally:

        if con is not None:
            con.close()


# ============================================================
# EXTRACT TOP POSITIVE AND NEGATIVE SHAP CONTRIBUTORS
# ============================================================


def get_top_shap_features(df, top_n=5):

    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    if df.empty:
        return pd.DataFrame(
            columns=[
                "season_id",
                "snapshot_date",
                "game_date",
                "team_id",
                "tri_code",
                "model_version",
                "feature",
                "shap_value",
                "impact"
            ]
        )

    df = df.copy()

    # Unique identifier for each original prediction record
    df["_record_id"] = np.arange(len(df))

    # Identify all SHAP feature columns
    # Exclude the SHAP base value
    shap_cols = [
        col for col in df.columns
        if col.startswith("shap_")
        and col != "shap_base_value"
    ]

    # Columns identifying each prediction
    id_cols = [
        "_record_id",
        "season_id",
        "snapshot_date",
        "game_date",
        "team_id",
        "tri_code",
        "model_version"
    ]

    # Convert wide SHAP data into long format
    long_df = df.melt(
        id_vars=id_cols,
        value_vars=shap_cols,
        var_name="feature",
        value_name="shap_value"
    )

    # Remove shap_ prefix from feature names
    long_df["feature"] = (
        long_df["feature"]
        .str.removeprefix("shap_")
    )

    # Ensure SHAP values are numeric
    long_df["shap_value"] = pd.to_numeric(
        long_df["shap_value"],
        errors="coerce"
    )

    # Remove missing SHAP values
    long_df = long_df.dropna(
        subset=["shap_value"]
    )

    # --------------------------------------------------------
    # TOP POSITIVE CONTRIBUTORS
    # --------------------------------------------------------
    positive = (
        long_df[long_df["shap_value"] > 0]
        .sort_values(
            "shap_value",
            ascending=False
        )
        .groupby("_record_id")
        .head(top_n)
        .copy()
    )

    positive["impact"] = "Positive"

    # --------------------------------------------------------
    # TOP NEGATIVE CONTRIBUTORS
    # --------------------------------------------------------
    negative = (
        long_df[long_df["shap_value"] < 0]
        .sort_values(
            "shap_value",
            ascending=True
        )
        .groupby("_record_id")
        .head(top_n)
        .copy()
    )

    negative["impact"] = "Negative"

    # --------------------------------------------------------
    # COMBINE RESULTS
    # --------------------------------------------------------
    top_shap = pd.concat(
        [positive, negative],
        ignore_index=True
    )

    top_shap = top_shap.drop(
        columns="_record_id"
    )

    return top_shap


def calculate_shap_consistency(shap_df):

    if shap_df.empty:
        return pd.DataFrame()

    df = shap_df.copy()

    df["game_date"] = pd.to_datetime(df["game_date"])

    # Keep the latest record for each team/date
    if "created_at" in df.columns:
        df = df.sort_values("created_at")

    df = df.drop_duplicates(
        subset=["season_id", "model_version", "tri_code", "game_date"],
        keep="last"
    )

    # Identify all SHAP feature columns
    shap_cols = [
        c for c in df.columns
        if c.startswith("shap_")
        and c != "shap_base_value"
    ]

    # Convert wide SHAP data to long format
    shap_long = df.melt(
        id_vars=[
            "season_id",
            "model_version",
            "tri_code",
            "game_date"
        ],
        value_vars=shap_cols,
        var_name="feature",
        value_name="shap_value"
    )

    shap_long["feature"] = (
        shap_long["feature"].str.removeprefix("shap_")
    )

    shap_long["shap_value"] = pd.to_numeric(
        shap_long["shap_value"],
        errors="coerce"
    )

    shap_long = shap_long.dropna(subset=["shap_value"])

    shap_long = shap_long.sort_values(
        ["season_id", "model_version", "tri_code", "feature", "game_date"]
    )

    group_cols = [
        "season_id",
        "model_version",
        "tri_code",
        "feature"
    ]

    grouped = shap_long.groupby(group_cols)

    # Running number of snapshots
    shap_long["snapshots"] = grouped.cumcount() + 1

    # Positive and negative indicators
    shap_long["positive"] = (
        shap_long["shap_value"] > 0
    ).astype(int)

    shap_long["negative"] = (
        shap_long["shap_value"] < 0
    ).astype(int)

    # Running counts
    shap_long["positive_count"] = (
        shap_long.groupby(group_cols)["positive"].cumsum()
    )

    shap_long["negative_count"] = (
        shap_long.groupby(group_cols)["negative"].cumsum()
    )

    # Running percentages
    shap_long["pct_positive"] = (
        shap_long["positive_count"] / shap_long["snapshots"]
    )

    shap_long["pct_negative"] = (
        shap_long["negative_count"] / shap_long["snapshots"]
    )

    # Running average contribution
    shap_long["avg_shap"] = (
        shap_long.groupby(group_cols)["shap_value"].cumsum()
        / shap_long["snapshots"]
    )

    return shap_long


# ============================================================
# OPTIONAL: TEST FUNCTIONS WHEN RUN DIRECTLY
# ============================================================


if __name__ == "__main__":

    season_id = 20262027
    model_version = "v6"

    # Fetch predictions
    predictions_df = fetch_current_predictions(
        season_id=season_id,
        model_version=model_version
    )

    print(f"Prediction records: {len(predictions_df):,}")

    # Fetch SHAP values
    shap_df = fetch_shap_values(
        season_id=season_id,
        model_version=model_version
    )

    print(f"SHAP records: {len(shap_df):,}")

    shap_summary_df = fetch_shap_summary(
        season_id=season_id,
        model_version=model_version
    )

    print(f"SHAP summary records: {len(shap_summary_df):,}")

    print("\nSample SHAP summary:")
    print(
        shap_summary_df.head(10).to_string(index=False)
    )

    # Extract top 5 positive and negative contributors
    top_shap = get_top_shap_features(
        shap_df,
        top_n=5
    )

    print(f"Top SHAP contributor records: {len(top_shap):,}")

    print("\nSample SHAP contributors:")
    print(top_shap.head(10).to_string(index=False))
