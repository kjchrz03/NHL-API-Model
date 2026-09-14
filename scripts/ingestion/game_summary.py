# ============================================================
# Imports
# ============================================================
# ============================================================
# Imports
# ============================================================
import requests
import numpy as np
import pandas as pd
from datetime import date, datetime
from typing import List
import pytz
import duckdb
from typing import Any
from config import DB_PATH

# ============================================================
# Main Entry Point (called by pipeline)
# ============================================================

def run_game_summary(
    *,
    con: duckdb.DuckDBPyConnection,
    season_id: int,
    season_start: pd.Timestamp,
    season_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Update all_game_summary for a given season.
    Handles FUT → OFF transitions only.
    """

    print(f"▶ Running game summary update for season {season_id}")

    # ============================================================
    # Load Existing FUT Games
    # ============================================================

    def load_games(season_id: int) -> pd.DataFrame:
        query = """
            SELECT *
            FROM all_game_summary
            WHERE season_id = ?
              AND game_date < CURRENT_DATE
              AND gameState = 'FUT'
            ORDER BY game_id DESC
        """
        df = con.execute(query, [season_id]).df()
        print(f"Loaded {len(df):,} FUT games from DuckDB")
        return df
    # ============================================================
    # Game Summary Schema
    # ============================================================

    GS_SCHEMA = {
        "game_id": "Int64",
        "season_id": "Int64",
        "away_team": "Int64",
        "away_tri_code": "string",
        "home_team": "Int64",
        "home_tri_code": "string",
        "awayTeam.score": "Int64",
        "homeTeam.score": "Int64",
        "gameState": "string",
        "link": "string",
        "game_date": "DATE",
        "api_snapshot_date": "DATE",
    }

    def enforce_gs_schema(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # keep only known cols
        df = df[[c for c in GS_SCHEMA if c in df.columns]]

        # add missing cols
        for col in GS_SCHEMA:
            if col not in df.columns:
                df[col] = pd.NA

        # enforce dtypes
        for col, dtype in GS_SCHEMA.items():
            if dtype == "string":
                df[col] = df[col].astype("string")
            elif dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "DATE":
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
            else:  # Int64
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        return df[list(GS_SCHEMA.keys())]


    # ============================================================
    # Fetch New / Updated Games from NHL API
    # ============================================================

    def fetch_new_games(past_games: pd.DataFrame) -> pd.DataFrame:
        base_url = "https://api-web.nhle.com/v1/schedule/"

        # Only check dates that still have unresolved FUT games
        dates_to_check = (
            pd.to_datetime(past_games["game_date"])
            .dt.normalize()
            .drop_duplicates()
            .sort_values()
        )

        if dates_to_check.empty:
            print("No unresolved game dates to check.")
            return pd.DataFrame()

        print(
            f"Checking {len(dates_to_check)} unresolved game date(s): "
            f"{dates_to_check.min().date()} → {dates_to_check.max().date()}"
        )

        all_games: list[dict] = []

        for d in dates_to_check:
            url = f"{base_url}{d.strftime('%Y-%m-%d')}"
            r = requests.get(url, timeout=10)

            if r.status_code != 200:
                print(f"⚠ Failed request for {url}")
                continue

            data = r.json()

            for week in data.get("gameWeek", []):
                for game in week.get("games", []):
                    all_games.append(game)

        if not all_games:
            print("No games returned from API.")
            return pd.DataFrame()

        print(f"Retrieved {len(all_games):,} games from API.")

        # -------------------------------------------------
        # Normalize to your DuckDB schema
        # -------------------------------------------------
        df = pd.json_normalize(all_games)

        df = df.drop_duplicates(subset=["id"]).copy()

        # regular season only
        df = df[df["gameType"] == 2].copy()

        # rename API fields -> table fields
        df = df.rename(columns={
            "awayTeam.id": "away_team",
            "awayTeam.abbrev": "away_tri_code",
            "homeTeam.id": "home_team",
            "homeTeam.abbrev": "home_tri_code",
        })

        # ids + season
        df["game_id"] = pd.to_numeric(df["id"], errors="raise").astype("int64")
        df.drop(columns=["id"], inplace=True)
        df["season_id"] = season_id

        # link
        df["link"] = (
            "https://api-web.nhle.com/v1/gamecenter/"
            + df["game_id"].astype(str)
            + "/play-by-play"
        )

        # game_date (ET)
        eastern = pytz.timezone("America/New_York")
        df["startTimeUTC"] = pd.to_datetime(df["startTimeUTC"], utc=True)
        df["game_date_time"] = df["startTimeUTC"].dt.tz_convert(eastern)
        df["game_date"] = df["game_date_time"].dt.strftime("%Y-%m-%d")

        # snapshot date
        df["api_snapshot_date"] = pd.Timestamp.now()

        df = df[
            [
                "game_id",
                "season_id",
                "away_team",
                "away_tri_code",
                "home_team",
                "home_tri_code",
                "awayTeam.score",
                "homeTeam.score",
                "gameState",
                "link",
                "game_date",
                "api_snapshot_date",
            ]
        ].copy()

        df = enforce_gs_schema(df)

        df.sort_values("game_id", inplace=True)
        df.reset_index(drop=True, inplace=True)

        print(f"Fetched {len(df):,} games from API")
        return df
    # ============================================================
    # Determine Which Games Need Replacement (FUT -> OFF)
    # ============================================================

    def split_insert_and_replace(
        past_games: pd.DataFrame,
        api_games: pd.DataFrame,
    ) -> pd.DataFrame:

        # keep only the columns we want to update from API
        api_updates = api_games[
            ["game_id", "season_id", "gameState", "awayTeam.score", "homeTeam.score", "api_snapshot_date"]
        ].copy()

        # start from the existing FUT rows (past_games is already filtered FUT in your load_games)
        game_updates = past_games.copy()

        # normalize ids
        game_updates["game_id"] = pd.to_numeric(game_updates["game_id"], errors="coerce").astype("int64")
        api_updates["game_id"] = pd.to_numeric(api_updates["game_id"], errors="raise").astype("int64")

        # merge in the latest state/scores
        game_updates = game_updates.merge(
            api_updates,
            on=["game_id", "season_id"],
            how="left",
            suffixes=("", "_api"),
        )

        # only rows that are now completed
        game_updates = game_updates[game_updates["gameState_api"] == "OFF"].copy()

        if game_updates.empty:
            return game_updates

        # overwrite fields with API values
        game_updates["gameState"] = game_updates["gameState_api"]
        game_updates["awayTeam.score"] = game_updates["awayTeam.score_api"]
        game_updates["homeTeam.score"] = game_updates["homeTeam.score_api"]
        game_updates["api_snapshot_date"] = game_updates["api_snapshot_date_api"]

        # drop helper cols
        game_updates.drop(
            columns=[
                "gameState_api",
                "awayTeam.score_api",
                "homeTeam.score_api",
                "api_snapshot_date_api",
            ],
            errors="ignore",
            inplace=True,
        )

        # enforce schema/types/order
        game_updates = enforce_gs_schema(game_updates)

        return game_updates


    # ============================================================
    # Replace Games in DuckDB
    # ============================================================

    def replace_games(df: pd.DataFrame) -> None:
        if df.empty:
            return

        df = df.drop_duplicates(subset=["game_id"]).copy()
        df = enforce_gs_schema(df)

        con.register("df_replace", df)

        con.execute("""
            DELETE FROM all_game_summary
            WHERE game_id IN (SELECT game_id FROM df_replace)
            AND season_id IN (SELECT season_id FROM df_replace)
        """)

        cols = ", ".join([f'"{c}"' for c in df.columns])
        con.execute(f"""
            INSERT INTO all_game_summary ({cols})
            SELECT {cols}
            FROM df_replace
        """)

        print(f"Replaced {len(df):,} completed games")


    # ============================================================
    # Execution
    # ============================================================

    past_games = load_games(season_id)

    if past_games.empty:
        print("✅ Game summary table already up to date.")
        return pd.DataFrame()

    new_games = fetch_new_games(past_games)

    if new_games.empty:
        print("✅ No API updates needed.")
        return pd.DataFrame()

    updated_games = split_insert_and_replace(
        past_games=past_games,
        api_games=new_games,
    )

    if updated_games.empty:
        print("✅ No completed games to update.")
        return pd.DataFrame()

    print(f"Updating {len(updated_games):,} games...")
    replace_games(updated_games)

    print("✅ Game summary update complete.")

    return updated_games