# ============================================================
# Imports
# ============================================================

import requests
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import os
import sys
import warnings
import duckdb
from typing import Any
from config import DB_PATH

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

# ============================================================
# Play-by-Play Runner
# ============================================================
def run_play_by_play(
    *,
    con: duckdb.DuckDBPyConnection,
    season_id: int,
    changed_games: pd.DataFrame | None = None,
    ):
    """
    Update play-by-play data for a given season.
    Assumes game summary has already been run.
    """

    print(f"▶ Running play-by-play for season {season_id}")

    # ============================================================
    # Helpers
    # ============================================================
    def load_missing_pbp_ids(season_id: int) -> list[int]:
        query = """
            SELECT DISTINCT gs.game_id
            FROM all_game_summary AS gs
            WHERE gs.season_id = ?
            AND gs.gameState = 'OFF'
            AND NOT EXISTS (
                SELECT 1
                FROM all_play_by_play AS pbp
                WHERE pbp.season_id = gs.season_id
                    AND pbp.game_id = gs.game_id
            )
            ORDER BY gs.game_id
        """

        df = con.execute(query, [season_id]).df()
        return df["game_id"].astype("int64").tolist()


    def load_last_pbp_id(season_id: int) -> int | None:
        query = """
            SELECT MAX(game_id)
            FROM all_play_by_play
            WHERE season_id = ?
        """
        result = con.execute(query, [season_id]).fetchone()
        max_id = result[0] if result else None
        print(f"Max PBP game_id: {max_id}")
        return max_id

    def load_last_game_summary_id(season_id: int) -> int | None:
        query = """
            SELECT MAX(game_id)
            FROM all_game_summary
            WHERE season_id = ?
              AND gameState = 'OFF'
        """
        result = con.execute(query, [season_id]).fetchone()
        max_id = result[0] if result else None
        print(f"Max completed GS game_id: {max_id}")
        return max_id

    def load_game_summary_df(season_id: int) -> pd.DataFrame:
        query = """
            SELECT *
            FROM all_game_summary
            WHERE season_id = ?
              AND gameState = 'OFF'
            ORDER BY game_id
        """
        df = con.execute(query, [season_id]).df()
        print(f"Loaded {len(df):,} completed games from summary")
        return df

    # ============================================================
    # SAFE REQUEST WITH BACKOFF
    # ============================================================
    def safe_request(url, max_retries=5):
        delay = 1

        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=7)

                if response.status_code == 200:
                    return response

                if response.status_code == 429:
                    print(f"⚠️ 429 Too Many Requests — waiting {delay}s...")
                    time.sleep(delay)
                    delay = min(delay * 2, 5)
                    continue

                print(f"⚠️ HTTP {response.status_code} for {url}")
                return None

            except KeyboardInterrupt:
                print("⚠️ Keyboard interrupt during request — aborting safely")
                return None

            except Exception as e:
                print(f"⚠️ Request error {e} — waiting {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, 5)

        print(f"❌ Failed after {max_retries} attempts: {url}")
        return None


    # ============================================================
    # PBP SCHEMA (DuckDB-aligned)
    # ============================================================

    PBP_SCHEMA = {
        "season_id": "Int64",
        "game_id": "Int64",
        "eventId": "Int64",
        "timeInPeriod": "string",
        "timeRemaining": "string",
        "situationCode": "Int64",
        "homeTeamDefendingSide": "string",
        "type_code": "Int64",
        "typeDescKey": "string",
        "sortOrder": "Int64",
        "period_number": "Int64",
        "period_type": "string",
        "max_regulation_periods": "Int64",
        "team_id": "Int64",
        "losing_player_id": "Int64",
        "winning_player_id": "Int64",
        "x_coord": "float64",
        "y_coord": "float64",
        "zone_code": "string",
        "hitter": "Int64",
        "hittee": "Int64",
        "shot_type": "string",
        "shooting_player": "Int64",
        "goalie": "Int64",
        "away_sog": "Int64",
        "home_sog": "Int64",
        "player_id": "Int64",
        "reason": "string",
        "blocking_player": "Int64",
        "desc_key": "string",
        "duration": "Int64",
        "committed_by": "Int64",
        "drawn_by": "Int64",
        "secondary_reason": "string",
        "scoring_player": "Int64",
        "scoring_player_total": "Int64",
        "assist_1": "Int64",
        "assist1_total": "Int64",
        "away_score": "Int64",
        "home_score": "Int64",
        "assist_2": "Int64",
        "assist2_total": "Int64",
        "served_by": "Int64",
        "situation": "string",
        "goalie_situation": "string",
        "game_in_seconds": "Int64",
        "home_team": "Int64",
        "away_team": "Int64",
    }

    def enforce_pbp_schema(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # -------------------------------------------------
        # Collapse duplicate columns (CRITICAL)
        # -------------------------------------------------
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        # -------------------------------------------------
        # Align columns to schema
        # -------------------------------------------------
        df = df[[c for c in PBP_SCHEMA if c in df.columns]]

        for col in PBP_SCHEMA:
            if col not in df.columns:
                df[col] = pd.NA

        # -------------------------------------------------
        # Enforce dtypes safely
        # -------------------------------------------------
        for col, dtype in PBP_SCHEMA.items():
            if dtype == "string":
                df[col] = df[col].astype("string")

            elif dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce")

            else:
                # Guard against non-Series objects
                if isinstance(df[col], pd.DataFrame):
                    df[col] = df[col].iloc[:, 0]

                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        return df[list(PBP_SCHEMA.keys())]

    # ============================================================
    # PLAY-BY-PLAY FETCHER (CORRECTED)
    # ============================================================

    # def play_by_play_events(game_summary: pd.DataFrame) -> pd.DataFrame:
    #     all_game_ids = game_summary["game_id"].unique().tolist()
    #     results = []

    #     # --------------------------------------------------
    #     # RENAME MAP (FIXED)
    #     # --------------------------------------------------
    #     rename_map = {
    #         # period info
    #         "periodDescriptor.number": "period_number",
    #         "periodDescriptor.periodType": "period_type",
    #         "periodDescriptor.maxRegulationPeriods": "max_regulation_periods",

    #         # team / players
    #         "details.eventOwnerTeamId": "event_team_id",
    #         "details.losingPlayerId": "losing_player_id",
    #         "details.winningPlayerId": "winning_player_id",

    #         # coordinates / zones
    #         "details.xCoord": "x_coord",
    #         "details.yCoord": "y_coord",
    #         "details.zoneCode": "zone_code",

    #         # shooting / scoring
    #         "details.shotType": "shot_type",
    #         "details.shootingPlayerId": "shooting_player",
    #         "details.goalieInNetId": "goalie",
    #         "details.scoringPlayerId": "scoring_player",
    #         "details.scoringPlayerTotal": "scoring_player_total",
    #         "details.assist1PlayerId": "assist_1",
    #         "details.assist1PlayerTotal": "assist1_total",
    #         "details.assist2PlayerId": "assist_2",
    #         "details.assist2PlayerTotal": "assist2_total",

    #         # scores
    #         "details.awayScore": "away_score",
    #         "details.homeScore": "home_score",

    #         # misc
    #         "details.blockingPlayerId": "blocking_player",
    #         "details.playerId": "player_id",
    #         "details.reason": "reason",
    #         "details.hittingPlayerId": "hitter",
    #         "details.hitteePlayerId": "hittee",
    #         "details.awaySOG": "away_sog",
    #         "details.homeSOG": "home_sog",
    #         "details.secondaryReason": "secondary_reason",
    #         "details.duration": "duration",
    #         "details.committedByPlayerId": "committed_by",
    #         "details.drawnByPlayerId": "drawn_by",
    #         "details.servedByPlayerId": "served_by",

    #         # 🔑 EVENT TYPE FIELDS (CRITICAL FIX)
    #         "typeCode": "type_code",                 # top-level
    #         "details.typeCode": "type_code",         # fallback
    #         "typeDescKey": "type_desc_key",           # top-level (GUARANTEED)
    #         "details.descKey": "desc_key",
    #     }

    #     situation_dictionary = {
    #         "1551": "5 on 5", "1451": "5 on 4", "1541": "5 on 4",
    #         "0651": "6 on 5", "1560": "6 on 5", "1441": "4 on 4",
    #         "1331": "3 on 3", "1460": "6 on 4", "1351": "5 on 3",
    #         "0641": "6 on 4", "1341": "4 on 3"
    #     }

    #     # --------------------------------------------------
    #     # PROCESS A SINGLE GAME
    #     # --------------------------------------------------
    #     def process_game(game_id: int) -> pd.DataFrame:
    #         match = game_summary.loc[game_summary["game_id"] == game_id, "link"]
    #         if match.empty:
    #             return pd.DataFrame()

    #         link = str(match.iloc[0]).strip()
    #         if not link.startswith("http"):
    #             return pd.DataFrame()

    #         response = safe_request(link)
    #         if response is None:
    #             return pd.DataFrame()

    #         data = response.json()
    #         if "plays" not in data:
    #             return pd.DataFrame()

    #         df = pd.json_normalize(data["plays"])
    #         df["game_id"] = game_id
    #         return df

    #     # --------------------------------------------------
    #     # FETCH ALL GAMES
    #     # --------------------------------------------------
    #     for i, game_id in enumerate(
    #         tqdm(all_game_ids, total=len(all_game_ids)),
    #         start=1
    #     ):
    #         tqdm.write(f"🎯 Processing game {i}/{len(all_game_ids)} — game_id={game_id}")
    #         result = process_game(game_id)
    #         if not result.empty:
    #             results.append(result)

    #     if not results:
    #         return pd.DataFrame()

    #     # --------------------------------------------------
    #     # CONCAT + RENAME (CRITICAL ORDER)
    #     # --------------------------------------------------
    #     df = pd.concat(results, ignore_index=True)

    #     # 🔑 RENAME FIRST — BEFORE SCHEMA ENFORCEMENT
    #     df = df.rename(columns=rename_map)

    #     # --------------------------------------------------
    #     # SEASON + SITUATION LOGIC
    #     # --------------------------------------------------
    #     df["season_id"] = season_id

    #     if "situationCode" in df.columns:
    #         sc = df["situationCode"].astype("string")

    #         df["situation"] = sc.map(situation_dictionary)

    #         df["goalie_situation"] = np.where(
    #             sc.str.startswith("0") | ((sc.str.len() >= 4) & (sc.str[3] == "0")),
    #             "pulled",
    #             "in net",
    #         )
    #     else:
    #         df["situation"] = pd.NA
    #         df["goalie_situation"] = pd.NA

    #     # --------------------------------------------------
    #     # TIME IN SECONDS
    #     # --------------------------------------------------
    #     df["game_in_seconds"] = df["timeInPeriod"].apply(
    #         lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1])
    #         if isinstance(x, str) and ":" in x else np.nan
    #     )

    #     return df


    #     for i, game_id in enumerate(
    #         tqdm(all_game_ids, total=len(all_game_ids)),
    #         start=1
    #     ):
    #         tqdm.write(f"🎯 Processing game {i}/{len(all_game_ids)} — game_id={game_id}")

    #         try:
    #             result = process_game(game_id)
    #             if not result.empty:
    #                 results.append(result)

    #         except KeyboardInterrupt:
    #             print("⚠️ Keyboard interrupt received — stopping PBP fetch gracefully")
    #             return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    #     if not results:
    #         return pd.DataFrame()

    #     df = pd.concat(results, ignore_index=True)
    #     df = df.rename(columns=rename_map)
    #     df["season_id"] = season_id
    #     if "situationCode" in df.columns:
    #         sc = df["situationCode"].astype("string")
    #         df["situation"] = sc.map(situation_dictionary)

    #         df["goalie_situation"] = np.where(
    #             sc.str.startswith("0") | ((sc.str.len() >= 4) & (sc.str[3] == "0")),
    #             "pulled",
    #             "in net",
    #         )
    #     else:
    #         df["situation"] = pd.NA
    #         df["goalie_situation"] = pd.NA


    #     df["game_in_seconds"] = df["timeInPeriod"].apply(
    #         lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1])
    #         if isinstance(x, str) and ":" in x else np.nan
    #     )

    #     return df

    # ============================================================
    # PLAY-BY-PLAY FETCH (RAW)
    # ============================================================

    def play_by_play_events(game_summary: pd.DataFrame) -> pd.DataFrame:
        all_game_ids = game_summary["game_id"].unique().tolist()
        results = []

        def process_game(game_id: int) -> pd.DataFrame:
            match = game_summary.loc[game_summary["game_id"] == game_id, "link"]
            if match.empty:
                return pd.DataFrame()

            link = str(match.iloc[0]).strip()
            response = safe_request(link)
            if response is None:
                return pd.DataFrame()

            data = response.json()
            if "plays" not in data:
                return pd.DataFrame()

            df = pd.json_normalize(data["plays"])
            df["game_id"] = game_id
            return df

        for game_id in all_game_ids:
            df = process_game(game_id)
            if not df.empty:
                results.append(df)

        if not results:
            return pd.DataFrame()

        out = pd.concat(results, ignore_index=True)
        out["season_id"] = season_id
        return out


    # ============================================================
    # PLAY-BY-PLAY NORMALIZATION (ALL RENAMES LIVE HERE)
    # ============================================================

    def normalize_play_by_play(df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            # -------- TOP-LEVEL EVENT FIELDS (CRITICAL) --------
            "typeCode": "type_code",


            # -------- PERIOD INFO --------
            "periodDescriptor.number": "period_number",
            "periodDescriptor.periodType": "period_type",
            "periodDescriptor.maxRegulationPeriods": "max_regulation_periods",

            # -------- TEAM / PLAYER --------
            "details.eventOwnerTeamId": "team_id",
            "details.losingPlayerId": "losing_player_id",
            "details.winningPlayerId": "winning_player_id",

            # -------- LOCATION --------
            "details.xCoord": "x_coord",
            "details.yCoord": "y_coord",
            "details.zoneCode": "zone_code",

            # -------- SHOOTING / SCORING --------
            "details.shotType": "shot_type",
            "details.shootingPlayerId": "shooting_player",
            "details.goalieInNetId": "goalie",
            "details.scoringPlayerId": "scoring_player",
            "details.scoringPlayerTotal": "scoring_player_total",
            "details.assist1PlayerId": "assist_1",
            "details.assist1PlayerTotal": "assist1_total",
            "details.assist2PlayerId": "assist_2",
            "details.assist2PlayerTotal": "assist2_total",

            # -------- SCORE --------
            "details.awayScore": "away_score",
            "details.homeScore": "home_score",

            # -------- MISC --------
            "details.blockingPlayerId": "blocking_player",
            "details.playerId": "player_id",
            "details.reason": "reason",
            "details.hittingPlayerId": "hitter",
            "details.hitteePlayerId": "hittee",
            "details.awaySOG": "away_sog",
            "details.homeSOG": "home_sog",
            "details.secondaryReason": "secondary_reason",
            "details.duration": "duration",
            "details.committedByPlayerId": "committed_by",
            "details.drawnByPlayerId": "drawn_by",
            "details.servedByPlayerId": "served_by",
        }

        df = df.rename(columns=rename_map)

        # -------- SITUATION / GOALIE --------
        if "situationCode" in df.columns:
            sc = df["situationCode"].astype("string")

            SITUATION_DICT = {
                "1551": "5 on 5", "1451": "5 on 4", "1541": "5 on 4",
                "0651": "6 on 5", "1560": "6 on 5", "1441": "4 on 4",
                "1331": "3 on 3", "1460": "6 on 4", "1351": "5 on 3",
                "0641": "6 on 4", "1341": "4 on 3"
            }

            df["situation"] = sc.map(SITUATION_DICT)

            df["goalie_situation"] = np.where(
                sc.str.startswith("0") | ((sc.str.len() >= 4) & (sc.str[3] == "0")),
                "pulled",
                "in net",
            )
        else:
            df["situation"] = pd.NA
            df["goalie_situation"] = pd.NA

        # -------- TIME IN SECONDS --------
        df["game_in_seconds"] = df["timeInPeriod"].apply(
            lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1])
            if isinstance(x, str) and ":" in x else np.nan
        )

        # -------- HARD GUARD (NO SILENT FAILURES) --------
        if "typeDescKey" not in df.columns:
            raise RuntimeError("Normalization failed: typeDescKey missing")

        return df


    # ============================================================
    # INGESTION / BACKFILL EXECUTION
    # ============================================================

    missing_ids = load_missing_pbp_ids(season_id)

    changed_ids = []

    if changed_games is not None and not changed_games.empty:
        changed_ids = (
            changed_games["game_id"]
            .dropna()
            .astype("int64")
            .unique()
            .tolist()
        )

    game_ids = sorted(set(changed_ids) | set(missing_ids))

    if not game_ids:
        print("✅ PBP and game summary are in sync.")
        return

    summary_df = load_game_summary_df(season_id)

    backfill_df = summary_df[
        summary_df["game_id"].isin(game_ids)
    ].copy()

    if backfill_df.empty:
        return

    print(f"Fetching play-by-play for {len(backfill_df):,} game(s).")

    # ---- RAW FETCH ----
    pbp_updates = play_by_play_events(backfill_df)
    if pbp_updates.empty:
        return

    # ---- NORMALIZE (CRITICAL STEP YOU WERE MISSING) ----
    pbp_updates = normalize_play_by_play(pbp_updates)

    # ---- SCHEMA ENFORCEMENT LAST ----
    pbp_updates = enforce_pbp_schema(pbp_updates)

    # ---- UPSERT ----
    con.register("pbp_updates", pbp_updates)

    con.execute("""
        DELETE FROM all_play_by_play
        WHERE game_id IN (SELECT DISTINCT game_id FROM pbp_updates)
    """)

    con.execute("""
        INSERT INTO all_play_by_play
        SELECT * FROM pbp_updates
    """)
    con.execute("""UPDATE all_play_by_play AS pbp
        SET 
            home_team = gs.home_team,
            away_team = gs.away_team
        FROM all_game_summary AS gs
        WHERE 
            CAST(pbp.game_id AS VARCHAR) = CAST(gs.game_id AS VARCHAR)
            AND CAST(pbp.season_id AS VARCHAR) = CAST(gs.season_id AS VARCHAR);
        """)

    print(f"Inserted {len(pbp_updates):,} play-by-play rows.")
