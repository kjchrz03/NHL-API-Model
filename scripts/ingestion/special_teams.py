import requests
import json
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta
import concurrent.futures
import base64
import os
import time
import random
import duckdb
from typing import Any
from config import DB_PATH

# ============================================================
# Play-by-Play Runner
# ============================================================
def run_special_teams(
    *,
    con: duckdb.DuckDBPyConnection,
    season_id: int,
):
    """
    Update PP / PK by-game data for a given season.
    Assumes game summary has already been run.
    """

    print(f"▶ Running special teams for season {season_id}")

    # ============================================================
    # Date helpers
    # ============================================================
    def load_last_date(table: str) -> pd.Timestamp | None:
        query = f"""
            SELECT MAX(game_date)
            FROM {table}
            WHERE season_id = ?
        """
        result = con.execute(query, [season_id]).fetchone()
        max_date = result[0]

        if max_date is None:
            return None

        return pd.to_datetime(max_date) + pd.Timedelta(days=1)

    def load_last_game_summary_date() -> pd.Timestamp | None:
        query = """
            SELECT MAX(game_date)
            FROM all_game_summary
            WHERE season_id = ?
              AND gameState = 'OFF'
        """
        result = con.execute(query, [season_id]).fetchone()
        return pd.to_datetime(result[0]) if result and result[0] else None

    # ============================================================
    # SAFE REQUEST
    # ============================================================
    def safe_request(url, params, max_retries=5):
        backoff = 1.0
        for _ in range(max_retries):
            r = requests.get(url, params=params)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                time.sleep(backoff)
                backoff *= 2
                continue
            time.sleep(backoff)
            backoff *= 2
        raise RuntimeError(f"Failed request: {url}")
    # ============================================================
    # PBP SCHEMA (DuckDB-aligned)
    # ============================================================

    PK_SCHEMA = {
        
        "season_id": "Int64",
        "game_id": "Int64",
        "team_id": "Int64",
        "game_date": "DATE",
        "home_road": "string",
        "opponent": "string",
        "pk_net_pct": "float64",
        "pk_pct": "float64",
        "pk_net_g": "Int64",
        "pk_toi": "float64",
        "pp_ga": "Int64",
        "sh_gf": "Int64",
        "team_name": "string",
        "times_sh": "Int64",

        }

    def enforce_pk_schema(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df[[c for c in PK_SCHEMA if c in df.columns]]
        for col in PK_SCHEMA:
            if col not in df.columns:
                df[col] = pd.NA
        for col, dtype in PK_SCHEMA.items():
            if dtype == "string":
                df[col] = df[col].astype("string")
            elif dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "DATE":
                df[col] = pd.to_datetime(df[col]).dt.date
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        return df[list(PK_SCHEMA.keys())]
    
    PP_SCHEMA = {
            "season_id": "Int64",
            "game_id": "Int64",
            "team_id": "Int64",
            "game_date": "DATE",
            "home_road": "string",
            "opponent": "string",
            "pp_gf": "Int64",
            "pp_net_pct": "float64",
            "pp_pct": "float64",
            "pp_net_g": "Int64",
            "pp_opportunities": "Int64",
            "pp_toi": "float64",
            "sh_ga": "Int64",
            "team_name": "string",

        
    }
    
    def enforce_pp_schema(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df[[c for c in PP_SCHEMA if c in df.columns]]
        for col in PP_SCHEMA:
            if col not in df.columns:
                df[col] = pd.NA
        for col, dtype in PP_SCHEMA.items():
            if dtype == "string":
                df[col] = df[col].astype("string")
            elif dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "DATE":
                df[col] = pd.to_datetime(df[col]).dt.date
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        return df[list(PP_SCHEMA.keys())]

    
# =========================================
# PK SCRAPER — CORRECT PAGINATION (start+=100)
# =========================================
    def get_pk_by_game(pk_start_date, end_date, base_delay=1.0):
        pk_start_str = pd.to_datetime(pk_start_date).strftime("%Y-%m-%d")
        end_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        url = "https://api.nhle.com/stats/rest/en/team/penaltykill"
        limit = 100  # NHL API always returns 100 rows per page

        cayenne = (
            f'gameDate<="{end_str} 23:59:59" '
            f'and gameDate>="{pk_start_str} 00:00:00" '
            f'and gameTypeId=2'
        )

        params = {
            "isAggregate": "false",
            "isGame": "true",
            "sort": '[{"property":"teamFullName","direction":"ASC"},{"property":"teamId","direction":"ASC"}]',
            "start": 0,
            "limit": limit,
            "cayenneExp": cayenne,
        }

        # -----------------------------------------
        # First call → get total row count
        # -----------------------------------------
        resp = safe_request(url, params)
        payload = resp.json()

        if "total" not in payload:
            raise ValueError("API response missing 'total' — cannot paginate.")

        total = payload["total"]
        print(f"  → API reports total rows = {total}")

        all_rows = []
        start = 0

        # -----------------------------------------
        # Loop through all pages
        # -----------------------------------------
        while start < total:

            params["start"] = start
            resp = safe_request(url, params)
            page = resp.json().get("data", [])

            print(f"     Fetching rows {start} to {start+limit} → got {len(page)} rows")

            if not page:
                print("⚠️ Empty page before reaching total — stopping early.")
                break

            all_rows.extend(page)

            start += limit   # increment by 100

            time.sleep(base_delay + random.uniform(0.3, 0.6))

        # Just in case API duplicates rows
        if len(all_rows) > total:
            all_rows = all_rows[:total]

        print(f"  → Downloaded {len(all_rows)} rows (expected {total})")

        return pd.DataFrame(all_rows)

        
        
        
    # =========================================
    # POWER PLAY SCRAPER — FULL PAGINATION
    # =========================================
    def get_pp_by_game(pp_start_date, end_date, base_delay=1.0):
        pp_start_str = pd.to_datetime(pp_start_date).strftime("%Y-%m-%d")
        end_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        
        url = "https://api.nhle.com/stats/rest/en/team/powerplay"
        limit = 100  # NHL API returns 100 rows per page

        cayenne = (
            f'gameDate<="{end_str} 23:59:59" '
            f'and gameDate>="{pp_start_str} 00:00:00" '

            f'and gameTypeId=2'
        )

        params = {
            "isAggregate": "false",
            "isGame": "true",
            "sort": '[{"property":"powerPlayPct","direction":"DESC"},{"property":"teamId","direction":"ASC"}]',
            "start": 0,
            "limit": limit,
            "cayenneExp": cayenne
        }

        # -------- First call to get total rows --------
        resp = safe_request(url, params)
        payload = resp.json()

        if "total" not in payload:
            raise Exception("API missing 'total' field — cannot paginate")

        total = payload["total"]
        print(f"  → API reports total rows = {total}")

        all_rows = []
        start = 0

        # -------- Loop pages until all rows fetched --------
        while start < total:
            params["start"] = start
            resp = safe_request(url, params)
            page = resp.json().get("data", [])

            print(f"     Fetching rows {start} to {start+limit} → got {len(page)} rows")

            if not page:
                print("⚠️ Empty page before reaching total — stopping early.")
                break

            all_rows.extend(page)
            start += limit

            time.sleep(base_delay + random.uniform(0.25, 0.5))

        # Trim duplicates if any extra
        if len(all_rows) > total:
            all_rows = all_rows[:total]

        print(f"  → Downloaded {len(all_rows)} rows (expected {total})")


        return pd.DataFrame(all_rows)

    # ============================================================
    # DATE WINDOW
    # ============================================================
    pk_start = load_last_date("all_pk")
    pp_start = load_last_date("all_pp")
    end_date = load_last_game_summary_date()

    if end_date is None:
        print("⚠ No completed games yet.")
        return

    if pk_start is None:
        pk_start = end_date
    if pp_start is None:
        pp_start = end_date

    if pk_start > end_date and pp_start > end_date:
        print("✅ Special teams already up to date.")
        return

    # ============================================================
    # FETCH DATA
    # ============================================================
    pk_df = get_pk_by_game(pk_start.date(), end_date.date())
    pp_df = get_pp_by_game(pp_start.date(), end_date.date())

    if pk_df.empty and pp_df.empty:
        print("⚠ No special teams updates.")
        return

    # ============================================================
    # CLEAN + NORMALIZE
    # ============================================================
    pp_rename_map = {
        "season": "season_id",
        "gameId": "game_id",
        "teamId": "team_id",
        "gameDate": "game_date",
        "teamFullName": "team_name",
        "homeRoad":"home_road",
        "opponentTeamAbbrev":"opponent",
        "powerPlayGoalsFor":"pp_gf",
        "powerPlayNetPct":"pp_net_pct",
        "powerPlayPct":"pp_pct",
        "ppNetGoals":"pp_net_g",
        "ppOpportunities":"pp_opportunities",
        "ppTimeOnIcePerGame":"pp_toi",
        "shGoalsAgainst":"sh_ga",
    }

    pk_rename_map = {
        "season": "season_id",
        "gameId": "game_id",
        "teamId": "team_id",
        "teamFullName": "team_name",
        "gameDate": "game_date",
        "homeRoad":"home_road",
        "opponentTeamAbbrev":"opponent",
        "penaltyKillNetPct":"pk_net_pct",
        "penaltyKillPct":"pk_pct",
        "pkNetGoals": "pk_net_g",
        "pkTimeOnIcePerGame": "pk_toi",
        "ppGoalsAgainst": "pp_ga",
        "shGoalsFor":"sh_gf",
        "timesShorthanded":"times_sh",

    }

    if not pk_df.empty:
        pk_df = pk_df.rename(columns=pk_rename_map)
        pk_df["season_id"] = season_id
        pk_df["game_date"] = pd.to_datetime(pk_df["game_date"]).dt.date
        pk_df = enforce_pk_schema(pk_df)

    if not pp_df.empty:
        pp_df = pp_df.rename(columns=pp_rename_map)
        pp_df["season_id"] = season_id
        pp_df["game_date"] = pd.to_datetime(pp_df["game_date"]).dt.date
        pp_df = enforce_pp_schema(pp_df)

        

    # ============================================================
    # INSERT PK
    # ============================================================
    if not pk_df.empty:
        
        con.register("pk_updates", pk_df)
        con.execute("""
            DELETE FROM all_pk
            WHERE season_id = ?
              AND game_date IN (SELECT DISTINCT game_date FROM pk_updates)
        """, [season_id])

        con.execute("INSERT INTO all_pk SELECT * FROM pk_updates")
        print(f"✔ Inserted {len(pk_df):,} PK rows")

    # ============================================================
    # INSERT PP
    # ============================================================
    if not pp_df.empty:
        con.register("pp_updates", pp_df)
        con.execute("""
            DELETE FROM all_pp
            WHERE season_id = ?
              AND game_date IN (SELECT DISTINCT game_date FROM pp_updates)
        """, [season_id])

        con.execute("INSERT INTO all_pp SELECT * FROM pp_updates")
        print(f"✔ Inserted {len(pp_df):,} PP rows")