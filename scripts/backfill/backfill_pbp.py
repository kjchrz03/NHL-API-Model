# ============================================================
# Standalone Play by Play Backfill (Set-Based)
# ============================================================

import requests
import numpy as np
import pandas as pd
import time
import random
import warnings
import duckdb

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
DB_PATH = r"C:\Users\kjchr\Desktop\hockey\new_hockey_env\API data\data\duckdb\nhl_data.duckdb"




# ============================================================
# SAFE REQUEST WITH BACKOFF
# ============================================================
def safe_request(url, params=None, max_retries=5):
    delay = 1.0
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
# LOADERS
# ============================================================
def load_game_summary_df(con, season_id: int) -> pd.DataFrame:
    return con.execute(
        """
        SELECT *
        FROM all_game_summary
        WHERE season_id = ?
          AND gameState = 'OFF'
        ORDER BY game_id
        """,
        [season_id],
    ).df()


def load_existing_pbp(con, season_id: int) -> set[int]:
    rows = con.execute(
        """
        SELECT DISTINCT game_id
        FROM all_play_by_play
        WHERE season_id = ?
        """,
        [season_id],
    ).fetchall()
    return {r[0] for r in rows}


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
# PLAY-BY-PLAY FETCH (RAW)
# ============================================================

def play_by_play_events(game_summary: pd.DataFrame, season_id: int) -> pd.DataFrame:

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

    total = len(all_game_ids)

    for i, game_id in enumerate(all_game_ids, start=1):
        print(f"  🎯 [{i}/{total}] Fetching PBP for game_id={game_id}")

        df = process_game(game_id)

        if df.empty:
            print(f"    ⚠️ No PBP returned for game_id={game_id}")
            continue

        results.append(df)

    if i % 25 == 0:
        print(f"  ⏳ Progress checkpoint: {i}/{total} games fetched")



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
        # -------- TOP-LEVEL EVENT FIELDS --------
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

    # =========================================================
    # SITUATION / GOALIE HANDLING (EARLY-SEASON SAFE)
    # =========================================================

    # Default assumption for ALL rows
    df["goalie_situation"] = "in net"
    df["situation"] = pd.NA

    # Only attempt logic if column exists AND has real values
    if "situationCode" in df.columns and df["situationCode"].notna().any():

        sc = df["situationCode"].astype("string")

        SITUATION_DICT = {
            "1551": "5 on 5", "1451": "5 on 4", "1541": "5 on 4",
            "0651": "6 on 5", "1560": "6 on 5", "1441": "4 on 4",
            "1331": "3 on 3", "1460": "6 on 4", "1351": "5 on 3",
            "0641": "6 on 4", "1341": "4 on 3"
        }

        df["situation"] = sc.map(SITUATION_DICT)

        pulled_mask = (
            sc.str.startswith("0", na=False)
            | ((sc.str.len() >= 4) & (sc.str[3] == "0"))
        )

        df.loc[pulled_mask.fillna(False), "goalie_situation"] = "pulled"

    # =========================================================
    # TIME IN SECONDS
    # =========================================================

    df["game_in_seconds"] = df["timeInPeriod"].apply(
        lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1])
        if isinstance(x, str) and ":" in x else pd.NA
    )

    # =========================================================
    # HARD GUARD — THIS SHOULD ALWAYS EXIST
    # =========================================================

    if "typeDescKey" not in df.columns:
        raise RuntimeError("Normalization failed: typeDescKey missing")

    return df


# ============================================================
# MAIN BACKFILL RUNNER
# ============================================================

def run_plays(con: duckdb.DuckDBPyConnection, season_id: int):

    print(f"\n🏒 Backfilling play_by_play for {season_id}")

    summary_df = load_game_summary_df(con, season_id)
    if summary_df.empty:
        print("⚠ No completed games.")
        return
    
    existing_pbp = load_existing_pbp(con, season_id)

    backfill_df = summary_df[
        ~summary_df["game_id"].isin(existing_pbp)
]

    if backfill_df.empty:
        return

    # ---- RAW FETCH ----
    pbp_updates = play_by_play_events(backfill_df, season_id)

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

    print(f"Inserted {len(pbp_updates):,} play-by-play rows.")


# ============================================================
# EXECUTION
# ============================================================
if __name__ == "__main__":

    con = duckdb.connect(DB_PATH)

    # for season_id in range(20232024+1):
    #     run_plays(con, season_id)
    season_ids = [20252026]

    for season_id in season_ids:
        run_plays(con, season_id)

    con.close()
    print("\n🏁 Historical play by plays backfill complete.")
