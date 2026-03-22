# ============================================================
# Standalone Special Teams Backfill (Set-Based)
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

DB_PATH = r"C:\Users\kjchr\Desktop\hockey\new_hockey_env\API data\duckdb\nhl_data.duckdb"

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
        SELECT game_id, game_date
        FROM all_game_summary
        WHERE season_id = ?
          AND gameState = 'OFF'
        ORDER BY game_id
        """,
        [season_id],
    ).df()


def load_existing_pk_game_ids(con, season_id: int) -> set[int]:
    rows = con.execute(
        """
        SELECT DISTINCT game_id
        FROM all_pk
        WHERE season_id = ?
        """,
        [season_id],
    ).fetchall()
    return {r[0] for r in rows}


def load_existing_pp_game_ids(con, season_id: int) -> set[int]:
    rows = con.execute(
        """
        SELECT DISTINCT game_id
        FROM all_pp
        WHERE season_id = ?
        """,
        [season_id],
    ).fetchall()
    return {r[0] for r in rows}


# ============================================================
# PK / PP API DOWNLOADERS (DATE RANGE)
# ============================================================
def get_pk_by_game(start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://api.nhle.com/stats/rest/en/team/penaltykill"
    limit = 100

    cayenne = (
        f'gameDate>="{start_date} 00:00:00" '
        f'and gameDate<="{end_date} 23:59:59" '
        f'and gameTypeId=2'
    )

    params = {
        "isAggregate": "false",
        "isGame": "true",
        "sort": '[{"property":"teamId","direction":"ASC"}]',
        "cayenneExp": cayenne,
        "start": 0,
        "limit": limit,
    }

    resp = safe_request(url, params)
    if not resp:
        return pd.DataFrame()

    payload = resp.json()
    total = payload.get("total", 0)

    rows = []
    start = 0

    while start < total:
        params["start"] = start
        resp = safe_request(url, params)
        if not resp:
            break

        page = resp.json().get("data", [])
        if not page:
            break

        rows.extend(page)
        start += limit
        time.sleep(0.75 + random.uniform(0.2, 0.4))

    return pd.DataFrame(rows)


def get_pp_by_game(start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://api.nhle.com/stats/rest/en/team/powerplay"
    limit = 100

    cayenne = (
        f'gameDate>="{start_date} 00:00:00" '
        f'and gameDate<="{end_date} 23:59:59" '
        f'and gameTypeId=2'
    )

    params = {
        "isAggregate": "false",
        "isGame": "true",
        "sort": '[{"property":"teamId","direction":"ASC"}]',
        "cayenneExp": cayenne,
        "start": 0,
        "limit": limit,
    }

    resp = safe_request(url, params)
    if not resp:
        return pd.DataFrame()

    payload = resp.json()
    total = payload.get("total", 0)

    rows = []
    start = 0

    while start < total:
        params["start"] = start
        resp = safe_request(url, params)
        if not resp:
            break

        page = resp.json().get("data", [])
        if not page:
            break

        rows.extend(page)
        start += limit
        time.sleep(0.75 + random.uniform(0.2, 0.4))

    return pd.DataFrame(rows)


# ============================================================
# SCHEMAS
# ============================================================
PK_SCHEMA = {
    "season_id": "Int64",
    "game_id": "Int64",
    "team_id": "Int64",
    "game_date": "DATE",
    "games_played": "Int64",
    "home_road": "string",
    "losses": "Int64",
    "opponent": "string",
    "ot_losses": "Int64",
    "pk_net_pct": "float64",
    "pk_pct": "float64",
    "pk_net_g": "Int64",
    "pk_net_gpg": "float64",
    "pk_toi_pg": "float64",
    "points": "Int64",
    "points_pct": "float64",
    "pp_ga": "Int64",
    "pp_ga_pg": "float64",
    "sh_gf": "Int64",
    "sh_gf_pg": "float64",
    "team_name": "string",
    "ties": "string",
    "times_sh": "Int64",
    "times_sh_pg": "float64",
    "wins": "Int64",
}


PP_SCHEMA = {
    "season_id": "Int64",
    "game_id": "Int64",
    "team_id": "Int64",
    "game_date": "DATE",
    "games_played": "Int64",
    "home_road": "string",
    "losses": "Int64",
    "opponent": "string",
    "ot_losses": "Int64",
    "point_pct": "float64",
    "points": "Int64",
    "pp_gf": "Int64",
    "pp_net_pct": "float64",
    "pp_pct": "float64",
    "pp_gpg": "float64",
    "pp_net_g": "Int64",
    "pp_net_gpg": "float64",
    "pp_opportunities": "Int64",
    "pp_opp_pg": "float64",
    "pp_toi": "float64",
    "sh_ga": "Int64",
    "sh_ga_pg": "float64",
    "team_name": "string",
    "ties": "string",
    "wins": "Int64",
}


def enforce_schema(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    df = df.copy()

    for col in schema:
        if col not in df.columns:
            df[col] = pd.NA

    for col, dtype in schema.items():
        if dtype == "string":
            df[col] = df[col].astype("string")
        elif dtype == "float64":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif dtype == "DATE":
            df[col] = pd.to_datetime(df[col]).dt.date
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df[list(schema.keys())]


# ============================================================
# MAIN BACKFILL RUNNER
# ============================================================
def run_special_teams(con: duckdb.DuckDBPyConnection, season_id: int):

    print(f"\n🏒 Backfilling special teams for {season_id}")

    summary_df = load_game_summary_df(con, season_id)
    if summary_df.empty:
        print("⚠ No completed games.")
        return

    existing_pk = load_existing_pk_game_ids(con, season_id)
    existing_pp = load_existing_pp_game_ids(con, season_id)

    missing = summary_df[
        ~summary_df["game_id"].isin(existing_pk)
        | ~summary_df["game_id"].isin(existing_pp)
    ]

    if missing.empty:
        print("✅ No missing special teams games.")
        return

    start_date = pd.to_datetime(missing["game_date"].min()).strftime("%Y-%m-%d")
    end_date = pd.to_datetime(missing["game_date"].max()).strftime("%Y-%m-%d")

    print(f"📅 Fetching {start_date} → {end_date}")

    pk_df = get_pk_by_game(start_date, end_date)
    pp_df = get_pp_by_game(start_date, end_date)

    if pk_df.empty and pp_df.empty:
        print("⚠ No data returned from API.")
        return

    # -----------------------------
    # Normalize PK
    # -----------------------------
    if not pk_df.empty:
        pk_df = pk_df.rename(columns={
            "season": "season_id",
            "gameId": "game_id",
            "teamId": "team_id",
            "gameDate": "game_date",
            "teamFullName": "team_name",
            "gamesPlayed": "games_played",
            "homeRoad": "home_road",
            "opponentTeamAbbrev": "opponent",
            "otLosses": "ot_losses",
            "penaltyKillNetPct": "pk_net_pct",
            "penaltyKillPct": "pk_pct",
            "pkNetGoals": "pk_net_g",
            "pkNetGoalsPerGame": "pk_net_gpg",
            "pkTimeOnIcePerGame": "pk_toi_pg",
            "pointsPct": "points_pct",
            "ppGoalsAgainst": "pp_ga",
            "ppGoalsAgainstPerGame": "pp_ga_pg",
            "shGoalsFor": "sh_gf",
            "shGoalsForPerGame": "sh_gf_pg",
            "timesShorthanded": "times_sh",
            "timesShorthandedPerGame": "times_sh_pg",
        })

        pk_df["season_id"] = season_id
        pk_df = enforce_schema(pk_df, PK_SCHEMA)

    # -----------------------------
    # Normalize PP
    # -----------------------------
    if not pp_df.empty:
        pp_df = pp_df.rename(columns={
            "season": "season_id",
            "gameId": "game_id",
            "teamId": "team_id",
            "gameDate": "game_date",
            "teamFullName": "team_name",
            "gamesPlayed": "games_played",
            "homeRoad": "home_road",
            "opponentTeamAbbrev": "opponent",
            "otLosses": "ot_losses",
            "pointPct": "point_pct",
            "powerPlayGoalsFor": "pp_gf",
            "powerPlayNetPct": "pp_net_pct",
            "powerPlayPct": "pp_pct",
            "ppGoalsPerGame": "pp_gpg",
            "ppNetGoals": "pp_net_g",
            "ppNetGoalsPerGame": "pp_net_gpg",
            "ppOpportunities": "pp_opportunities",
            "ppOpportunitiesPerGame": "pp_opp_pg",
            "ppTimeOnIcePerGame": "pp_toi",
            "shGoalsAgainst": "sh_ga",
            "shGoalsAgainstPerGame": "sh_ga_pg",
        })

        pp_df["season_id"] = season_id
        pp_df = enforce_schema(pp_df, PP_SCHEMA)

    # -----------------------------
    # INSERT PK
    # -----------------------------
    if not pk_df.empty:
        con.register("pk_updates", pk_df)
        con.execute("""
            DELETE FROM all_pk
            WHERE game_id IN (SELECT DISTINCT game_id FROM pk_updates)
        """)
        con.execute("INSERT INTO all_pk SELECT * FROM pk_updates")
        print(f"✔ Inserted {len(pk_df):,} PK rows")

    # -----------------------------
    # INSERT PP
    # -----------------------------
    if not pp_df.empty:
        con.register("pp_updates", pp_df)
        con.execute("""
            DELETE FROM all_pp
            WHERE game_id IN (SELECT DISTINCT game_id FROM pp_updates)
        """)
        con.execute("INSERT INTO all_pp SELECT * FROM pp_updates")
        print(f"✔ Inserted {len(pp_df):,} PP rows")


# ============================================================
# EXECUTION
# ============================================================
if __name__ == "__main__":

    con = duckdb.connect(DB_PATH)

    for season_id in range(20122013, 20122013 + 1):
        run_special_teams(con, season_id)

    con.close()
    print("\n🏁 Historical special teams backfill complete.")
