import requests
import numpy as np
import pandas as pd
import time
import random
import duckdb


# ============================================================
# Goalie Stats Runner
# ============================================================
def run_goalie_stats(
    *,
    con: duckdb.DuckDBPyConnection,
    season_id: int,
):
    """
    Update goalie stats for a given season.
    Assumes game summary has already been run.
    """

    print(f"▶ Running goalie stats for season {season_id}")

    # ============================================================
    # Helpers
    # ============================================================

    def load_last_game_summary_id(season_id: int) -> int | None:
        result = con.execute(
            """
            SELECT MAX(game_id)
            FROM all_game_summary
            WHERE season_id = ?
              AND gameState = 'OFF'
            """,
            [season_id],
        ).fetchone()
        return result[0] if result else None

    def load_last_goalie_game_id(season_id: int) -> int | None:
        result = con.execute(
            """
            SELECT MAX(game_id)
            FROM all_goalie_games
            WHERE season_id = ?
            """,
            [season_id],
        ).fetchone()
        return result[0] if result else None

    def load_game_summary_df(season_id: int) -> pd.DataFrame:
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
    # SCHEMA
    # ============================================================
    goalie_schema = {
        "game_id": "Int64",
        "games_played": "Int64",
        "games_started": "Int64",
        "goalie_name": "string",
        "player_id": "Int64",
        "tri_code": "string",
        "toi": "Int64",
        "wins": "Int64",
        "losses": "Int64",
        "ot_losses": "Int64",
        "gaa": "float64",
        "saves": "Int64",
        "shutouts": "Int64",
        "save_pct": "float64",
        "shots_against": "Int64",
        "goals_against": "Int64",
        "season_id": "Int64",
        "team_id": "Int64",
    }


    def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in goalie_schema:
            if col not in df.columns:
                df[col] = pd.NA
        for col, dtype in goalie_schema.items():
            if dtype == "string":
                df[col] = df[col].astype("string")
            elif dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        return df[list(goalie_schema.keys())]

    # ============================================================
    # DOWNLOAD GOALIE GAMES
    # ============================================================
    def get_goalie_games(start_date: str, end_date: str) -> pd.DataFrame:
        url = "https://api.nhle.com/stats/rest/en/goalie/summary"
        limit = 100

        cayenne = (
            f'gameDate>="{start_date} 00:00:00" '
            f'and gameDate<="{end_date} 23:59:59"'
        )

        params = {
            "isAggregate": "false",
            "isGame": "true",
            "sort": '[{"property":"gameId","direction":"ASC"}]',
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
            time.sleep(0.75 + random.uniform(0.1, 0.3))

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = df.rename(
            columns={
            'gameId': 'game_id',
            'gamesPlayed':'games_played', 
            'gamesStarted': 'games_started',
            'goalieFullName':'goalie_name', 
            'playerId':'player_id',
            'teamAbbrev':'tri_code', 
            'timeOnIce':'toi', 
            'otLosses':'ot_losses',
            'goalsAgainstAverage':'gaa', 
            'savePct':'save_pct', 
            'shotsAgainst':'shots_against',
            'goalsAgainst':'goals_against', 
            'season':'season_id'
            }
        )

        df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")

        # Regular season only (02)
        df = df[df["game_id"].astype(str).str[4:6] == "02"]

        return df

    # ============================================================
    # PROGRESS CHECK
    # ============================================================
    goalie_max = load_last_goalie_game_id(season_id)
    summary_max = load_last_game_summary_id(season_id)

    if goalie_max == summary_max:
        print("✅ Goalie stats up to date.")
        return

    summary_df = load_game_summary_df(season_id)

    missing = summary_df[
        summary_df["game_id"] > (goalie_max or 0)
    ]

    if missing.empty:
        return

    start_date = missing["game_date"].min()
    end_date = missing["game_date"].max()

    goalie_updates = get_goalie_games(start_date, end_date)
    if goalie_updates.empty:
        return

    goalie_updates = enforce_schema(goalie_updates)

    con.register("goalie_updates", goalie_updates)

    con.execute("""
        DELETE FROM all_goalie_games
        WHERE game_id IN (SELECT DISTINCT game_id FROM goalie_updates)
    """)

    con.execute("""
        INSERT INTO all_goalie_games
        SELECT * FROM goalie_updates
    """)
    con.execute("""
    UPDATE all_goalie_games g
    SET team_id = t.team_id
    FROM all_teams t
    WHERE g.tri_code = t.tri_code
      AND g.tri_code NOT IN ('UTA')
      AND (g.team_id IS NULL OR g.team_id <> t.team_id);
    """)
    con.execute("""
    UPDATE all_goalie_games
    SET team_id =
        CASE
            WHEN season_id = 20242025 THEN 59
            WHEN season_id >= 20252026 THEN 68
            ELSE team_id
        END
    WHERE tri_code = 'UTA';
    """)
    print(f"✅ Inserted {len(goalie_updates):,} goalie rows.")
