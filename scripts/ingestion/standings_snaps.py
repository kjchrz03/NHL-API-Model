# ============================================================
# Imports
# ============================================================
import requests
import numpy as np
import pandas as pd
import time
import random
import warnings
import duckdb
from typing import Any
from config import DB_PATH

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
# ============================================================
# Play-by-Play Runner
# ============================================================
def run_standings_snaps(
    *,
    con: duckdb.DuckDBPyConnection,
    season_id: int,
):
    """
    Update standings stanpshot data for a given season.
    Assumes game summary has already been run.
    """

    print(f"▶ Running standings snapshots for season {season_id}")

    # ============================================================
    # Helpers
    # ============================================================

    def load_last_snap_date(season_id: int) -> pd.Timestamp | None:
        query = """
            SELECT MAX(game_date)
            FROM all_standings_daily
            WHERE season_id = ?
        """
        result = con.execute(query, [season_id]).fetchone()
        max_date = result[0]

        if max_date is None:
            return None

        max_date = pd.to_datetime(max_date) + pd.Timedelta(days=1)
        print(f"Last snapshot date +1: {max_date.date()}")
        return max_date


    def load_last_game_summary_date(season_id: int) -> pd.Timestamp | None:

        query = """
            SELECT MAX(game_date)
            FROM all_game_summary
            WHERE season_id = ?
              AND gameState = 'OFF'
        """
        result = con.execute(query, [season_id]).fetchone()
        max_date = result[0] if result else None
        print(f"Max completed gs date: {max_date}")
        return max_date
    # =========================================
    # SAFE REQUEST
    # =========================================
    def safe_get(url, retries=5):
        wait = 1.0

        for _ in range(retries):
            r = requests.get(url)

            if r.status_code == 200:
                time.sleep(0.15 + random.uniform(0, 0.15))
                return r

            if r.status_code == 429:
                print(f"[429] Rate limited — waiting {wait:.1f}s…")
                time.sleep(wait)
                wait *= 2
                continue

            print(f"[{r.status_code}] Error fetching {url}")
            return None

        print("❌ Max retries reached:", url)
        return None


    # ============================================================
    # Daily Snaps SCHEMA (DuckDB-aligned)
    # ============================================================

    Snap_SCHEMA = {

        'game_date': "DATE",
        'season_id': "Int64",
        'tri_code': "string",
        'team_name': "string",
        'games_played': "Int64",
        'road_gp': "Int64",
        'division': "string",
        'division_rank': "Int64",
        'division_home_rank': "Int64",
        'division_road_rank': "Int64",
        'division_last_10_rank': "Int64",
        'conference': "string",
        'conference_rank': "Int64",
        'conference_home_rank': "Int64",
        'conference_road_rank': "Int64",
        'conference_last_10_rank': "Int64",
        'league_rank': "Int64",
        'league_home_rank': "Int64",
        'league_road_rank': "Int64",
        'league_last_10_rank': "Int64",
        'wildcard_rank': "Int64",
        'wins': "Int64",
        'wins_pctg': "float64",
        'rw': "Int64",
        'losses': "Int64",
        'ot_losses': "Int64",
        'rotw': "Int64",
        'rotw_pctg': "float64",
        'road_losses': "Int64",
        'road_ot_losses': "Int64",
        'road_rotw': "Int64",
        'road_rw': "Int64",
        'points': "Int64",
        'points_pct': "float64",
        'home_points': "Int64",
        'road_points': "Int64",
        'goals_for': "Int64",
        'gf_pctg': "float64",
        'home_gf': "Int64",
        'road_gf': "Int64",
        'goals_against': "Int64",
        'home_ga': "Int64",
        'road_ga': "Int64",
        'goal_diff': "Int64",
        'goal_diff_pctg': "float64",
        'home_goal_diff': "Int64",
        'road_goal_diff': "Int64",
        'streak': "string",
        'streak_count': "float64",
        'last_10_gp': "Int64",
        'last_10_wins': "Int64",
        'last_10_goal_diff': "Int64",
        'last_10_ga': "Int64",
        'last_10_gf': "Int64",
        'last_10_losses': "Int64",
        'last_10_otl': "Int64",
        'last_10_points': "Int64",
        'last_10_rotw': "Int64",
        'last_10_rw': "Int64",
        'home_gp': "Int64",
        'home_wins': "Int64",
        'home_losses': "Int64",
        'home_otlosses': "Int64",
        'home_rotw': "Int64",	
        'home_rw': "Int64",
        'road_wins': "Int64",
        'so_losses': "Int64",
        'so_wins': "Int64",
        'in_playoff_spot': "Int64",
        'games_remaining': "Int64",
        'ppg': "float64",
        'projected_points': "Int64",
        'team_id': "Int64",
    }

    def enforce_snap_schema(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Keep only known columns first
        df = df[[c for c in Snap_SCHEMA if c in df.columns]]

        # Add missing columns
        for col in Snap_SCHEMA:
            if col not in df.columns:
                df[col] = pd.NA

        # Enforce types
        for col, dtype in Snap_SCHEMA.items():
            if dtype == "string":
                df[col] = df[col].astype("string")

            elif dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce")

            elif dtype == "DATE":
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

            else:  # Int64
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        # Enforce column order
        return df[list(Snap_SCHEMA.keys())]


    # =========================================
    # DAILY SNAPSHOT SCRAPER
    # =========================================
    def get_daily_standings_snapshots(start_date, end_date):
        base_url = "https://api-web.nhle.com/v1/standings/"
        all_rows = []

        # Cache last valid standings day (for no-game days or API gaps)
        last_valid_day = None

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        print("\n==============================")
        print(f"📅 Processing season {season_id}")
        print(f"   From {start_date.date()} → {end_date.date()}")
        print("==============================\n")

        for dt in pd.date_range(start_date, end_date, freq="D"):
            date_str = dt.strftime("%Y-%m-%d")
            url = f"{base_url}{date_str}"

            r = safe_get(url)
            if r is None:
                print(f"⚠ Skipping {date_str} (request failed)")
                continue

            data = r.json().get("standings", [])

            # --------------------------------------------------
            # Fallback: carry forward previous standings
            # --------------------------------------------------
            if not data:
                if last_valid_day is None:
                    print(f"⚠ No standings and no prior data for {date_str}")
                    continue

                print(f"↩ Carrying forward standings for {date_str}")

                for row in last_valid_day:
                    row_copy = row.copy()
                    row_copy["game_date"] = date_str
                    row_copy["date"] = date_str
                    all_rows.append(row_copy)

                continue

            # --------------------------------------------------
            # Normal path: API returned standings
            # --------------------------------------------------
            day_rows = []

            for team in data:
                row = {
                    "game_date": date_str,
                    "season_id": season_id,

                    "tri_code": team.get("teamAbbrev", {}).get("default"),
                    "team_name": team.get("teamName", {}).get("default"),
                    "games_played": team.get("gamesPlayed", 0),
                    "road_gp": team.get("roadGamesPlayed", 0),

                    "division": team.get("divisionName"),
                    "division_rank": team.get("divisionSequence", 0),
                    "division_home_rank": team.get("divisionHomeSequence", 0),
                    "division_road_rank": team.get("divisionRoadSequence", 0),
                    "division_last_10_rank": team.get("divisionL10Sequence", 0),

                    "conference": team.get("conferenceName"),
                    "conference_rank": team.get("conferenceSequence", 0),
                    "conference_home_rank": team.get("conferenceHomeSequence", 0),
                    "conference_road_rank": team.get("conferenceRoadSequence", 0),
                    "conference_last_10_rank": team.get("conferenceL10Sequence", 0),

                    "league_rank": team.get("leagueSequence", 0),
                    "league_home_rank": team.get("leagueHomeSequence", 0),
                    "league_road_rank": team.get("leagueRoadSequence", 0),
                    "league_last_10_rank": team.get("leagueL10Sequence", 0),
                    "wildcard_rank": team.get("wildcardSequence", 0),

                    "wins": team.get("wins", 0),
                    "wins_pctg": team.get("winPctg", 0.0),
                    "rw": team.get("regulationWins", 0),
                    "losses": team.get("losses", 0),
                    "ot_losses": team.get("otLosses", 0),
                    "rotw": team.get("regulationPlusOtWins", 0),
                    "rotw_pctg": team.get("regulationPlusOtWinPctg", 0.0),

                    "road_losses": team.get("roadLosses", 0),
                    "road_ot_losses": team.get("roadOtLosses", 0),
                    "road_rotw": team.get("roadRegulationPlusOtWins", 0),
                    "road_rw": team.get("roadRegulationWins", 0),

                    "points": team.get("points", 0),
                    "points_pct": team.get("pointPctg", 0.0),
                    "home_points": team.get("homePoints", 0),
                    "road_points": team.get("roadPoints", 0),

                    "goals_for": team.get("goalFor", 0),
                    "gf_pctg": team.get("goalsForPctg", 0.0),
                    "home_gf": team.get("homeGoalsFor", 0),
                    "road_gf": team.get("roadGoalsFor", 0),

                    "goals_against": team.get("goalAgainst", 0),
                    "home_ga": team.get("homeGoalsAgainst", 0),
                    "road_ga": team.get("roadGoalsAgainst", 0),

                    "goal_diff": team.get("goalDifferential", 0),
                    "goal_diff_pctg": team.get("goalDifferentialPctg", 0.0),
                    "home_goal_diff": team.get("homeGoalDifferential", 0),
                    "road_goal_diff": team.get("roadGoalDifferential", 0),

                    "streak": team.get("streakCode"),
                    "streak_count": team.get("streakCount"),

                    "last_10_gp": team.get("l10GamesPlayed", 0),
                    "last_10_wins": team.get("l10Wins", 0),
                    "last_10_goal_diff": team.get("l10GoalDifferential", 0),
                    "last_10_ga": team.get("l10GoalsAgainst", 0),
                    "last_10_gf": team.get("l10GoalsFor", 0),
                    "last_10_losses": team.get("l10Losses", 0),
                    "last_10_otl": team.get("l10OtLosses", 0),
                    "last_10_points": team.get("l10Points", 0),
                    "last_10_rotw": team.get("l10RegulationPlusOtWins", 0),
                    "last_10_rw": team.get("l10RegulationWins", 0),

                    "home_gp": team.get("homeGamesPlayed", 0),
                    "home_wins": team.get("homeWins", 0),
                    "home_losses": team.get("homeLosses", 0),
                    "home_otlosses": team.get("homeOtLosses", 0),
                    "home_rotw": team.get("homeRegulationPlusOtWins", 0),
                    "home_rw": team.get("homeRegulationWins", 0),

                    "road_wins": team.get("roadWins", 0),

                    "so_losses": team.get("shootoutLosses", 0),
                    "so_wins": team.get("shootoutWins", 0),

                    "in_playoff_spot": 1
                    if (
                        (team.get("divisionRank") is not None and team.get("divisionRank") <= 3)
                        or (team.get("wildCardRank") is not None and team.get("wildCardRank") <= 2)
                    )
                    else 0,

                    "team_id": team.get("teamId",0),
                }

                day_rows.append(row)
                all_rows.append(row)

            # Cache this day's standings for carry-forward use
            last_valid_day = day_rows

            print(f"✔ Completed standings for {date_str}")

        df = pd.DataFrame(all_rows)
        print(f"\n📦 Total snapshot rows collected: {len(df):,}")
        return df

        
    # ============================================================
    # COMPARE PROGRESS
    # ============================================================
    start_date = load_last_snap_date(season_id)
    end_date = load_last_game_summary_date(season_id)

    # ----------------------------------
    # Guard clauses FIRST
    # ----------------------------------
    if end_date is None:
        print("⚠ No completed games yet — skipping standings.")
        return

    if start_date is None:
        # Bootstrap from first completed game
        start_date = end_date

    # ----------------------------------
    # Normalize types AFTER None handling
    # ----------------------------------
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # ----------------------------------
    # Date window validation
    # ----------------------------------
    if start_date > end_date:
        print("✅ standings snapshots already up to date.")
        return

    # ----------------------------------
    # Build snapshots
    # ----------------------------------
    snap_updates = get_daily_standings_snapshots(start_date, end_date)
    if snap_updates.empty:
        return
    # -------------------------------------------------
    # Derived standings metrics (DAILY, NOT FORWARDED)
    # -------------------------------------------------
    snap_updates["games_remaining"] = 82 - snap_updates["games_played"].fillna(0)

    snap_updates["ppg"] = np.where(
        snap_updates["games_played"] > 0,
        snap_updates["points"] / snap_updates["games_played"],
        0.0
    )

    snap_updates["projected_points"] = (
        snap_updates["ppg"] * 82
    ).round().astype("Int64")

    snap_updates = enforce_snap_schema(snap_updates)

    con.register("snap_updates", snap_updates)

    con.execute("""
        DELETE FROM all_standings_daily
        WHERE game_date IN (SELECT DISTINCT game_date FROM snap_updates)
    """)

    con.execute("""
        INSERT INTO all_standings_daily
        SELECT * FROM snap_updates
    """)
    
    #20132014 last season of PHX --> ARI
    #20252026 UTA is 68, previous version is 2425 season

    con.execute("""UPDATE all_standings_daily s
    SET team_id = t.team_id
    FROM all_teams t
    WHERE s.tri_code = t.tri_code
    AND s.tri_code != 'UTA';
        """)

    con.execute("""UPDATE all_standings_daily
    SET team_id = 
        CASE
            WHEN season_id = 20242025 THEN 59
            WHEN season_id >= 20252026 THEN 68
            ELSE team_id
        END
    WHERE tri_code = 'UTA';
        """)
    con.execute("""SELECT season_id, tri_code, team_id
    FROM all_standings_daily
    WHERE tri_code = 'UTA'
    ORDER BY season_id;
    """)

    print(f"Inserted {len(snap_updates):,} daily standings updated rows.")
