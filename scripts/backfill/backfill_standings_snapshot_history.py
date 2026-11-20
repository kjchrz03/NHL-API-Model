# ============================================================
# Standalone Standings Snapshots Backfill
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
# Safe request helper
# ============================================================

def safe_get(url, retries=5):
    wait = 1.0
    for _ in range(retries):
        r = requests.get(url)
        if r.status_code == 200:
            time.sleep(0.15 + random.uniform(0, 0.15))
            return r
        if r.status_code == 429:
            print(f"[429] Rate limited — sleeping {wait:.1f}s")
            time.sleep(wait)
            wait *= 2
            continue
        print(f"[{r.status_code}] Error fetching {url}")
        return None
    print("❌ Max retries reached:", url)
    return None
def get_regular_season_games(season_id: int) -> int:

    # COVID-shortened season
    if season_id == 20202021:
        return 56

    # 84-game schedule beginning in 2026-27
    if season_id >= 20262027:
        return 84

    # Standard NHL schedule
    return 82

# ============================================================
# Schema
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

CLINCH_SCHEMA = {
    "clinch_indicator": "boolean",
    "tri_code": "string",
    "clinch_type": "string",   # 'x' or 'e'
    "clinch_date": "DATE",
    "season_id": "Int64",
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

def enforce_clinch_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Keep only known columns
    df = df[[c for c in CLINCH_SCHEMA if c in df.columns]]

    # Add missing columns
    for col in CLINCH_SCHEMA:
        if col not in df.columns:
            df[col] = pd.NA

    # Enforce types
    for col, dtype in CLINCH_SCHEMA.items():
        if dtype == "string":
            df[col] = df[col].astype("string")

        elif dtype == "boolean":
            df[col] = df[col].astype("boolean")

        elif dtype == "DATE":
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

        else:  # Int64
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df[list(CLINCH_SCHEMA.keys())]


# ============================================================
# Standings snapshot builder (FULL SEASON)
# ============================================================

def build_season_standings(con, season_id: int):
    clinch_rows = []
    season_games = get_regular_season_games(season_id)

    print(f"🏒 Regular-season length: {season_games} games")
    print(f"\n🏒 Backfilling standings for {season_id}")

    # --------------------------------------------------------
    # Determine season date range from important_dates
    # --------------------------------------------------------

    query = """
        SELECT
            regular_start,
            regular_end
        FROM important_dates
        WHERE season_id = ?
    """

    dates = con.execute(query, [season_id]).fetchone()

    if dates is None:
        print(f"⚠ No important_dates record for {season_id} — skipping")
        return

    start_date, end_date = dates

    if start_date is None or end_date is None:
        print(
            f"⚠ Missing regular_start or regular_end "
            f"for {season_id} — skipping"
        )
        return

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    print(f"📅 {start_date.date()} → {end_date.date()}")

    base_url = "https://api-web.nhle.com/v1/standings/"
    all_rows = []
    last_valid_day = None

    for dt in pd.date_range(start_date, end_date, freq="D"):
        date_str = dt.strftime("%Y-%m-%d")
        r = safe_get(f"{base_url}{date_str}")

        if r is None:
            continue

        data = r.json().get("standings", [])

        if not data:
            if last_valid_day is None:
                continue
            for row in last_valid_day:
                row_copy = row.copy()
                row_copy["game_date"] = date_str
                all_rows.append(row_copy)
            continue

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

            clinch_indicator = team.get("clinchIndicator")

            if clinch_indicator:
                print(
                    date_str,
                    team.get("teamAbbrev", {}).get("default"),
                    clinch_indicator
                )

            if clinch_indicator in ("x", "e"):
                clinch_rows.append({
                    "clinch_indicator": True,
                    "tri_code": team.get("teamAbbrev", {}).get("default"),
                    "clinch_type": clinch_indicator,
                    "clinch_date": pd.to_datetime(date_str).date(),
                    "season_id": season_id,
                })


            day_rows.append(row)
            all_rows.append(row)

        last_valid_day = day_rows
        print(f"✔ {date_str}")

    df = pd.DataFrame(all_rows)

    df["games_remaining"] = (
    season_games
    - df["games_played"].fillna(0)
    ).clip(lower=0).astype("Int64")

    df["ppg"] = np.where(
        df["games_played"] > 0,
        df["points"] / df["games_played"],
        0.0,
    )

    df["projected_points"] = (
        df["ppg"] * season_games
    ).round().astype("Int64")

    df = enforce_snap_schema(df)
    # --------------------------------------------------------
    # Insert clinch events (first appearance only)
    # --------------------------------------------------------
    if clinch_rows:
        clinch_df = enforce_clinch_schema(pd.DataFrame(clinch_rows))


        # Keep the earliest date per team (first time indicator appears)
        clinch_df = (
            clinch_df.sort_values(["season_id", "tri_code", "clinch_date"])
                    .drop_duplicates(subset=["season_id", "tri_code"], keep="first")
        )

        con.register("clinch_updates", clinch_df)

        # Idempotent insert: don't insert if already exists for that team+season
        con.execute("""
            INSERT INTO all_clinch_history
            SELECT *
            FROM clinch_updates c
            WHERE NOT EXISTS (
                SELECT 1
                FROM all_clinch_history h
                WHERE h.season_id = c.season_id
                AND h.tri_code = c.tri_code
            )
        """)

        print(f"🏁 Clinch events captured: {len(clinch_df)} for {season_id}")


    # --------------------------------------------------------
    # Replace season in DB
    # --------------------------------------------------------
    
    con.execute("DELETE FROM all_standings_daily WHERE season_id = ?", [season_id])
    con.register("snap_updates", df)

    con.execute("""
        INSERT INTO all_standings_daily
        SELECT * FROM snap_updates
    """)
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

    print(f"✅ Inserted {len(df):,} rows for {season_id}")


# ============================================================
# RUN HISTORICAL BACKFILL
# ============================================================

if __name__ == "__main__":

    con = duckdb.connect(DB_PATH)
    season_ids = [20262027]
    # for season_id in range(20002001, 20042005 + 1):
    for season_id in season_ids:
        build_season_standings(con, season_id)

    con.close()
    print("\n🏁 Historical standings backfill complete.")
