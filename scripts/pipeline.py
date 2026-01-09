# pipeline.py
import sys
from pathlib import Path

# --------------------------------------------------
# Ensure project root is on PYTHONPATH FIRST
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --------------------------------------------------
# Now imports from scripts will work
# --------------------------------------------------
import duckdb
import argparse
from datetime import date

from scripts.game_summary import run_game_summary
from scripts.play_by_play import run_play_by_play
from scripts.goalie_stats import run_goalie_stats
from scripts.standings_snaps import run_standings_snaps
from scripts.special_teams import run_special_teams
from scripts.prediction_data import run_prediction_data


DB_PATH = r"C:\Users\kjchr\Desktop\hockey\new_hockey_env\API data\duckdb\nhl_data.duckdb"

def load_season_context(con, season_override: int | None):
    if season_override:
        query = """
            SELECT *
            FROM important_dates
            WHERE season_id = ?
        """
        row = con.execute(query, [season_override]).fetchone()
        if not row:
            raise ValueError(f"Season {season_override} not found")
    else:
        query = """
            SELECT *
            FROM important_dates
            WHERE current_season = TRUE
        """
        row = con.execute(query).fetchone()
        if not row:
            raise RuntimeError("No current season set")

    cols = [d[0] for d in con.execute("DESCRIBE important_dates").fetchall()]
    return dict(zip(cols, row))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, help="Override season_id")
    args = parser.parse_args()

    con = duckdb.connect(DB_PATH)

    try:
        season = load_season_context(con, args.season)

        print(f"📅 Using season: {season['season_id']}")
        print(f"📆 Today: {date.today()}")

        # 1️⃣ Game Summary
        run_game_summary(
            season_id=season["season_id"],
            season_start=season["regular_start"],
            season_end=season["regular_end"],
            con=con
        )

        # 2️⃣ Play-by-play
        run_play_by_play(
            season_id=season["season_id"],
            con=con

        )
        
        # 3️⃣ Goalie Stats
        run_goalie_stats(
            season_id=season["season_id"],
            con=con
        )

        # 4️⃣ Special Teams
        run_special_teams(
            season_id=season["season_id"],
            con=con
        )
        
        # 5️⃣ Daily Standings Snap Shots
        run_standings_snaps(
            season_id=season["season_id"],
            con=con
        )
        
        # 6️⃣ Update the prediction data table
        run_prediction_data(
            season_id=season["season_id"],
            con=con            
        )



        print("✅ Pipeline finished")

    finally:
        con.close()

if __name__ == "__main__":
    main()
