# ============================================================
# pipeline.py
# ============================================================
import sys
from pathlib import Path



# --------------------------------------------------
# Ensure project root is on PYTHONPATH FIRST
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent  # goes from scripts → API data

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --------------------------------------------------
# Imports
# --------------------------------------------------
import duckdb
import argparse
from datetime import date, datetime

from scripts.game_summary import run_game_summary
from scripts.play_by_play import run_play_by_play
from scripts.goalie_stats import run_goalie_stats
from scripts.standings_snaps import run_standings_snaps
from scripts.special_teams import run_special_teams
from scripts.prediction_data import run_prediction_data
from prediction_model.daily_playoff_probability import run_probability_scores
from prediction_model.daily_feature_predictions import run_prediction_features
from prediction_model.inference import run_inference
from prediction_model.compute_shap import compute_shap_values
# --------------------------------------------------
# Config
# --------------------------------------------------
DB_PATH = r"C:\Users\kjchr\Desktop\hockey\new_hockey_env\API data\duckdb\nhl_data.duckdb"


# ============================================================
# Utility: Safe Step Runner
# ============================================================
def run_step(name, fn, *, continue_on_error=False, **kwargs):
    print(f"\n▶ Running {name}")
    try:
        fn(**kwargs)
        print(f"✅ {name} complete")
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        if not continue_on_error:
            raise


# ============================================================
# Season Context Loader
# ============================================================
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


# ============================================================
# Main Pipeline
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    # --- core args ---
    parser.add_argument("--season", type=int, help="Override season_id")

    # --- skip flags ---
    parser.add_argument("--skip-game-summary", action="store_true")
    parser.add_argument("--skip-pbp", action="store_true")
    parser.add_argument("--skip-goalie", action="store_true")
    parser.add_argument("--skip-special-teams", action="store_true")
    parser.add_argument("--skip-standings", action="store_true")
    parser.add_argument("--skip-prediction-data", action="store_true")
    parser.add_argument("--skip-probabilities", action="store_true")
    parser.add_argument("--skip-features", action="store_true")

    # --- behavior ---
    parser.add_argument("--continue-on-error", action="store_true")

    args = parser.parse_args()

    print("==================================================")
    print(f"🚀 PIPELINE START")
    print(f"🕒 Timestamp: {datetime.now()}")
    print("==================================================")

    con = duckdb.connect(DB_PATH)

    try:
        season = load_season_context(con, args.season)

        print(f"📅 Using season: {season['season_id']}")
        print(f"📆 Today: {date.today()}")

        # ============================================================
        # Pipeline Steps
        # ============================================================

        if not args.skip_game_summary:
            run_step(
                "Game Summary",
                run_game_summary,
                continue_on_error=args.continue_on_error,
                season_id=season["season_id"],
                season_start=season["regular_start"],
                season_end=season["regular_end"],
                con=con
            )

        if not args.skip_pbp:
            run_step(
                "Play-by-Play",
                run_play_by_play,
                continue_on_error=args.continue_on_error,
                season_id=season["season_id"],
                con=con
            )

        if not args.skip_goalie:
            run_step(
                "Goalie Stats",
                run_goalie_stats,
                continue_on_error=args.continue_on_error,
                season_id=season["season_id"],
                con=con
            )

        if not args.skip_special_teams:
            run_step(
                "Special Teams",
                run_special_teams,
                continue_on_error=args.continue_on_error,
                season_id=season["season_id"],
                con=con
            )

        if not args.skip_standings:
            run_step(
                "Standings Snapshots",
                run_standings_snaps,
                continue_on_error=args.continue_on_error,
                season_id=season["season_id"],
                con=con
            )

        if not args.skip_prediction_data:
            run_step(
                "Prediction Data",
                run_prediction_data,
                continue_on_error=args.continue_on_error,
                season_id=season["season_id"],
                con=con
            )

        if not args.skip_probabilities:
            run_step(
                "Playoff Probabilities",
                run_probability_scores,
                continue_on_error=args.continue_on_error,
                season_id=season["season_id"],
                con=con
            )

        if not args.skip_features:
            run_step(
                "Prediction Features",
                run_prediction_features,
                continue_on_error=args.continue_on_error,
                season_id=season["season_id"],
                con=con
            )

        print("\n==================================================")
        print("✅ PIPELINE FINISHED SUCCESSFULLY")
        print("==================================================")

    finally:
        con.close()
        print("🔒 DuckDB connection closed")


# ============================================================
# Entrypoint
# ============================================================
if __name__ == "__main__":
    main()