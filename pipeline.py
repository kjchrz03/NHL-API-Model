# ============================================================
# pipeline.py
# ============================================================

# --------------------------------------------------
# Imports
# --------------------------------------------------
import duckdb
import argparse
from datetime import date, datetime

# Ingestion
from scripts.ingestion.game_summary import run_game_summary
from scripts.ingestion.play_by_play import run_play_by_play
from scripts.ingestion.goalie_stats import run_goalie_stats
from scripts.ingestion.standings_snaps import run_standings_snaps
from scripts.ingestion.special_teams import run_special_teams

# Feature engineering
from scripts.features.prediction_data import run_prediction_data

# Model - inference
from prediction_model.inference.daily_playoff_probability import run_probability_scores
from prediction_model.inference import score_dataframe

# Model - explainability
from prediction_model.explainability.daily_feature_predictions import run_prediction_features
from prediction_model.explainability.compute_shap import run_shap_summary

# --------------------------------------------------
# Config
# --------------------------------------------------
from config import DB_PATH


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

        if not args.skip_prediction_data:########################################is the sos still wrong? NA values found
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