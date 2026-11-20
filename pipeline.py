# ============================================================
# pipeline.py
# ============================================================

# --------------------------------------------------
# Imports
# --------------------------------------------------
import os
import sys
import duckdb
import pandas as pd
import argparse
import uuid
from datetime import date, datetime
import time
import random
import requests

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

# Model - explainability
from prediction_model.explainability.daily_feature_predictions import run_prediction_features
from prediction_model.explainability.shap_summary import update_shap_summary


# --------------------------------------------------
# Config
# --------------------------------------------------
from config import DB_PATH, CURRENT_SEASON_ID

# --------------------------------------------------
# PyInstaller / Script Directory
# --------------------------------------------------
if getattr(sys, "frozen", False):
    script_dir = os.path.dirname(sys.executable)
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))

print(f"🔍 Script is running from: {script_dir}")



def run_step(name, fn, *, continue_on_error=False, max_attempts=4, failures=None, **kwargs):
    print(f"\n▶ Running {name}")

    for attempt in range(1, max_attempts + 1):
        try:
            result = fn(**kwargs)
            print(f"✅ {name} complete")
            return result

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            retryable = True
            error = e

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            retryable = status in {429, 500, 502, 503, 504}
            error = e

        except Exception as e:
            retryable = False
            error = e

        if not retryable or attempt == max_attempts:
            print(f"❌ {name} failed: {error}")
            if failures is not None:
                failures.append(f"{name}: {type(error).__name__}: {error}")
            if not continue_on_error:
                raise error
            return None

        delay = min(5 * (2 ** (attempt - 1)), 60) + random.uniform(0, 1)

        print(
            f"⚠️ {name} failed on attempt {attempt}/{max_attempts}: {error}\n"
            f"🔄 Retrying in {delay:.1f} seconds..."
        )

        time.sleep(delay)

# ============================================================
# Season Context Loader
# ============================================================
def load_season_context(con, season_override: int | None = None):
    season_id = season_override if season_override is not None else CURRENT_SEASON_ID

    row = con.execute("""
        SELECT *
        FROM important_dates
        WHERE season_id = ?
    """, [season_id]).fetchone()

    if not row:
        raise ValueError(f"Season {season_id} not found in important_dates")

    cols = [d[0] for d in con.execute("DESCRIBE important_dates").fetchall()]
    return dict(zip(cols, row))

# ============================================================
# Persistent Pipeline Run Logging
# ============================================================
def ensure_run_log(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_run_log (
            run_id UUID PRIMARY KEY,
            season_id BIGINT,
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            status VARCHAR,
            error_message VARCHAR
        )
    """)


def finish_run_log(con, run_id, status, error_message=None):
    con.execute("""
        UPDATE pipeline_run_log
        SET finished_at = CURRENT_TIMESTAMP,
            status = ?,
            error_message = ?
        WHERE run_id = ?
    """, [status, error_message, run_id])


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

    # --------------------------------------------------
    # Database Connection
    # --------------------------------------------------
    from pathlib import Path

    # Actual project directory
    if getattr(sys, "frozen", False):
        project_dir = Path(sys.executable).resolve().parent
    else:
        project_dir = Path(__file__).resolve().parent

    # Database location
    db_path = project_dir / "data" / "duckdb" / "nhl_data.duckdb"

    print(f"📁 Project directory: {project_dir}")
    print(f"🗄️ Database: {db_path}")

    if not db_path.is_file():
        raise FileNotFoundError(
            f"Database not found: {db_path}"
        )

    con = duckdb.connect(str(db_path))
    run_id = str(uuid.uuid4())
    failures = []
    blocked_steps = []
    log_created = False

    try:
        ensure_run_log(con)
        con.execute("""
            INSERT INTO pipeline_run_log (run_id, season_id, started_at, status)
            VALUES (?, ?, CURRENT_TIMESTAMP, 'RUNNING')
        """, [run_id, args.season if args.season is not None else CURRENT_SEASON_ID])
        log_created = True

        season = load_season_context(con, args.season)
        print(f"📅 Using season: {season['season_id']}")
        print(f"📆 Today: {date.today()}")
        print(f"🧾 Pipeline run ID: {run_id}")

        # ============================================================
        # Pipeline Steps
        # ============================================================
        changed_games = None

        if not args.skip_game_summary:
            changed_games = run_step(
                "Game Summary",
                run_game_summary,
                continue_on_error=args.continue_on_error,
                failures=failures,
                season_id=season["season_id"],
                season_start=season["regular_start"],
                season_end=season["regular_end"],
                con=con
            )


        # ---------------------------------------------------------
        # Determine downstream ingestion eligibility
        # ---------------------------------------------------------
        if args.skip_game_summary:
            # Explicit skip: downstream ingestions can still recover
            # missing games and refresh recently completed games.
            changed_games = pd.DataFrame(columns=["game_id"])
            print("ℹ️ Game Summary skipped — checking downstream recovery/refresh queues.")

        elif changed_games is None:
            # No usable result: do not trust game-dependent downstream data.
            if not any(f.startswith("Game Summary:") for f in failures):
                failures.append("Game Summary: no result returned")
            print("⚠️ Game Summary did not return a result — skipping dependent ingestion steps.")

        elif changed_games.empty:
            print("ℹ️ No newly completed games — checking downstream recovery/refresh queues.")

        else:
            print(f"🔄 {len(changed_games)} newly completed game(s) detected.")

        can_run_game_ingestion = changed_games is not None

        # ---------------------------------------------------------
        # Play-by-Play
        # ---------------------------------------------------------
        if not args.skip_pbp and can_run_game_ingestion:
            run_step(
                "Play-by-Play",
                run_play_by_play,
                continue_on_error=args.continue_on_error,
                failures=failures,
                season_id=season["season_id"],
                changed_games=changed_games,
                con=con
            )

        # ---------------------------------------------------------
        # Goalie Stats
        # ---------------------------------------------------------
        if not args.skip_goalie and can_run_game_ingestion:
            run_step(
                "Goalie Stats",
                run_goalie_stats,
                continue_on_error=args.continue_on_error,
                failures=failures,
                season_id=season["season_id"],
                changed_games=changed_games,
                con=con
            )

        # ---------------------------------------------------------
        # Special Teams
        # ---------------------------------------------------------
        if not args.skip_special_teams and can_run_game_ingestion:
            run_step(
                "Special Teams",
                run_special_teams,
                continue_on_error=args.continue_on_error,
                failures=failures,
                season_id=season["season_id"],
                changed_games=changed_games,
                con=con
            )
        # ---------------------------------------------------------
        # Standings Snapshots
        # ---------------------------------------------------------
        # This always gets its own check in case it is a day off in the season
        if not args.skip_standings:
            run_step(
                "Standings Snapshots",
                run_standings_snaps,
                continue_on_error=args.continue_on_error,
                failures=failures,
                season_id=season["season_id"],
                con=con
            )
        # ---------------------------------------------------------
        # Dependency gates for downstream model stages
        # ---------------------------------------------------------
        # Only attempted ingestion failures block prediction stages.
        # Explicit --skip-* flags are intentional and do not count as failures.
        ingestion_failures = list(failures)

        def block_step(name, reason):
            blocked_steps.append(f"{name}: {reason}")
            print(f"⏭️ Skipping {name} — {reason}")

        # ---------------------------------------------------------
        # Create Data used for Prediction
        # ---------------------------------------------------------
        prediction_data_failed = False
        if not args.skip_prediction_data:
            if ingestion_failures:
                block_step("Prediction Data", "an ingestion dependency failed")
            else:
                before = len(failures)
                run_step(
                    "Prediction Data", run_prediction_data,
                    continue_on_error=args.continue_on_error, failures=failures,
                    season_id=season["season_id"], con=con
                )
                prediction_data_failed = len(failures) > before

        # ---------------------------------------------------------
        # Probabilities
        # ---------------------------------------------------------
        probabilities_failed = False
        if not args.skip_probabilities:
            if ingestion_failures or prediction_data_failed:
                block_step("Playoff Probabilities", "an upstream dependency failed")
            else:
                before = len(failures)
                run_step(
                    "Playoff Probabilities", run_probability_scores,
                    continue_on_error=args.continue_on_error, failures=failures,
                    season_id=season["season_id"], con=con
                )
                probabilities_failed = len(failures) > before

        # ---------------------------------------------------------
        # Prediction Features
        # ---------------------------------------------------------
        prediction_features_failed = False

        if not args.skip_features:
            if ingestion_failures or prediction_data_failed or probabilities_failed:
                block_step(
                    "Prediction Features",
                    "an upstream dependency failed"
                )
            else:
                before = len(failures)

                run_step(
                    "Prediction Features",
                    run_prediction_features,
                    continue_on_error=args.continue_on_error,
                    failures=failures,
                    season_id=season["season_id"],
                    con=con
                )

                prediction_features_failed = len(failures) > before
                
        # ---------------------------------------------------------
        # Playoff SHAP Summary
        # ---------------------------------------------------------
        if not args.skip_features:
            if (
                ingestion_failures
                or prediction_data_failed
                or probabilities_failed
                or prediction_features_failed
            ):
                block_step(
                    "SHAP Summary",
                    "an upstream dependency failed"
                )
            else:
                run_step(
                    "SHAP Summary",
                    update_shap_summary,
                    continue_on_error=args.continue_on_error,
                    failures=failures,
                    season_id=season["season_id"],
                    con=con
                )

        status = "PARTIAL_FAILURE" if failures or blocked_steps else "SUCCESS"
        log_messages = failures + [f"BLOCKED: {item}" for item in blocked_steps]
        error_message = "\n".join(log_messages) if log_messages else None
        finish_run_log(con, run_id, status, error_message)

        print("\n==================================================")
        print(f"{'⚠️' if failures else '✅'} PIPELINE FINISHED: {status}")
        print("==================================================")

    except Exception as exc:
        if log_created:
            try:
                details = "\n".join(failures + [f"{type(exc).__name__}: {exc}"])
                finish_run_log(con, run_id, "FAILED", details)
            except Exception as log_exc:
                print(f"⚠️ Could not record pipeline failure: {log_exc}")
        raise

    finally:
        con.close()
        print("🔒 DuckDB connection closed")

# ============================================================
# Entrypoint
# ============================================================
if __name__ == "__main__":
    try:
        main()

    except Exception:
        import traceback

        print("\n❌ PIPELINE FAILED")
        traceback.print_exc()

        if getattr(sys, "frozen", False):
            input("\nPress Enter to close...")

        sys.exit(1)