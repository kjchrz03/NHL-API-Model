# ============================================================
# daily_shap_values.py
# ============================================================

import sys
from pathlib import Path

# ------------------------------------------------------------
# Ensure project root is on PYTHONPATH
# ------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import duckdb
import pandas as pd
import shap
from joblib import load
from datetime import date

from config import DB_PATH, MODEL_PATH
from feature_engineering import build_features

FEATURE_TARGET_TABLE = "playoff_shap_values"


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_shap_values(
    season_id: int,
    con: duckdb.DuckDBPyConnection,
    backfill: bool = False,
    backfill_end_date: str | None = None
):

    print(f"▶ Running SHAP values for season {season_id}")

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    bundle = load(MODEL_PATH)

    xgb_pipeline = bundle["xgb_pipeline"]
    FEATURE_COLS = bundle["features"]

    xgb_model = xgb_pipeline.named_steps["model"]
    scaler = xgb_pipeline.named_steps.get("scaler", None)

    print(f"📦 Model version: {bundle['version']}")
    print(f"📊 Feature count: {len(FEATURE_COLS)}")

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    df_raw = con.execute(f"""
        SELECT *
        FROM team_daily_stats
        WHERE season_id = {season_id}
    """).df()

    df_scored = con.execute(f"""
        SELECT *
        FROM playoff_daily_predictions
        WHERE season_id = {season_id}
    """).df()

    if df_raw.empty or df_scored.empty:
        print("⚠️ No data available for SHAP")
        return

    # ------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------
    df_features = build_features(df_raw)

    df_features = df_features.merge(
        df_scored[[
            "team_id",
            "game_date",
            "playoff_prob"
        ]],
        on=["team_id", "game_date"],
        how="left"
    )

    df_features = df_features.dropna(subset=["playoff_prob"])
    df_features["game_date"] = pd.to_datetime(df_features["game_date"]).dt.date

    # ------------------------------------------------------------
    # Enforce TEAM-DAY grain
    # ------------------------------------------------------------
    df_features = (
        df_features
        .sort_values(["team_id", "game_date"])
        .drop_duplicates(subset=["season_id", "game_date", "team_id"], keep="last")
    )

    dupes = df_features.duplicated(
        subset=["season_id", "game_date", "team_id"]
    ).sum()

    assert dupes == 0, f"❌ Found {dupes} duplicate rows before SHAP"

    # ------------------------------------------------------------
    # BACKFILL MODE
    # ------------------------------------------------------------
    if backfill:
        print("🔄 FULL BACKFILL MODE")

        if backfill_end_date:
            cutoff = pd.to_datetime(backfill_end_date).date()
        else:
            cutoff = date.today()

        df_features = df_features[df_features["game_date"] <= cutoff]

        print(f"Backfilling through: {cutoff}")

        # 🔥 Drop table to avoid schema drift
        print("🧹 Dropping existing SHAP table")
        con.execute(f"DROP TABLE IF EXISTS {FEATURE_TARGET_TABLE}")

    # ------------------------------------------------------------
    # INCREMENTAL MODE
    # ------------------------------------------------------------
    else:
        existing = con.execute(f"""
            SELECT DISTINCT game_date
            FROM {FEATURE_TARGET_TABLE}
            WHERE season_id = {season_id}
        """).df()

        if not existing.empty:
            existing_dates = set(pd.to_datetime(existing["game_date"]).dt.date)

            df_features = df_features[
                ~df_features["game_date"].isin(existing_dates)
            ]

        if df_features.empty:
            print("✅ SHAP values already up to date.")
            return

    print("🧠 Computing SHAP values")

    # ------------------------------------------------------------
    # Build feature matrix (MATCH TRAINING EXACTLY)
    # ------------------------------------------------------------
    LEAK_COLS = [
        "game_id", "game_date", "date",
        "clinch_type", "made_playoffs",
        "team_name", "opponent", "season_id"
    ]

    ID_COLS = ["team_id", "opponent_id", "tri_code"]

    X = (
        df_features
        .drop(columns=LEAK_COLS + ID_COLS, errors="ignore")
        .select_dtypes(include="number")
    )

    # Match training NA handling
    X = X.loc[:, X.isna().mean() < 0.20].fillna(0)

    # Align to model features
    X = X.reindex(columns=FEATURE_COLS, fill_value=0)

    # 🔥 CRITICAL CHECK
    assert list(X.columns) == FEATURE_COLS, "❌ Feature misalignment with model"

    # ------------------------------------------------------------
    # Scale
    # ------------------------------------------------------------
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values

    # ------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_scaled).values

    # ------------------------------------------------------------
    # Build SHAP table
    # ------------------------------------------------------------
    shap_wide = pd.DataFrame(
        shap_values,
        columns=FEATURE_COLS,
        index=df_features.index
    )

    # Metadata
    shap_wide.insert(0, "season_id", season_id)
    shap_wide.insert(1, "game_date", df_features["game_date"].values)
    shap_wide.insert(2, "team_id", df_features["team_id"].values)
    shap_wide.insert(3, "tri_code", df_features["tri_code"].values)

    shap_wide["model_version"] = bundle["version"]
    shap_wide["model_run_ts"] = pd.Timestamp.utcnow()

    # ------------------------------------------------------------
    # Debug: check zero columns
    # ------------------------------------------------------------
    zero_mask = (shap_wide[FEATURE_COLS] == 0).all()

    zero_cols = zero_mask[zero_mask].index.tolist()
    print(f"⚠️ All-zero SHAP columns: {len(zero_cols)}")

    print(len(FEATURE_COLS))
    print(FEATURE_COLS[:20])

    print("\nZero columns:")
    print(zero_cols)
    
    # Check what model actually used
    model = xgb_pipeline.named_steps["model"]

    booster = model.get_booster()

    raw_importance = booster.get_score(importance_type="weight")

    # Map f0 → actual feature names
    importance = {
        FEATURE_COLS[int(k[1:])]: v
        for k, v in raw_importance.items()
    }

    unused = [col for col in FEATURE_COLS if col not in importance]

    print("\n🧠 Model Usage (fixed)")
    print("Used features:", len(importance))
    print("Unused features:", len(unused))
    print("Example unused:", unused[:20])

    print("\n🧠 Model Usage")
    print("Unused features:", len(unused))
    print(unused[:20])
    # ------------------------------------------------------------
    # Create table
    # ------------------------------------------------------------
    feature_cols_sql = ",\n".join([f"{col} DOUBLE" for col in FEATURE_COLS])

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {FEATURE_TARGET_TABLE} (
            season_id BIGINT,
            game_date DATE,
            team_id BIGINT,
            tri_code VARCHAR,
            {feature_cols_sql},
            model_version VARCHAR,
            model_run_ts TIMESTAMP,
            PRIMARY KEY (season_id, game_date, team_id)
        )
    """)

    # ------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------
    print("🧪 Writing SHAP values to DuckDB")

    con.register("shap_df", shap_wide)

    cols = shap_wide.columns.tolist()
    col_str = ", ".join(cols)

    con.execute(f"""
        INSERT INTO {FEATURE_TARGET_TABLE} ({col_str})
        SELECT {col_str} FROM shap_df
    """)

    con.unregister("shap_df")

    print(f"✅ SHAP values stored")
    print(f"Rows written: {len(shap_wide)}")
    print(f"Date range: {shap_wide['game_date'].min()} → {shap_wide['game_date'].max()}")


# ============================================================
# RUN (MANUAL BACKFILL)
# ============================================================

if __name__ == "__main__":

    con = duckdb.connect(DB_PATH)

    try:
        run_shap_values(
            season_id=20252026,
            con=con,
            backfill=True,
            backfill_end_date="2026-03-20"
        )
    finally:
        con.close()
        print("🔒 DuckDB connection closed")