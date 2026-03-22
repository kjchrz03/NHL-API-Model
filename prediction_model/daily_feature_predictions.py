# ============================================================
# daily_shap_values.py
# ============================================================

import duckdb
import pandas as pd
import shap
from joblib import load
from datetime import datetime

from config import DB_PATH, MODEL_PATH
from feature_engineering import build_features

FEATURE_TARGET_TABLE = "daily_game_features"

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def run_prediction_features(season_id: int, con: duckdb.DuckDBPyConnection):

    print(f"▶ Running SHAP values for season {season_id}")

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    bundle = load(MODEL_PATH)

    xgb_pipeline = bundle["xgb_pipeline"]
    FEATURE_COLS = bundle["features"]

    xgb_model = xgb_pipeline.named_steps["model"]
    scaler = xgb_pipeline.named_steps.get("scaler", None)

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

    # ------------------------------------------------------------
    # Determine missing dates (incremental/backfill)
    # ------------------------------------------------------------
    existing = con.execute(f"""
        SELECT DISTINCT game_date
        FROM {FEATURE_TARGET_TABLE}
        WHERE season_id = {season_id}
    """).df()

    if not existing.empty:
        existing_dates = set(pd.to_datetime(existing["game_date"]).dt.date)
        df_features["game_date"] = pd.to_datetime(df_features["game_date"]).dt.date

        df_features = df_features[
            ~df_features["game_date"].isin(existing_dates)
        ]

    if df_features.empty:
        print("✅ SHAP values already up to date.")
        return

    print("🧠 Computing SHAP values")

    # ------------------------------------------------------------
    # Build feature matrix
    # ------------------------------------------------------------
    LEAK_COLS = [
        "game_id","game_date","date",
        "clinch_type","made_playoffs",
        "team_name","opponent","season_id"
    ]
    ID_COLS = ["team_id","opponent_id","tri_code"]

    X = (
        df_features
        .drop(columns=LEAK_COLS + ID_COLS, errors="ignore")
        .select_dtypes(include="number")
        .reindex(columns=FEATURE_COLS, fill_value=0)
        .fillna(0)
    )

    if scaler is not None:
        X_in = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
    else:
        X_in = X

    # ------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_in).values

    # ------------------------------------------------------------
    # Build wide SHAP table
    # ------------------------------------------------------------
    shap_wide = pd.DataFrame(
        shap_values,
        columns=FEATURE_COLS,
        index=df_features.index
    )

    shap_wide.insert(0, "season_id", season_id)
    shap_wide.insert(1, "game_date", df_features["game_date"].values)
    shap_wide.insert(2, "team_id", df_features["team_id"].values)
    shap_wide.insert(3, "tri_code", df_features["tri_code"].values)

    shap_wide["model_version"] = "wc_field_pressure_v4"
    shap_wide["model_run_ts"] = pd.Timestamp.utcnow()

    # ------------------------------------------------------------
    # Create table if needed
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
    # Upsert
    # ------------------------------------------------------------
    print("🧪 Writing SHAP values to DuckDB")

    con.register("shap_df", shap_wide)

    con.execute(f"""
        DELETE FROM {FEATURE_TARGET_TABLE}
        WHERE (season_id, game_date, team_id) IN (
            SELECT season_id, game_date, team_id FROM shap_df
        )
    """)

    con.execute(f"""
        INSERT INTO {FEATURE_TARGET_TABLE}
        SELECT * FROM shap_df
    """)

    con.unregister("shap_df")

    print(f"✅ SHAP values stored")
    print(f"Rows written: {len(shap_wide)}")
    print(f"Date range: {shap_wide['game_date'].min()} → {shap_wide['game_date'].max()}")