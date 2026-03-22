# ============================================================
# daily_feature_predictions.py
# ============================================================

import duckdb
import pandas as pd
import shap
from joblib import load
from datetime import datetime

from config import MODEL_PATH
from feature_engineering import build_features

# ------------------------------------------------------------
# Tables
# ------------------------------------------------------------

FEATURE_SOURCE_TABLE = "team_daily_stats"      # your feature base
FEATURE_TARGET_TABLE = "playoff_shap_values"  # shap output

# ------------------------------------------------------------
# Load model bundle
# ------------------------------------------------------------

bundle = load(MODEL_PATH)

xgb_pipeline = bundle["xgb_pipeline"]
FEATURE_COLS = bundle["features"]

xgb_model = xgb_pipeline.named_steps["model"]
scaler = xgb_pipeline.named_steps.get("scaler", None)

# ------------------------------------------------------------
# Main Runner
# ------------------------------------------------------------

def run_prediction_features(
    *,
    con: duckdb.DuckDBPyConnection,
    season_id: int,
) -> None:
    print(f"▶ Running SHAP values for season {season_id}")

    # ------------------------------------------------------------
    # Pull latest snapshot (one row per team)
    # ------------------------------------------------------------

    df_raw = con.execute(f"""
        SELECT *
        FROM {FEATURE_SOURCE_TABLE}
        WHERE season_id = {season_id}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY team_id
            ORDER BY game_date DESC
        ) = 1
    """).df()

    if df_raw.empty:
        raise RuntimeError("❌ No data found for SHAP")

    # ------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------

    df_features = build_features(df_raw)

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

    # Apply scaler if present
    if scaler is not None:
        X_in = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
    else:
        X_in = X

    # ------------------------------------------------------------
    # SHAP calculation
    # ------------------------------------------------------------

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_in).values

    shap_df = pd.DataFrame(
        shap_values,
        columns=FEATURE_COLS,
        index=X_in.index
    )

    # ------------------------------------------------------------
    # Add identifiers back
    # ------------------------------------------------------------

    shap_df["team_id"] = df_features["team_id"].values
    shap_df["tri_code"] = df_features["tri_code"].values
    shap_df["season_id"] = season_id
    shap_df["game_date"] = df_features["game_date"].values
    shap_df["created_at"] = datetime.utcnow()

    # ------------------------------------------------------------
    # WRITE (REPLACE TABLE — avoids schema mismatch)
    # ------------------------------------------------------------

    con.execute(f"DROP TABLE IF EXISTS {FEATURE_TARGET_TABLE}")

    con.execute(f"""
        CREATE TABLE {FEATURE_TARGET_TABLE} AS
        SELECT * FROM shap_df
    """)

    print(f"✅ SHAP values written to {FEATURE_TARGET_TABLE}")