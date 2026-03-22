import duckdb
import pandas as pd
import shap
from joblib import load

from config import MODEL_PATH
from feature_engineering import build_features

# --------------------------------------------------
# Load model bundle
# --------------------------------------------------
bundle = load(MODEL_PATH)

xgb_pipeline = bundle["xgb_pipeline"]
FEATURE_COLS = bundle["features"]

xgb_model = xgb_pipeline.named_steps["model"]
scaler = xgb_pipeline.named_steps.get("scaler", None)

# --------------------------------------------------
# Function wrapper (pipeline-friendly)
# --------------------------------------------------
def run_shap_summary(
    *,
    con,
    season_id: int,
) -> None:
    print(f"▶ Running SHAP summary for season {season_id}")

    # --------------------------------------------------
    # Pull latest snapshot (one row per team)
    # --------------------------------------------------
    df_raw = con.execute("""
    SELECT *
    FROM team_daily_stats
    WHERE season_id = ?
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY team_id
        ORDER BY game_date DESC
    ) = 1
    """, [season_id]).df()

    if df_raw.empty:
        raise RuntimeError("❌ No data found for SHAP")

    # --------------------------------------------------
    # Feature engineering
    # --------------------------------------------------
    df_features = build_features(df_raw)

    # --------------------------------------------------
    # Build SHAP feature matrix
    # --------------------------------------------------
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

    # --------------------------------------------------
    # SHAP
    # --------------------------------------------------
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_in).values

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    shap.summary_plot(shap_values, X_in)

    print("✅ SHAP summary complete")