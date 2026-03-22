import duckdb
import pandas as pd
import shap
from joblib import load
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from config import DB_PATH, CURRENT_SEASON_ID
from feature_engineering import build_features
import os
from datetime import datetime, timezone
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
# --------------------------------------------------
# Load model bundle
# --------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
model_path = os.path.join(script_dir, "models", "playoff_models_wc_field_pressure_v4.joblib")

bundle = load(model_path)
xgb_pipeline = bundle["xgb_pipeline"]
FEATURE_COLS = bundle["features"]

xgb_model = xgb_pipeline.named_steps["model"]
scaler = xgb_pipeline.named_steps.get("scaler", None)

# --------------------------------------------------
# Pull latest snapshot (one row per team)
# --------------------------------------------------
con = duckdb.connect(DB_PATH)

# df_raw = con.execute(f"""
# SELECT *
# FROM team_daily_stats
# WHERE season_id = {CURRENT_SEASON_ID}
# QUALIFY ROW_NUMBER() OVER (
#     PARTITION BY team_id
#     ORDER BY game_date DESC
# ) = 1
# """).df()


df_raw = con.execute(f"""
SELECT *
FROM team_daily_stats
WHERE season_id = {CURRENT_SEASON_ID}
ORDER BY game_date, team_id
""").df()


# df_scored = con.execute(f"""
#                         SELECT *
# FROM playoff_daily_predictions
# WHERE season_id = {CURRENT_SEASON_ID}
# QUALIFY ROW_NUMBER() OVER (
#     PARTITION BY team_id
#     ORDER BY game_date DESC
# ) = 1
# """).df()

df_scored = con.execute(f"""
SELECT *
FROM playoff_daily_predictions
WHERE season_id = {CURRENT_SEASON_ID}
ORDER BY game_date, team_id
""").df()


assert df_raw["team_id"].nunique() == 32
assert df_scored["team_id"].nunique() == 32
# --------------------------------------------------
# Feature engineering
# --------------------------------------------------
df_features = build_features(df_raw)


df_features = df_features.merge(
    df_scored[[
        "team_id",
        "game_date",
        "playoff_prob",
        "playoff_prob_xgb",
        "playoff_prob_rf",
        "playoff_prob_display",
        "playoff_tier"
    ]],
    # on="team_id",
    on=["team_id", "game_date"],
    how="left"
)

assert df_features["playoff_prob"].notna().all(), "Missing probabilities after merge"
# --------------------------------------------------
# Build feature matrix (must match training features)
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




# ---- Build the exact numeric matrix you want to explain with SHAP
if scaler is not None:
    X_shap_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )
else:
    X_shap_scaled  = X

# --------------------------------------------------
# SHAP
# --------------------------------------------------
explainer = shap.TreeExplainer(xgb_model)
shap_expl = explainer(X_shap_scaled)  # <-- Explanation object

# --------------------------------------------------
# Explain function
# --------------------------------------------------
def explain_team(
    team_code: str,
    current_df: pd.DataFrame,
    X_shap_scaled: pd.DataFrame,
    shap_expl,
    feature_cols: list,
    n: int = 8
):
    team_mask = current_df["tri_code"].str.upper() == team_code.upper()
    if not team_mask.any():
        raise ValueError(f"Team {team_code} not found")

    row_idx = current_df.loc[team_mask].index[-1]

    if row_idx not in X_shap_scaled.index:
        raise ValueError(
            f"Row index {row_idx} not found in X_shap_scaled. "
            "Make sure you didn't reset index on either df."
        )

    row_pos = X_shap_scaled.index.get_loc(row_idx)

    shap_row = pd.Series(
        shap_expl.values[row_pos, :],
        index=feature_cols
    ).sort_values()

    most_harmful = shap_row.head(n)
    most_helpful = shap_row.tail(n)

    # season stage
    if "season_bucket" in current_df.columns:
        season_stage = current_df.loc[row_idx, "season_bucket"]
    elif "games_played" in current_df.columns:
        gp = current_df.loc[row_idx, "games_played"]
        season_stage = "early" if gp <= 20 else "mid" if gp <= 50 else "late"
    else:
        season_stage = "unknown"

    # probability (now exists because we added playoff_prob)
    playoff_prob = current_df.loc[row_idx, "playoff_prob"] if "playoff_prob" in current_df.columns else None

    header = f"\n📊 {team_code.upper()} PLAYOFF EXPLANATION ({season_stage.upper()} SEASON)\n"
    if playoff_prob is not None and pd.notna(playoff_prob):
        header += f"Model Odds: {playoff_prob:.1%}\n"
    print(header)

    print("🔻 Most Harmful Drivers")
    display(most_harmful.round(3))

    print("\n🔺 Most Helpful Drivers")
    display(most_helpful.round(3))

    return most_helpful, most_harmful

