from pathlib import Path
import pandas as pd
from joblib import load
from config import MODEL_PATH

LEAK_COLS = [
    "game_id", "game_date", "date", "season_id",
    "team_name", "opponent", "clinch_type", "made_playoffs"
]

ID_COLS = ["team_id", "opponent_id", "tri_code"]

def score_version(df_features: pd.DataFrame, version: str = "v6") -> pd.DataFrame:
    if version not in {"v6", "v7"}:
        raise ValueError(f"Unsupported model version: {version}")

    path = (
        Path(MODEL_PATH) if version == "v6"
        else Path(MODEL_PATH).with_name("playoff_models_v7.joblib")
    )

    bundle = load(path)
    features = list(bundle["features"])

    df = df_features.copy()
    X = (
        df.drop(columns=LEAK_COLS + ID_COLS, errors="ignore")
          .select_dtypes(include="number")
          .copy()
    )

    missing = [c for c in features if c not in X.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    X = X.reindex(columns=features).fillna(0)

    df["playoff_prob_xgb"] = bundle["xgb_pipeline"].predict_proba(X)[:, 1]
    df["playoff_prob_rf"] = bundle["rf_pipeline"].predict_proba(X)[:, 1]
    df["playoff_prob"] = (
        df["playoff_prob_xgb"] + df["playoff_prob_rf"]
    ) / 2

    # Preserve v6 behavior; v7 displays unadjusted probabilities.
    if version == "v6":
        early = df["games_played"].isna() | df["games_played"].le(20)
        df["playoff_prob_display"] = df["playoff_prob"].where(
            ~early, 0.45 * df["playoff_prob"] + 0.275
        )
    else:
        df["playoff_prob_display"] = df["playoff_prob"]

    df["model_version"] = version
    return df
