import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from joblib import load

from config import MODEL_PATH


_bundle = load(MODEL_PATH)

XGB_PIPELINE = _bundle["xgb_pipeline"]
RF_PIPELINE = _bundle["rf_pipeline"]
FEATURE_COLS = _bundle["features"]


LEAK_COLS = [
    "game_id",
    "game_date",
    "date",
    "season_id",
    "team_name",
    "opponent",
    "clinch_type",
    "made_playoffs",
]

ID_COLS = [
    "team_id",
    "opponent_id",
    "tri_code",
]


def score_dataframe(df_features: pd.DataFrame) -> pd.DataFrame:

    df = df_features.copy()

    X = (
        df
        .drop(columns=LEAK_COLS + ID_COLS, errors="ignore")
        .select_dtypes(include="number")
        .copy()
    )

    X = X.fillna(0)

    missing_cols = [col for col in FEATURE_COLS if col not in X.columns]
    extra_cols = [col for col in X.columns if col not in FEATURE_COLS]

    if missing_cols:
        for col in missing_cols:
            X[col] = 0

    X = X.reindex(columns=FEATURE_COLS, fill_value=0)

    if list(X.columns) != FEATURE_COLS:
        raise ValueError("❌ Feature alignment failed")

    if missing_cols:
        print(f"⚠️ Filled {len(missing_cols)} missing features with 0")

    if extra_cols:
        print(f"ℹ️ Ignored {len(extra_cols)} extra features not used by model")

    df["playoff_prob_xgb"] = XGB_PIPELINE.predict_proba(X)[:, 1]
    df["playoff_prob_rf"] = RF_PIPELINE.predict_proba(X)[:, 1]

    df["playoff_prob"] = (
        df["playoff_prob_xgb"] + df["playoff_prob_rf"]
    ) / 2

    return df