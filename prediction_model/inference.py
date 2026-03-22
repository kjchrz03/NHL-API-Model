# ============================================================
# inference.py
# ============================================================

import pandas as pd
from joblib import load

from config import MODEL_PATH

# ------------------------------------------------------------
# Load model bundle ONCE (module import time)
# ------------------------------------------------------------

_bundle = load(MODEL_PATH)

XGB_PIPELINE = _bundle["xgb_pipeline"]
RF_PIPELINE = _bundle["rf_pipeline"]
FEATURE_COLS = _bundle["features"]

# ------------------------------------------------------------
# Constants (must match training)
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def score_dataframe(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Align engineered features to the frozen model contract
    and score playoff probabilities using both models.
    """

    df = df_features.copy()

    # --------------------------------------------------------
    # 1. Build numeric feature matrix
    # --------------------------------------------------------

    X = (
        df
        .drop(columns=LEAK_COLS + ID_COLS, errors="ignore")
        .select_dtypes(include="number")
        .fillna(0)
    )

    # --------------------------------------------------------
    # 2. Enforce frozen feature contract
    # --------------------------------------------------------

    X = X.reindex(columns=FEATURE_COLS, fill_value=0)

    if list(X.columns) != FEATURE_COLS:
        raise ValueError("❌ Feature alignment failed")

    # --------------------------------------------------------
    # 3. Predict probabilities (per-model)
    # --------------------------------------------------------

    df["playoff_prob_xgb"] = XGB_PIPELINE.predict_proba(X)[:, 1]
    df["playoff_prob_rf"] = RF_PIPELINE.predict_proba(X)[:, 1]

    # --------------------------------------------------------
    # 4. Ensemble (primary probability)
    # --------------------------------------------------------

    df["playoff_prob"] = (
        df["playoff_prob_xgb"] + df["playoff_prob_rf"]
    ) / 2

    return df