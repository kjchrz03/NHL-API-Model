# =====================================================
# PLAYOFF MODEL — WC FIELD PRESSURE v1 (LOCKED)
# =====================================================
# Library-style module:
#  - Explicit training
#  - Default inference
#  - No side effects on import
# =====================================================

import os
import json
import pandas as pd
from datetime import datetime, timezone
from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, brier_score_loss

from xgboost import XGBClassifier

# =====================================================
# CONSTANTS
# =====================================================

MODEL_VERSION = "wc_field_pressure_v2"
MODEL_PATH = f"models/playoff_xgb_pipeline_{MODEL_VERSION}.joblib"
FEATURE_PATH = f"models/playoff_features_{MODEL_VERSION}.json"

LEAK_COLS = [
    "game_id", "game_date", "date",
    "clinch_type", "made_playoffs",
    "team_name", "opponent",
    "season_id"
]

ID_COLS = [
    "team_id", "opponent_id", "tri_code"
]

# =====================================================
# FEATURE ENGINEERING — LOCKED v2
# =====================================================

def add_wc_field_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    wc_eligible = out[out["division_rank"] > 3]

    wc_cutline = (
        wc_eligible
        .sort_values(
            ["season_id", "game_date", "conference", "points", "points_pct"],
            ascending=[True, True, True, False, False]
        )
        .groupby(["season_id", "game_date", "conference"], as_index=False)
        .nth(1)  # 2nd wild card
        .reset_index(drop=True)
        [["season_id", "game_date", "conference", "points", "points_pct"]]
        .rename(columns={
            "points": "wc2_points",
            "points_pct": "wc2_points_pct"
        })
    )

    out = out.merge(
        wc_cutline,
        on=["season_id", "game_date", "conference"],
        how="left"
    )

    out["wc_points_margin"] = out["points"] - out["wc2_points"]
    out["wc_points_pct_margin"] = out["points_pct"] - out["wc2_points_pct"]

    # ✅ CORRECT FIELD NAME: wildcard_rank
    out["in_wc_spot"] = (out["wildcard_rank"] <= 2).astype(int)
    out["in_div_top3"] = (out["division_rank"] <= 3).astype(int)
    out["on_wc_bubble"] = out["wildcard_rank"].between(1, 4).astype(int)

    return out


def add_wc_density_features(
    df: pd.DataFrame,
    windows=(2, 4, 6)
) -> pd.DataFrame:
    out = df.copy()
    wc_mask = out["division_rank"] > 3

    for w in windows:
        col = f"wc_teams_within_{w}pts"

        out[col] = (
            out
            .assign(diff=lambda x: (x["points"] - x["wc2_points"]).abs())
            .where(wc_mask)
            .groupby(["season_id", "game_date", "conference"])["diff"]
            .transform(lambda s: (s <= w).sum())
            .fillna(0)
            .astype(int)
        )

    return out

# =====================================================
# TRAINING — EXPLICIT ONLY
# =====================================================

def train_playoff_model(train_df: pd.DataFrame):
    df = (
        train_df
        .sort_values(["season_id", "team_id", "game_date"])
        .reset_index(drop=True)
    )

    df = add_wc_field_pressure_features(df)
    df = add_wc_density_features(df)

    X = (
        df
        .drop(columns=LEAK_COLS + ID_COLS, errors="ignore")
        .select_dtypes(include="number")
    )

    X = X.loc[:, X.isna().mean() < 0.20].fillna(0)
    y = df["made_playoffs"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n✅ MODEL REPORT\n")
    print(classification_report(y_test, pipeline.predict(X_test)))
    print("Brier score:", brier_score_loss(y_test, y_prob))

    return pipeline, X.columns.tolist()


def save_model_bundle(pipeline, feature_cols):
    os.makedirs("models", exist_ok=True)

    dump(
        {
            "pipeline": pipeline,
            "features": feature_cols,
            "version": MODEL_VERSION,
            "trained_on": datetime.now(timezone.utc).isoformat(),
            "notes": "WC cutline + density features; race-aware calibration",
        },
        MODEL_PATH
    )

    with open(FEATURE_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"💾 Model saved → {MODEL_PATH}")
    print(f"📐 Features saved → {FEATURE_PATH}")

# =====================================================
# INFERENCE — DEFAULT ENTRY POINT
# =====================================================

def load_and_score(current_df: pd.DataFrame) -> pd.DataFrame:
    bundle = load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    feature_cols = bundle["features"]

    df = add_wc_field_pressure_features(current_df)
    df = add_wc_density_features(df)

    X = (
        df
        .drop(columns=LEAK_COLS + ID_COLS, errors="ignore")
        .select_dtypes(include="number")
        .fillna(0)
        .reindex(columns=feature_cols, fill_value=0)
    )

    df["playoff_prob_row"] = pipeline.predict_proba(X)[:, 1]
    return df
