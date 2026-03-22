# ============================================================
# train_playoff_model.py
# ============================================================

"""
Train + evaluate NHL playoff probability models (XGBoost + Random Forest)
and export a joblib bundle.

Outputs:
- models/playoff_models_<MODEL_VERSION>.joblib

Bundle contents:
{
  "xgb_pipeline": sklearn Pipeline,
  "rf_pipeline": sklearn Pipeline,
  "features": frozen feature column list,
  "trained_on": UTC timestamp,
  "model_type": "XGB+RF-playoff-models",
  "snapshot": "daily",
  "version": MODEL_VERSION,
  "source_table": SOURCE_TABLE,
  "target": TARGET_COL,
  "metrics": { ... }
}
"""
# ============================================================
# train_playoff_model.py
# ============================================================

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional

import duckdb
import numpy as np
import pandas as pd
import shap

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, brier_score_loss
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from joblib import dump

from feature_engineering import build_features

# ============================================================
# Configuration
# ============================================================

DB_PATH = r"C:\Users\kjchr\Desktop\hockey\new_hockey_env\API data\duckdb\nhl_data.duckdb"

SOURCE_TABLE = "team_daily_stats"
TARGET_COL = "made_playoffs"

EXCLUDE_SEASON_ID = 20252026
MODEL_VERSION = "wc_field_pressure_v5"  # bumped version

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUT_PATH = MODEL_DIR / f"playoff_models_{MODEL_VERSION}.joblib"

LEAK_COLS = [
    "game_id",
    "game_date",
    "date",
    "season_id",
    "team_name",
    "opponent",
    "clinch_type",
    TARGET_COL,
]

ID_COLS = [
    "team_id",
    "opponent_id",
    "tri_code",
]
# ============================================================
# Helpers
# ============================================================

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_feature_matrix(
    df_features: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:

    X = (
        df_features
        .drop(columns=LEAK_COLS + ID_COLS, errors="ignore")
        .select_dtypes(include="number")
    )

    X = X.loc[:, X.isna().mean() < 0.20].fillna(0)

    if feature_cols is None:
        return X, X.columns.tolist()

    X = X.reindex(columns=feature_cols, fill_value=0)

    if list(X.columns) != feature_cols:
        raise ValueError("❌ Feature alignment failed")

    return X, feature_cols


def _make_xgb_pipeline() -> Pipeline:
    return Pipeline(
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


def _make_rf_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("model", RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )


# ============================================================
# SHAP FEATURE SELECTION
# ============================================================

def select_features_from_shap(X, shap_values, min_usage=0.05, min_impact=0.001):

    shap_df = pd.DataFrame(shap_values, columns=X.columns)

    usage = (shap_df != 0).mean()
    impact = shap_df.abs().mean()

    keep_mask = (usage >= min_usage) & (impact >= min_impact)

    selected = keep_mask[keep_mask].index.tolist()

    print("\n🔍 SHAP Feature Selection Results")
    print(f"Selected {len(selected)} / {X.shape[1]} features")

    return selected



# ============================================================
# Main
# ============================================================

def main() -> None:
    print("🔌 Connecting to DuckDB")
    con = duckdb.connect(DB_PATH)

    df_raw = con.execute(f"SELECT * FROM {SOURCE_TABLE}").df()

    if "season_id" in df_raw.columns:
        df_raw = df_raw[df_raw["season_id"] != EXCLUDE_SEASON_ID].copy()

    if df_raw.empty:
        raise RuntimeError("❌ Training set is empty")

    if "game_id" in df_raw.columns:
        df_raw = df_raw.sort_values("game_id").reset_index(drop=True)

    print("🧠 Applying feature engineering")
    df_features = build_features(df_raw)

    y = df_features[TARGET_COL].astype(int)
    
    # --------------------------------------------------------
    # Build initial feature matrix
    # --------------------------------------------------------
    
    X, FEATURE_COLS = _build_feature_matrix(df_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------------
    # Train XGBoost
    # --------------------------------------------------------
    print("🏋️ Training INITIAL XGBoost model")
    xgb_pipeline = _make_xgb_pipeline()
    xgb_pipeline.fit(X_train, y_train)

    # --------------------------------------------------------
    # SHAP FEATURE SELECTION (NEW)
    # --------------------------------------------------------
    print("🔍 Running SHAP feature selection")

    scaler = xgb_pipeline.named_steps["scaler"]
    model = xgb_pipeline.named_steps["model"]

    X_train_scaled = scaler.transform(X_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train_scaled).values

    selected_features = select_features_from_shap(
        X_train,
        shap_values,
        min_usage=0.05,
        min_impact=0.001
    )

    # Reduce dataset
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    FEATURE_COLS = selected_features

    print(f"✅ Final feature count: {len(FEATURE_COLS)}")
    
    
    # --------------------------------------------------------
    # RETRAIN FINAL MODEL
    # --------------------------------------------------------

    print("🔁 Retraining XGBoost with selected features")
    xgb_pipeline = _make_xgb_pipeline()
    xgb_pipeline.fit(X_train, y_train)

    xgb_prob = xgb_pipeline.predict_proba(X_test)[:, 1]

    print("\n✅ XGBOOST REPORT\n")
    print(classification_report(y_test, xgb_pipeline.predict(X_test)))

    xgb_brier = float(brier_score_loss(y_test, xgb_prob))
    print("XGB Brier score:", xgb_brier)

    # --------------------------------------------------------
    # Train Random Forest
    # --------------------------------------------------------

    print("🌲 Training Random Forest model")
    rf_pipeline = _make_rf_pipeline()
    rf_pipeline.fit(X_train, y_train)

    rf_prob = rf_pipeline.predict_proba(X_test)[:, 1]

    print("\n✅ RANDOM FOREST REPORT\n")
    print(classification_report(y_test, rf_pipeline.predict(X_test)))

    rf_brier = float(brier_score_loss(y_test, rf_prob))
    print("RF Brier score:", rf_brier)

    # --------------------------------------------------------
    # Save bundle
    # --------------------------------------------------------

    bundle = {
        "xgb_pipeline": xgb_pipeline,
        "rf_pipeline": rf_pipeline,
        "features": FEATURE_COLS,
        "trained_on": _utc_now(),
        "model_type": "XGB+RF-playoff-models",
        "snapshot": "daily",
        "version": MODEL_VERSION,
        "source_table": SOURCE_TABLE,
        "target": TARGET_COL,
        "metrics": {
            "xgb_brier": xgb_brier,
            "rf_brier": rf_brier,
            "rows_train": int(len(X_train)),
            "rows_test": int(len(X_test)),
            "positive_rate": float(y.mean()),
        },
    }

    dump(bundle, MODEL_OUT_PATH)

    print(f"\n💾 Model bundle saved: {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()