
import sys
from pathlib import Path
from datetime import datetime, timezone
import json

from joblib import load

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import MODEL_PATH, MODEL_VERSION


# ============================================================
# MODEL MANIFEST CONFIGURATION
# ============================================================

OUTPUT_PATH = Path(MODEL_PATH).parent / f"playoff_models_{MODEL_VERSION}_manifest.json"

MODEL_NAME = "playoff_models_xgb_rf"
SOURCE_TABLE = "team_daily_stats"
SNAPSHOT_GRAIN = "team-day"
TARGET = "made_playoffs"

FEATURE_ENGINEERING_VERSION = MODEL_VERSION

NOTES = (
    "Ensemble playoff probability model (XGBoost + Random Forest). "
    "Feature set includes standings, schedule context, possession, "
    "special teams, shooting, goaltending, wildcard pressure, rush, "
    "and interaction features. "
    "SHAP-based feature selection applied using historical training seasons. "
    "Model evaluated out-of-time on the 2025-26 season, then retrained "
    "on all completed seasons through 2025-26 for production. "
    "Early, mid, and late season SHAP importance retained for model analysis. "
    "Daily (team-day) time-series output."
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

bundle = load(MODEL_PATH)

FEATURE_COLS = bundle["features"]

# Preserve the actual training timestamp.
# Do not substitute the manifest creation date.
trained_on = bundle.get("trained_on")

manifest_created_on = datetime.now(timezone.utc).isoformat()


# ============================================================
# TRAINING METRICS
# ============================================================

metric_defaults = {
    "xgb_brier": None,
    "rf_brier": None,
    "evaluation_train_rows": None,
    "evaluation_test_rows": None,
    "final_training_rows": None,
    "final_positive_rate": None,
    "evaluation_train_positive_rate": None,
    "evaluation_test_positive_rate": None,
    "test_season_id": None,
}

training_metrics = {
    **metric_defaults,
    **(bundle.get("metrics") or {}),
}


# ============================================================
# FEATURE SELECTION / PREPROCESSING METADATA
# ============================================================

feature_selection = bundle.get("feature_selection") or {
    "method": "SHAP",
    "min_usage": 0.03,
    "min_impact": 0.0005,
}

preprocessing = bundle.get("preprocessing") or {
    "numeric_only": True,
    "max_missing_fraction": 0.20,
    "fillna": 0,
}


# ============================================================
# BUILD MODEL MANIFEST
# ============================================================

manifest = {
    "model_name": MODEL_NAME,
    "version": MODEL_VERSION,
    "trained_on": trained_on,
    "manifest_created_on": manifest_created_on,

    "source_table": SOURCE_TABLE,
    "snapshot_grain": SNAPSHOT_GRAIN,
    "target": TARGET,

    "features": FEATURE_COLS,
    "feature_count": len(FEATURE_COLS),

    "model_type": bundle.get("model_type", "XGB+RF-playoff-pipeline"),

    "inference_outputs": [
        "playoff_prob",
        "playoff_prob_xgb",
        "playoff_prob_rf",
        "playoff_prob_display",
        "playoff_tier",
    ],

    "training_metrics": training_metrics,
    "importance_by_bucket": bundle.get("importance_by_bucket", {}),

    "feature_selection": feature_selection,
    "preprocessing": preprocessing,

    "categoricals": {
        "rank_bucket": [
            "tier1_elite",
            "tier2_strong",
            "tier3_bubble",
            "tier4_bottom",
        ],
        "conference": [
            "Eastern",
            "Western",
        ],
    },

    "excluded_columns": [
        "game_id",
        "game_date",
        "date",
        "season_id",
        "team_name",
        "opponent",
        "clinch_type",
        "made_playoffs",
    ],

    "feature_engineering_version": FEATURE_ENGINEERING_VERSION,
    "notes": NOTES,
}


# ============================================================
# VALIDATE MANIFEST BEFORE WRITING
# ============================================================

assert isinstance(manifest["features"], list), "Features must be a list"
assert len(manifest["features"]) > 0, "Feature list cannot be empty"
assert manifest["version"] == MODEL_VERSION, "Model version mismatch"

assert len(manifest["features"]) == len(set(manifest["features"])), (
    "Duplicate feature names detected"
)

assert manifest["feature_count"] == len(manifest["features"]), (
    "Feature count mismatch"
)

assert isinstance(manifest["training_metrics"], dict), (
    "Training metrics must be a dictionary"
)

# Verify that the manifest can be serialized before creating the file.
json.dumps(manifest, indent=2)

print("✅ Manifest validation passed")


# ============================================================
# WRITE MANIFEST
# ============================================================

output_path = Path(OUTPUT_PATH)
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"✅ Model manifest written to: {output_path.resolve()}")
print(f"📊 Model version: {MODEL_VERSION}")
print(f"📐 Feature count: {len(FEATURE_COLS):,}")
print(f"🕒 Trained on: {trained_on or 'Not recorded in model bundle'}")
print(f"📝 Manifest created: {manifest_created_on}")
