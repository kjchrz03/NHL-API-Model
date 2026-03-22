import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
from datetime import datetime, timezone
from pathlib import Path
from joblib import load

from config import MODEL_PATH, MODEL_VERSION


OUTPUT_PATH = (
    Path(MODEL_PATH).parent
    / f"playoff_models_{MODEL_VERSION}_manifest.json"
)

MODEL_NAME = "playoff_models_xgb_rf"

SOURCE_TABLE = "team_daily_stats"
SNAPSHOT_GRAIN = "team-day"
TARGET = "made_playoffs"

FEATURE_ENGINEERING_VERSION = MODEL_VERSION

NOTES = (
    "Ensemble playoff probability model (XGBoost + Random Forest). "
    "Feature set includes engineered possession, special teams, "
    "wildcard pressure, rush, and interaction features. "
    "Cumulative stats excluded to prevent temporal leakage. "
    "SHAP-based feature selection applied. "
    "Daily (team-day) time-series output."
)


bundle = load(MODEL_PATH)

FEATURE_COLS = bundle["features"]
trained_on = bundle.get("trained_on")

if trained_on is None:
    trained_on = datetime.now(timezone.utc).isoformat()


training_metrics = bundle.get(
    "metrics",
    {
        "xgb_brier": None,
        "rf_brier": None,
        "rows_train": None,
        "rows_test": None,
        "positive_rate": None,
    }
)


manifest = {
    "model_name": MODEL_NAME,
    "version": MODEL_VERSION,
    "trained_on": trained_on,
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

    "feature_selection": {
        "method": "SHAP",
        "min_usage": 0.03,
        "min_impact": 0.0005
    },

    "preprocessing": {
        "numeric_only": True,
        "fillna": 0,
        "dropped_prefixes": ["c_"]
    },

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


output_path = Path(OUTPUT_PATH)
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"✅ Model manifest written to: {output_path.resolve()}")


assert isinstance(manifest["features"], list)
assert len(manifest["features"]) > 0
assert manifest["version"] == MODEL_VERSION

print("✅ Manifest validation passed")