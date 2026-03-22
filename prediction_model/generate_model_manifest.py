# ============================================================
# Generate Model Manifest (Authoritative Contract) — v4
# ============================================================

import json
from datetime import datetime, timezone
from pathlib import Path
from joblib import load

from config import MODEL_PATH

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

OUTPUT_PATH = (
    Path(MODEL_PATH).parent
    / "playoff_models_wc_field_pressure_v4_manifest.json"
)

MODEL_NAME = "playoff_models_xgb_rf"
MODEL_VERSION = "wc_field_pressure_v4"

SOURCE_TABLE = "team_daily_stats"
SNAPSHOT_GRAIN = "team-day"
TARGET = "made_playoffs"

FEATURE_ENGINEERING_VERSION = "v4"

NOTES = (
    "Ensemble playoff probability model (XGBoost + Random Forest). "
    "Includes WC field pressure + density features. "
    "Daily (team-day) time-series output."
)

# ------------------------------------------------------------
# LOAD MODEL BUNDLE
# ------------------------------------------------------------

bundle = load(MODEL_PATH)

FEATURE_COLS = bundle["features"]
trained_on = bundle.get("trained_on")

if trained_on is None:
    trained_on = datetime.now(timezone.utc).isoformat()

# ------------------------------------------------------------
# Pull metrics if present
# ------------------------------------------------------------

training_metrics = bundle.get(
    "metrics",
    {
        "brier_score": None,
        "train_rows": None,
        "test_rows": None,
        "positive_rate": None,
    }
)

# ------------------------------------------------------------
# BUILD MANIFEST
# ------------------------------------------------------------

manifest = {
    "model_name": MODEL_NAME,
    "version": MODEL_VERSION,
    "trained_on": trained_on,
    "source_table": SOURCE_TABLE,
    "snapshot_grain": SNAPSHOT_GRAIN,
    "target": TARGET,

    # 🔑 Authoritative feature contract
    "features": FEATURE_COLS,

    # Model identity
    "model_type": bundle.get("model_type", "XGB+RF-playoff-pipeline"),
    "inference_outputs": [
        "playoff_prob",
        "playoff_prob_xgb",
        "playoff_prob_rf",
        "playoff_prob_display",
        "playoff_tier",
    ],

    # Training diagnostics
    "training_metrics": training_metrics,
    "importance_by_bucket": bundle.get("importance_by_bucket", {}),

    # Categoricals used in feature engineering
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

    # Explicit leakage / identifier exclusions
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

# ------------------------------------------------------------
# WRITE TO DISK
# ------------------------------------------------------------

output_path = Path(OUTPUT_PATH)
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"✅ Model manifest written to: {output_path.resolve()}")

# ------------------------------------------------------------
# SAFETY CHECKS
# ------------------------------------------------------------

assert isinstance(manifest["features"], list)
assert len(manifest["features"]) > 0
assert manifest["version"].endswith("v4")

print("✅ Manifest validation passed")