
from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from datetime import datetime, timezone


import duckdb
import pandas as pd
import numpy as np

import shap

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, brier_score_loss
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from joblib import dump

from config import DB_PATH, MODEL_VERSION, MODEL_PATH, EXCLUDE_SEASON_ID
from feature_engineering import build_features


SOURCE_TABLE = "team_daily_stats"
TARGET_COL = "made_playoffs"

# Candidate model — keep production v6 unchanged
TRAINING_MODEL_VERSION = "v7"

MODEL_OUT_PATH = MODEL_PATH.with_name(
    f"playoff_models_{TRAINING_MODEL_VERSION}{MODEL_PATH.suffix}"
)

MODEL_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_feature_matrix(
    df_features: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:

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


def select_features_from_shap(X, shap_values, min_usage=0.03, min_impact=0.0005):

    shap_df = pd.DataFrame(shap_values, columns=X.columns)

    usage = (shap_df != 0).mean()
    impact = shap_df.abs().mean()

    keep_mask = (usage >= min_usage) & (impact >= min_impact)

    selected = keep_mask[keep_mask].index.tolist()

    shap_importance = pd.DataFrame({
        "feature": X.columns,
        "impact": impact
    }).sort_values("impact", ascending=False)

    print("\n🔍 SHAP Feature Selection Results")
    print(f"Selected {len(selected)} / {X.shape[1]} features")
    print(shap_importance.head(15))

    return selected

def compute_importance_by_bucket(
    pipeline: Pipeline,
    X: pd.DataFrame,
    games_played: pd.Series,
) -> dict:

    # ------------------------------------------------------------
    # Align games_played to X
    # ------------------------------------------------------------
    games_played = games_played.reindex(X.index)

    # ------------------------------------------------------------
    # Season buckets
    # ------------------------------------------------------------
    buckets = pd.cut(
        games_played,
        bins=[0, 20, 50, float("inf")],
        labels=["early", "mid", "late"],
    )

    # ------------------------------------------------------------
    # Transform X exactly as model sees it
    # ------------------------------------------------------------
    X_scaled = pipeline.named_steps["scaler"].transform(X)

    explainer = shap.TreeExplainer(
        pipeline.named_steps["model"]
    )

    shap_values = explainer(X_scaled).values

    shap_df = pd.DataFrame(
        shap_values,
        columns=X.columns,
        index=X.index,
    )

    # ------------------------------------------------------------
    # Calculate mean absolute SHAP by season bucket
    # ------------------------------------------------------------
    importance_by_bucket = {}

    for bucket in ["early", "mid", "late"]:

        mask = buckets.eq(bucket)

        if not mask.any():
            importance_by_bucket[bucket] = {}
            continue

        importance = (
            shap_df.loc[mask]
            .abs()
            .mean()
            .sort_values(ascending=False)
        )

        importance_by_bucket[bucket] = {
            feature: float(value)
            for feature, value in importance.items()
        }

        print(f"\n📊 {bucket.upper()} season SHAP importance")
        print(importance.head(15))

    return importance_by_bucket


def build_canonical_training_features(df_raw):
    """
    Calculate features using complete daily team snapshots,
    then retain only the original team-game observations.

    Does not modify the source DataFrame or database.
    """
    import pandas as pd

    keys = ["season_id", "game_date", "team_id"]

    source = df_raw.copy()
    source["game_date"] = pd.to_datetime(
        source["game_date"]
    ).dt.normalize()

    if source.duplicated(keys).any():
        raise ValueError("Duplicate source team-date keys")

    if source[keys].isna().any().any():
        raise ValueError("Null source team-date keys")

    parts = []

    for season_id, games in source.groupby(
        "season_id", sort=True
    ):
        dates = pd.DataFrame({
            "game_date": sorted(games["game_date"].unique())
        })
        dates["game_date"] = pd.to_datetime(
            dates["game_date"]
        )

        teams = pd.DataFrame({
            "team_id": sorted(games["team_id"].unique())
        })

        spine = dates.merge(teams, how="cross")
        spine["season_id"] = season_id

        spine = spine.sort_values(
            ["game_date", "team_id"]
        ).reset_index(drop=True)

        history = games.sort_values(
            ["game_date", "team_id"]
        ).reset_index(drop=True)

        snapshots = pd.merge_asof(
            spine,
            history.drop(columns=["season_id"]),
            on="game_date",
            by="team_id",
            direction="backward",
            allow_exact_matches=True
        )

        snapshots = snapshots.loc[
            snapshots["games_played"].notna()
            & snapshots["tri_code"].notna()
        ].copy()

        parts.append(snapshots)

    canonical = pd.concat(parts, ignore_index=True)

    print(
        f"Canonical snapshots: {len(canonical):,}"
    )

    engineered = build_features(canonical)

    if len(engineered) != len(canonical):
        raise RuntimeError(
            "Feature engineering changed snapshot count"
        )

    if engineered.duplicated(keys).any():
        raise RuntimeError(
            "Duplicate engineered team-date keys"
        )

    game_keys = source[keys].drop_duplicates()

    training = engineered.merge(
        game_keys,
        on=keys,
        how="inner",
        validate="one_to_one"
    )

    if len(training) != len(source):
        raise RuntimeError(
            f"Training row mismatch: "
            f"{len(training):,} vs {len(source):,}"
        )

    # Restore original game-row ordering.
    source_order = source[keys].copy()
    source_order["_source_order"] = range(len(source_order))

    training = training.merge(
        source_order,
        on=keys,
        how="left",
        validate="one_to_one"
    )

    training = (
        training.sort_values("_source_order")
        .drop(columns="_source_order")
        .reset_index(drop=True)
    )


    for col in ["game_id", TARGET_COL, "games_played"]:
        if col not in source.columns:
            raise RuntimeError(f"Missing source column: {col}")
        if col not in training.columns:
            raise RuntimeError(f"Missing training column: {col}")

        left = pd.to_numeric(
            source[col].reset_index(drop=True),
            errors="coerce"
        )

        right = pd.to_numeric(
            training[col].reset_index(drop=True),
            errors="coerce"
        )

        if not np.array_equal(
            left.to_numpy(),
            right.to_numpy(),
            equal_nan=True
        ):
            raise RuntimeError(
                f"Original {col} values were not preserved"
            )


    print(f"Training rows: {len(training):,}")
    print(f"Training columns: {len(training.columns)}")
    print("Canonical training dataset validated")

    return training

def main() -> None:
    print("🔌 Connecting to DuckDB")
    print(f"Model version: {TRAINING_MODEL_VERSION}")
    print(f"Model output: {MODEL_OUT_PATH}")
    print(f"Excluded season: {EXCLUDE_SEASON_ID}")

    con = duckdb.connect(DB_PATH)


    df_raw = con.execute(f"SELECT * FROM {SOURCE_TABLE}").df()

    if "season_id" in df_raw.columns:
        df_raw = df_raw[~df_raw["season_id"].isin(
            [EXCLUDE_SEASON_ID] if isinstance(EXCLUDE_SEASON_ID, int) else EXCLUDE_SEASON_ID
        )].copy()

    if df_raw.empty:
        raise RuntimeError("❌ Training set is empty")

    if "game_id" in df_raw.columns:
        df_raw = df_raw.sort_values("game_id").reset_index(drop=True)

    print("🧠 Building canonical daily training features")

    df_features = build_canonical_training_features(df_raw)

    # ============================================================
    # Time-based train/test split
    #
    # Train: all completed seasons before 2025-26
    # Test:  2025-26
    # 2026-27 is already excluded via EXCLUDE_SEASON_ID
    # ============================================================

    TEST_SEASON_ID = 20252026

    train_mask = (
        df_features["season_id"] < TEST_SEASON_ID
    )

    test_mask = (
        df_features["season_id"] == TEST_SEASON_ID
    )

    X, FEATURE_COLS = _build_feature_matrix(df_features)
    
    check_features = [
        "games_played",
        "games_remaining",
        "conference_rank",
        "sos_future",
        "c_shot_pct",
        "goal_diff",
        "corsi_diff_per_game",
        "SCF_pg",
        "rush_for_per_game",
        "so_losses",
        "home_goal_diff",
        "c_sv_pct",
    ]

    print("\n🔎 Season-stage feature check")

    for feature in check_features:
        print(
            f"{feature:25} "
            f"{'✅ included' if feature in FEATURE_COLS else '❌ missing'}"
        )

    # ============================================================
    # Target sanity check
    # ============================================================

    print("\n🎯 TARGET SANITY CHECK")

    target_check = (
        df_features
        .groupby(["season_id", "team_id"], as_index=False)
        .agg(
            target_values=(TARGET_COL, "nunique"),
            made_playoffs=(TARGET_COL, "max"),
        )
    )

    bad_targets = target_check[
        target_check["target_values"] != 1
    ]

    if not bad_targets.empty:
        print("\n❌ Teams with inconsistent made_playoffs values:")
        print(bad_targets.to_string(index=False))
        raise RuntimeError(
            "❌ made_playoffs changes within a team-season"
        )

    season_targets = (
        target_check
        .groupby("season_id")
        .agg(
            teams=("team_id", "nunique"),
            playoff_teams=("made_playoffs", "sum"),
        )
    )

    print("\nPlayoff target counts by season:")
    print(season_targets.to_string())
    
    y = df_features[TARGET_COL].astype(int)

    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()

    y_train = y.loc[train_mask].copy()
    y_test = y.loc[test_mask].copy()
    
    if X_train.empty:
        raise RuntimeError("❌ Training split is empty")

    if X_test.empty:
        raise RuntimeError(
            f"❌ Test season {TEST_SEASON_ID} contains no rows"
        )

    if y_train.nunique() < 2:
        raise RuntimeError("❌ Training target contains only one class")

    if y_test.nunique() < 2:
        raise RuntimeError("❌ Test target contains only one class")

    print("\n📊 Training split")
    print(
        f"Train seasons: "
        f"{df_features.loc[train_mask, 'season_id'].min()} → "
        f"{df_features.loc[train_mask, 'season_id'].max()}"
    )
    print(f"Test season: {TEST_SEASON_ID}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Training positive rate: {y_train.mean():.3f}")
    print(f"Test positive rate: {y_test.mean():.3f}")

    print("🏋️ Training INITIAL XGBoost model")
    xgb_pipeline = _make_xgb_pipeline()
    xgb_pipeline.fit(X_train, y_train)

    print("🔍 Running SHAP feature selection")

    X_train_scaled = xgb_pipeline.named_steps["scaler"].transform(X_train)
    explainer = shap.TreeExplainer(xgb_pipeline.named_steps["model"])
    shap_values = explainer(X_train_scaled).values

    selected_features = select_features_from_shap(
        X_train,
        shap_values,
        min_usage=0.03,
        min_impact=0.0005
    )
    
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    FEATURE_COLS = selected_features
    
    print("\n🔎 Selected season-stage features")

    for feature in check_features:
        print(
            f"{feature:25} "
            f"{'✅ selected' if feature in FEATURE_COLS else '❌ dropped'}"
        )
        
    print(f"✅ Final feature count: {len(FEATURE_COLS)}")

    print("🔁 Retraining XGBoost with selected features")
    xgb_pipeline = _make_xgb_pipeline()
    xgb_pipeline.fit(X_train, y_train)

    xgb_prob = xgb_pipeline.predict_proba(X_test)[:, 1]

    print("\n✅ XGBOOST REPORT\n")
    print(classification_report(y_test, xgb_pipeline.predict(X_test)))

    xgb_brier = float(brier_score_loss(y_test, xgb_prob))
    print("XGB Brier score:", xgb_brier)
    
    print("\n🔬 Computing season-stage SHAP importance")

    importance_by_bucket = compute_importance_by_bucket(
        pipeline=xgb_pipeline,
        X=X_test,
        games_played=df_features.loc[test_mask, "games_played"],
    )
    print("🌲 Training Random Forest model")
    rf_pipeline = _make_rf_pipeline()
    rf_pipeline.fit(X_train, y_train)

    rf_prob = rf_pipeline.predict_proba(X_test)[:, 1]

    print("\n✅ RANDOM FOREST REPORT\n")
    print(classification_report(y_test, rf_pipeline.predict(X_test)))

    rf_brier = float(brier_score_loss(y_test, rf_prob))
    print("RF Brier score:", rf_brier)
    
    # ============================================================
    # Season-stage holdout evaluation
    # ============================================================

    print("\n🔬 Evaluating holdout performance by season stage")

    # Keep games_played from the original feature dataframe because
    # it may not survive SHAP feature selection.
    holdout_eval = df_features.loc[
        test_mask,
        [
            "season_id",
            "team_id",
            "game_id",
            "game_date",
            "games_played",
            TARGET_COL,
        ],
    ].copy()

    # Predictions are in the same row order as X_test / y_test
    holdout_eval["playoff_prob_xgb"] = xgb_prob
    holdout_eval["playoff_prob_rf"] = rf_prob

    holdout_eval["playoff_prob"] = (
        holdout_eval["playoff_prob_xgb"]
        + holdout_eval["playoff_prob_rf"]
    ) / 2


    # ------------------------------------------------------------
    # Season buckets
    # ------------------------------------------------------------

    holdout_eval["season_bucket"] = pd.cut(
        holdout_eval["games_played"],
        bins=[0, 20, 50, float("inf")],
        labels=["early", "mid", "late"],
        include_lowest=True,
    )


    # ------------------------------------------------------------
    # Evaluate each bucket
    # ------------------------------------------------------------

    bucket_results = []

    for bucket in ["early", "mid", "late"]:

        bucket_df = holdout_eval[
            holdout_eval["season_bucket"] == bucket
        ].copy()

        if bucket_df.empty:
            continue

        y_bucket = bucket_df[TARGET_COL].astype(int)

        xgb_bucket = bucket_df["playoff_prob_xgb"]
        rf_bucket = bucket_df["playoff_prob_rf"]
        ensemble_bucket = bucket_df["playoff_prob"]

        ensemble_class = (
            ensemble_bucket >= 0.50
        ).astype(int)

        bucket_results.append({
            "season_bucket": bucket,
            "rows": len(bucket_df),

            "brier_xgb": brier_score_loss(
                y_bucket,
                xgb_bucket,
            ),

            "brier_rf": brier_score_loss(
                y_bucket,
                rf_bucket,
            ),

            "brier_ensemble": brier_score_loss(
                y_bucket,
                ensemble_bucket,
            ),

            "accuracy_ensemble": (
                ensemble_class == y_bucket
            ).mean(),

            "mean_predicted_prob": (
                ensemble_bucket.mean()
            ),

            "actual_playoff_rate": (
                y_bucket.mean()
            ),
        })


    bucket_results = pd.DataFrame(bucket_results)

    print("\n📊 HOLDOUT PERFORMANCE BY SEASON STAGE\n")

    print(
        bucket_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
        
        
    # ============================================================
    # Probability calibration by season stage
    # ============================================================

    print("\n🎯 Evaluating probability calibration")

    # ------------------------------------------------------------
    # Probability bands
    # ------------------------------------------------------------

    prob_bins = [
        0.00,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.01,
    ]

    prob_labels = [
        "0-10%",
        "10-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%",
    ]

    holdout_eval["probability_band"] = pd.cut(
        holdout_eval["playoff_prob"],
        bins=prob_bins,
        labels=prob_labels,
        right=False,
        include_lowest=True,
    )


    # ------------------------------------------------------------
    # Calibration table
    # ------------------------------------------------------------

    calibration = (
        holdout_eval
        .groupby(
            ["season_bucket", "probability_band"],
            observed=True,
        )
        .agg(
            rows=("made_playoffs", "size"),
            mean_predicted=("playoff_prob", "mean"),
            actual_playoff_rate=("made_playoffs", "mean"),
        )
        .reset_index()
    )

    calibration["calibration_error"] = (
        calibration["actual_playoff_rate"]
        - calibration["mean_predicted"]
    )

    calibration["absolute_error"] = (
        calibration["calibration_error"].abs()
    )


    # ------------------------------------------------------------
    # Display each season stage
    # ------------------------------------------------------------

    for bucket in ["early", "mid", "late"]:

        print(f"\n📊 {bucket.upper()} SEASON CALIBRATION\n")

        bucket_calibration = calibration[
            calibration["season_bucket"] == bucket
        ].copy()

        print(
            bucket_calibration[
                [
                    "probability_band",
                    "rows",
                    "mean_predicted",
                    "actual_playoff_rate",
                    "calibration_error",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )


    # ------------------------------------------------------------
    # Weighted calibration error by season stage
    # ------------------------------------------------------------

    calibration_summary = (
        calibration
        .assign(
            weighted_error=lambda x:
                x["absolute_error"] * x["rows"]
        )
        .groupby(
            "season_bucket",
            observed=True,
        )
        .agg(
            rows=("rows", "sum"),
            weighted_error=("weighted_error", "sum"),
        )
        .reset_index()
    )

    calibration_summary["mean_absolute_calibration_error"] = (
        calibration_summary["weighted_error"]
        / calibration_summary["rows"]
    )

    print("\n📏 CALIBRATION SUMMARY\n")

    print(
        calibration_summary[
            [
                "season_bucket",
                "rows",
                "mean_absolute_calibration_error",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    
    # ============================================================
    # Test current early-season display adjustment
    # ============================================================

    print("\n🧪 Testing current display probability adjustment")

    holdout_eval["playoff_prob_display_test"] = (
        holdout_eval["playoff_prob"]
    )

    early_mask = (
        holdout_eval["season_bucket"] == "early"
    )

    holdout_eval.loc[
        early_mask,
        "playoff_prob_display_test"
    ] = (
        0.45
        * holdout_eval.loc[early_mask, "playoff_prob"]
        + 0.275
    )


    # ------------------------------------------------------------
    # Compare raw vs adjusted Brier
    # ------------------------------------------------------------

    adjustment_results = []

    for bucket in ["early", "mid", "late"]:

        bucket_df = holdout_eval[
            holdout_eval["season_bucket"] == bucket
        ]

        if bucket_df.empty:
            continue

        y_bucket = bucket_df[TARGET_COL].astype(int)

        adjustment_results.append({
            "season_bucket": bucket,
            "rows": len(bucket_df),

            "raw_brier": brier_score_loss(
                y_bucket,
                bucket_df["playoff_prob"],
            ),

            "display_brier": brier_score_loss(
                y_bucket,
                bucket_df["playoff_prob_display_test"],
            ),
        })


    adjustment_results = pd.DataFrame(
        adjustment_results
    )

    adjustment_results["brier_change"] = (
        adjustment_results["display_brier"]
        - adjustment_results["raw_brier"]
    )

    print("\n📊 RAW VS DISPLAY PROBABILITY\n")

    print(
        adjustment_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    # ============================================================
    # Inspect largest holdout prediction errors
    # ============================================================

    print("\n🚨 Inspecting largest holdout prediction errors")

    holdout_eval["prediction_error"] = (
        holdout_eval["playoff_prob"]
        - holdout_eval[TARGET_COL]
    ).abs()

    largest_errors = (
        holdout_eval
        .sort_values(
            "prediction_error",
            ascending=False,
        )
        .head(30)
    )

    print(
        largest_errors[
            [
                "game_date",
                "team_id",
                "games_played",
                "season_bucket",
                TARGET_COL,
                "playoff_prob",
                "prediction_error",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


    # ============================================================
    # Team probability responsiveness test
    # ============================================================

    print("\n📈 Testing probability responsiveness for teams 7 and 17")

    RESPONSIVENESS_TEAMS = [7, 17]

    response_cols = [
        "season_id",
        "team_id",
        "tri_code",
        "game_id",
        "game_date",
        "games_played",
        "conference_rank",
        "points_pct",
        "goal_diff",
        "sos_future",
        "c_shot_pct",
        "SCF_pg",
        "rush_for_per_game",
        "c_sv_pct",
    ]

    # Pull the original model inputs for the holdout season
    response_df = (
        df_features.loc[
            test_mask,
            [c for c in response_cols if c in df_features.columns],
        ]
        .copy()
    )

    # Add the holdout probabilities.
    # Index alignment keeps each probability attached to its original row.
    response_df["playoff_prob"] = holdout_eval["playoff_prob"]

    response_df = (
        response_df[
            response_df["team_id"].isin(RESPONSIVENESS_TEAMS)
        ]
        .sort_values(["team_id", "game_date"])
        .copy()
    )

    print("\n📊 RESPONSIVENESS TEAMS")

    print(
        response_df[
            [
                "team_id",
                "tri_code",
            ]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )

    # ============================================================
    # Checkpoint snapshots
    # ============================================================

    checkpoint_dates = pd.to_datetime([
        "2025-11-27",  # Thanksgiving
        "2025-12-15",
        "2026-01-15",
        "2026-02-15",
        "2026-03-01",
        "2026-03-15",
        "2026-04-01",
    ])


    checkpoint_rows = []

    for team_id in RESPONSIVENESS_TEAMS:

        team_df = (
            response_df[
                response_df["team_id"] == team_id
            ]
            .sort_values("game_date")
            .copy()
        )

        for checkpoint in checkpoint_dates:

            available = team_df[
                pd.to_datetime(team_df["game_date"]) <= checkpoint
            ]

            if available.empty:
                continue

            row = available.iloc[-1].copy()
            row["checkpoint"] = checkpoint.date()

            checkpoint_rows.append(row)


    checkpoints = pd.DataFrame(checkpoint_rows)

    display_cols = [
        "checkpoint",
        "game_date",
        "team_id",
        "tri_code",
        "games_played",
        "playoff_prob",
        "conference_rank",
        "points_pct",
        "goal_diff",
        "sos_future",
        "c_shot_pct",
        "SCF_pg",
        "rush_for_per_game",
        "c_sv_pct",
    ]

    display_cols = [
        c for c in display_cols
        if c in checkpoints.columns
    ]


    for team_id in RESPONSIVENESS_TEAMS:

        team_checkpoints = checkpoints[
            checkpoints["team_id"] == team_id
        ]

        tri_code = (
            team_checkpoints["tri_code"].iloc[0]
            if "tri_code" in team_checkpoints.columns
            else str(team_id)
        )

        print(
            f"\n🏒 {tri_code} — TEAM {team_id} "
            f"PROBABILITY TRAJECTORY\n"
        )

        print(
            team_checkpoints[
                display_cols
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )
            
            
    # ============================================================
    # Game-to-game probability stability
    # ============================================================

    print("\n🌊 Evaluating game-to-game probability stability")

    stability_df = (
        holdout_eval
        .sort_values(["team_id", "game_date", "game_id"])
        .copy()
    )

    # Previous prediction for the same team
    stability_df["previous_playoff_prob"] = (
        stability_df
        .groupby("team_id")["playoff_prob"]
        .shift(1)
    )

    # Change after each new game
    stability_df["prob_change"] = (
        stability_df["playoff_prob"]
        - stability_df["previous_playoff_prob"]
    )

    stability_df["abs_prob_change"] = (
        stability_df["prob_change"].abs()
    )

    # Remove each team's first observation
    stability_valid = (
        stability_df[
            stability_df["previous_playoff_prob"].notna()
        ]
        .copy()
    )

    # ============================================================
    # Overall stability by season stage
    # ============================================================

    stability_summary = (
        stability_valid
        .groupby("season_bucket", observed=True)
        .agg(
            rows=("abs_prob_change", "size"),
            mean_change=("abs_prob_change", "mean"),
            median_change=("abs_prob_change", "median"),
            p90_change=("abs_prob_change", lambda x: x.quantile(0.90)),
            p95_change=("abs_prob_change", lambda x: x.quantile(0.95)),
            max_change=("abs_prob_change", "max"),
        )
        .reindex(["early", "mid", "late"])
        .reset_index()
    )

    print("\n📊 GAME-TO-GAME PROBABILITY STABILITY\n")

    print(
        stability_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    # ============================================================
    # Size of probability movements
    # ============================================================

    change_bands = pd.cut(
        stability_valid["abs_prob_change"],
        bins=[
            0,
            0.025,
            0.05,
            0.10,
            0.20,
            1.0,
        ],
        labels=[
            "0-2.5 pts",
            "2.5-5 pts",
            "5-10 pts",
            "10-20 pts",
            "20+ pts",
        ],
        include_lowest=True,
    )

    movement_distribution = (
        stability_valid
        .assign(change_band=change_bands)
        .groupby(
            ["season_bucket", "change_band"],
            observed=True,
        )
        .size()
        .reset_index(name="rows")
    )

    movement_distribution["pct_of_stage"] = (
        movement_distribution["rows"]
        / movement_distribution
            .groupby("season_bucket", observed=True)["rows"]
            .transform("sum")
    )

    print("\n📏 PROBABILITY MOVEMENT DISTRIBUTION\n")

    print(
        movement_distribution.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


    # ============================================================
    # Largest single-game probability movements
    # ============================================================

    largest_moves = (
        stability_valid
        .sort_values(
            "abs_prob_change",
            ascending=False,
        )
        .head(30)
    )

    print("\n🚨 LARGEST SINGLE-GAME PROBABILITY MOVEMENTS\n")

    largest_move_cols = [
        "game_date",
        "team_id",
        "tri_code",
        "games_played",
        "season_bucket",
        "previous_playoff_prob",
        "playoff_prob",
        "prob_change",
        "abs_prob_change",
    ]

    largest_move_cols = [
        c for c in largest_move_cols
        if c in largest_moves.columns
    ]

    print(
        largest_moves[
            largest_move_cols
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


    # ============================================================
    # Early-season stability by games played
    # ============================================================

    print("\n🌱 Evaluating early-season stability by games played")

    early_stability = stability_valid[
        stability_valid["games_played"] <= 20
    ].copy()

    early_stability["game_window"] = pd.cut(
        early_stability["games_played"],
        bins=[0, 5, 10, 15, 20],
        labels=[
            "Games 1-5",
            "Games 6-10",
            "Games 11-15",
            "Games 16-20",
        ],
    )

    early_window_summary = (
        early_stability
        .groupby(
            "game_window",
            observed=True,
        )
        .agg(
            rows=("abs_prob_change", "size"),
            mean_change=("abs_prob_change", "mean"),
            median_change=("abs_prob_change", "median"),
            p90_change=("abs_prob_change", lambda x: x.quantile(0.90)),
            p95_change=("abs_prob_change", lambda x: x.quantile(0.95)),
            max_change=("abs_prob_change", "max"),
        )
        .reset_index()
    )

    print("\n📊 EARLY-SEASON STABILITY BY GAME WINDOW\n")

    print(
        early_window_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    
    # ============================================================
    # Test early-season display adjustment by game window
    # ============================================================

    print("\n🧪 Testing early-season display adjustment strength")

    early_adjustment_test = holdout_eval[
        holdout_eval["games_played"] <= 20
    ].copy()

    # Current adjustment used for all games 1-20
    early_adjustment_test["display_current"] = (
        0.45 * early_adjustment_test["playoff_prob"] + 0.275
    )

    # Stronger opening adjustment for games 1-5
    early_adjustment_test["display_opening"] = (
        early_adjustment_test["playoff_prob"]
    )

    opening_mask = early_adjustment_test["games_played"] <= 5

    early_adjustment_test.loc[
        opening_mask,
        "display_opening"
    ] = (
        0.25
        * early_adjustment_test.loc[opening_mask, "playoff_prob"]
        + 0.375
    )

    # Games 6-20 retain current adjustment
    later_early_mask = early_adjustment_test["games_played"].between(6, 20)

    early_adjustment_test.loc[
        later_early_mask,
        "display_opening"
    ] = (
        0.45
        * early_adjustment_test.loc[later_early_mask, "playoff_prob"]
        + 0.275
    )

    early_adjustment_test["game_window"] = pd.cut(
        early_adjustment_test["games_played"],
        bins=[0, 5, 10, 15, 20],
        labels=[
            "Games 1-5",
            "Games 6-10",
            "Games 11-15",
            "Games 16-20",
        ],
    )

    adjustment_window_results = []

    for window, window_df in early_adjustment_test.groupby(
        "game_window",
        observed=True,
    ):

        y_window = window_df[TARGET_COL].astype(int)

        adjustment_window_results.append({
            "game_window": str(window),
            "rows": len(window_df),

            "raw_brier": brier_score_loss(
                y_window,
                window_df["playoff_prob"],
            ),

            "current_display_brier": brier_score_loss(
                y_window,
                window_df["display_current"],
            ),

            "stronger_opening_brier": brier_score_loss(
                y_window,
                window_df["display_opening"],
            ),
        })

    adjustment_window_results = pd.DataFrame(
        adjustment_window_results
    )

    print("\n📊 EARLY DISPLAY ADJUSTMENT BY GAME WINDOW\n")

    print(
        adjustment_window_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    
    # ============================================================
    # SAVE HISTORICAL HOLDOUT EVALUATION
    # ============================================================

    EVAL_DIR = ROOT_DIR / "evaluation" / TRAINING_MODEL_VERSION
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    holdout_eval.to_parquet(EVAL_DIR / "holdout_predictions.parquet", index=False)
    bucket_results.to_csv(EVAL_DIR / "season_stage_brier.csv", index=False)
    calibration.to_csv(EVAL_DIR / "calibration.csv", index=False)
    calibration_summary.to_csv(EVAL_DIR / "calibration_summary.csv", index=False)
    adjustment_results.to_csv(EVAL_DIR / "display_adjustment.csv", index=False)
    adjustment_window_results.to_csv(EVAL_DIR / "early_adjustment_windows.csv", index=False)
    stability_summary.to_csv(EVAL_DIR / "stability_summary.csv", index=False)
    early_window_summary.to_csv(EVAL_DIR / "early_stability.csv", index=False)

    print(f"\n💾 Historical evaluation saved to: {EVAL_DIR}")
    
    # ============================================================
    # Final production training
    #
    # Evaluation above uses 2025-26 as the holdout season.
    # Now retrain the production models on ALL completed seasons
    # through 2025-26 using the selected feature set.
    # ============================================================

    print("\n🏁 Retraining final production models on all completed seasons")

    X_final = X[FEATURE_COLS].copy()
    y_final = y.copy()

    print(
        f"Final training seasons: "
        f"{df_features['season_id'].min()} → "
        f"{df_features['season_id'].max()}"
    )
    print(f"Final training rows: {len(X_final):,}")
    print(f"Final feature count: {len(FEATURE_COLS)}")


    # ------------------------------------------------------------
    # Final XGBoost
    # ------------------------------------------------------------

    print("🔁 Training final XGBoost model")

    xgb_pipeline = _make_xgb_pipeline()
    xgb_pipeline.fit(X_final, y_final)


    # ------------------------------------------------------------
    # Final Random Forest
    # ------------------------------------------------------------

    print("🌲 Training final Random Forest model")

    rf_pipeline = _make_rf_pipeline()
    rf_pipeline.fit(X_final, y_final)
    bundle = {
        "xgb_pipeline": xgb_pipeline,
        "rf_pipeline": rf_pipeline,
        "features": FEATURE_COLS,
        "trained_on": _utc_now(),
        "model_type": "XGB+RF-playoff-models",
        "snapshot": "daily",
        "version": TRAINING_MODEL_VERSION,
        "source_table": SOURCE_TABLE,
        "target": TARGET_COL,
        "importance_by_bucket": importance_by_bucket,
        "metrics": {
            "holdout_xgb_brier": xgb_brier,
            "holdout_rf_brier": rf_brier,
            "evaluation_train_rows": int(len(X_train)),
            "evaluation_test_rows": int(len(X_test)),
            "final_training_rows": int(len(X_final)),
            "final_positive_rate": float(y_final.mean()),
            "evaluation_train_positive_rate": float(y_train.mean()),
            "evaluation_test_positive_rate": float(y_test.mean()),
            "test_season_id": TEST_SEASON_ID,
        },
    }

    dump(bundle, MODEL_OUT_PATH)

    print(f"\n💾 Model bundle saved: {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()