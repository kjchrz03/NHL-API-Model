# ============================================================
# metadata.py
# ============================================================

from typing import Dict, List
from config import MODEL_VERSION



# ------------------------------------------------------------
# Model identity
# ------------------------------------------------------------

MODEL_METADATA = {
    "model_name": "playoff_models",
    "version": MODEL_VERSION,
    "snapshot_grain": "team-day",
    "target": "made_playoffs",
    "description": (
        "Predicts the probability that an NHL team will qualify "
        "for the playoffs based on daily performance trends, "
        "standings position, and wild-card field pressure. "
        "Includes both XGBoost and Random Forest models."
    ),
    "league_context": {
        "playoff_structure": "Top 3 per division + 2 wild cards per conference",
        "conference_playoff_slots": 8,
    },
    "models_included": [
        "xgboost",
        "random_forest",
    ],
    "primary_probability": "playoff_prob",
    "ensemble_method": "mean(xgb, rf)",
}

# ------------------------------------------------------------
# Feature groups (for explanation + SHAP narratives)
# ------------------------------------------------------------

FEATURE_GROUPS: Dict[str, List[str]] = {
    "Standings Position": [
        "conference_rank",
        "division_rank",
        "wildcard_rank",
        "points",
        "points_pct",
    ],
    "Schedule & Context": [
        "sos_past",
        "sos_future",
        "games_played",
        "games_remaining",
    ],
    "Goal Differential": [
        "goal_diff",
        "home_goal_diff",
        "road_goal_diff",
    ],
    "Team Strength (5v5)": [
        "corsi_diff_per_game",
        "SCF_pg",
        "rush_for_per_game",
        "rush_against_per_game",
        "rush_diff_per_game",
    ],
    "Special Teams": [
        "pp_pct",
        "pk_pct",
        "ppng_rank",
        "pk_rank",
        "times_sh_pg",
    ],
    "Goaltending & Shooting": [
        "c_shot_pct",
        "c_sv_pct",
        "c_pdo",
    ],
    "Wild Card Pressure": [
        "wc_points_margin",
        "wc_points_pct_margin",
        "wc_teams_within_2pts",
        "wc_teams_within_4pts",
        "wc_teams_within_6pts",
        "in_wc_spot",
        "on_wc_bubble",
        "in_div_top3",
    ],
}

# ------------------------------------------------------------
# Feature descriptions (plain English)
# ------------------------------------------------------------

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "conference_rank": "Current rank within the conference",
    "division_rank": "Current rank within the division",
    "wildcard_rank": "Rank among wild-card eligible teams",
    "points": "Total standings points",
    "points_pct": "Points earned as a percentage of possible points",

    "sos_past": "Strength of schedule already played",
    "sos_future": "Strength of remaining schedule",

    "goal_diff": "Total goal differential",
    "home_goal_diff": "Goal differential in home games",
    "road_goal_diff": "Goal differential in road games",

    "corsi_diff_per_game": "Shot attempt differential per game",
    "SCF_pg": "Scoring chances for per game",
    "rush_for_per_game": "Rush chances generated per game",
    "rush_against_per_game": "Rush chances allowed per game",
    "rush_diff_per_game": "Net rush chance differential",

    "pp_pct": "Power play success rate",
    "pk_pct": "Penalty kill success rate",
    "ppng_rank": "League rank in power-play net goals",
    "pk_rank": "League rank in penalty-kill effectiveness",
    "times_sh_pg" : "Average number of times short handed per game",

    "c_shot_pct": "Team shooting percentage over the season",
    "c_sv_pct": "Team save percentage over the season",
    "c_pdo": "Combined shooting and save percentage",

    "wc_points_margin": "Points above or below the 2nd wild-card cutline",
    "wc_points_pct_margin": "Points percentage margin versus WC cutline",
    "wc_teams_within_2pts": "Number of teams within 2 points of WC cutline",
    "wc_teams_within_4pts": "Number of teams within 4 points of WC cutline",
    "wc_teams_within_6pts": "Number of teams within 6 points of WC cutline",
    "in_wc_spot": "Currently occupying a wild-card spot",
    "on_wc_bubble": "Within realistic striking distance of a wild-card spot",
    "in_div_top3": "Currently in a guaranteed divisional playoff position",
}

# ------------------------------------------------------------
# Probability interpretation
# ------------------------------------------------------------

PROBABILITY_TIERS = {
    "Lock":      (0.85, 1.01),
    "Likely":    (0.65, 0.85),
    "Bubble":    (0.45, 0.65),
    "Danger":    (0.25, 0.45),
    "Long Shot": (0.00, 0.25),
}

EARLY_SEASON_NOTE = (
    "Early-season probabilities emphasize structural strength "
    "over results and should be interpreted with caution."
)

# ------------------------------------------------------------
# Helpers for explanation layers
# ------------------------------------------------------------

def feature_to_group(feature: str) -> str:
    for group, feats in FEATURE_GROUPS.items():
        if feature in feats:
            return group
    return "Other"


def explain_shap_row(shap_series, top_n: int = 5):
    """
    Convert a SHAP value Series into grouped narrative drivers.
    """
    drivers = []

    for feat, val in shap_series.sort_values(ascending=False).head(top_n).items():
        drivers.append({
            "feature": feat,
            "group": feature_to_group(feat),
            "impact": float(val),
            "description": FEATURE_DESCRIPTIONS.get(feat, ""),
        })

    return drivers