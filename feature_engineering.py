# ============================================================
# Feature Engineering — Playoff Model (v5)
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# Core helpers
# ============================================================

def normalize_made_playoffs(df: pd.DataFrame) -> pd.DataFrame:
    if "made_playoffs" not in df.columns:
        return df

    final_outcomes = (
        df
        .groupby(["season_id", "tri_code"])["made_playoffs"]
        .apply(lambda s: 1 if (s == 1).any() else 0)
        .reset_index()
    )

    return (
        df
        .drop(columns=["made_playoffs"])
        .merge(final_outcomes, on=["season_id", "tri_code"], how="left")
    )


def add_conference_baseline(df: pd.DataFrame) -> pd.DataFrame:
    conference_sizes = (
        df.groupby(["season_id", "conference"])["tri_code"]
        .nunique()
        .reset_index()
        .rename(columns={"tri_code": "conference_team_count"})
    )

    df = df.merge(conference_sizes, on=["season_id", "conference"], how="left")

    df["conf_playoff_chance_baseline"] = (
        8 / df["conference_team_count"].replace(0, np.nan)
    )

    return df


def add_streak_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "streak" not in df.columns or "streak_count" not in df.columns:
        df["streak_signed"] = 0
        return df

    df["streak_direction"] = (
        df["streak"]
        .map({"W": 1, "L": -1})
        .fillna(0)
        .astype(int)
    )

    df["streak_signed"] = (
        df["streak_direction"]
        * df["streak_count"].fillna(0).astype(int)
    ).clip(-7, 7)

    df.drop(columns=["streak", "streak_direction"], inplace=True, errors="ignore")
    return df


def add_rank_bucket(df: pd.DataFrame) -> pd.DataFrame:
    def bucket(rank):
        if rank <= 3:
            return "tier1_elite"
        elif rank <= 6:
            return "tier2_strong"
        elif rank <= 10:
            return "tier3_bubble"
        else:
            return "tier4_bottom"

    df = df.copy()
    df["rank_bucket"] = df["conference_rank"].apply(bucket)
    return df


# ============================================================
# Wild-card field pressure
# ============================================================

def add_wc_field_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    wc_eligible = out[out["division_rank"] > 3].copy()

    wc_cutline = (
        wc_eligible
        .sort_values(
            ["season_id", "game_date", "conference", "points", "points_pct"],
            ascending=[True, True, True, False, False]
        )
        .groupby(["season_id", "game_date", "conference"], as_index=False)
        .nth(1)
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

    out["in_wc_spot"] = (
    (out["wildcard_rank"] <= 2)
    .fillna(False)
    .astype(int)
    )

    out["in_div_top3"] = (
        (out["division_rank"] <= 3)
        .fillna(False)
        .astype(int)
    )

    out["on_wc_bubble"] = (
        out["wildcard_rank"]
        .between(1, 4)
        .fillna(False)
        .astype(int)
    )

    return out


def add_wc_density_features(df: pd.DataFrame, windows=(2, 4, 6)) -> pd.DataFrame:
    out = df.copy()

    wc_mask = out["division_rank"] > 3

    for w in windows:
        out[f"wc_teams_within_{w}pts"] = (
            out
            .assign(diff=lambda x: (x["points"] - x["wc2_points"]).abs())
            .where(wc_mask)
            .groupby(["season_id", "game_date", "conference"])["diff"]
            .transform(lambda s: (s <= w).sum())
            .fillna(0)
            .astype(int)
        )

    return out


# ============================================================
# Special teams
# ============================================================
def add_special_teams_features(df):
    df = df.copy()

    def safe_col(col, default=0.0):
        return df[col] if col in df.columns else default

    # -----------------------------
    # Core strength
    # -----------------------------
    df["special_teams_net"] = (
        safe_col("pp_net_gpg") + safe_col("pk_net_gpg")
    )

    # -----------------------------
    # Discipline
    # -----------------------------
    df["penalty_burden"] = safe_col("times_sh_pg")

    # -----------------------------
    # Efficiency
    # -----------------------------
    df["pp_eff_x_volume"] = (
        safe_col("pp_pct") * safe_col("pp_opp_pg")
    )

    df["pk_eff_x_volume"] = (
        safe_col("pk_pct") * safe_col("times_sh_pg")
    )

    # -----------------------------
    # Risk
    # -----------------------------
    df["penalty_risk"] = (
        safe_col("pp_ga_pg") / (safe_col("times_sh_pg") + 1e-6)
    )

    # -----------------------------
    # Interaction
    # -----------------------------
    df["discipline_x_pk"] = (
        safe_col("times_sh_pg") * safe_col("pk_pct")
    )

    df["pk_stress_handling"] = safe_col("pk_load_x_pk_pct")

    return df
# ============================================================
# Season context
# ============================================================

def add_season_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["season_pct"] = df["games_played"] / 82

    return df

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def safe_col(col, default=0.0):
        return df[col] if col in df.columns else default

    # Special teams importance changes over season
    df["special_teams_x_season"] = (
        safe_col("special_teams_net") * safe_col("season_pct")
    )

    # WC pressure x performance
    df["wc_pressure_x_performance"] = (
        safe_col("wc_points_pct_margin") * safe_col("season_pct")
    )

    # Discipline becomes more costly late
    df["penalties_x_season"] = (
        safe_col("times_sh_pg") * safe_col("season_pct")
    )

    # PK stress interaction
    df["pk_stress_x_season"] = (
        safe_col("pk_load_x_pk_pct") * safe_col("season_pct")
    )

    return df
# ============================================================
# Rush
# ============================================================

def add_rush_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def safe_col(col, default=0.0):
        return df[col] if col in df.columns else default

    # Core strength
    df["rush_net"] = safe_col("rush_diff_per_game")

    # Balance
    df["rush_balance_ratio"] = (
        safe_col("rush_for_per_game") /
        (safe_col("rush_against_per_game") + 1e-6)
    )

    # Optional relative (only if exists)
    if "league_avg_rush_diff" in df.columns:
        df["rush_diff_rel"] = (
            safe_col("rush_diff_per_game") - df["league_avg_rush_diff"]
        )
    else:
        df["rush_diff_rel"] = 0.0

    return df


def add_rush_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def safe_col(col, default=0.0):
        return df[col] if col in df.columns else default

    df["rush_net_x_season"] = (
        safe_col("rush_diff_per_game") * safe_col("season_pct")
    )

    df["rush_x_special_teams"] = (
        safe_col("rush_diff_per_game") * safe_col("special_teams_net")
    )

    return df

# ============================================================
# Corsi (possession backbone)
# ============================================================

def add_corsi_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def safe(col, default=0.0):
        return df[col] if col in df.columns else default

    df["corsi_net"] = safe("corsi_diff_per_game")

    df["corsi_ratio"] = (
        safe("corsi_for") /
        (safe("corsi_for") + safe("corsi_against") + 1e-6)
    )

    return df
def add_corsi_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    group = ["season_id", "game_date"]

    for col in ["corsi_net", "corsi_ratio"]:
        if col in df.columns:
            df[f"{col}_rel"] = df[col] - df.groupby(group)[col].transform("mean")
        else:
            df[f"{col}_rel"] = 0.0

    return df
def add_corsi_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def safe(col, default=0.0):
        return df[col] if col in df.columns else default

    df["corsi_x_season"] = safe("corsi_net") * safe("season_pct")
    df["corsi_x_special_teams"] = safe("corsi_net") * safe("special_teams_net")
    df["corsi_x_pdo"] = safe("corsi_net") * safe("pdo_dev")

    return df
# ============================================================
# Shot quality
# ============================================================

def add_shot_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def safe(col, default=0.0):
        return df[col] if col in df.columns else default

    df["sc_net"] = safe("SCF_pg") - safe("SCA_pg")
    df["hd_net"] = safe("HDCF") - safe("HDCA")

    df["danger_net"] = (
        (safe("HDCF") * 3 + safe("MDCF") * 2 + safe("LDCF"))
        - (safe("HDCA") * 3 + safe("MDCA") * 2 + safe("LDCA"))
    )

    df["pdo_dev"] = safe("pdo") - 1.0

    return df

def add_shot_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    group = ["season_id", "game_date"]

    for col in ["danger_net", "pdo_dev"]:
        if col in df.columns:
            df[f"{col}_rel"] = df[col] - df.groupby(group)[col].transform("mean")
        else:
            df[f"{col}_rel"] = 0.0

    return df

def add_shot_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def safe(col, default=0.0):
        return df[col] if col in df.columns else default

    df["pdo_x_corsi"] = safe("pdo_dev") * safe("corsi_net")
    df["danger_x_season"] = safe("danger_net") * safe("season_pct")

    return df

# ============================================================
# Standings
# ============================================================

def add_standings_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def safe(col, default=0.0):
        return df[col] if col in df.columns else default

    df["goal_diff_per_game"] = safe("goal_diff") / (safe("games_played") + 1e-6)

    df["last10_points_pct"] = (
        safe("last_10_points") / (safe("last_10_gp") * 2 + 1e-6)
    )

    df["momentum"] = df["last10_points_pct"] - safe("points_pct")

    df["rank_pressure"] = safe("conference_rank") - 8
    df["urgency"] = (1 - safe("season_pct")) * df["rank_pressure"]

    return df

def add_standings_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def safe(col, default=0.0):
        return df[col] if col in df.columns else default

    df["momentum_x_season"] = safe("momentum") * safe("season_pct")
    df["urgency_x_points"] = safe("urgency") * safe("points_pct")

    return df
# ============================================================
# MASTER
# ============================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # -----------------------------
    # Core features
    # -----------------------------
    out = normalize_made_playoffs(out)
    out = add_conference_baseline(out)
    out = add_streak_features(out)
    out = add_rank_bucket(out)

    # -----------------------------
    # Wildcard pressure
    # -----------------------------
    out = add_wc_field_pressure_features(out)
    out = add_wc_density_features(out)

    # -----------------------------
    # Special teams + context
    # -----------------------------
    out = add_special_teams_features(out)
    out = add_season_context(out)
    out = add_interaction_features(out)

    # -----------------------------
    # Rush
    # -----------------------------
    out = add_rush_features(out)
    out = add_rush_interactions(out)

    # -----------------------------
    # Possession (Corsi)
    # -----------------------------
    out = add_corsi_features(out)
    out = add_corsi_relative_features(out)
    out = add_corsi_interactions(out)

    # -----------------------------
    # Shot quality
    # -----------------------------
    out = add_shot_quality_features(out)
    out = add_shot_relative_features(out)
    out = add_shot_interactions(out)

    # -----------------------------
    # Standings
    # -----------------------------
    out = add_standings_features(out)
    out = add_standings_interactions(out)

    # -----------------------------
    # FINAL CLEANING (VERY IMPORTANT)
    # -----------------------------
    # ------------------------------------------------------------
    # Drop cumulative + redundant + rank columns
    # ------------------------------------------------------------
# # ------------------------------------------------------------
# # Drop cumulative + redundant + rank + directional + extras
# # ------------------------------------------------------------

#     drop_cumulative = [col for col in out.columns if col.startswith("c_")]
#     # ============================================================
#     # FINAL CLEANING (OPTIMIZED)
#     # ============================================================

#     drop_cumulative = [col for col in out.columns if col.startswith("c_")]

#     # -----------------------------------
#     # Redundant / raw stats
#     # -----------------------------------
#     drop_redundant = [
#         "wins",
#         "losses",
#         "ot_losses",
#         "so_wins",
#         "so_losses",
#         "points",
#         # "games_played",
#         "wins_pctg",
#         "games_before",
#         "shot_attempts_for",
#         "shot_attempts_against",
#         "shots_for",
#         "shots_against",
#         "rush_shots_for",
#         "rush_shots_against",
#         "rush_shot_differential",
#         "SCF",
#         "SCA",
#         "SCSF",
#         "SCGF",
#         "HDCF",
#         "HDCA",
#         "MDCF",
#         "MDCA",
#         "LDCF",
#         "LDCA",
#         "goals_for",
#         "goals_against",
#         "home_gp",
#         "home_wins",
#         "home_losses",
#         "home_otlosses",
#         "home_points",
#         "road_gp",
#         "road_wins",
#         "road_losses",
#         "road_ot_losses",
#         "road_points",
#     ]

#     # -----------------------------------
#     # Rank leakage
#     # -----------------------------------
#     drop_ranks = [
#         "ppng_rank",
#         "pk_rank",
#         "division_rank",
#         "division_home_rank",
#         "division_road_rank",
#         "division_last_10_rank",
#         "conference_rank",
#         "conference_home_rank",
#         "conference_road_rank",
#         "conference_last_10_rank",
#         "league_rank",
#         "league_home_rank",
#         "league_road_rank",
#         "league_last_10_rank",
#         "wildcard_rank",
#     ]

#     # -----------------------------------
#     # Directional duplicates
#     # -----------------------------------
#     drop_directional = [
#         "rush_for_per_game",
#         "rush_against_per_game",
#         "shot_attempts_for_per_game",
#         "shot_attempts_against_per_game",
#         "pp_opp_pg",
#         "SCF_pg",
#         "SCA_pg",
#     ]

#     # -----------------------------------
#     # Special teams duplication
#     # -----------------------------------
#     drop_special_teams_extra = [
#         "pp_gpg",
#         "pp_pct",
#         "pp_net_gpg",
#         "pk_pct",
#         "pk_net_gpg",
#     ]

#     # -----------------------------------
#     # Interaction overload
#     # -----------------------------------
#     drop_interactions = [
#         "corsi_x_season",
#         "corsi_x_special_teams",
#         "corsi_x_pdo",
#         "pdo_x_corsi",
#         "danger_x_season",
#         "momentum_x_season",
#         "urgency_x_points",
#     ]

#     # -----------------------------------
#     # Standings overload
#     # -----------------------------------
#     drop_standings_overload = [
#         "wins_before",
#         "rotw",
#         "rotw_pctg",
#         "rw",
#         "projected_points",
#         "goal_diff",
#         "goal_diff_pctg",
#     ]

#     # -----------------------------------
#     # Home / road splits
#     # -----------------------------------
#     drop_home_road = [
#         "home_rotw", "home_rw", "home_gf", "home_ga", "home_goal_diff",
#         "road_rotw", "road_rw", "road_gf", "road_ga", "road_goal_diff",
#     ]

#     # -----------------------------------
#     # Last 10 redundancy
#     # -----------------------------------
#     drop_last10 = [
#         "last_10_wins",
#         "last_10_losses",
#         "last_10_points",
#         "last_10_gf",
#         "last_10_ga",
#         "last_10_goal_diff",
#         "last_10_rw",
#     ]

#     # -----------------------------------
#     # Misc noise
#     # -----------------------------------
#     drop_misc_noise = [
#         "games_remaining",
#         "gf_pctg",
#         "ot_wins",
#         "overtimes",
#         "ot_games_pct",
#         "pt_pct_x_sos",
#         "gaa_x_sv",
#     ]

#     # -----------------------------------
#     # FINAL MICRO TRIM (important)
#     # -----------------------------------
#     drop_final_micro = [
#         "pk_eff_x_volume",
#         "pp_eff_x_volume",
#         "rush_balance_ratio",
#         "rush_x_special_teams",
#         "rank_pressure",
#     ]

#     # -----------------------------------
#     # Remove low-signal constants
#     # -----------------------------------
#     drop_constant_context = [
#         "conference_team_count",
#         "conf_playoff_chance_baseline",
#     ]

#     # ============================================================
#     # APPLY DROP
#     # ============================================================

#     # out = out.drop(
#     #     columns=(
#     #         drop_cumulative
#     #         + drop_redundant
#     #         + drop_ranks
#     #         + drop_directional
#     #         + drop_special_teams_extra
#     #         + drop_interactions
#     #         + drop_standings_overload
#     #         + drop_home_road
#     #         + drop_last10
#     #         + drop_misc_noise
#     #         + drop_final_micro
#     #         + drop_constant_context
#     #     ),
#     #     errors="ignore"
#     # )

#     # ============================================================
#     # FINAL FILL
#     # ============================================================

#     # out = out.fillna(0)
#     # out = out.drop(
#     #     columns=(
#     #         drop_cumulative
#     #         + drop_redundant
#     #         + drop_ranks
#     #         + drop_directional
#     #         + drop_special_teams_extra
#     #         + drop_interactions
#     #         + drop_standings_overload
#     #         + drop_home_road
#     #         + drop_last10
#     #         + drop_misc_noise
#     #     ),
#     #     errors="ignore"
#     # )


    
#     # out = out.fillna(0)

    return out