"""
NHL Playoff Prediction Model — Feature Metadata

Centralized feature definitions shared by:
    - Streamlit dashboard (current_predictions.py)
    - Game-level SHAP explanations
    - Other explainability modules

Contains:
    - Feature labels and descriptions
    - Game-level statistical mappings
    - Game contribution and comparison mappings
    - Per-game and ranking context definitions

This module contains metadata only.
No database connections or calculations are performed here.
"""
# ============================================================
# FEATURE LABELS
# ============================================================

FEATURE_LABELS = {
    "c_corsi_diff": "Cumulative Corsi Differential",
    "c_hd_goals": "Cumulative High-Danger Goals",
    "c_hd_sog": "Cumulative High-Danger Shots on Goal",
    "c_HDCA": "Cumulative High-Danger Chances Against",
    "c_HDCF": "Cumulative High-Danger Chances For",
    "c_ld_against": "Cumulative Low-Danger Chances Against",
    "c_ld_goals": "Cumulative Low-Danger Goals",
    "c_ld_sog": "Cumulative Low-Danger Shots on Goal",
    "c_LDCA": "Cumulative Low-Danger Chances Against",
    "c_md_goals": "Cumulative Medium-Danger Goals",
    "c_md_sog": "Cumulative Medium-Danger Shots on Goal",
    "c_pdo": "Cumulative PDO",
    "c_pp_ga": "Cumulative Power Play Goals Against",
    "c_pp_net_g": "Cumulative Net Power Play Goals",
    "c_rush_diff": "Cumulative Rush Differential",
    "c_rush_shots_against": "Cumulative Rush Shots Against",
    "c_rush_shots_for": "Cumulative Rush Shots For",
    "c_shot_pct": "Cumulative Shooting Percentage",
    "c_shots_diff": "Cumulative Shot Differential",
    "c_sv_pct": "Cumulative Save Percentage",
    "c_times_sh": "Cumulative Times Shorthanded",
    "conf_playoff_chance_baseline": "Conference Playoff Probability Baseline",
    "conference_home_rank": "Conference Home Rank",
    "conference_rank": "Conference Rank",
    "conference_road_rank": "Conference Road Ranking",
    "conference_team_count": "Number of Conference Teams",
    "corsi_diff_per_game": "Corsi Differential per Game",
    "corsi_net_rel": "Relative Net Corsi",
    "corsi_x_season": "Corsi × Season Progress",
    "corsi_x_special_teams": "Corsi × Special Teams",
    "division_home_rank": "Division Home Ranking",
    "division_rank": "Division Ranking",
    "division_road_rank": "Division Road Ranking",
    "games_remaining": "Games Remaining",
    "gf_pctg": "Goals For Percentage",
    "goal_diff": "Goal Differential",
    "goal_diff_pctg": "Goal Differential Percentage",
    "home_ga": "Home Goals Against",
    "home_gf": "Home Goals For",
    "home_goal_diff": "Home Goal Differential",
    "home_gp": "Home Games Played",
    "home_losses": "Home Losses",
    "home_otlosses": "Home Overtime Losses",
    "home_points": "Home Points",
    "home_rotw": "Home Regulation and Overtime Wins",
    "home_rw": "Home Regulation Wins",
    "home_wins": "Home Wins",
    "league_home_rank": "League Home Ranking",
    "league_last_10_rank": "League Ranking — Last 10 Games",
    "league_rank": "League Ranking",
    "league_road_rank": "League Road Ranking",
    "losses": "Losses",
    "ot_games_pct": "Percentage of Games Reaching Overtime",
    "ot_losses": "Overtime Losses",
    "ot_wins": "Overtime Wins",
    "overtimes": "Overtime Games",
    "pdo_x_corsi": "PDO × Corsi",
    "penalties_x_season": "Penalties × Season Progress",
    "penalty_risk": "Penalty Risk",
    "pk_load_x_pk_pct": "Penalty Kill Workload × Kill Percentage",
    "pk_net_gpg": "Net Penalty Kill Goals per Game",
    "pk_net_pct": "Net Penalty Kill Percentage",
    "pk_stress_x_season": "Penalty Kill Stress × Season Progress",
    "points_pct": "Points Percentage",
    "pp_ga_pg": "Power Play Goals Against per Game",
    "pp_gpg": "Power Play Goals per Game",
    "pp_net_gpg": "Net Power Play Goals per Game",
    "pp_net_pct": "Net Power Play Percentage",
    "pp_opp_pg": "Power Play Opportunities per Game",
    "ppng_rank": "Net Power Play Goals Ranking",
    "projected_points": "Projected Season Points",
    "pt_pct_x_sos": "Points Percentage × Strength of Schedule",
    "road_ga": "Road Goals Against",
    "road_gf": "Road Goals For",
    "road_goal_diff": "Road Goal Differential",
    "road_gp": "Road Games Played",
    "road_losses": "Road Losses",
    "road_ot_losses": "Road Overtime Losses",
    "road_points": "Road Points",
    "road_rotw": "Road Regulation and Overtime Wins",
    "road_rw": "Road Regulation Wins",
    "road_wins": "Road Wins",
    "rotw": "Regulation and Overtime Wins",
    "rotw_pctg": "Regulation and Overtime Win Percentage",
    "rush_against_per_game": "Rush Chances Against per Game",
    "rush_balance_ratio": "Rush Chance Balance Ratio",
    "rush_diff_per_game": "Rush Differential per Game",
    "rush_for_per_game": "Rush Chances For per Game",
    "rush_net_x_season": "Season-Adjusted Net Rush Chances",
    "rush_x_special_teams": "Rush Chances × Special Teams",
    "rw": "Regulation Wins",
    "SCA_pg": "Scoring Chances Against per Game",
    "sc_net": "Net Scoring Chances",
    "SCF_pg": "Scoring Chances For per Game",
    "sh_ga_pg": "Shorthanded Goals Against per Game",
    "sh_gf_pg": "Shorthanded Goals For per Game",
    "shot_attempts_against_per_game": "Shot Attempts Against per Game",
    "shot_attempts_for_per_game": "Shot Attempts For per Game",
    "shots_diff_per_game": "Shot Differential per Game",
    "so_losses": "Shootout Losses",
    "so_wins": "Shootout Wins",
    "sos_future": "Remaining Strength of Schedule",
    "sos_past": "Past Strength of Schedule",
    "special_teams_net": "Net Special Teams Performance",
    "special_teams_x_season": "Special Teams × Season Progress",
    "times_sh_pg": "Times Shorthanded per Game",
    "urgency": "Playoff Urgency",
    "urgency_x_points": "Playoff Urgency × Points",
    "wc2_points": "Second Wild Card Team Points",
    "wc2_points_pct": "Second Wild Card Team Points Percentage",
    "wc_points_margin": "Wild Card Points Margin",
    "wc_points_pct_margin": "Wild Card Points Percentage Margin",
    "wc_pressure_x_performance": "Wild Card Pressure × Performance",
    "wc_teams_within_2pts": "Wild Card Teams Within 2 Points",
    "wc_teams_within_4pts": "Wild Card Teams Within 4 Points",
    "wc_teams_within_6pts": "Wild Card Teams Within 6 Points",
    "wildcard_rank": "Wild Card Ranking",
    "win_pct_before": "Previous Win Percentage",
    "wins": "Wins",
    "wins_before": "Previous Wins",
    "wins_pctg": "Win Percentage",
}
def get_feature_label(feature):
    return FEATURE_LABELS.get(
        feature,
        feature.replace("_", " ").title()
    )


# ============================================================
# FEATURE DESCRIPTIONS
# ============================================================

FEATURE_DESCRIPTIONS = {

    # --------------------------------------------------------
    # CURRENT PERFORMANCE
    # --------------------------------------------------------

    "c_corsi_diff": (
        "Corsi is a way to measure a team's possession of the puck. This feature compares all shot attempts made by the team to all shot attempts made against the team over the course of the season. "
        "If the team's cumulative shot attempts for exceeds cumulative shots against, playoff chances improve."
    ),

    "c_hd_goals": (
        "The cumulative number of goals for the team from high danger scoring areas. The more goals scored here by the team improves playoff chances."
    ),

    "c_hd_sog": (
        "A cumulative count of shots on goal by the team from dangerous scoring areas. More shots on goal in this location improves playoff chances."
    ),

    "c_HDCA": (
        "The cumulative number of high danger scoring chances against the team over the course of the season. More scoring chances against the team hurts playoff chances. "
    ),

    "c_HDCF": (
        "The cumulative number of high-danger scoring chances created by the team over the season. "
        "Creating more chances from dangerous areas generally improves playoff chances."
    ),

    "c_ld_against": (
        "Tracks the team's cumulative low-danger opportunities allowed to opponents. Allowing fewer "
        "opportunities from these areas generally supports playoff chances, although the impact is "
        "usually smaller than allowing high-danger chances."
    ),

    "c_ld_goals": (
      "The cumulative number of goals scored by the team from low-danger areas. Scoring more goals "
        "from these locations improves playoff chances."
    ),

    "c_ld_sog": (
          "The cumulative number of shots on goal taken by the team from low-danger areas. More shots "
        "can support playoff chances, but these attempts are less likely to become goals than "
        "high-danger shots."
    ),

    "c_LDCA": (
        "The cumulative number of low-danger scoring chances allowed to opponents. Allowing fewer "
        "chances helps limit opposing offense and generally supports playoff chances."
    ),

    "c_md_goals": (
        "The cumulative number of goals scored by the team from medium-danger areas. More goals from "
        "these areas improve playoff chances."
    ),

    "c_md_sog": (
        "The cumulative number of shots on goal taken by the team from medium-danger areas. "
        "Generating more of these shots generally supports playoff chances by creating scoring "
        "opportunities."
    ),

    "c_pdo": (
       "PDO combines a team's shooting percentage and save percentage to show how effectively it "
        "scores on its shots and stops opponents' shots. A higher PDO generally supports playoff "
        "chances, although unusually high or low values can be difficult to sustain."
    ),

    "c_pp_ga": (
        "The cumulative number of goals allowed by the team during its own power plays. Allowing "
        "fewer shorthanded goals to opponents helps protect scoring advantages and improves playoff "
        "chances."
    ),

    "c_pp_net_g": (
        "The cumulative difference between goals scored and goals allowed during the team's power "
        "plays. A larger positive difference means the power play is producing more goals than it "
        "concedes, improving playoff chances."
    ),

    "c_rush_diff": (
        "The cumulative difference between rush opportunities created by the team and those allowed "
        "to opponents. Creating more rush opportunities than the team allows generally improves "
        "playoff chances."
    ),
    
    "c_rush_shots_against": (
         "The cumulative number of shots on goal allowed when opponents attack off the rush. Allowing "
        "fewer rush shots reduces dangerous scoring opportunities and generally improves playoff "
        "chances."
    ),

    "c_rush_shots_for": (
        "The cumulative number of shots on goal generated when the team attacks off the rush. More "
        "rush shots create additional scoring opportunities and generally improve playoff chances."

    ),

    "c_shot_pct": (
         "The percentage of the team's shots on goal that become goals. "
        "A higher shooting percentage means the team converts more of its "
        "shots into goals, generally improving playoff chances."
    ),

    "c_shots_diff": (
        "The cumulative difference between shots on goal for and against. "
        "A positive difference means the team puts more shots on net than "
        "its opponents, generally supporting playoff chances."
    ),
        
    # --------------------------------------------------------
    # PLAYOFF RACE AND URGENCY
    # --------------------------------------------------------

    "urgency": (
        "Measures the pressure the team faces based on its position in the playoff race. Greater "
        "urgency can indicate that the team needs points quickly, but does not by itself improve "
        "playoff chances; the impact depends on the team's standings position."
    ),

    "urgency_x_points": (
        "Combines the team's playoff-race urgency with its standings points. It helps the model "
        "assess how much the team's point total matters under its current playoff pressure, rather "
        "than treating urgency alone as a positive or negative sign."
    ),

    "wc2_points": (
        "The number of standings points held by the team in the second wild-card playoff position. A "
        "higher cutoff means teams chasing that spot generally need more points, which can make "
        "qualifying harder."
    ),

    "wc2_points_pct": (
        "The points percentage of the team holding the second wild-card playoff position. A higher "
        "cutoff percentage indicates a stronger pace is needed to qualify and can make the playoff "
        "race more difficult."
    ),

    "wc_points_margin": (
        "The difference in standings points between the team and the wild-card cutoff. A more "
        "positive margin means the team is farther above the cutoff, generally improving playoff "
        "chances; a negative margin means it is chasing a playoff spot."
    ),

    "wc_points_pct_margin": (
        "The difference between the team's points percentage and the wild-card cutoff points "
        "percentage. A more positive margin generally improves playoff chances and helps compare "
        "teams that have played different numbers of games."
    ),

    "wc_pressure_x_performance": (
        "Combines the pressure of the wild-card race with the team's current performance. Strong "
        "performance under greater playoff pressure generally supports playoff chances, while falling "
        "behind in a tight race hurts them."
    ),

    "wc_teams_within_2pts": (
        "The number of teams within two standings points of the wild-card cutoff. More teams close to "
        "the cutoff generally indicate a tighter playoff race and more competition for the available "
        "spots."
    ),

    "wc_teams_within_4pts": (
        "The number of teams within four standings points of the wild-card cutoff. More teams in this "
        "range generally indicate greater competition for wild-card positions."
    ),

    "wc_teams_within_6pts": (
        "The number of teams within six standings points of the wild-card cutoff. More teams in this "
        "range suggest a crowded playoff race, although not every team is equally likely to qualify."
    ),

    "wildcard_rank": (
        "The team's current ranking in the wild-card race. A better ranking generally improves "
        "playoff chances because the team is closer to securing one of the available playoff spots."
    ),
}



def get_feature_description(feature):
    return FEATURE_DESCRIPTIONS.get(
        feature,
        "This statistic helps the model evaluate the team's "
        "playoff chances. Green means it is raising the model's "
        "prediction, while red means it is lowering it."
    )

# ============================================================
# GAME-LEVEL FEATURE STATISTICS
# ============================================================

FEATURE_GAME_STATS = {
    # Team performance
    "goal_diff": ["goals_for", "goals_against", "game_goal_diff"],
    "points_pct": ["game_win", "points", "points_pct"],
    "win_pct_before": ["wins_before", "games_before"],
    "c_sv_pct": ["game_savePct", "shots_against", "sv_pct_vs_avg"],

    # Rankings
    "league_rank": ["league_rank"],
    "division_rank": ["division_rank"],
    "conference_rank": ["conference_rank"],
    "division_road_rank": ["division_road_rank"],
    "division_home_rank": ["division_home_rank"],
    "league_road_rank": ["league_road_rank"],
    "league_home_rank": ["league_home_rank"],

    # Rush chances
    "rush_for_per_game": [
        "rush_shots_for",
        "rush_for_vs_avg",
    ],
    "rush_against_per_game": [
        "rush_shots_against",
        "rush_against_vs_avg",
    ],
    "rush_diff_per_game": [
        "rush_shot_differential",
        "rush_diff_vs_avg",
    ],

    # Scoring chances
    "sc_net": ["SCF", "SCA", "game_sc_net"],
    "SCF_pg": ["SCF", "scf_vs_avg"],
    "SCA_pg": ["SCA", "sca_vs_avg"],

    # Shot attempts
    "corsi_diff_per_game": [
        "corsi_for",
        "corsi_against",
        "game_corsi_diff",
        "corsi_diff_vs_avg",
    ],
    "shot_attempts_for_per_game": [
        "shot_attempts_for",
        "shot_attempts_for_vs_avg",
    ],
    "shot_attempts_against_per_game": [
        "shot_attempts_against",
        "shot_attempts_against_vs_avg",
    ],
    "shots_diff_per_game": [
        "shots_for",
        "shots_against",
        "game_shots_diff",
        "shots_diff_vs_avg",
    ],

    # Special teams
    "c_pp_net_g": ["pp_gf", "sh_ga", "pp_opp", "game_pp_net"],
    "pp_net_gpg": ["pp_gf", "sh_ga", "game_pp_net"],
    "pp_net_pct": ["pp_gf", "sh_ga", "pp_opp", "game_pp_net_pct"],
    "pk_net_gpg": ["sh_gf", "pp_ga", "times_sh", "game_pk_net"],
    "pk_net_pct": ["sh_gf", "pp_ga", "times_sh", "game_pk_net_pct"],
    "special_teams_net": [
        "game_pp_net",
        "game_pk_net",
        "game_special_teams_net",
    ],
}

# ============================================================
# GAME CONTRIBUTION MAPPINGS
# ============================================================

GAME_CONTRIBUTION_VALUES = {
    # Goals and results
    "goal_diff": "game_goal_diff",
    "points_pct": "game_points_earned",
    "win_pct_before": "game_win",
    "wins": "game_win",
    "wins_before": "game_win",

    # Shots and scoring chances
    "c_shots_diff": "game_shots_diff",
    "c_corsi_diff": "game_corsi_diff",
    "sc_net": "game_sc_net",
    "c_rush_diff": "rush_shot_differential",

    # Special teams
    "c_pp_net_g": "game_pp_net",
    "special_teams_net": "game_special_teams_net",
}


# ============================================================
# CUMULATIVE RATE COMPARISONS
# ============================================================

CUMULATIVE_RATE_VALUES = {
    "c_sv_pct": "game_savePct",
    "c_shot_pct": "game_shootingPct",
    "c_pdo": "game_pdo",
    "points_pct": "game_points_pct",
    "wins_pctg": "game_win",
    "gf_pctg": "game_gf_pct",
    "pp_net_pct": "game_pp_net_pct",
    "pk_net_pct": "game_pk_net_pct",
}


# ============================================================
# PER-GAME AVERAGE COMPARISONS
# ============================================================

GAME_AVERAGE_VALUES = {
    # Rush performance
    "rush_for_per_game": "rush_shots_for",
    "rush_against_per_game": "rush_shots_against",
    "rush_diff_per_game": "rush_shot_differential",

    # Scoring chances
    "SCF_pg": "SCF",
    "SCA_pg": "SCA",

    # Shot attempts
    "corsi_diff_per_game": "game_corsi_diff",
    "shot_attempts_for_per_game": "shot_attempts_for",
    "shot_attempts_against_per_game": "shot_attempts_against",
    "shots_diff_per_game": "game_shots_diff",

    # Special teams
    "pp_gpg": "pp_gf",
    "pp_ga_pg": "pp_ga",
    "pp_net_gpg": "game_pp_net",
    "pp_opp_pg": "pp_opp",
    "pk_net_gpg": "game_pk_net",
    "sh_gf_pg": "sh_gf",
    "sh_ga_pg": "sh_ga",
    "times_sh_pg": "times_sh",
}


# ============================================================
# GAME RATE VALUES
# ============================================================

GAME_RATE_VALUES = {
    "c_sv_pct": "game_savePct",
    "c_shot_pct": "game_shootingPct",
    "c_pdo": "game_pdo",
    "points_pct": "game_points_pct",
    "wins_pctg": "game_win",
    "pp_net_pct": "game_pp_net_pct",
    "pk_net_pct": "game_pk_net_pct",
}

# ============================================================
# PER-GAME FEATURE CONTEXT
# ============================================================

PER_GAME_CONTEXT = {
    # Rush chances
    "rush_for_per_game": (
        "rush_shots_for",
        "rush chances generated"
    ),
    "rush_against_per_game": (
        "rush_shots_against",
        "rush chances allowed"
    ),
    "rush_diff_per_game": (
        "rush_shot_differential",
        "net rush chances"
    ),

    # Scoring chances
    "SCF_pg": (
        "SCF",
        "scoring chances generated"
    ),
    "SCA_pg": (
        "SCA",
        "scoring chances allowed"
    ),

    # Shot attempts
    "corsi_diff_per_game": (
        "game_corsi_diff",
        "net shot attempts"
    ),
    "shot_attempts_for_per_game": (
        "shot_attempts_for",
        "shot attempts generated"
    ),
    "shot_attempts_against_per_game": (
        "shot_attempts_against",
        "shot attempts allowed"
    ),
    "shots_diff_per_game": (
        "game_shots_diff",
        "net shots on goal"
    ),

    # Special teams
    "pp_gpg": (
        "pp_gf",
        "power-play goals scored"
    ),
    "pp_ga_pg": (
        "pp_ga",
        "power-play goals allowed"
    ),
    "pp_net_gpg": (
        "game_pp_net",
        "net power-play goals"
    ),
    "pp_opp_pg": (
        "pp_opp",
        "power-play opportunities"
    ),
    "pk_net_gpg": (
        "game_pk_net",
        "net penalty-kill goals"
    ),
    "sh_gf_pg": (
        "sh_gf",
        "shorthanded goals scored"
    ),
    "sh_ga_pg": (
        "sh_ga",
        "shorthanded goals allowed"
    ),
    "times_sh_pg": (
        "times_sh",
        "times shorthanded"
    ),
}


# ============================================================
# RANKING FEATURE CONTEXT
# ============================================================

RANK_CONTEXT = {
    # League rankings
    "league_rank": "league standings",
    "league_home_rank": "league home standings",
    "league_road_rank": "league road standings",
    "league_last_10_rank": "league standings over the last 10 games",

    # Conference rankings
    "conference_rank": "conference standings",
    "conference_home_rank": "conference home standings",
    "conference_road_rank": "conference road standings",

    # Division rankings
    "division_rank": "division standings",
    "division_home_rank": "division home standings",
    "division_road_rank": "division road standings",

    # Wild-card rankings
    "wildcard_rank": "wild-card race",

    # Special teams rankings
    "ppng_rank": "net power-play goals rankings",
}