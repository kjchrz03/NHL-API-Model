import requests
import numpy as np
import pandas as pd
from datetime import date
import os
import time
import random
import duckdb
from IPython.display import display, HTML
import html
import logging

logger = logging.getLogger(__name__)



# ============================================================
# SAFE REQUEST WITH BACKOFF
# ============================================================
def safe_request(url, params=None, max_retries=5):
    delay = 1

    for _ in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=10)

            if r.status_code == 200:
                return r

            if r.status_code == 429:
                time.sleep(delay)
                delay = min(delay * 2, 5)
                continue

            return None

        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 5)

    return None


# ============================================================
# DATABASE CONFIGURATION
# ============================================================
DB_PATH = (
    r"C:\Users\kjchr\Desktop\hockey\new_hockey_env"
    r"\API data\data\duckdb\nhl_data.duckdb"
)

pd.options.mode.chained_assignment = None

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


# ============================================================
# FETCH CURRENT PLAYOFF PREDICTIONS
# ============================================================
def fetch_current_predictions(season_id, model_version="v6"):

    con = None

    try:
        logger.info(
            "Fetching playoff predictions | Season: %s | Model: %s",
            season_id,
            model_version
        )

        con = duckdb.connect(DB_PATH, read_only=True)

        # ====================================================
        # 1. FETCH GAME SUMMARY
        # ====================================================
        daily_summary = con.execute("""
            SELECT *
            FROM all_game_summary
            WHERE season_id = ?
        """, [season_id]).df()

        logger.info(
            "Loaded %s game summary records",
            len(daily_summary)
        )

        # ====================================================
        # 2. CREATE HOME TEAM PERSPECTIVE
        # ====================================================
        home_games = daily_summary[
            [
                "game_id",
                "season_id",
                "game_date",
                "home_tri_code",
                "away_tri_code",
                "homeTeam.score",
                "awayTeam.score",
                "gameState"
            ]
        ].copy()

        home_games = home_games.rename(columns={
            "home_tri_code": "tri_code",
            "away_tri_code": "opponent",
            "homeTeam.score": "team_score",
            "awayTeam.score": "opponent_score"
        })

        home_games["home_away"] = "Home"

        # ====================================================
        # 3. CREATE AWAY TEAM PERSPECTIVE
        # ====================================================
        away_games = daily_summary[
            [
                "game_id",
                "season_id",
                "game_date",
                "away_tri_code",
                "home_tri_code",
                "awayTeam.score",
                "homeTeam.score",
                "gameState"
            ]
        ].copy()

        away_games = away_games.rename(columns={
            "away_tri_code": "tri_code",
            "home_tri_code": "opponent",
            "awayTeam.score": "team_score",
            "homeTeam.score": "opponent_score"
        })

        away_games["home_away"] = "Away"

        # ====================================================
        # 4. COMBINE HOME AND AWAY GAMES
        # ====================================================
        team_games = pd.concat(
            [home_games, away_games],
            ignore_index=True
        )

        team_games["game_date"] = pd.to_datetime(
            team_games["game_date"]
        ).dt.normalize()

        # Remove duplicate game records, if any
        team_games = team_games.drop_duplicates(
            subset=["season_id", "game_id", "tri_code"]
        )

        logger.info(
            "Prepared %s team-game records",
            len(team_games)
        )

        # Check for multiple games per team per day
        duplicate_game_keys = team_games.duplicated(
            subset=["season_id", "game_date", "tri_code"]
        ).sum()

        if duplicate_game_keys:
            logger.warning(
                "Found %s duplicate team/date game keys",
                duplicate_game_keys
            )
        
        # ====================================================
        # 5. FETCH TEAM INFORMATION
        # ====================================================
        teams = con.execute("""
            SELECT *
            FROM all_teams
        """).df()
        
        teams = teams[
            ["team_id","tri_code", "team_name"]
            ].drop_duplicates(subset=["team_id"])
        

        # ====================================================
        # ADD OPPONENT NAMES
        # ====================================================

        opponents = (
            teams[["tri_code", "team_name"]]
            .drop_duplicates(subset=["tri_code"])
            .rename(columns={
                "tri_code": "opponent",
                "team_name": "opponent_name"
            })
        )

        team_games = team_games.merge(
            opponents,
            on="opponent",
            how="left",
            validate="many_to_one"
        )

        # ====================================================
        # 6. FETCH PLAYOFF PREDICTIONS
        # ====================================================
        predictions = con.execute("""
            SELECT *
            FROM playoff_daily_predictions
            WHERE season_id = ?
              AND model_version = ?
        """, [season_id, model_version]).df()

        logger.info(
            "Loaded %s prediction records",
            len(predictions)
        )

        # Avoid introducing duplicate team_name columns
        if "team_name" in predictions.columns:
            predictions = predictions.drop(
                columns=["team_name"]
            )


        predictions = predictions.merge(
            teams[["team_id", "team_name"]],
            on="team_id",
            how="left",
            validate="many_to_one"
        )

        predictions = predictions.sort_values(
            by="game_date"
        ).reset_index(drop=True)

        # ====================================================
        # 7. NORMALIZE DATES
        # ====================================================
        predictions["game_date"] = pd.to_datetime(
            predictions["game_date"]
        ).dt.normalize()

        # ====================================================
        # 8. CHECK PREDICTION KEYS
        # ====================================================
        duplicate_prediction_keys = predictions.duplicated(
            subset=["season_id", "game_date", "tri_code"]
        ).sum()

        if duplicate_prediction_keys:
            logger.warning(
                "Found %s duplicate prediction team/date keys",
                duplicate_prediction_keys
            )

        # ====================================================
        # 9. MERGE PREDICTIONS WITH GAMES
        # ====================================================
        predictions_df = predictions.merge(
            team_games,
            on=["season_id", "game_date", "tri_code"],
            how="left",
            validate="many_to_many"
        )

        logger.info(
            "Merge completed | Predictions: %s | Merged rows: %s",
            len(predictions),
            len(predictions_df)
        )

        if len(predictions_df) > len(predictions):
            logger.warning(
                "Game merge increased row count by %s",
                len(predictions_df) - len(predictions)
            )

        # ====================================================
        # 10. IDENTIFY COMPLETED GAMES
        # ====================================================
        predictions_df["played_game"] = (
            predictions_df["game_id"].notna()
            & predictions_df["gameState"].eq("OFF")
            & predictions_df["team_score"].notna()
            & predictions_df["opponent_score"].notna()
        )

        # ====================================================
        # 11. FORMAT GAME IDS
        # ====================================================
        predictions_df["game_id"] = (
            pd.to_numeric(
                predictions_df["game_id"],
                errors="coerce"
            )
            .astype("Int64")
            .astype("string")
        )

        logger.info(
            "Completed successfully | Total rows: %s | Completed-game rows: %s",
            len(predictions_df),
            predictions_df["played_game"].sum()
        )

        return predictions_df

    except Exception:
        logger.exception(
            "Failed to fetch playoff predictions | Season: %s | Model: %s",
            season_id,
            model_version
        )
        raise

    finally:
        if con is not None:
            con.close()

# ============================================================
# FETCH SHAP VALUES
# ============================================================
def fetch_shap_values(season_id, model_version="v6"):

    con = duckdb.connect(DB_PATH, read_only=True)

    try:
        shap_df = con.execute("""
            SELECT *
            FROM playoff_shap_values_v2
            WHERE season_id = ?
              AND model_version = ?
            ORDER BY game_date, tri_code
        """, [season_id, model_version]).df()

        return shap_df

    finally:
        con.close()

# ============================================================
# FETCH SHAP SUMMARY
# ============================================================

def fetch_shap_summary(season_id, model_version="v6"):

    # Match the version-specific tables in shap_summary.py
    if model_version == "v6":
        table_name = "playoff_shap_summary"
    else:
        import re

        if not re.fullmatch(r"v[0-9]+", model_version):
            raise ValueError(
                f"Invalid model version: {model_version}"
            )

        table_name = f"playoff_shap_summary_{model_version}"

    con = None

    try:
        logger.info(
            "Fetching SHAP summary | Season: %s | Model: %s",
            season_id,
            model_version
        )

        con = duckdb.connect(DB_PATH, read_only=True)

        shap_summary_df = con.execute(f"""
            SELECT
                season_id,
                snapshot_date,
                game_date,
                team_id,
                tri_code,
                model_version,
                playoff_prob,
                feature,
                shap_value,
                direction,
                rank
            FROM "{table_name}"
            WHERE season_id = ?
              AND model_version = ?
            ORDER BY
                snapshot_date,
                tri_code,
                direction,
                rank
        """, [season_id, model_version]).df()

        # Normalize dates
        for col in ["snapshot_date", "game_date"]:
            shap_summary_df[col] = pd.to_datetime(
                shap_summary_df[col]
            ).dt.normalize()

        # Check for duplicate rankings
        duplicate_keys = shap_summary_df.duplicated(
            subset=[
                "season_id",
                "snapshot_date",
                "team_id",
                "model_version",
                "direction",
                "rank"
            ]
        ).sum()

        if duplicate_keys:
            raise ValueError(
                f"Found {duplicate_keys} duplicate "
                "SHAP summary rankings"
            )

        logger.info(
            "Loaded %s SHAP summary records across %s snapshots",
            len(shap_summary_df),
            shap_summary_df[
                ["snapshot_date", "team_id"]
            ].drop_duplicates().shape[0]
        )

        return shap_summary_df

    except Exception:

        logger.exception(
            "Failed to fetch SHAP summary | Season: %s | Model: %s",
            season_id,
            model_version
        )

        raise

    finally:

        if con is not None:
            con.close()

# ============================================================
# EXTRACT TOP POSITIVE AND NEGATIVE SHAP CONTRIBUTORS
# ============================================================
def get_top_shap_features(df, top_n=5):

    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    if df.empty:
        return pd.DataFrame(
            columns=[
                "season_id",
                "snapshot_date",
                "game_date",
                "team_id",
                "tri_code",
                "model_version",
                "feature",
                "shap_value",
                "impact"
            ]
        )

    df = df.copy()

    # Unique identifier for each original prediction record
    df["_record_id"] = np.arange(len(df))

    # Identify all SHAP feature columns
    # Exclude the SHAP base value
    shap_cols = [
        col for col in df.columns
        if col.startswith("shap_")
        and col != "shap_base_value"
    ]

    # Columns identifying each prediction
    id_cols = [
        "_record_id",
        "season_id",
        "snapshot_date",
        "game_date",
        "team_id",
        "tri_code",
        "model_version"
    ]

    # Convert wide SHAP data into long format
    long_df = df.melt(
        id_vars=id_cols,
        value_vars=shap_cols,
        var_name="feature",
        value_name="shap_value"
    )

    # Remove shap_ prefix from feature names
    long_df["feature"] = (
        long_df["feature"]
        .str.removeprefix("shap_")
    )

    # Ensure SHAP values are numeric
    long_df["shap_value"] = pd.to_numeric(
        long_df["shap_value"],
        errors="coerce"
    )

    # Remove missing SHAP values
    long_df = long_df.dropna(
        subset=["shap_value"]
    )

    # --------------------------------------------------------
    # TOP POSITIVE CONTRIBUTORS
    # --------------------------------------------------------
    positive = (
        long_df[long_df["shap_value"] > 0]
        .sort_values(
            "shap_value",
            ascending=False
        )
        .groupby("_record_id")
        .head(top_n)
        .copy()
    )

    positive["impact"] = "Positive"

    # --------------------------------------------------------
    # TOP NEGATIVE CONTRIBUTORS
    # --------------------------------------------------------
    negative = (
        long_df[long_df["shap_value"] < 0]
        .sort_values(
            "shap_value",
            ascending=True
        )
        .groupby("_record_id")
        .head(top_n)
        .copy()
    )

    negative["impact"] = "Negative"

    # --------------------------------------------------------
    # COMBINE RESULTS
    # --------------------------------------------------------
    top_shap = pd.concat(
        [positive, negative],
        ignore_index=True
    )

    top_shap = top_shap.drop(
        columns="_record_id"
    )

    return top_shap

def calculate_shap_consistency(shap_df):

    if shap_df.empty:
        return pd.DataFrame()

    df = shap_df.copy()

    df["game_date"] = pd.to_datetime(df["game_date"])

    # Keep the latest record for each team/date
    if "created_at" in df.columns:
        df = df.sort_values("created_at")

    df = df.drop_duplicates(
        subset=["season_id", "model_version", "tri_code", "game_date"],
        keep="last"
    )

    # Identify all SHAP feature columns
    shap_cols = [
        c for c in df.columns
        if c.startswith("shap_")
        and c != "shap_base_value"
    ]

    # Convert wide SHAP data to long format
    shap_long = df.melt(
        id_vars=[
            "season_id",
            "model_version",
            "tri_code",
            "game_date"
        ],
        value_vars=shap_cols,
        var_name="feature",
        value_name="shap_value"
    )

    shap_long["feature"] = (
        shap_long["feature"].str.removeprefix("shap_")
    )

    shap_long["shap_value"] = pd.to_numeric(
        shap_long["shap_value"],
        errors="coerce"
    )

    shap_long = shap_long.dropna(subset=["shap_value"])

    shap_long = shap_long.sort_values(
        ["season_id", "model_version", "tri_code", "feature", "game_date"]
    )

    group_cols = [
        "season_id",
        "model_version",
        "tri_code",
        "feature"
    ]

    grouped = shap_long.groupby(group_cols)

    # Running number of snapshots
    shap_long["snapshots"] = grouped.cumcount() + 1

    # Positive and negative indicators
    shap_long["positive"] = (
        shap_long["shap_value"] > 0
    ).astype(int)

    shap_long["negative"] = (
        shap_long["shap_value"] < 0
    ).astype(int)

    # Running counts
    shap_long["positive_count"] = (
        shap_long.groupby(group_cols)["positive"].cumsum()
    )

    shap_long["negative_count"] = (
        shap_long.groupby(group_cols)["negative"].cumsum()
    )

    # Running percentages
    shap_long["pct_positive"] = (
        shap_long["positive_count"] / shap_long["snapshots"]
    )

    shap_long["pct_negative"] = (
        shap_long["negative_count"] / shap_long["snapshots"]
    )

    # Running average contribution
    shap_long["avg_shap"] = (
        shap_long.groupby(group_cols)["shap_value"].cumsum()
        / shap_long["snapshots"]
    )

    return shap_long

FEATURE_LABELS = {
    "c_corsi_diff": "Current Corsi Differential",
    "c_hd_goals": "Current High-Danger Goals",
    "c_hd_sog": "Current High-Danger Shots on Goal",
    "c_HDCA": "Current High-Danger Chances Against",
    "c_HDCF": "Current High-Danger Chances For",
    "c_ld_against": "Current Low-Danger Chances Against",
    "c_ld_goals": "Current Low-Danger Goals",
    "c_ld_sog": "Current Low-Danger Shots on Goal",
    "c_LDCA": "Current Low-Danger Chances Against",
    "c_md_goals": "Current Medium-Danger Goals",
    "c_md_sog": "Current Medium-Danger Shots on Goal",
    "c_pdo": "Current PDO",
    "c_pp_ga": "Current Power Play Goals Against",
    "c_pp_net_g": "Current Net Power Play Goals",
    "c_rush_diff": "Current Rush Differential",
    "c_rush_shots_against": "Current Rush Shots Against",
    "c_rush_shots_for": "Current Rush Shots For",
    "c_shot_pct": "Current Shooting Percentage",
    "c_shots_diff": "Current Shot Differential",
    "c_sv_pct": "Current Save Percentage",
    "c_times_sh": "Current Times Shorthanded",
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
        "The percentage of the team's shots on goal that become goals. A higher shooting percentage "
        "means the team converts more of its shots into goals, generally improving playoff chances."
        "allows. A positive difference means the team puts more shots on net than its opponents and "
        "generally improves playoff chances."
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


# ============================================================
# GET FEATURE DESCRIPTION
# ============================================================

def get_feature_description(feature):
    return FEATURE_DESCRIPTIONS.get(
        feature,
        "This statistic helps the model evaluate the team's "
        "playoff chances. Green means it is raising the model's "
        "prediction, while red means it is lowering it."
    )


# ============================================================
# OPTIONAL: TEST FUNCTIONS WHEN RUN DIRECTLY
# ============================================================
if __name__ == "__main__":

    season_id = 20262027
    model_version = "v6"

    # Fetch predictions
    predictions_df = fetch_current_predictions(
        season_id=season_id,
        model_version=model_version
    )

    print(f"Prediction records: {len(predictions_df):,}")

    # Fetch SHAP values
    shap_df = fetch_shap_values(
        season_id=season_id,
        model_version=model_version
    )

    print(f"SHAP records: {len(shap_df):,}")
    
    shap_summary_df = fetch_shap_summary(
        season_id=season_id,
        model_version=model_version
    )

    print(f"SHAP summary records: {len(shap_summary_df):,}")

    print("\nSample SHAP summary:")
    print(
        shap_summary_df.head(10).to_string(index=False)
    )

    # Extract top 5 positive and negative contributors
    top_shap = get_top_shap_features(
        shap_df,
        top_n=5
    )

    print(f"Top SHAP contributor records: {len(top_shap):,}")

    print("\nSample SHAP contributors:")
    print(top_shap.head(10).to_string(index=False))