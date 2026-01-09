import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import duckdb


def run_prediction_data(
    *,
    con: duckdb.DuckDBPyConnection,
    season_id: int,
):

    # ============================================================
    # Fetch Data
    # ============================================================
    def load_last_game_summary_id(season_id: int) -> int | None:
            query = """
                SELECT MAX(game_id)
                FROM all_game_summary
                WHERE season_id = ?
                AND gameState = 'OFF'
            """
            result = con.execute(query, [season_id]).fetchone()
            max_id = result[0] if result else None
            print(f"Max completed GS game_id: {max_id}")
            return max_id
    
    def load_last_game(season_id: int) -> int | None:
                query = """
                    SELECT MAX(game_id)
                    FROM prediction_data
                    WHERE season_id = ?
                """
                result = con.execute(query, [season_id]).fetchone()
                max_id = result[0] if result else None
                print(f"Max prediction game_id: {max_id}")
                return max_id
    
    def load_game_summary_df(season_id: int) -> pd.DataFrame:
        query = """
            SELECT *
            FROM all_game_summary
            WHERE season_id = ?
              AND gameState = 'OFF'
            ORDER BY game_id
        """
        game_summary_df = con.execute(query, [season_id]).df()
        return game_summary_df
    
    def load_teams(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        return con.execute("""
            SELECT team_id, tri_code
            FROM all_teams
        """).df()

            
    def fetch_tables(con, season_id: int) -> dict[str, pd.DataFrame]:
        print(f"📦 Loading tables for season {season_id}")

        tables = {}

        tables["penalty_kill"] = con.execute("""
            SELECT pk.*, t.tri_code
            FROM all_pk pk
            JOIN all_teams t ON t.team_id = pk.team_id
            WHERE pk.season_id = ?
            ORDER BY pk.game_id
        """, [season_id]).df()

        tables["power_play"] = con.execute("""
            SELECT pp.*, t.tri_code
            FROM all_pp pp
            JOIN all_teams t ON t.team_id = pp.team_id
            WHERE pp.season_id = ?
            ORDER BY pp.game_id
        """, [season_id]).df()

        tables["daily_standings"] = con.execute("""
            SELECT
                game_date, season_id, team_id, tri_code, team_name,
                games_remaining, wins, wins_pctg, losses, ot_losses,
                rotw, rotw_pctg, rw, so_losses, so_wins,
                points, points_pct, projected_points, in_playoff_spot,
                goals_for, gf_pctg, goals_against,
                goal_diff, goal_diff_pctg,
                streak, streak_count,
                last_10_gp, last_10_wins, last_10_losses, last_10_otl,
                last_10_points, last_10_gf, last_10_ga, last_10_goal_diff,
                last_10_rotw, last_10_rw,
                home_gp, home_wins, home_losses, home_otlosses,
                home_rotw, home_rw, home_points, home_gf, home_ga, home_goal_diff,
                road_gp, road_wins, road_losses, road_ot_losses,
                road_rotw, road_rw, road_points, road_gf, road_ga, road_goal_diff,
                division, division_rank, division_home_rank,
                division_road_rank, division_last_10_rank,
                conference, conference_rank, conference_home_rank,
                conference_road_rank, conference_last_10_rank,
                league_rank, league_home_rank, league_road_rank,
                league_last_10_rank, wildcard_rank
            FROM all_standings_daily
            WHERE season_id = ?
            ORDER by game_date
        """, [season_id]).df()

        tables["play_by_play"] = con.execute("""
            SELECT
                *
            FROM all_play_by_play
            WHERE season_id = ?
            ORDER BY game_id, sortOrder
        """, [season_id]).df()

        tables["goalie_stats"] = con.execute("""
            SELECT *
            FROM all_goalie_games
            WHERE season_id = ?
            ORDER BY goalie_name
        """, [season_id]).df()
        
        return tables
   
    # ============================================================
    # PBP SCHEMA (DuckDB-aligned)
    # ============================================================

    DATA_SCHEMA = {
        "season_id": "BIGINT",
        "game_id": "BIGINT",
        "game_date": "TIMESTAMP",
        "team_id": "BIGINT",
        "tri_code": "VARCHAR",
        "team_name": "VARCHAR",
        "game_win": "BIGINT",
        "wins_before": "BIGINT",
        "games_before": "BIGINT",
        "win_pct_before": "DOUBLE",
        "sos_past": "DOUBLE",
        "sos_future": "DOUBLE",
        "pp_opp": "BIGINT",
        "pp_opp_pg": "DOUBLE",
        "pp_toi": "DOUBLE",
        "pp_gf": "BIGINT",
        "pp_gpg": "DOUBLE",
        "pp_net_g": "BIGINT",
        "pp_net_gpg": "DOUBLE",
        "pp_pct": "DOUBLE",
        "pp_net_pct": "DOUBLE",
        "sh_ga": "BIGINT",
        "sh_ga_pg": "DOUBLE",
        "times_sh": "BIGINT",
        "times_sh_pg": "DOUBLE",
        "pk_net_pct": "DOUBLE",
        "pk_pct": "DOUBLE",
        "pk_net_g": "BIGINT",
        "pk_net_gpg": "DOUBLE",
        "pk_toi_pg": "DOUBLE",
        "pp_ga": "BIGINT",
        "pp_ga_pg": "DOUBLE",
        "sh_gf": "BIGINT",
        "sh_gf_pg": "DOUBLE",
        "c_pp_opps": "DOUBLE",
        "c_pp_net_g": "DOUBLE",
        "net_pp_pct": "DOUBLE",
        "ppng_rank": "BIGINT",
        "c_pp_ga": "DOUBLE",
        "c_times_sh": "DOUBLE",
        "net_pk_pct": "DOUBLE",
        "pk_rank": "BIGINT",
        "game_savePct": "DOUBLE",
        "game_GAA": "DOUBLE",
        "rush_shots_for": "BIGINT",
        "rush_shots_against": "BIGINT",
        "rush_shot_differential": "BIGINT",
        "c_rush_shots_for": "BIGINT",
        "c_rush_shots_against": "BIGINT",
        "c_rush_diff": "BIGINT",
        "rush_for_per_game": "DOUBLE",
        "rush_against_per_game": "DOUBLE",
        "rush_diff_per_game": "DOUBLE",
        "shot_attempts_for": "BIGINT",
        "shot_attempts_against": "BIGINT",
        "shot_attempts_differential": "BIGINT",
        "corsi_for": "BIGINT",
        "corsi_against": "BIGINT",
        "corsi_diff": "BIGINT",
        "c_shots_diff": "BIGINT",
        "games_played": "BIGINT",
        "shot_attempts_for_per_game": "DOUBLE",
        "shot_attempts_against_per_game": "DOUBLE",
        "shots_diff_per_game": "DOUBLE",
        "SCF": "BIGINT",
        "SCSF": "BIGINT",
        "SCGF": "BIGINT",
        "SCA": "BIGINT",
        "HDCF": "BIGINT",
        "MDCF": "BIGINT",
        "LDCF": "BIGINT",
        "HDCA": "BIGINT",
        "MDCA": "BIGINT",
        "LDCA": "BIGINT",
        "HDCF_pct": "DOUBLE",
        "HDCA_pct": "DOUBLE",
        "MDCF_pct": "DOUBLE",
        "MDCA_pct": "DOUBLE",
        "LDCF_pct": "DOUBLE",
        "LDCA_pct": "DOUBLE",
        "c_SCF": "BIGINT",
        "c_SCSF": "BIGINT",
        "c_SCGF": "BIGINT",
        "c_SCA": "BIGINT",
        "SCF_pg": "DOUBLE",
        "SCA_pg": "DOUBLE",
        "c_HDCF": "BIGINT",
        "c_HDCA": "BIGINT",
        "c_MDCF": "BIGINT",
        "c_MDCA": "BIGINT",
        "c_LDCF": "BIGINT",
        "c_LDCA": "BIGINT",
        "shots_for": "BIGINT",
        "hd_sog": "BIGINT",
        "md_sog": "BIGINT",
        "ld_sog": "BIGINT",
        "hd_goals": "BIGINT",
        "md_goals": "BIGINT",
        "ld_goals": "BIGINT",
        "shot_pct": "DOUBLE",
        "shots_against": "BIGINT",
        "hd_against": "BIGINT",
        "md_against": "BIGINT",
        "ld_against": "BIGINT",
        "sv_pct": "DOUBLE",
        "pdo": "DOUBLE",
        "c_goals_for": "BIGINT",
        "c_hd_sog": "BIGINT",
        "c_md_sog": "BIGINT",
        "c_ld_sog": "BIGINT",
        "c_hd_goals": "BIGINT",
        "c_md_goals": "BIGINT",
        "c_ld_goals": "BIGINT",
        "c_goals_against": "BIGINT",
        "c_hd_against": "BIGINT",
        "c_md_against": "BIGINT",
        "c_ld_against": "BIGINT",
        "c_shot_pct": "DOUBLE",
        "c_sv_pct": "DOUBLE",
        "c_pdo": "DOUBLE",
        "c_shots_for": "BIGINT",
        "c_shots_against": "BIGINT",
        "c_corsi_for": "BIGINT",
        "c_corsi_against": "BIGINT",
        "c_corsi_diff": "BIGINT",
        "corsi_diff_per_game": "DOUBLE",
        "games_remaining": "BIGINT",
        "wins": "BIGINT",
        "wins_pctg": "DOUBLE",
        "losses": "BIGINT",
        "ot_losses": "BIGINT",
        "rotw": "BIGINT",
        "rotw_pctg": "DOUBLE",
        "rw": "BIGINT",
        "so_losses": "BIGINT",
        "so_wins": "BIGINT",
        "points": "BIGINT",
        "points_pct": "DOUBLE",
        "projected_points": "BIGINT",
        "in_playoff_spot": "BIGINT",
        "gf_pctg": "DOUBLE",
        "goal_diff": "BIGINT",
        "goal_diff_pctg": "DOUBLE",
        "streak": "VARCHAR",
        "streak_count": "DOUBLE",
        "last_10_gp": "BIGINT",
        "last_10_wins": "BIGINT",
        "last_10_losses": "BIGINT",
        "last_10_otl": "BIGINT",
        "last_10_points": "BIGINT",
        "last_10_gf": "BIGINT",
        "last_10_ga": "BIGINT",
        "last_10_goal_diff": "BIGINT",
        "last_10_rotw": "BIGINT",
        "last_10_rw": "BIGINT",
        "home_gp": "BIGINT",
        "home_wins": "BIGINT",
        "home_losses": "BIGINT",
        "home_otlosses": "BIGINT",
        "home_rotw": "BIGINT",
        "home_rw": "BIGINT",
        "home_points": "BIGINT",
        "home_gf": "BIGINT",
        "home_ga": "BIGINT",
        "home_goal_diff": "BIGINT",
        "road_gp": "BIGINT",
        "road_wins": "BIGINT",
        "road_losses": "BIGINT",
        "road_ot_losses": "BIGINT",
        "road_rotw": "BIGINT",
        "road_rw": "BIGINT",
        "road_points": "BIGINT",
        "road_gf": "BIGINT",
        "road_ga": "BIGINT",
        "road_goal_diff": "BIGINT",
        "division": "VARCHAR",
        "division_rank": "BIGINT",
        "division_home_rank": "BIGINT",
        "division_road_rank": "BIGINT",
        "division_last_10_rank": "BIGINT",
        "conference": "VARCHAR",
        "conference_rank": "BIGINT",
        "conference_home_rank": "BIGINT",
        "conference_road_rank": "BIGINT",
        "conference_last_10_rank": "BIGINT",
        "league_rank": "BIGINT",
        "league_home_rank": "BIGINT",
        "league_road_rank": "BIGINT",
        "league_last_10_rank": "BIGINT",
        "wildcard_rank": "BIGINT",
        "made_playoffs": "BIGINT",
        "goals_for": "BIGINT",
        "goals_against": "BIGINT",
        "ot_wins": "BIGINT",
        "overtimes": "BIGINT",
        "ot_games_pct": "DOUBLE",
        "pt_pct_x_sos": "DOUBLE",
        "gaa_x_sv": "DOUBLE",
        "gd_x_rushdiff": "BIGINT",
        "pk_load_x_pk_pct": "DOUBLE",
    }


    def enforce_data_schema(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df[[c for c in DATA_SCHEMA if c in df.columns]]

        for col in DATA_SCHEMA:
            if col not in df.columns:
                df[col] = pd.NA

        for col, dtype in DATA_SCHEMA.items():
            if dtype in ("VARCHAR", "STRING"):
                df[col] = df[col].astype("string")
            elif dtype in ("DOUBLE", "FLOAT", "REAL"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype in ("TIMESTAMP", "DATETIME"):
                df[col] = pd.to_datetime(df[col], errors="coerce")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        return df[list(DATA_SCHEMA.keys())]

    
    # ============================================================
    # Calculate Features from PBP
    # ============================================================   
    
    def build_rush_stats(play_by_play: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identify rush-shot events and compute per-game rush shot stats.

        Returns:
            rush_events: event-level DataFrame
            rush_stats: game × team aggregated stats
        """

        df = play_by_play.copy()

        # --------------------------------------
        # 1. Define event groups
        # --------------------------------------
        shot_events = {
            "shot-on-goal",
            "missed-shot",
            "goal",
            "blocked-shot",
            "failed-shot-attempt",
        }

        transition_events = {"giveaway", "takeaway", "hit"}

        stoppage_events = {
            "faceoff",
            "stoppage",
            "period-end",
            "game-end",
            "penalty",
            "delayed-penalty",
        }

        valid_prev_events = shot_events | transition_events | stoppage_events

        # --------------------------------------
        # 2. Filter + sort
        # --------------------------------------
        plays = df[df["typeDescKey"].isin(valid_prev_events)].copy()
        plays = plays.sort_values(["game_id", "sortOrder"])

        # --------------------------------------
        # 3. Previous-event context
        # --------------------------------------
        g = plays.groupby("game_id", sort=False)

        plays["prev_event_time"] = g["game_in_seconds"].shift(1)
        plays["prev_event_type"] = g["typeDescKey"].shift(1)
        plays["prev_zone"] = g["zone_code"].shift(1)
        plays["prev_team_id"] = g["team_id"].shift(1)

        plays["prev_event_was_stoppage"] = plays["prev_event_type"].isin(stoppage_events)
        plays["time_diff"] = plays["game_in_seconds"] - plays["prev_event_time"]

        # --------------------------------------
        # 4. Rush classification
        # --------------------------------------
        is_shot = plays["typeDescKey"].isin(shot_events)
        quick = plays["time_diff"].between(0, 5, inclusive="both")

        possession_change = (
            plays["prev_team_id"].notna()
            & (plays["prev_team_id"] != plays["team_id"])
        )

        same_team_from_dz_nz = (
            (plays["prev_team_id"] == plays["team_id"])
            & plays["prev_zone"].isin(["D", "N"])
        )

        no_stoppage = ~plays["prev_event_was_stoppage"]

        rush_mask = (
            is_shot
            & quick
            & no_stoppage
            & (possession_change | same_team_from_dz_nz)
        )

        rush_events = (
            plays.loc[rush_mask]
            .copy()
            .drop_duplicates(subset=["season_id", "game_id", "team_id", "sortOrder"])
        )

        rush_events["rush_shot"] = 1

        # --------------------------------------
        # Rush shots FOR
        # --------------------------------------
        rush_for = (
            rush_events
            .groupby(["season_id", "game_id", "team_id"], as_index=False)
            .size()
            .rename(columns={"size": "rush_shots_for"})
        )

        # --------------------------------------
        # Rush shots AGAINST
        # --------------------------------------
        rush_against = (
            rush_events[rush_events["prev_team_id"].notna()]
            .groupby(["season_id", "game_id", "prev_team_id"], as_index=False)
            .size()
            .rename(columns={
                "prev_team_id": "team_id",
                "size": "rush_shots_against"
            })
        )

        # --------------------------------------
        # Merge FOR + AGAINST  ✅ OUTER
        # --------------------------------------
        rush_stats = (
            rush_for
            .merge(
                rush_against,
                on=["season_id", "game_id", "team_id"],
                how="outer",
            )
            .fillna(0)
        )

        rush_stats["rush_shots_for"] = rush_stats["rush_shots_for"].astype(int)
        rush_stats["rush_shots_against"] = rush_stats["rush_shots_against"].astype(int)

        rush_stats["rush_shot_differential"] = (
            rush_stats["rush_shots_for"] - rush_stats["rush_shots_against"]
        )

        return rush_events, rush_stats

    def cumulative_rush(rush_stats: pd.DataFrame) -> pd.DataFrame:
        rush_to_date = rush_stats.copy()
        rush_to_date = rush_to_date.drop_duplicates(
            subset=["season_id", "game_id", "team_id"]
        )

        # Ensure correct ordering
        rush_to_date = rush_to_date.sort_values(
            ["season_id", "team_id", "game_date", "game_id"]
            if "game_date" in rush_to_date.columns
            else ["season_id", "team_id", "game_id"]
        )

        # Group by season + team (NOT game)
        g = rush_to_date.groupby(["season_id", "team_id"])

        # Cumulative totals
        rush_to_date["c_rush_shots_for"] = g["rush_shots_for"].cumsum()
        rush_to_date["c_rush_shots_against"] = g["rush_shots_against"].cumsum()
        rush_to_date["c_rush_diff"] = g["rush_shot_differential"].cumsum()

        # Games played to date
        rush_to_date["games_played"] = g.cumcount() + 1

        # Per-game rates
        rush_to_date["rush_for_per_game"] = (
            rush_to_date["c_rush_shots_for"] / rush_to_date["games_played"]
        )

        rush_to_date["rush_against_per_game"] = (
            rush_to_date["c_rush_shots_against"] / rush_to_date["games_played"]
        )

        rush_to_date["rush_diff_per_game"] = (
            rush_to_date["c_rush_diff"] / rush_to_date["games_played"]
        )

        return rush_to_date
    
    def extract_shot_attempts(play_by_play: pd.DataFrame,) -> tuple[pd.DataFrame, pd.DataFrame]:
        SHOT_CODES = [505, 506, 507, 508, 537]

        pbp = play_by_play.sort_values(["game_id", "sortOrder"]).copy()

        g = pbp.groupby("game_id", sort=False)
        pbp["prev_team_id"] = g["team_id"].shift(1)

        shot_events = pbp[pbp["type_code"].isin(SHOT_CODES)].copy()
        shot_events["shot_attempt"] = 1

        shots_for = (
            shot_events
            .groupby(["season_id", "game_id", "team_id"], as_index=False)
            .size()
            .rename(columns={"size": "shot_attempts_for"})
        )

        shots_against = (
            shot_events[shot_events["prev_team_id"].notna()]
            .groupby(["season_id", "game_id", "prev_team_id"], as_index=False)
            .size()
            .rename(columns={
                "prev_team_id": "team_id",
                "size": "shot_attempts_against"
            })
        )

        shot_stats = (
            shots_for
            .merge(shots_against, on=["season_id", "game_id", "team_id"], how="outer")
            .fillna(0)
        )

        shot_stats["shot_attempts_for"] = shot_stats["shot_attempts_for"].astype(int)
        shot_stats["shot_attempts_against"] = shot_stats["shot_attempts_against"].astype(int)

        shot_stats["shot_attempts_differential"] = (
            shot_stats["shot_attempts_for"] - shot_stats["shot_attempts_against"]
        )

        corsi_for = (
            shot_events[shot_events["zone_code"] == "O"]
            .groupby(["season_id", "game_id", "team_id"], as_index=False)
            .size()
            .rename(columns={"size": "corsi_for"})
        )

        corsi_against = (
            shot_events[(shot_events["zone_code"] == "O") & (shot_events["prev_team_id"].notna())]
            .groupby(["season_id", "game_id", "prev_team_id"], as_index=False)
            .size()
            .rename(columns={"prev_team_id": "team_id", "size": "corsi_against"})
        )

        shot_stats = (
            shot_stats
            .merge(corsi_for, on=["season_id", "game_id", "team_id"], how="left")
            .merge(corsi_against, on=["season_id", "game_id", "team_id"], how="left")
            .fillna(0)
        )

        shot_stats["corsi_for"] = shot_stats["corsi_for"].astype(int)
        shot_stats["corsi_against"] = shot_stats["corsi_against"].astype(int)

        shot_stats["corsi_diff"] = shot_stats["corsi_for"] - shot_stats["corsi_against"]

        return shot_stats, shot_events

    def cumulative_shot_attempts(shot_stats: pd.DataFrame) -> pd.DataFrame:
        shots_to_date = shot_stats.copy()

        # -------------------------------------------------
        # Ensure required columns exist
        # -------------------------------------------------
        required_cols = [
            "shot_attempts_for", "shot_attempts_against",
            "shot_attempts_differential",
            "corsi_for", "corsi_against"
        ]
        for col in required_cols:
            if col not in shots_to_date.columns:
                shots_to_date[col] = 0

        # -------------------------------------------------
        # Ensure correct ordering
        # -------------------------------------------------
        shots_to_date = shots_to_date.sort_values(
            ["season_id", "team_id", "game_id"]
        )

        # -------------------------------------------------
        # Group by season + team
        # -------------------------------------------------
        g = shots_to_date.groupby(["season_id", "team_id"])

        # -------------------------------------------------
        # Cumulative totals
        # -------------------------------------------------
        shots_to_date["c_shots_for"] = g["shot_attempts_for"].cumsum()
        shots_to_date["c_shots_against"] = g["shot_attempts_against"].cumsum()
        shots_to_date["c_shots_diff"] = g["shot_attempts_differential"].cumsum()

        shots_to_date["c_corsi_for"] = g["corsi_for"].cumsum()
        shots_to_date["c_corsi_against"] = g["corsi_against"].cumsum()
        shots_to_date["c_corsi_diff"] = (
            shots_to_date["c_corsi_for"] - shots_to_date["c_corsi_against"]
        )

        # -------------------------------------------------
        # Games played
        # -------------------------------------------------
        shots_to_date["games_played"] = g.cumcount() + 1

        # -------------------------------------------------
        # Per-game rates
        # -------------------------------------------------
        shots_to_date["shot_attempts_for_per_game"] = np.where(
            shots_to_date["games_played"] > 0,
            shots_to_date["c_shots_for"] / shots_to_date["games_played"],
            0.0
        )

        shots_to_date["shot_attempts_against_per_game"] = np.where(
            shots_to_date["games_played"] > 0,
            shots_to_date["c_shots_against"] / shots_to_date["games_played"],
            0.0
        )

        shots_to_date["shots_diff_per_game"] = np.where(
            shots_to_date["games_played"] > 0,
            shots_to_date["c_shots_diff"] / shots_to_date["games_played"],
            0.0
        )

        shots_to_date["corsi_diff_per_game"] = np.where(
            shots_to_date["games_played"] > 0,
            shots_to_date["c_corsi_diff"] / shots_to_date["games_played"],
            0.0
        )

        return shots_to_date

    def preprocess_shot_dataframe(shot_events: pd.DataFrame) -> pd.DataFrame:
    # This identifies the location of shot events to determine danger zones

    #This is just identifying the location of shot events to determine their danger zone 
        required_cols = {
            "x_coord", "y_coord", "game_id",
            "period_number", "game_in_seconds", 'sortOrder'
        }
        missing = required_cols - set(shot_events.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        danger = shot_events.copy()

        # -----------------------------
        # 1. Boolean danger regions
        # -----------------------------
        hd = (
            ((danger["x_coord"] >= 66.5) & (danger["x_coord"] <= 89) &
            (danger["y_coord"].between(-7, 7)))
            |
            ((danger["x_coord"].between(-89, -66.5)) &
            (danger["y_coord"].between(-7, 7)))
        )

        md = (
            # Right side
            ((danger["x_coord"].between(69, 89)) & danger["y_coord"].between(-22, -7))
            |
            ((danger["x_coord"].between(54, 69)) & danger["y_coord"].between(-22, 22))
            |
            ((danger["x_coord"].between(40, 54)) & danger["y_coord"].between(-7, 7))
            |
            # Left side mirrored
            ((danger["x_coord"].between(-89, -69)) & danger["y_coord"].between(-22, -7))
            |
            ((danger["x_coord"].between(-69, -54)) & danger["y_coord"].between(-22, 22))
            |
            ((danger["x_coord"].between(-54, -40)) & danger["y_coord"].between(-7, 7))
        )

        # Low danger = not HD or MD
        ld = ~(hd | md)

        # -----------------------------
        # 2. Assign numeric danger level
        # -----------------------------
        danger["hd_shot"] = hd.astype(int)
        danger["md_shot"] = md.astype(int)
        danger["ld_shot"] = ld.astype(int)

        # -----------------------------
        # 3. Timing features
        # -----------------------------
        danger = danger.sort_values(by=['game_id', 'sortOrder'])
        
        danger["game_time"] = (danger["period_number"] - 1) * 1200 + danger["game_in_seconds"]

        danger["time_since_last_shot"] = danger.groupby("game_id")["game_time"].diff()

        danger["is_rebound"] = (danger["time_since_last_shot"] <= 3).astype(int)

        return danger

    def add_scoring_chances(combined_chances_data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds scoring chance classifications at the EVENT level.

        Produces:
            - base_scoring_chance_score
            - adj_scoring_chance_score
            - scoring_chance
            - rush_scoring_chance
        """

        df = combined_chances_data.copy()

        # -------------------------------------------------
        # Sort for previous-team logic
        # -------------------------------------------------
        df = df.sort_values(["game_id", "sortOrder"])

        df["prev_team_id"] = df.groupby("game_id")["team_id"].shift(1)

        # -------------------------------------------------
        # Base scoring chance score
        # HD=3, MD=2, LD=1
        # -------------------------------------------------
        df["base_scoring_chance_score"] = (
            3 * df["hd_shot"]
            + 2 * df["md_shot"]
            + 1 * df["ld_shot"]
        )

        # -------------------------------------------------
        # Binary scoring chance (SCF / SCA)
        # -------------------------------------------------
        df["scoring_chance"] = (df["base_scoring_chance_score"] >= 2).astype(int)

        # -------------------------------------------------
        # Context bonus (rebound + rush)
        # -------------------------------------------------
        df["context_bonus"] = (
            (df["is_rebound"] == 1).astype(int)
            + (df["rush_shot"] == 1).astype(int)
        )

        # -------------------------------------------------
        # Adjusted scoring chance score
        # -------------------------------------------------
        df["adj_scoring_chance_score"] = (
            df["base_scoring_chance_score"] + df["context_bonus"]
        )

        # -------------------------------------------------
        # Outcome flags
        # -------------------------------------------------
        df["scoring_chance_goal"] = (df["type_code"] == 505).astype(int)

        df["shot_on_goal"] = (
            (df["type_code"] == 506).astype(int)
            + df["scoring_chance_goal"]
        )

        # -------------------------------------------------
        # Rush scoring chance
        # -------------------------------------------------
        df["rush_scoring_chance"] = (
            (df["scoring_chance"] == 1)
            & (df["rush_shot"] == 1)
        ).astype(int)

        return df

    def build_danger_corsi_and_scf_stats(
        scoring_chances_events: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Produces:
            - HDCF, HDCA
            - MDCF, MDCA
            - LDCF, LDCA
            - SCF, SCSF, SCGF
            - SCA

        Modeling assumptions:
            - Blocked shots (type_code == 508) are EXCLUDED
            - Danger / scoring chances are based on unblocked shot attempts only
            - Shooting team is the event team for all remaining events
        """

        # -------------------------------------------------
        # 0. Copy + REMOVE blocked shots
        # -------------------------------------------------
        d = scoring_chances_events.copy()
        d = d[d["type_code"] != 508].copy()

        # -------------------------------------------------
        # 1. Shooting team assignment
        # -------------------------------------------------
        d["shooting_team_id"] = d["team_id"]

        # -------------------------------------------------
        # 2. Shot / goal flags
        # -------------------------------------------------
        d["shot_on_goal"] = d["type_code"].isin([505, 506]).astype(int)
        d["scoring_chance_goal"] = (d["type_code"] == 505).astype(int)

        # -------------------------------------------------
        # 3. FOR — scoring chances
        # -------------------------------------------------
        scf = (
            d.groupby(["season_id", "game_id", "shooting_team_id"], as_index=False)
            .agg(
                SCF=("scoring_chance", "sum"),
                SCSF=("shot_on_goal", "sum"),
                SCGF=("scoring_chance_goal", "sum"),
            )
            .rename(columns={"shooting_team_id": "team_id"})
        )

        # -------------------------------------------------
        # 4. AGAINST — scoring chances
        # -------------------------------------------------
        # Home team allows all away scoring chances; away team allows all home scoring chances
        sca_home = (
            d.groupby(["season_id", "game_id", "home_team"], as_index=False)
            .agg(SCA=("scoring_chance", "sum"))
            .rename(columns={"home_team": "team_id"})
        )

        sca_away = (
            d.groupby(["season_id", "game_id", "away_team"], as_index=False)
            .agg(SCA=("scoring_chance", "sum"))
            .rename(columns={"away_team": "team_id"})
        )

        sca = (
            pd.concat([sca_home, sca_away])
            .groupby(["season_id", "game_id", "team_id"], as_index=False)
            .agg(SCA=("SCA", "sum"))
        )

        # -------------------------------------------------
        # 5. Danger-split Corsi FOR
        # -------------------------------------------------
        corsi_for = (
            d.groupby(["season_id", "game_id", "team_id"], as_index=False)
            .agg(
                HDCF=("hd_shot", "sum"),
                MDCF=("md_shot", "sum"),
                LDCF=("ld_shot", "sum"),
            )
        )

        # -------------------------------------------------
        # 6. Danger-split Corsi AGAINST
        # -------------------------------------------------
        corsi_against = (
            d.groupby(["season_id", "game_id", "shooting_team_id"], as_index=False)
            .agg(
                HDCA=("hd_shot", "sum"),
                MDCA=("md_shot", "sum"),
                LDCA=("ld_shot", "sum"),
            )
            .rename(columns={"shooting_team_id": "team_id"})
        )

        # -------------------------------------------------
        # 7. Final merge
        # -------------------------------------------------
        final = (
            scf
            .merge(sca, on=["season_id", "game_id", "team_id"], how="left")
            .merge(corsi_for, on=["season_id", "game_id", "team_id"], how="left")
            .merge(corsi_against, on=["season_id", "game_id", "team_id"], how="left")
            .fillna(0)
        )

        # -------------------------------------------------
        # 8. Percentages (guard against divide-by-zero)
        # -------------------------------------------------
        final["HDCF_pct"] = final["HDCF"] / (final["HDCF"] + final["HDCA"]).replace(0, np.nan)
        final["MDCF_pct"] = final["MDCF"] / (final["MDCF"] + final["MDCA"]).replace(0, np.nan)
        final["LDCF_pct"] = final["LDCF"] / (final["LDCF"] + final["LDCA"]).replace(0, np.nan)

        return final

    def cumulative_scoring_chances(danger_scoring: pd.DataFrame,
                                corsi_stats: pd.DataFrame) -> pd.DataFrame:
        scoring = danger_scoring.copy()
        shots = corsi_stats.copy()

        # -------------------------------------------------
        # Sort both dataframes consistently
        # -------------------------------------------------
        sort_cols = (
            ["season_id", "team_id", "game_date", "game_id"]
            if "game_date" in scoring.columns
            else ["season_id", "team_id", "game_id"]
        )

        scoring = scoring.sort_values(sort_cols)
        shots = shots.sort_values(sort_cols)

        # -------------------------------------------------
        # Grouping
        # -------------------------------------------------
        g_sc = scoring.groupby(["season_id", "team_id"])
        g_cf = shots.groupby(["season_id", "team_id"])

        # -------------------------------------------------
        # Cumulative scoring-chance metrics
        # -------------------------------------------------
        scoring["c_SCF"] = g_sc["SCF"].cumsum()
        scoring["c_SCSF"] = g_sc['SCSF'].cumsum()
        scoring['c_SCGF'] = g_sc['SCGF'].cumsum()
        scoring['c_SCA'] = g_sc['SCA'].cumsum()
        
        scoring["games_played"] = g_sc.cumcount() + 1

        scoring["SCF_pg"] = scoring["c_SCF"] / scoring["games_played"]
        scoring["SCA_pg"] = scoring["c_SCA"] / scoring["games_played"]
        
        
        scoring["c_HDCF"] = g_sc["HDCF"].cumsum()
        scoring["c_HDCA"] = g_sc["HDCA"].cumsum()
        scoring["c_MDCF"] = g_sc["MDCF"].cumsum()
        scoring["c_MDCA"] = g_sc["MDCA"].cumsum()
        scoring["c_LDCF"] = g_sc["LDCF"].cumsum()
        scoring["c_LDCA"] = g_sc["LDCA"].cumsum()


        # -------------------------------------------------
        # Cumulative Corsi metrics
        # -------------------------------------------------
        shots["c_corsi_for"] = g_cf["corsi_for"].cumsum()
        shots["c_corsi_against"] = g_cf["corsi_against"].cumsum()
        shots["c_corsi_diff"] = (
            shots["c_corsi_for"] - shots["c_corsi_against"]
        )

        shots["games_played"] = g_cf.cumcount() + 1
        shots["corsi_diff_per_game"] = (
            shots["c_corsi_diff"] / shots["games_played"]
        )




        # -------------------------------------------------
        # Merge back together
        # -------------------------------------------------
        merge_cols = ["season_id", "team_id", "game_id"]
        if "game_date" in scoring.columns and "game_date" in shots.columns:
            merge_cols.append("game_date")

        scoring = scoring.merge(
            shots[
                merge_cols + [
                    "c_corsi_for",
                    "c_corsi_against",
                    "c_corsi_diff",
                    "corsi_diff_per_game",

                ]
            ],
            on=merge_cols,
            how="left",
        )
        scoring = scoring.drop(columns=['games_played'])

        return scoring

    def build_team_shooting_stats(
        scoring_chances_events: pd.DataFrame
    ) -> pd.DataFrame:
        df = scoring_chances_events.copy()

        # -------------------------------------------------
        # Shot + goal flags
        # -------------------------------------------------
        df["goal"] = df["scoring_chance_goal"]
        df["sog"] = df["shot_on_goal"]

        df["hd_sog"] = ((df["sog"] == 1) & (df["hd_shot"] == 1)).astype(int)
        df["md_sog"] = ((df["sog"] == 1) & (df["md_shot"] == 1)).astype(int)
        df["ld_sog"] = ((df["sog"] == 1) & (df["ld_shot"] == 1)).astype(int)

        df["hd_goal"] = ((df["goal"] == 1) & (df["hd_shot"] == 1)).astype(int)
        df["md_goal"] = ((df["goal"] == 1) & (df["md_shot"] == 1)).astype(int)
        df["ld_goal"] = ((df["goal"] == 1) & (df["ld_shot"] == 1)).astype(int)

        # -------------------------------------------------
        # Opponent mapping (SAFE)
        # -------------------------------------------------
        teams = (
            df[["game_id", "team_id"]]
            .drop_duplicates()
            .merge(
                df[["game_id", "team_id"]].drop_duplicates(),
                on="game_id",
                suffixes=("", "_opp"),
            )
            .query("team_id != team_id_opp")
        )

        df = (
            df.merge(
                teams,
                on=["game_id", "team_id"],
                how="left",
            )
            .rename(columns={"team_id_opp": "opponent_team_id"})
        )

        # -------------------------------------------------
        # FOR
        # -------------------------------------------------
        stats_for = (
            df.groupby(["season_id", "game_id", "team_id"], as_index=False)
            .agg(
                shots_for=("sog", "sum"),
                goals_for=("goal", "sum"),
                hd_sog=("hd_sog", "sum"),
                md_sog=("md_sog", "sum"),
                ld_sog=("ld_sog", "sum"),
                hd_goals=("hd_goal", "sum"),
                md_goals=("md_goal", "sum"),
                ld_goals=("ld_goal", "sum"),
            )
        )

        stats_for["shot_pct"] = (
            100 * stats_for["goals_for"] / stats_for["shots_for"].replace(0, np.nan)
        )

        # -------------------------------------------------
        # AGAINST
        # -------------------------------------------------
        stats_against = (
            df.groupby(["season_id", "game_id", "opponent_team_id"], as_index=False)
            .agg(
                shots_against=("sog", "sum"),
                goals_against=("goal", "sum"),
                hd_against=("hd_sog", "sum"),
                md_against=("md_sog", "sum"),
                ld_against=("ld_sog", "sum"),
            )
            .rename(columns={"opponent_team_id": "team_id"})
        )

        # -------------------------------------------------
        # Merge
        # -------------------------------------------------
        shooting_stats = (
            stats_for
            .merge(
                stats_against,
                on=["season_id", "game_id", "team_id"],
                how="left",
            )
            .fillna(0)
        )

        shooting_stats["sv_pct"] = (
            100
            - 100
            * shooting_stats["goals_against"]
            / shooting_stats["shots_against"].replace(0, np.nan)
        )

        shooting_stats["pdo"] = shooting_stats["shot_pct"] + shooting_stats["sv_pct"]

        return shooting_stats


    def cumulative_sog(sog_stats: pd.DataFrame) -> pd.DataFrame:

        sog = sog_stats.copy()

        # -------------------------------------------------
        # Sort BEFORE cumulative math
        # -------------------------------------------------
        sort_cols = (
            ["season_id", "team_id", "game_date", "game_id"]
            if "game_date" in sog.columns
            else ["season_id", "team_id", "game_id"]
        )
        sog = sog.sort_values(sort_cols)

        # -------------------------------------------------
        # Group by season + team
        # -------------------------------------------------
        g = sog.groupby(["season_id", "team_id"])

        # -------------------------------------------------
        # Games played
        # -------------------------------------------------
        sog["games_played"] = g.cumcount() + 1

        # -------------------------------------------------
        # Cumulative FOR
        # -------------------------------------------------
        sog["c_shots_for"]  = g["shots_for"].cumsum()
        sog["c_goals_for"]  = g["goals_for"].cumsum()

        sog["c_hd_sog"] = g["hd_sog"].cumsum()
        sog["c_md_sog"] = g["md_sog"].cumsum()
        sog["c_ld_sog"] = g["ld_sog"].cumsum()

        sog["c_hd_goals"] = g["hd_goals"].cumsum()
        sog["c_md_goals"] = g["md_goals"].cumsum()
        sog["c_ld_goals"] = g["ld_goals"].cumsum()

        # -------------------------------------------------
        # Cumulative AGAINST
        # -------------------------------------------------
        sog["c_shots_against"]  = g["shots_against"].cumsum()
        sog["c_goals_against"]  = g["goals_against"].cumsum()

        sog["c_hd_against"] = g["hd_against"].cumsum()
        sog["c_md_against"] = g["md_against"].cumsum()
        sog["c_ld_against"] = g["ld_against"].cumsum()

        # -------------------------------------------------
        # Recompute percentages (DO NOT cumsum these)
        # -------------------------------------------------
        sog["c_shot_pct"] = np.where(
            sog["c_shots_for"] > 0,
            sog["c_goals_for"] / sog["c_shots_for"],
            0.0
        )

        sog["c_sv_pct"] = np.where(
            sog["c_shots_against"] > 0,
            1 - (sog["c_goals_against"] / sog["c_shots_against"]),
            0.0
        )

        sog["c_pdo"] = (sog["c_shot_pct"] + sog["c_sv_pct"]) * 100
        sog=sog.drop(columns=['games_played'])

        return sog
    
    def build_enriched_shot_events(
        shot_events: pd.DataFrame,
        danger_events: pd.DataFrame,
        rush_events: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Join danger and rush context onto shot events.

        shot_events is the canonical event spine.
        """

        key_cols = ["season_id", "game_id", "team_id", "sortOrder"]

        combined_chances_data = shot_events.merge(
            danger_events[
                key_cols + ["hd_shot", "md_shot", "ld_shot", "is_rebound"]
            ],
            on=key_cols,
            how="left",
        ).merge(
            rush_events[
                key_cols + ["rush_shot"]
            ],
            on=key_cols,
            how="left",
        )

        return combined_chances_data

    # ============================================================
    # Rush Orchestrator
    # ============================================================   
    def build_transition_offense_features(
        play_by_play: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """
        Canonical orchestrator for rush, shot, danger, and scoring chance features.

        Returns final cumulative outputs only — intended for final merge.
        """

        print("\n▶ Running transition offense orchestrator")

        # ---------------------------------------------------------
        # Event-level construction
        # ---------------------------------------------------------
        rush_events, rush_stats = build_rush_stats(play_by_play)
        shot_stats, shot_events = extract_shot_attempts(play_by_play)

        danger_events = preprocess_shot_dataframe(shot_events)

        combined_chances_data = build_enriched_shot_events(
            shot_events,
            danger_events,
            rush_events,
        )

        scoring_chances_events = add_scoring_chances(combined_chances_data)

        # ---------------------------------------------------------
        # Team/game aggregations (non-cumulative)
        # ---------------------------------------------------------
        danger_stats = build_danger_corsi_and_scf_stats(scoring_chances_events)
        sog_stats = build_team_shooting_stats(scoring_chances_events)

        # ---------------------------------------------------------
        # Pre-cumulative sanity checks (CRITICAL)
        # ---------------------------------------------------------
        print("\n📋 PRE-CUMULATIVE CHECKS")

        checks = {
            "rush_stats": rush_stats,
            "shot_stats": shot_stats,
            "danger_stats": danger_stats,
            "sog_stats": sog_stats,
        }

        for name, df in checks.items():
            if df is None:
                print(f"🚨 {name} is None")
                continue

            print(
                f"{name.upper():>14}: rows={len(df):>6} | "
                f"cols={len(df.columns):>3}"
            )

            missing_keys = {"game_id", "team_id"} - set(df.columns)
            if missing_keys:
                print(f"🚨 {name} missing keys: {missing_keys}")

        # ---------------------------------------------------------
        # FINAL cumulative outputs
        # ---------------------------------------------------------
        cumulative_outputs = {
            "cumulative_rush": cumulative_rush(rush_stats),
            "cumulative_shooting": cumulative_shot_attempts(shot_stats),
            "cumulative_scoring": cumulative_scoring_chances(
                danger_stats,
                shot_stats,
            ),
            "cumulative_sog": cumulative_sog(sog_stats),
        }

        # ---------------------------------------------------------
        # Post-cumulative validation (THIS IS WHERE IT BREAKS)
        # ---------------------------------------------------------
        print("\n📊 TRANSITION OFFENSE OUTPUT CHECK")

        for name, df in cumulative_outputs.items():
            if df is None:
                print(f"🚨 {name} returned None")
                continue

            print(f"{name.upper():>24}: {df.shape}")

            if df.empty:
                print(f"🚨 {name} returned EMPTY DataFrame")

            else:
                missing_keys = {"game_id", "team_id"} - set(df.columns)
                if missing_keys:
                    print(f"🚨 {name} missing keys: {missing_keys}")

        return cumulative_outputs

    # ============================================================
    # Calculate Features from Game Summary
    # ============================================================   

    def strength_of_schedule(backfill_df: pd.DataFrame) -> pd.DataFrame:
        sos = backfill_df.copy()

        games_sos = []

        # -------------------------------------------------
        # Build two rows per game (HOME + AWAY)
        # -------------------------------------------------
        for _, row in sos.iterrows():

            # AWAY team row
            games_sos.append({
                "season_id": row["season_id"],
                "game_id": int(row["game_id"]),
                "team_id": int(row["away_team"]),
                "tri_code": row["away_tri_code"],
                "opponent_id": int(row["home_team"]),
                "opponent": row["home_tri_code"],
                "goals_for": row["awayTeam.score"],
                "goals_against": row["homeTeam.score"],
            })

            # HOME team row
            games_sos.append({
                "season_id": row["season_id"],
                "game_id": int(row["game_id"]),
                "team_id": int(row["home_team"]),
                "tri_code": row["home_tri_code"],
                "opponent_id": int(row["away_team"]),
                "opponent": row["away_tri_code"],
                "goals_for": row["homeTeam.score"],
                "goals_against": row["awayTeam.score"],
            })

        games_sos = pd.DataFrame(games_sos)

        # -------------------------------------------------
        # Sort + win calculation
        # -------------------------------------------------
        games_sos = games_sos.sort_values(
            ["season_id", "team_id", "tri_code", "game_id"]
        )

        games_sos["win"] = (
            games_sos["goals_for"] > games_sos["goals_against"]
        ).astype(int)

        # -------------------------------------------------
        # Wins BEFORE (leave NaN for first game)
        # -------------------------------------------------
        games_sos["wins_before"] = (
            games_sos
            .groupby(["season_id", "team_id", "tri_code"])["win"]
            .transform(lambda s: s.cumsum().shift(1))
            .astype("Int64")
        )


        # -------------------------------------------------
        # Games BEFORE / played
        # -------------------------------------------------
        games_sos["games_before"] = (
            games_sos
            .groupby(["season_id", "team_id", "tri_code"])
            .cumcount()
        )

        games_sos["games_played"] = games_sos["games_before"] + 1

        # -------------------------------------------------
        # Win% BEFORE (NaN if no prior games)
        # -------------------------------------------------
        games_sos["win_pct_before"] = np.where(
            games_sos["games_before"] > 0,
            games_sos["wins_before"] / games_sos["games_before"],
            np.nan
        )

        # -------------------------------------------------
        # Deduplicate opponent win% BEFORE merge
        # -------------------------------------------------
        opp_win = (
            games_sos[["season_id", "tri_code", "game_id", "win_pct_before"]]
            .drop_duplicates(subset=["season_id", "tri_code", "game_id"])
        )

        games_sos = games_sos.merge(
            opp_win.rename(columns={
                "tri_code": "opponent",
                "win_pct_before": "opponent_win_pct_before"
            }),
            on=["season_id", "opponent", "game_id"],
            how="left"
        )

        # -------------------------------------------------
        # SOS — PAST (NaN until at least one prior game)
        # -------------------------------------------------
        games_sos["sos_past"] = (
            games_sos
            .groupby(["season_id", "tri_code"])["opponent_win_pct_before"]
            .expanding()
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

        # -------------------------------------------------
        # SOS — FUTURE (reverse expanding, valid from game 1)
        # -------------------------------------------------
        games_sos["sos_future"] = (
            games_sos.iloc[::-1]
            .groupby(["season_id", "team_id", "tri_code"])["opponent_win_pct_before"]
            .expanding()
            .mean()
            .reset_index(level=[0, 1, 2], drop=True)
            .iloc[::-1]
        )

        # -------------------------------------------------
        # Final cleanup
        # -------------------------------------------------
        games_sos = (
            games_sos
            .drop(columns=['opponent_id', 'opponent', 'goals_for', 'goals_against', 'opponent_win_pct_before'])
            .sort_values(by=["season_id", "game_id"])
        )

        return games_sos

    # ============================================================
    # Calculate Features from Goalie Stats
    # ============================================================  
     
    def build_team_goalie_stats(goalies: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
        def safe_weighted_mean(values, weights):
            weights = pd.to_numeric(weights, errors="coerce").fillna(0)
            values = pd.to_numeric(values, errors="coerce")
            weights = weights.replace(0, np.nan)
            num = (values * weights).sum()
            den = weights.sum()
            return num / den if den > 0 else np.nan

        g = goalies.copy()

        goalies_indiv = (
            g.groupby(["goalie_name", "season_id", "tri_code", "game_id"], as_index=False)
            .agg(
                i_savePct=("save_pct", lambda x: safe_weighted_mean(x, g.loc[x.index, "shots_against"])),
                i_GAA=("gaa", lambda x: safe_weighted_mean(x, g.loc[x.index, "toi"])),
                i_gamesPlayed=("games_played", "sum"),
                i_gamesStarted=("games_started", "sum"),
                i_wins=("wins", "sum"),
                i_losses=("losses", "sum"),
                i_otlosses=("ot_losses", "sum"),
                i_shutouts=("shutouts", "sum"),
                i_saves=("saves", "sum"),
                i_shotsAgainst=("shots_against", "sum"),
                i_timeOnIce=("toi", "sum"),
            )
        )

        game_goalie_stats = (
            goalies_indiv.groupby(["season_id", "game_id", "tri_code"], as_index=False)
            .agg(
                game_savePct=("i_savePct", lambda x: safe_weighted_mean(x, goalies_indiv.loc[x.index, "i_shotsAgainst"])),
                game_GAA=("i_GAA", lambda x: safe_weighted_mean(x, goalies_indiv.loc[x.index, "i_timeOnIce"])),
                game_wins=("i_wins", "sum"),
                game_losses=("i_losses", "sum"),
                game_shutouts=("i_shutouts", "sum"),
                game_saves=("i_saves", "sum"),
                game_shotsAgainst=("i_shotsAgainst", "sum"),
            )
        )

        goalies = game_goalie_stats[["season_id", "game_id", "tri_code", "game_savePct", "game_GAA"]]
        goalies = goalies.merge(teams[["team_id", "tri_code"]].drop_duplicates(), on="tri_code", how="left")

        return goalies

    
    # ============================================================
    # Calculate Features from Special Teams
    # ============================================================   
    def build_special_teams_stats(power_play: pd.DataFrame, penalty_kill: pd.DataFrame) -> pd.DataFrame:
        pp = power_play.copy()
        pk = penalty_kill.copy()

        pp = pp.sort_values(["season_id", "team_id", "game_date"])
        pk = pk.sort_values(["season_id", "team_id", "game_date"])

        pp["games_played"] = pp.groupby(["season_id", "team_id"]).cumcount() + 1

        merged = pp.merge(
            pk,
            on=["season_id", "game_id", "team_id", "game_date"],
            how="inner",
            suffixes=("_pp", "_pk")
        )

        pp_rank = pp[
            ["season_id", "game_date", "team_id", "tri_code", "games_played", "pp_net_g", "pp_opportunities"]
        ].copy()

        pp_rank["c_pp_net_g"] = pp_rank.groupby(["season_id", "team_id"])["pp_net_g"].cumsum()
        pp_rank["c_pp_opps"] = pp_rank.groupby(["season_id", "team_id"])["pp_opportunities"].cumsum()

        pp_rank["net_pp_pct"] = np.where(
            pp_rank["c_pp_opps"] > 0,
            pp_rank["c_pp_net_g"] / pp_rank["c_pp_opps"],
            np.nan
        )

        pk_rank = pk[["season_id", "game_date", "team_id", "pp_ga", "times_sh"]].copy()
        pk_rank["c_pp_ga"] = pk_rank.groupby(["season_id", "team_id"])["pp_ga"].cumsum()
        pk_rank["c_times_sh"] = pk_rank.groupby(["season_id", "team_id"])["times_sh"].cumsum()

        pk_rank["net_pk_pct"] = np.where(
            pk_rank["c_times_sh"] > 0,
            pk_rank["c_pp_ga"] / pk_rank["c_times_sh"],
            np.nan
        )

        all_dates = sorted(pp_rank["game_date"].dropna().unique().tolist())
        seasons = sorted(pp_rank["season_id"].dropna().unique().tolist())
        teams = sorted(pp_rank["team_id"].dropna().unique().tolist())

        grid = pd.MultiIndex.from_product(
            [seasons, all_dates, teams],
            names=["season_id", "game_date", "team_id"]
        )

        pp_full = (
            pp_rank
            .set_index(["season_id", "game_date", "team_id"])
            .reindex(grid)
            .sort_index()
        )

        pk_full = (
            pk_rank
            .set_index(["season_id", "game_date", "team_id"])
            .reindex(grid)
            .sort_index()
        )

        for col in ["c_pp_net_g", "c_pp_opps", "net_pp_pct", "tri_code", "games_played"]:
            if col in pp_full.columns:
                pp_full[col] = pp_full.groupby(["season_id", "team_id"])[col].ffill()

        for col in ["c_pp_ga", "c_times_sh", "net_pk_pct"]:
            if col in pk_full.columns:
                pk_full[col] = pk_full.groupby(["season_id", "team_id"])[col].ffill()

        pp_full["net_pp_pct"] = (pp_full["net_pp_pct"] * 100).round(1)
        pk_full["net_pk_pct"] = (pk_full["net_pk_pct"] * 100).round(1)

        pp_full["ppng_rank"] = (
            pp_full.groupby(["season_id", "game_date"])["net_pp_pct"]
            .rank(method="dense", ascending=False)
        )

        pk_full["pk_rank"] = (
            pk_full.groupby(["season_id", "game_date"])["net_pk_pct"]
            .rank(method="dense", ascending=True)
        )

        pp_features = pp_full.reset_index()[
            ["season_id", "game_date", "team_id", "c_pp_opps", "c_pp_net_g", "net_pp_pct", "ppng_rank"]
        ]

        pk_features = pk_full.reset_index()[
            ["season_id", "game_date", "team_id", "c_pp_ga", "c_times_sh", "net_pk_pct", "pk_rank"]
        ]

        special_teams = (
            merged
            .merge(pp_features, on=["season_id", "game_date", "team_id"], how="left")
            .merge(pk_features, on=["season_id", "game_date", "team_id"], how="left")
        )

        special_teams = special_teams.drop(
            columns=[
                'home_road_pp', 'losses_pp', 'opponent_pp', 'ot_losses_pp',
                'point_pct', 'points_pp', 'team_name_pp', 'ties_pp', 'wins_pp',
                'games_played_pk', 'home_road_pk', 'losses_pk', 'opponent_pk',
                'ot_losses_pk', 'points_pk', 'points_pct', 'team_name_pk',
                'ties_pk', 'wins_pk', 'tri_code_pk', 'games_played_pp'
            ],
            errors="ignore"
        )

        special_teams = special_teams.rename(columns={"pp_opportunities": "pp_opp", "tri_code_pp": "tri_code"})

        return special_teams


# ===============================================================
# Build Final DF
# ===============================================================
    def build_core_df(     
        play_by_play: pd.DataFrame, 
        backfill_df: pd.DataFrame,
        games_sos: pd.DataFrame,
        special_teams: pd.DataFrame,

        cumulative_rush_stats: pd.DataFrame,
        cumulative_shooting_stats: pd.DataFrame,
        cumulative_scoring: pd.DataFrame,
        cumulative_shots_on_goal: pd.DataFrame,
        cumulative_goalie_stats: pd.DataFrame
    ) -> pd.DataFrame:
        


        games = play_by_play[['season_id','game_id','team_id']].drop_duplicates()
        games=games[games['team_id'].notna()].reset_index(drop=True)
        print("GAMES:", games.shape)
        print("BACKFILL:", backfill_df.shape)
        print("SOS:", games_sos.shape)
        print("SPECIAL TEAMS:", special_teams.shape)
        print("GOALIES:", cumulative_goalie_stats.shape)
        print("RUSH:", cumulative_rush_stats.shape)
        print("SHOOTING:", cumulative_shooting_stats.shape)
        print("SCORING:", cumulative_scoring.shape)
        print("SOG:", cumulative_shots_on_goal.shape)        
        
        def coalesce_duplicates(core_df, left="_x", right="_y"):
            for col in core_df.columns:
                if col.endswith(left):
                    base = col[:-len(left)]
                    other = base + right
                    if other in core_df.columns:
                        core_df[base] = core_df[col].combine_first(core_df[other])
                        core_df.drop([col, other], axis=1, inplace=True)
            return core_df

        core_df = (
            games
            .merge(backfill_df[['game_id', 'game_date']], on='game_id', how='inner')
            .merge(games_sos, on=['season_id', 'game_id', 'team_id'], how='inner')
            .merge(special_teams, on=['season_id', 'game_id', 'team_id','tri_code'], how='inner')
            .merge(
                cumulative_goalie_stats[['game_id', 'team_id', 'game_savePct', 'game_GAA']],
                on=['game_id', 'team_id'],
                how='inner'
            )
            .merge(cumulative_rush_stats, on=['season_id','game_id', 'team_id'], how='inner')
            .merge(cumulative_shooting_stats, on=['season_id','game_id', 'team_id'], how = 'inner' )
            .merge(cumulative_scoring, on=['season_id', 'game_id', 'team_id'], how='left')
            .merge(cumulative_shots_on_goal, on=['season_id', 'game_id', 'team_id',], how='inner' )
        )

        core_df = coalesce_duplicates(core_df)
        
        return core_df
            
    def build_final_df(
        core_df: pd.DataFrame,
        standings: pd.DataFrame
    ):
        print("CORE DF SHAPE:", core_df.shape)
        print("STANDINGS SHAPE:", standings.shape)

        print("CORE game_date sample:")
        # print(core_df["game_date"].head())
        print(core_df["game_date"].dtype)

        print("STANDINGS game_date sample:")
        # print(standings["game_date"].head())
        print(standings["game_date"].dtype)

        print("CORE team_id sample:", core_df["team_id"].unique()[:5])
        print("STANDINGS team_id sample:", standings["team_id"].unique()[:5])


        final_updates = core_df.merge(
            standings,
            on=['season_id', 'team_id', 'game_date'],
            how='inner'
        )
        
        def coalesce_dupes(df, left="_x", right="_y"):
            for col in df.columns:
                if col.endswith(left):
                    base = col[:-len(left)]
                    other = base + right
                    if other in df.columns:
                        df[base] = df[col].combine_first(df[other])
                        df.drop([col, other], axis=1, inplace=True)
            return df

        final_updates = coalesce_dupes(final_updates)
        return final_updates


    #special features

    def special_features(final_updates):
        final_updates['ot_wins'] = (final_updates['rotw']-final_updates['rw']).clip(lower=0)
        final_updates['overtimes'] = final_updates['ot_losses']+final_updates['ot_wins']
        final_updates["ot_games_pct"] = (
            final_updates["overtimes"] / final_updates["games_played"]
        ).fillna(0).round(2)


        # Point% × Strength of Schedule
        final_updates["pt_pct_x_sos"] = final_updates["points_pct"] * final_updates["sos_future"]

        # Goalie GA × SV% interaction
        final_updates["gaa_x_sv"] = final_updates["game_GAA"] * final_updates["game_savePct"]

        # Goal differential × Rush differential
        final_updates["gd_x_rushdiff"] = (
            final_updates["goal_diff"] * final_updates["rush_shot_differential"]
        )

        # PK workload × PK%
        final_updates["pk_load_x_pk_pct"] = (
            final_updates["times_sh_pg"] * final_updates["pk_pct"]
        )
        
        print(final_updates.tail())
        return final_updates

    # ============================================================
    # Execution
    # ============================================================
    # ============================================================
    # 1. Determine backfill window
    # ============================================================
    prediction_data = load_last_game(season_id)
    gs_max_id = load_last_game_summary_id(season_id)
    game_summary = load_game_summary_df(season_id)

    if prediction_data == gs_max_id:
        print("✅ prediction data and game summary are in sync.")
        return

    missing_ids = list(range((prediction_data or 0) + 1, gs_max_id + 1))
    backfill_df = game_summary[game_summary["game_id"].isin(missing_ids)]

    if backfill_df.empty:
        return

    # ============================================================
    # 2. Load season tables
    # ============================================================
    print("📥 Loading tables...")
    tables = fetch_tables(con, season_id)

    # ============================================================
    # 3. Filter all tables to backfill games
    # ============================================================
    def filter_to_games(df: pd.DataFrame, game_ids: list[int]) -> pd.DataFrame:
        if "game_id" not in df.columns:
            return df
        return df[df["game_id"].isin(game_ids)]

    for key in [
        "play_by_play",
        "goalie_stats",
        "power_play",
        "penalty_kill",
        "daily_standings",
    ]:
        tables[key] = filter_to_games(tables[key], missing_ids)

    # ============================================================
    # 4. Transition / Rush / Shot features
    # ============================================================
    print("Running rush data...")

    play_by_play = tables["play_by_play"]
    print(play_by_play.head())
    transition_features = build_transition_offense_features(play_by_play)

    cumulative_rush_stats     = transition_features["cumulative_rush"]
    cumulative_shooting_stats = transition_features["cumulative_shooting"]
    cumulative_scoring        = transition_features["cumulative_scoring"]
    cumulative_shots_on_goal  = transition_features["cumulative_sog"]

    # ============================================================
    # 5. Strength of Schedule
    # ============================================================
    print("Running strength of schedule data...")
    games_sos = strength_of_schedule(backfill_df)

    # ============================================================
    # 6. Goalie stats
    # ============================================================
    print("Running goalie data...")
    goalies = tables["goalie_stats"]
    teams   = load_teams(con)


    cumulative_goalie_stats = build_team_goalie_stats(goalies, teams)

    # ============================================================
    # 7. Special teams
    # ============================================================
    print("Running special teams data...")
    penalty_kill = tables["penalty_kill"]
    power_play   = tables["power_play"]

    special_teams = build_special_teams_stats(power_play, penalty_kill)

    # ============================================================
    # 8. Build final feature frame
    # ============================================================
    print("Running final updates merging...")    
    core_df = build_core_df(
        play_by_play=play_by_play,
        backfill_df=backfill_df,
        games_sos=games_sos,
        special_teams=special_teams,

        cumulative_rush_stats=cumulative_rush_stats,
        cumulative_shooting_stats=cumulative_shooting_stats,
        cumulative_scoring=cumulative_scoring,
        cumulative_shots_on_goal=cumulative_shots_on_goal,
        cumulative_goalie_stats=cumulative_goalie_stats,
    )

    final_updates = build_final_df(
        core_df=core_df,
        standings=tables["daily_standings"]
    )
    

    # ============================================================
    # 9. Derived / interaction features
    # ============================================================
    prediction_data_updates = special_features(final_updates)

    if prediction_data_updates.empty:
        print(prediction_data_updates.tail())
        return

    # ============================================================
    # 10. Enforce schema
    # ============================================================
    prediction_data_updates = enforce_data_schema(prediction_data_updates)

    # ============================================================
    # 11. Upsert into prediction_data (game_id + team_id)
    # ============================================================
    con.register("prediction_data_updates", prediction_data_updates)

    con.execute("""
        DELETE FROM prediction_data
        WHERE (game_id, team_id) IN (
            SELECT game_id, team_id
            FROM prediction_data_updates
        )
    """)

    con.execute("""
        INSERT INTO prediction_data
        SELECT *
        FROM prediction_data_updates
    """)

    print(f"Inserted {len(prediction_data_updates):,} Prediction data rows.")







