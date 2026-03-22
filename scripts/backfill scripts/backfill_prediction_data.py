import requests
import numpy as np
import pandas as pd
import time
import random
import warnings
import duckdb

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

DB_PATH = r"C:\Users\kjchr\Desktop\hockey\new_hockey_env\API data\duckdb\nhl_data.duckdb"


# ============================================================
# LOADERS
# ============================================================
def load_game_summary_df(con, season_id: int) -> pd.DataFrame:
    return con.execute(
        """
        SELECT *
        FROM all_game_summary
        WHERE season_id = ?
          AND gameState = 'OFF'
        ORDER BY game_id
        """,
        [season_id],
    ).df()


def load_existing_predictions(con, season_id: int) -> set[int]:
    rows = con.execute(
        """
        SELECT DISTINCT game_id
        FROM prediction_data
        WHERE season_id = ?
        """,
        [season_id],
    ).fetchall()
    return {r[0] for r in rows}



def load_teams(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("""
        SELECT team_id, tri_code
        FROM all_teams
    """).df()


    # ============================================================
    # Fetch Data
    # ============================================================
def fetch_tables(con: duckdb.DuckDBPyConnection, season_id: int) -> dict[str, pd.DataFrame]:
    print(f"📦 Loading tables for season {season_id}")
    tables: dict[str, pd.DataFrame] = {}

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
        SELECT *
        FROM all_standings_daily
        WHERE season_id = ?
        ORDER BY game_date
    """, [season_id]).df()

    tables["play_by_play"] = con.execute("""
        SELECT *
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
# Load offsets for incremental (last row per team pre-window)
# ============================================================
def load_prediction_offsets(season_id: int, start_game_id: int) -> pd.DataFrame:
    q = """
        SELECT *
        FROM team_daily_stats
        WHERE season_id = ?
            AND game_id < ?
        QUALIFY
            ROW_NUMBER() OVER (
                PARTITION BY team_id
                ORDER BY game_id DESC
            ) = 1
    """
    df = con.execute(q, [season_id, start_game_id]).df()

    if df.empty:
        return pd.DataFrame(columns=["team_id", "games_played_offset"])

    if "games_played" not in df.columns:
        raise KeyError("team_daily_stats is missing games_played")

    df["games_played_offset"] = (
        pd.to_numeric(df["games_played"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    KEY_COLS = {"season_id", "game_id", "team_id", "games_played", "games_played_offset"}

    df = df.rename(columns={c: f"{c}_offset" for c in df.columns if c not in KEY_COLS})

    keep_cols = ["team_id"] + [c for c in df.columns if c.endswith("_offset")]
    df = df.loc[:, keep_cols]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.drop_duplicates(subset=["team_id"]).reset_index(drop=True)

    return df

# ============================================================
# Games played spine (canonical)
# ============================================================
def build_games_played_spine(
    backfill_df: pd.DataFrame,
    offsets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    for _, row in backfill_df.iterrows():
        rows.append({
            "season_id": int(row["season_id"]),
            "game_id": int(row["game_id"]),
            "team_id": int(row["away_team"]),
            "game_date": pd.to_datetime(row["game_date"]),
        })
        rows.append({
            "season_id": int(row["season_id"]),
            "game_id": int(row["game_id"]),
            "team_id": int(row["home_team"]),
            "game_date": pd.to_datetime(row["game_date"]),
        })

    spine = pd.DataFrame(rows).sort_values(
        ["season_id", "team_id", "game_date", "game_id"]
    ).reset_index(drop=True)

    spine["games_before_local"] = spine.groupby(["season_id", "team_id"], sort=False).cumcount()
    spine["games_played_local"] = spine["games_before_local"] + 1

    spine["games_played_offset"] = 0
    if offsets is not None and not offsets.empty and "games_played_offset" in offsets.columns:
        off = offsets[["team_id", "games_played_offset"]].drop_duplicates("team_id")
        spine = spine.merge(off, on="team_id", how="left", suffixes=("", "_off"))
        spine["games_played_offset"] = spine["games_played_offset_off"].fillna(0).astype(int)
        spine = spine.drop(columns=["games_played_offset_off"])

    spine["games_before"] = spine["games_before_local"] + spine["games_played_offset"]
    spine["games_played"] = spine["games_played_local"] + spine["games_played_offset"]

    return spine[
        ["season_id", "game_id", "team_id", "game_date", "games_before", "games_played", "games_played_offset"]
    ]
    

# ============================================================
# Schema enforcement (DuckDB-aligned)
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
    "pk_toi": "DOUBLE",
    "pp_ga": "BIGINT",
    "pp_ga_pg": "DOUBLE",
    "sh_gf": "BIGINT",
    "sh_gf_pg": "DOUBLE",
    "c_pp_opps": "DOUBLE",
    "c_pp_net_g": "DOUBLE",
    "ppng_rank": "BIGINT",
    "c_pp_ga": "DOUBLE",
    "c_times_sh": "DOUBLE",
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

    # ============================================================
    # ✅ SAFE OFFSET FETCH (fixes your crash)
    # df.get("gp_offset", 0) returns an int if missing → no .fillna()
    # ============================================================

    if "gp_offset" in df.columns:
        gp_offset = pd.to_numeric(df["gp_offset"], errors="coerce").fillna(0)
    else:
        gp_offset = pd.Series(0, index=df.index)

    if "wins_offset" in df.columns:
        wins_offset = pd.to_numeric(df["wins_offset"], errors="coerce").fillna(0)
    else:
        wins_offset = pd.Series(0, index=df.index)


    # ============================================================
    # ✅ STEP 1: FIX game_win
    # ============================================================

    if "win" in df.columns:
        df["game_win"] = pd.to_numeric(df["win"], errors="coerce").fillna(0).astype("Int64")

    elif "game_win" in df.columns:
        df["game_win"] = pd.to_numeric(df["game_win"], errors="coerce").fillna(0).astype("Int64")

    else:
        df["game_win"] = pd.Series(0, index=df.index, dtype="Int64")


    # ============================================================
    # ✅ STEP 2: games_played + games_before WITH OFFSET
    # ============================================================

    if "gp_cum" in df.columns:
        df["games_played"] = (
            pd.to_numeric(df["gp_cum"], errors="coerce").fillna(0)
            + gp_offset
        )

    elif "games_played" in df.columns:
        df["games_played"] = (
            pd.to_numeric(df["games_played"], errors="coerce").fillna(0)
            + gp_offset
        )

    else:
        df["games_played"] = pd.Series(pd.NA, index=df.index)

    df["games_before"] = df["games_played"] - 1


    # ============================================================
    # ✅ STEP 3: wins_before WITH OFFSET
    # ============================================================

    if "wins_before" in df.columns:
        df["wins_before"] = (
            pd.to_numeric(df["wins_before"], errors="coerce").fillna(0)
            + wins_offset
        )

    elif "wins_std" in df.columns:
        df["wins_before"] = (
            pd.to_numeric(df["wins_std"], errors="coerce").fillna(0)
            - df["game_win"].fillna(0)
            + wins_offset
        )

    else:
        df["wins_before"] = wins_offset


    # ============================================================
    # ✅ STEP 4: win_pct_before recompute
    # ============================================================

    df["win_pct_before"] = pd.NA
    mask = df["games_before"] > 0

    df.loc[mask, "win_pct_before"] = (
        df.loc[mask, "wins_before"] / df.loc[mask, "games_before"]
    )


    # ============================================================
    # ✅ STEP 5: SPECIAL TEAMS TOTALS + PER GAME FIX
    # ============================================================

    # ---- TIMES SH ----
    if "times_sh" not in df.columns or df["times_sh"].isna().all():
        df["times_sh"] = (
            pd.to_numeric(df.get("c_times_sh", 0), errors="coerce")
            - pd.to_numeric(df.get("c_times_sh_offset", 0), errors="coerce")
        ).fillna(0)

    if "c_times_sh_offset" in df.columns:
        df["c_times_sh"] = (
            pd.to_numeric(df["times_sh"], errors="coerce").fillna(0).cumsum()
            + pd.to_numeric(df["c_times_sh_offset"], errors="coerce").fillna(0)
        )

    df["times_sh_pg"] = df["c_times_sh"] / df["games_played"]


    # ---- PP OPP ----
    if "pp_opp" not in df.columns or df["pp_opp"].isna().all():
        df["pp_opp"] = (
            pd.to_numeric(df.get("c_pp_opps", 0), errors="coerce")
            - pd.to_numeric(df.get("c_pp_opps_offset", 0), errors="coerce")
        ).fillna(0)

    if "c_pp_opps_offset" in df.columns:
        df["c_pp_opps"] = (
            pd.to_numeric(df["pp_opp"], errors="coerce").fillna(0).cumsum()
            + pd.to_numeric(df["c_pp_opps_offset"], errors="coerce").fillna(0)
        )

    df["pp_opp_pg"] = df["c_pp_opps"] / df["games_played"]


    # ---- NET PP GOALS ----
    if "pp_net_g" not in df.columns or df["pp_net_g"].isna().all():
        df["pp_net_g"] = (
            pd.to_numeric(df.get("c_pp_net_g", 0), errors="coerce")
            - pd.to_numeric(df.get("c_pp_net_g_offset", 0), errors="coerce")
        ).fillna(0)

    if "c_pp_net_g_offset" in df.columns:
        df["c_pp_net_g"] = (
            pd.to_numeric(df["pp_net_g"], errors="coerce").fillna(0).cumsum()
            + pd.to_numeric(df["c_pp_net_g_offset"], errors="coerce").fillna(0)
        )

    df["pp_net_pct"] = (df["c_pp_net_g"] / df["c_pp_opps"]) * 100


    # ============================================================
    # ✅ STEP 6: FINAL DUCKDB TYPE ENFORCEMENT
    # ============================================================

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
# Transition / Rush / Shot features
# ============================================================
def build_rush_stats(play_by_play: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = play_by_play.copy()

    shot_events = {"shot-on-goal", "missed-shot", "goal", "blocked-shot", "failed-shot-attempt"}
    transition_events = {"giveaway", "takeaway", "hit"}
    stoppage_events = {"faceoff", "stoppage", "period-end", "game-end", "penalty", "delayed-penalty"}

    valid_prev_events = shot_events | transition_events | stoppage_events

    plays = df[df["typeDescKey"].isin(valid_prev_events)].copy()
    plays = plays.sort_values(["game_id", "sortOrder"])

    g = plays.groupby("game_id", sort=False)

    plays["prev_event_time"] = g["game_in_seconds"].shift(1)
    plays["prev_event_type"] = g["typeDescKey"].shift(1)
    plays["prev_zone"] = g["zone_code"].shift(1)
    plays["prev_team_id"] = g["team_id"].shift(1)

    plays["prev_event_was_stoppage"] = plays["prev_event_type"].isin(stoppage_events)
    plays["time_diff"] = plays["game_in_seconds"] - plays["prev_event_time"]

    is_shot = plays["typeDescKey"].isin(shot_events)
    quick = plays["time_diff"].between(0, 5, inclusive="both")

    possession_change = plays["prev_team_id"].notna() & (plays["prev_team_id"] != plays["team_id"])
    same_team_from_dz_nz = (plays["prev_team_id"] == plays["team_id"]) & plays["prev_zone"].isin(["D", "N"])
    no_stoppage = ~plays["prev_event_was_stoppage"]

    rush_mask = is_shot & quick & no_stoppage & (possession_change | same_team_from_dz_nz)

    rush_events = (
        plays.loc[rush_mask]
        .drop_duplicates(subset=["season_id", "game_id", "team_id", "sortOrder"])
        .copy()
    )
    rush_events["rush_shot"] = 1

    rush_for = (
        rush_events
        .groupby(["season_id", "game_id", "team_id"], as_index=False)
        .size()
        .rename(columns={"size": "rush_shots_for"})
    )

    rush_against = (
        rush_events[rush_events["prev_team_id"].notna()]
        .groupby(["season_id", "game_id", "prev_team_id"], as_index=False)
        .size()
        .rename(columns={"prev_team_id": "team_id", "size": "rush_shots_against"})
    )

    rush_stats = (
        rush_for.merge(rush_against, on=["season_id", "game_id", "team_id"], how="outer")
        .fillna(0)
    )
    rush_stats["rush_shots_for"] = rush_stats["rush_shots_for"].astype(int)
    rush_stats["rush_shots_against"] = rush_stats["rush_shots_against"].astype(int)
    rush_stats["rush_shot_differential"] = rush_stats["rush_shots_for"] - rush_stats["rush_shots_against"]

    return rush_events, rush_stats

def cumulative_rush(rush_stats: pd.DataFrame, offsets: pd.DataFrame | None = None) -> pd.DataFrame:
    rush = rush_stats.copy()
    rush = rush.drop_duplicates(subset=["season_id", "game_id", "team_id"])

    sort_cols = (
        ["season_id", "team_id", "game_date", "game_id"]
        if "game_date" in rush.columns
        else ["season_id", "team_id", "game_id"]
    )
    rush = rush.sort_values(sort_cols)

    REQUIRED_OFFSETS = [
        "c_rush_shots_for_offset",
        "c_rush_shots_against_offset",
        "c_rush_diff_offset",
    ]

    if offsets is not None and not offsets.empty:
        available = ["team_id"] + [c for c in REQUIRED_OFFSETS if c in offsets.columns]
        rush = rush.merge(offsets[available], on="team_id", how="left")

    for c in REQUIRED_OFFSETS:
        if c not in rush.columns:
            rush[c] = 0
    rush[REQUIRED_OFFSETS] = rush[REQUIRED_OFFSETS].fillna(0)

    if "games_played" not in rush.columns:
        raise KeyError("games_played missing: merge games_spine before calling cumulative_rush()")

    g = rush.groupby(["season_id", "team_id"], sort=False)

    rush["c_rush_shots_for"] = g["rush_shots_for"].cumsum() + rush["c_rush_shots_for_offset"]
    rush["c_rush_shots_against"] = g["rush_shots_against"].cumsum() + rush["c_rush_shots_against_offset"]
    rush["c_rush_diff"] = g["rush_shot_differential"].cumsum() + rush["c_rush_diff_offset"]

    rush["rush_for_per_game"] = np.where(
        rush["games_played"] > 0, rush["c_rush_shots_for"] / rush["games_played"], 0.0
    )
    rush["rush_against_per_game"] = np.where(
        rush["games_played"] > 0, rush["c_rush_shots_against"] / rush["games_played"], 0.0
    )
    rush["rush_diff_per_game"] = np.where(
        rush["games_played"] > 0, rush["c_rush_diff"] / rush["games_played"], 0.0
    )

    return rush

def extract_shot_attempts(play_by_play: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    SHOT_CODES = [505, 506, 507, 508, 537]
    pbp = play_by_play.sort_values(["game_id", "sortOrder"]).copy()      

    shot_events = pbp[pbp["type_code"].isin(SHOT_CODES)].copy()
    shot_events["shot_attempt"] = 1

    teams = (
        pbp[["season_id", "game_id", "team_id"]]
        .drop_duplicates()
        )

    opponents = (
        teams
        .merge(
            teams,
            on=["season_id", "game_id"],
            suffixes=("", "_opp")
        )
        .query("team_id != team_id_opp")
        .rename(columns={"team_id_opp": "opponent_team_id"})
        )
    
    shot_events = shot_events.merge(
        opponents,
        on=["season_id", "game_id", "team_id"],
        how="left"
        )
    
    bad_games = (
        opponents
        .groupby(["season_id","game_id"])
        .size()
        .reset_index(name="rows")
        .query("rows != 2")
        )

    if not bad_games.empty:
        raise ValueError("Invalid team count in some games")

    shots_for = (
        shot_events.groupby(["season_id", "game_id", "team_id"], as_index=False)
        .size()
        .rename(columns={"size": "shot_attempts_for"})
    )

    shots_against = (
        shot_events[shot_events["opponent_team_id"].notna()]
        .groupby(["season_id", "game_id", "opponent_team_id"], as_index=False)
        .size()
        .rename(columns={"opponent_team_id": "team_id", "size": "shot_attempts_against"})
    )

    shot_stats = (
        shots_for.merge(shots_against, on=["season_id", "game_id", "team_id"], how="outer")
        .fillna(0)
    )
    shot_stats["shot_attempts_for"] = shot_stats["shot_attempts_for"].astype(int)
    shot_stats["shot_attempts_against"] = shot_stats["shot_attempts_against"].astype(int)
    shot_stats["shot_attempts_differential"] = shot_stats["shot_attempts_for"] - shot_stats["shot_attempts_against"]

    corsi_for = (
        shot_events[shot_events["zone_code"] == "O"]
        .groupby(["season_id", "game_id", "team_id"], as_index=False)
        .size()
        .rename(columns={"size": "corsi_for"})
    )

    corsi_against = (
        shot_events[(shot_events["zone_code"] == "O") & (shot_events["opponent_team_id"].notna())]
        .groupby(["season_id", "game_id", "opponent_team_id"], as_index=False)
        .size()
        .rename(columns={"opponent_team_id": "team_id", "size": "corsi_against"})
    )

    shot_stats = (
        shot_stats.merge(corsi_for, on=["season_id", "game_id", "team_id"], how="left")
        .merge(corsi_against, on=["season_id", "game_id", "team_id"], how="left")
        .fillna(0)
    )
    shot_stats["corsi_for"] = shot_stats["corsi_for"].astype(int)
    shot_stats["corsi_against"] = shot_stats["corsi_against"].astype(int)
    shot_stats["corsi_diff"] = shot_stats["corsi_for"] - shot_stats["corsi_against"]

    return shot_stats, shot_events

def cumulative_shot_attempts(shot_stats: pd.DataFrame, offsets: pd.DataFrame | None = None) -> pd.DataFrame:
    shots = shot_stats.copy()

    BASE_COLS = ["shot_attempts_for", "shot_attempts_against", "shot_attempts_differential", "corsi_for", "corsi_against"]
    for c in BASE_COLS:
        if c not in shots.columns:
            shots[c] = 0

    shots = shots.drop_duplicates(subset=["season_id", "game_id", "team_id"])

    sort_cols = (
        ["season_id", "team_id", "game_date", "game_id"]
        if "game_date" in shots.columns
        else ["season_id", "team_id", "game_id"]
    )
    shots = shots.sort_values(sort_cols)

    REQUIRED_OFFSETS = [
        "c_shots_for_offset",
        "c_shots_against_offset",
        "c_shots_diff_offset",
        "c_corsi_for_offset",
        "c_corsi_against_offset",
    ]

    if offsets is not None and not offsets.empty:
        available = ["team_id"] + [c for c in REQUIRED_OFFSETS if c in offsets.columns]
        shots = shots.merge(offsets[available], on="team_id", how="left")

    for c in REQUIRED_OFFSETS:
        if c not in shots.columns:
            shots[c] = 0
    shots[REQUIRED_OFFSETS] = shots[REQUIRED_OFFSETS].fillna(0)

    if "games_played" not in shots.columns:
        raise KeyError("games_played missing: merge games_spine before calling cumulative_shot_attempts()")

    g = shots.groupby(["season_id", "team_id"], sort=False)

    shots["c_shots_for"] = g["shot_attempts_for"].cumsum() + shots["c_shots_for_offset"]
    shots["c_shots_against"] = g["shot_attempts_against"].cumsum() + shots["c_shots_against_offset"]
    shots["c_shots_diff"] = g["shot_attempts_differential"].cumsum() + shots["c_shots_diff_offset"]

    shots["c_corsi_for"] = g["corsi_for"].cumsum() + shots["c_corsi_for_offset"]
    shots["c_corsi_against"] = g["corsi_against"].cumsum() + shots["c_corsi_against_offset"]
    shots["c_corsi_diff"] = shots["c_corsi_for"] - shots["c_corsi_against"]

    shots["shot_attempts_for_per_game"] = np.where(
        shots["games_played"] > 0, shots["c_shots_for"] / shots["games_played"], 0.0
    )
    shots["shot_attempts_against_per_game"] = np.where(
        shots["games_played"] > 0, shots["c_shots_against"] / shots["games_played"], 0.0
    )
    shots["shots_diff_per_game"] = np.where(
        shots["games_played"] > 0, shots["c_shots_diff"] / shots["games_played"], 0.0
    )
    shots["corsi_diff_per_game"] = np.where(
        shots["games_played"] > 0, shots["c_corsi_diff"] / shots["games_played"], 0.0
    )

    return shots

def preprocess_shot_dataframe(shot_events: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"x_coord", "y_coord", "game_id", "period_number", "game_in_seconds", "sortOrder"}
    missing = required_cols - set(shot_events.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    danger = shot_events.copy()

    hd = (
        ((danger["x_coord"] >= 66.5) & (danger["x_coord"] <= 89) & danger["y_coord"].between(-7, 7))
        | (danger["x_coord"].between(-89, -66.5) & danger["y_coord"].between(-7, 7))
    )

    md = (
        ((danger["x_coord"].between(69, 89)) & danger["y_coord"].between(-22, -7))
        | ((danger["x_coord"].between(54, 69)) & danger["y_coord"].between(-22, 22))
        | ((danger["x_coord"].between(40, 54)) & danger["y_coord"].between(-7, 7))
        | ((danger["x_coord"].between(-89, -69)) & danger["y_coord"].between(-22, -7))
        | ((danger["x_coord"].between(-69, -54)) & danger["y_coord"].between(-22, 22))
        | ((danger["x_coord"].between(-54, -40)) & danger["y_coord"].between(-7, 7))
    )

    ld = ~(hd | md)

    danger["hd_shot"] = hd.astype(int)
    danger["md_shot"] = md.astype(int)
    danger["ld_shot"] = ld.astype(int)

    danger = danger.sort_values(["game_id", "sortOrder"])
    danger["game_time"] = (danger["period_number"] - 1) * 1200 + danger["game_in_seconds"]
    danger["time_since_last_shot"] = danger.groupby("game_id")["game_time"].diff()
    danger["is_rebound"] = danger["time_since_last_shot"].le(3).astype(int)

    return danger

def build_enriched_shot_events(
    shot_events: pd.DataFrame,
    danger_events: pd.DataFrame,
    rush_events: pd.DataFrame,
) -> pd.DataFrame:
    key_cols = ["season_id", "game_id", "team_id", "sortOrder"]

    combined = (
        shot_events
        .merge(danger_events[key_cols + ["hd_shot", "md_shot", "ld_shot", "is_rebound"]], on=key_cols, how="left")
        .merge(rush_events[key_cols + ["rush_shot"]], on=key_cols, how="left")
    )

    combined["hd_shot"] = combined["hd_shot"].fillna(0).astype(int)
    combined["md_shot"] = combined["md_shot"].fillna(0).astype(int)
    combined["ld_shot"] = combined["ld_shot"].fillna(0).astype(int)
    combined["is_rebound"] = combined["is_rebound"].fillna(0).astype(int)
    combined["rush_shot"] = combined["rush_shot"].fillna(0).astype(int)

    return combined

def add_scoring_chances(combined_chances_data: pd.DataFrame) -> pd.DataFrame: ### scoring chance for and against are not being calculated correctly
    df = combined_chances_data.copy()
    df = df.sort_values(["game_id", "sortOrder"])

    df["base_scoring_chance_score"] = 3 * df["hd_shot"] + 2 * df["md_shot"] + 1 * df["ld_shot"]
    df["scoring_chance"] = (df["base_scoring_chance_score"] >= 2).astype(int)

    df["context_bonus"] = (df["is_rebound"] == 1).astype(int) + (df["rush_shot"] == 1).astype(int)
    df["blocked_penalty"] = (df["type_code"] == 508).astype(int)
    
    df["adj_scoring_chance_score"] = df["base_scoring_chance_score"] + df["context_bonus"]- df["blocked_penalty"]

    ## Outcome flags
    df["scoring_chance_goal"] = (df["type_code"] == 505).astype(int)
    df["shot_on_goal"] = (df["type_code"] == 506).astype(int) + df["scoring_chance_goal"]

    df["rush_scoring_chance"] = ((df["scoring_chance"] == 1) & (df["rush_shot"] == 1)).astype(int)
    return df

def build_danger_corsi_and_scf_stats(scoring_chances_events: pd.DataFrame) -> pd.DataFrame: ########THIS NEEDS TO BE FIXED, THE CF AND CA ARE COMING OUT THE SAME
    d = scoring_chances_events.copy()
    # -----------------------------
    # Assign shooting team correctly
    # -----------------------------
    d["shooting_team_id"] = d["team_id"]

    mask_block = d["type_code"] == 508
    d.loc[mask_block, "shooting_team_id"] = d.loc[mask_block, "opponent_team_id"]

    # -----------------------------
    # Scoring chance shot + goal flags
    # -----------------------------
    d["shot_on_goal"] = d["type_code"].isin([505, 506]).astype(int)
    d["scoring_chance_goal"] = (d["type_code"] == 505).astype(int)

    # -----------------------------
    # FOR — scoring chances
    # -----------------------------
    scf = (
        d.groupby(["season_id", "game_id", "shooting_team_id"], as_index=False)
        .agg(
            SCF=("scoring_chance", "sum"),
            SCSF=("shot_on_goal", "sum"),
            SCGF=("scoring_chance_goal", "sum"),
        )
        .rename(columns={"shooting_team_id": "team_id"})
    )

    # -----------------------------
    # AGAINST — scoring chances (opponent-based)
    # -----------------------------
    sca = (
        d.groupby(["season_id", "game_id", "opponent_team_id"], as_index=False)
        .agg(SCA=("scoring_chance", "sum"))
        .rename(columns={"opponent_team_id": "team_id"})
    )

    # -----------------------------
    # Danger-split Corsi FOR
    # -----------------------------
    d["hd_corsi"] = d["hd_shot"]
    d["md_corsi"] = d["md_shot"]
    d["ld_corsi"] = d["ld_shot"]

    corsi_for = (
        d.groupby(["season_id", "game_id", "team_id"], as_index=False)
        .agg(
            HDCF=("hd_corsi", "sum"),
            MDCF=("md_corsi", "sum"),
            LDCF=("ld_corsi", "sum"),
        )
    )

    # -----------------------------
    # Danger-split Corsi AGAINST (opponent-based)
    # -----------------------------
    corsi_against = (
        d.groupby(["season_id", "game_id", "opponent_team_id"], as_index=False)
        .agg(
            HDCA=("hd_corsi", "sum"),
            MDCA=("md_corsi", "sum"),
            LDCA=("ld_corsi", "sum"),
        )
        .rename(columns={"opponent_team_id": "team_id"})
    )

    # -----------------------------
    # FINAL MERGE
    # -----------------------------
    final = (
        scf
        .merge(sca, on=["season_id", "game_id", "team_id"], how="left")
        .merge(corsi_for, on=["season_id", "game_id", "team_id"], how="left")
        .merge(corsi_against, on=["season_id", "game_id", "team_id"], how="left")
        .fillna(0)
    )

    # -----------------------------
    # Percentages
    # -----------------------------
    final["HDCF_pct"] = final["HDCF"] / (final["HDCF"] + final["HDCA"]).replace(0, np.nan)
    final["HDCA_pct"] = final["HDCA"] / (final["HDCF"] + final["HDCA"]).replace(0, np.nan)

    final["MDCF_pct"] = final["MDCF"] / (final["MDCF"] + final["MDCA"]).replace(0, np.nan)
    final["MDCA_pct"] = final["MDCA"] / (final["MDCF"] + final["MDCA"]).replace(0, np.nan)

    final["LDCF_pct"] = final["LDCF"] / (final["LDCF"] + final["LDCA"]).replace(0, np.nan)
    final["LDCA_pct"] = final["LDCA"] / (final["LDCF"] + final["LDCA"]).replace(0, np.nan)

    return final

def cumulative_scoring_chances(
    danger_scoring: pd.DataFrame,
    corsi_stats: pd.DataFrame,
    offsets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    scoring = danger_scoring.copy()
    shots = corsi_stats.copy()

    sort_cols = (
        ["season_id", "team_id", "game_date", "game_id"]
        if "game_date" in scoring.columns
        else ["season_id", "team_id", "game_id"]
    )

    scoring = scoring.drop_duplicates(subset=["season_id", "game_id", "team_id"]).sort_values(sort_cols)
    shots = shots.drop_duplicates(subset=["season_id", "game_id", "team_id"]).sort_values(sort_cols)

    REQUIRED_OFFSETS = [
        "c_SCF_offset", "c_SCSF_offset", "c_SCGF_offset", "c_SCA_offset",
        "c_HDCF_offset", "c_HDCA_offset",
        "c_MDCF_offset", "c_MDCA_offset",
        "c_LDCF_offset", "c_LDCA_offset",
        "c_corsi_for_offset", "c_corsi_against_offset",
    ]

    if offsets is not None and not offsets.empty:
        available = ["team_id"] + [c for c in REQUIRED_OFFSETS if c in offsets.columns]
        scoring = scoring.merge(offsets[available], on="team_id", how="left")
        shots = shots.merge(offsets[available], on="team_id", how="left")

    for c in REQUIRED_OFFSETS:
        if c not in scoring.columns:
            scoring[c] = 0
        if c not in shots.columns:
            shots[c] = 0

    scoring[REQUIRED_OFFSETS] = scoring[REQUIRED_OFFSETS].fillna(0)
    shots[REQUIRED_OFFSETS] = shots[REQUIRED_OFFSETS].fillna(0)

    if "games_played" not in scoring.columns:
        raise KeyError("games_played missing: merge games_spine before calling cumulative_scoring_chances()")

    g_sc = scoring.groupby(["season_id", "team_id"], sort=False)
    g_cf = shots.groupby(["season_id", "team_id"], sort=False)

    scoring["c_SCF"] = g_sc["SCF"].cumsum() + scoring["c_SCF_offset"]
    scoring["c_SCSF"] = g_sc["SCSF"].cumsum() + scoring["c_SCSF_offset"]
    scoring["c_SCGF"] = g_sc["SCGF"].cumsum() + scoring["c_SCGF_offset"]
    scoring["c_SCA"] = g_sc["SCA"].cumsum() + scoring["c_SCA_offset"]

    scoring["c_HDCF"] = g_sc["HDCF"].cumsum() + scoring["c_HDCF_offset"]
    scoring["c_HDCA"] = g_sc["HDCA"].cumsum() + scoring["c_HDCA_offset"]
    scoring["c_MDCF"] = g_sc["MDCF"].cumsum() + scoring["c_MDCF_offset"]
    scoring["c_MDCA"] = g_sc["MDCA"].cumsum() + scoring["c_MDCA_offset"]
    scoring["c_LDCF"] = g_sc["LDCF"].cumsum() + scoring["c_LDCF_offset"]
    scoring["c_LDCA"] = g_sc["LDCA"].cumsum() + scoring["c_LDCA_offset"]

    scoring["SCF_pg"] = np.where(scoring["games_played"] > 0, scoring["c_SCF"] / scoring["games_played"], 0.0)
    scoring["SCA_pg"] = np.where(scoring["games_played"] > 0, scoring["c_SCA"] / scoring["games_played"], 0.0)

    shots["c_corsi_for"] = g_cf["corsi_for"].cumsum() + shots["c_corsi_for_offset"]
    shots["c_corsi_against"] = g_cf["corsi_against"].cumsum() + shots["c_corsi_against_offset"]
    shots["c_corsi_diff"] = shots["c_corsi_for"] - shots["c_corsi_against"]

    shots["corsi_diff_per_game"] = np.where(
        shots["games_played"] > 0, shots["c_corsi_diff"] / shots["games_played"], 0.0
    )

    merge_cols = ["season_id", "team_id", "game_id"]
    if "game_date" in scoring.columns and "game_date" in shots.columns:
        merge_cols.append("game_date")

    scoring = scoring.merge(
        shots[merge_cols + ["c_corsi_for", "c_corsi_against", "c_corsi_diff", "corsi_diff_per_game"]],
        on=merge_cols,
        how="left",
    )

    return scoring

def build_team_shooting_stats(scoring_chances_events: pd.DataFrame) -> pd.DataFrame:
    df = scoring_chances_events.copy()

    df["goal"] = df["scoring_chance_goal"]
    df["sog"] = df["shot_on_goal"]

    df["hd_sog"] = ((df["sog"] == 1) & (df["hd_shot"] == 1)).astype(int)
    df["md_sog"] = ((df["sog"] == 1) & (df["md_shot"] == 1)).astype(int)
    df["ld_sog"] = ((df["sog"] == 1) & (df["ld_shot"] == 1)).astype(int)

    df["hd_goal"] = ((df["goal"] == 1) & (df["hd_shot"] == 1)).astype(int)
    df["md_goal"] = ((df["goal"] == 1) & (df["md_shot"] == 1)).astype(int)
    df["ld_goal"] = ((df["goal"] == 1) & (df["ld_shot"] == 1)).astype(int)

    teams_map = (
        df[["game_id", "team_id"]].drop_duplicates()
        .merge(df[["game_id", "team_id"]].drop_duplicates(), on="game_id", suffixes=("", "_opp"))
        .query("team_id != team_id_opp")
        .rename(columns={"team_id_opp": "opponent_team_id"})
    )
    df = df.merge(teams_map, on=["game_id", "team_id"], how="left")

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
    stats_for["shot_pct"] = 100 * stats_for["goals_for"] / stats_for["shots_for"].replace(0, np.nan)

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

    shooting_stats = (
        stats_for.merge(stats_against, on=["season_id", "game_id", "team_id"], how="left")
        .fillna(0)
    )
    shooting_stats["sv_pct"] = 100 - 100 * shooting_stats["goals_against"] / shooting_stats["shots_against"].replace(0, np.nan)
    shooting_stats["pdo"] = shooting_stats["shot_pct"] + shooting_stats["sv_pct"]

    return shooting_stats

def cumulative_sog(sog_stats: pd.DataFrame, offsets: pd.DataFrame | None = None) -> pd.DataFrame:
    sog = sog_stats.copy()

    sog = sog.drop_duplicates(subset=["season_id", "game_id", "team_id"])
    sort_cols = (
        ["season_id", "team_id", "game_date", "game_id"]
        if "game_date" in sog.columns
        else ["season_id", "team_id", "game_id"]
    )
    sog = sog.sort_values(sort_cols)

    REQUIRED_OFFSETS = [
        "c_shots_for_offset",
        "c_goals_for_offset",
        "c_hd_sog_offset",
        "c_md_sog_offset",
        "c_ld_sog_offset",
        "c_hd_goals_offset",
        "c_md_goals_offset",
        "c_ld_goals_offset",
        "c_shots_against_offset",
        "c_goals_against_offset",
        "c_hd_against_offset",
        "c_md_against_offset",
        "c_ld_against_offset",
    ]

    if offsets is not None and not offsets.empty:
        available = ["team_id"] + [c for c in REQUIRED_OFFSETS if c in offsets.columns]
        sog = sog.merge(offsets[available], on="team_id", how="left")

    for c in REQUIRED_OFFSETS:
        if c not in sog.columns:
            sog[c] = 0
    sog[REQUIRED_OFFSETS] = sog[REQUIRED_OFFSETS].fillna(0)

    if "games_played" not in sog.columns:
        raise KeyError("games_played missing: merge games_spine before calling cumulative_sog()")

    g = sog.groupby(["season_id", "team_id"], sort=False)

    sog["c_shots_for"] = g["shots_for"].cumsum() + sog["c_shots_for_offset"]
    sog["c_goals_for"] = g["goals_for"].cumsum() + sog["c_goals_for_offset"]

    sog["c_hd_sog"] = g["hd_sog"].cumsum() + sog["c_hd_sog_offset"]
    sog["c_md_sog"] = g["md_sog"].cumsum() + sog["c_md_sog_offset"]
    sog["c_ld_sog"] = g["ld_sog"].cumsum() + sog["c_ld_sog_offset"]

    sog["c_hd_goals"] = g["hd_goals"].cumsum() + sog["c_hd_goals_offset"]
    sog["c_md_goals"] = g["md_goals"].cumsum() + sog["c_md_goals_offset"]
    sog["c_ld_goals"] = g["ld_goals"].cumsum() + sog["c_ld_goals_offset"]

    sog["c_shots_against"] = g["shots_against"].cumsum() + sog["c_shots_against_offset"]
    sog["c_goals_against"] = g["goals_against"].cumsum() + sog["c_goals_against_offset"]

    sog["c_hd_against"] = g["hd_against"].cumsum() + sog["c_hd_against_offset"]
    sog["c_md_against"] = g["md_against"].cumsum() + sog["c_md_against_offset"]
    sog["c_ld_against"] = g["ld_against"].cumsum() + sog["c_ld_against_offset"]

    sog["c_shot_pct"] = np.where(sog["c_shots_for"] > 0, sog["c_goals_for"] / sog["c_shots_for"], 0.0)
    sog["c_sv_pct"] = np.where(sog["c_shots_against"] > 0, 1 - (sog["c_goals_against"] / sog["c_shots_against"]), 0.0)
    sog["c_pdo"] = (sog["c_shot_pct"] + sog["c_sv_pct"]) * 100

    return sog

def build_transition_offense_features(
    play_by_play: pd.DataFrame,
    games_spine: pd.DataFrame,
    offsets: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:

    print("\n▶ Running transition offense orchestrator")

    rush_events, rush_stats = build_rush_stats(play_by_play)
    shot_stats, shot_events = extract_shot_attempts(play_by_play)
    # ✅ Attach canonical games_played spine BEFORE cumulative calcs
    rush_stats = rush_stats.merge(
        games_spine[["season_id", "game_id", "team_id", "games_played", "game_date"]],
        on=["season_id", "game_id", "team_id"],
        how="left"
    )

    shot_stats = shot_stats.merge(
        games_spine[["season_id", "game_id", "team_id", "games_played", "game_date"]],
        on=["season_id", "game_id", "team_id"],
        how="left"
    )

    danger_events = preprocess_shot_dataframe(shot_events)
    combined_chances_data = build_enriched_shot_events(shot_events, danger_events, rush_events)
    scoring_chances_events = add_scoring_chances(combined_chances_data)

    danger_stats = build_danger_corsi_and_scf_stats(scoring_chances_events)
    sog_stats = build_team_shooting_stats(scoring_chances_events)
    danger_stats = danger_stats.merge(
        games_spine[["season_id", "game_id", "team_id", "games_played", "game_date"]],
        on=["season_id", "game_id", "team_id"],
        how="left"
    )

    sog_stats = sog_stats.merge(
        games_spine[["season_id", "game_id", "team_id", "games_played", "game_date"]],
        on=["season_id", "game_id", "team_id"],
        how="left"
    )

    return {
        "cumulative_rush": cumulative_rush(rush_stats, offsets=offsets),
        "cumulative_shooting": cumulative_shot_attempts(shot_stats, offsets=offsets),
        "cumulative_scoring": cumulative_scoring_chances(danger_stats, shot_stats, offsets=offsets),
        "cumulative_sog": cumulative_sog(sog_stats, offsets=offsets),
    }

# ============================================================
# Strength of schedule (uses games_before)
# ============================================================
def strength_of_schedule(backfill_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in backfill_df.iterrows():
        rows.append({
            "season_id": int(row["season_id"]),
            "game_id": int(row["game_id"]),
            "game_date": pd.to_datetime(row["game_date"]),
            "team_id": int(row["away_team"]),
            "tri_code": row["away_tri_code"],
            "opponent": row["home_tri_code"],
            "goals_for": row["awayTeam.score"],
            "goals_against": row["homeTeam.score"],
        })
        rows.append({
            "season_id": int(row["season_id"]),
            "game_id": int(row["game_id"]),
            "game_date": pd.to_datetime(row["game_date"]),
            "team_id": int(row["home_team"]),
            "tri_code": row["home_tri_code"],
            "opponent": row["away_tri_code"],
            "goals_for": row["homeTeam.score"],
            "goals_against": row["awayTeam.score"],
        })

    games_sos = pd.DataFrame(rows).sort_values(
        ["season_id", "team_id", "game_date", "game_id"]
    ).reset_index(drop=True)

    games_sos["win"] = (games_sos["goals_for"] > games_sos["goals_against"]).astype(int)

    g = games_sos.groupby(["season_id", "team_id"], sort=False)
    games_sos["games_before"] = g.cumcount()
    games_sos["games_played"] = games_sos["games_before"] + 1

    games_sos["wins_before"] = g["win"].cumsum().shift(1).fillna(0)
    games_sos["win_pct_before"] = np.where(
        games_sos["games_before"] > 0,
        games_sos["wins_before"] / games_sos["games_before"],
        np.nan,
    )

    opp_win = (
        games_sos[["season_id", "tri_code", "game_id", "win_pct_before"]]
        .rename(columns={"tri_code": "opponent", "win_pct_before": "opponent_win_pct_before"})
    )

    games_sos = games_sos.merge(
        opp_win[["season_id", "opponent", "game_id", "opponent_win_pct_before"]],
        on=["season_id", "opponent", "game_id"],
        how="left",
    )

    games_sos["sos_past"] = (
        games_sos.groupby(["season_id", "team_id"])["opponent_win_pct_before"]
        .expanding().mean()
        .reset_index(level=[0, 1], drop=True)
    )

    games_sos["sos_future"] = (
        games_sos.iloc[::-1]
        .groupby(["season_id", "team_id"])["opponent_win_pct_before"]
        .expanding().mean()
        .reset_index(level=[0, 1], drop=True)
        .iloc[::-1]
    )

    return games_sos.drop(columns=["goals_for", "goals_against", "opponent_win_pct_before"])
# ============================================================
# Calculate Features from Goalie Stats
# ============================================================

def build_team_goalie_stats(
    goalies: pd.DataFrame,
    teams: pd.DataFrame,
    games_spine: pd.DataFrame,
) -> pd.DataFrame:

    def safe_weighted_mean(values, weights):
        values = pd.to_numeric(values, errors="coerce")
        weights = pd.to_numeric(weights, errors="coerce")
        mask = (values.notna()) & (weights.notna()) & (weights > 0)
        if not mask.any():
            return np.nan
        return (values[mask] * weights[mask]).sum() / weights[mask].sum()

    g = goalies.copy()

    # -------------------------------------------------
    # Individual goalie stats per game
    # -------------------------------------------------
    goalies_indiv = (
        g.groupby(
            ["goalie_name", "season_id", "game_id", "tri_code"],
            as_index=False,
            sort=False,
        )
        .agg(
            i_savePct=("save_pct", lambda x: safe_weighted_mean(x, g.loc[x.index, "shots_against"])),
            i_GAA=("gaa", lambda x: safe_weighted_mean(x, g.loc[x.index, "toi"])),
            i_wins=("wins", "sum"),
            i_losses=("losses", "sum"),
            i_otlosses=("ot_losses", "sum"),
            i_shutouts=("shutouts", "sum"),
            i_saves=("saves", "sum"),
            i_shotsAgainst=("shots_against", "sum"),
            i_timeOnIce=("toi", "sum"),
        )
    )

    # -------------------------------------------------
    # Team-level goalie stats per game
    # -------------------------------------------------
    team_goalie_games = (
        goalies_indiv.groupby(
            ["season_id", "game_id", "tri_code"],
            as_index=False,
            sort=False,
        )
        .agg(
            game_savePct=(
                "i_savePct",
                lambda x: safe_weighted_mean(
                    x, goalies_indiv.loc[x.index, "i_shotsAgainst"]
                ),
            ),
            game_GAA=(
                "i_GAA",
                lambda x: safe_weighted_mean(
                    x, goalies_indiv.loc[x.index, "i_timeOnIce"]
                ),
            ),
            game_wins=("i_wins", "sum"),
            game_losses=("i_losses", "sum"),
            game_shutouts=("i_shutouts", "sum"),
            game_saves=("i_saves", "sum"),
            game_shotsAgainst=("i_shotsAgainst", "sum"),
        )
    )

    # -------------------------------------------------
    # Attach team_id
    # -------------------------------------------------
    team_goalie_games = (
        team_goalie_games
        .merge(
            teams[["team_id", "tri_code"]].drop_duplicates(),
            on="tri_code",
            how="left",
        )
        .drop_duplicates(subset=["season_id", "game_id", "team_id"])
    )

    # -------------------------------------------------
    # Merge canonical games_played spine
    # -------------------------------------------------
    team_goalie_games = team_goalie_games.merge(
        games_spine[
            ["season_id", "game_id", "team_id", "game_date", "games_played"]
        ],
        on=["season_id", "game_id", "team_id"],
        how="left",
    )

    return (
        team_goalie_games
        .sort_values(["season_id", "game_date", "game_id", "team_id"])
        .reset_index(drop=True)
    )
# ============================================================
# Calculate Features from Special Teams
# ============================================================
def build_special_teams_stats(
    power_play: pd.DataFrame,
    penalty_kill: pd.DataFrame,
    games_spine: pd.DataFrame,
    offsets: pd.DataFrame | None = None,
) -> pd.DataFrame:

    pp = power_play.copy()
    pk = penalty_kill.copy()
    
    # -------------------------------------------------
    # ✅ Ensure required keys exist
    # -------------------------------------------------
    for df in (pp, pk):
        for col in ["season_id", "game_id", "team_id"]:
            if col not in df.columns:
                raise KeyError(f"❌ {col} missing from special teams input table")

    # -------------------------------------------------
    # ✅ Ensure game_date exists (prefer PP/PK; fallback to games_spine)
    #    Also attach canonical games_played from games_spine
    # -------------------------------------------------
    # Normalize games_spine to the 4 columns we need (and coalesce if it has _x/_y)
    spine = games_spine.copy()
    
    # If someone accidentally passed backfill_df/game_summary instead of games_spine,
    # build a minimal spine from it (away/home rows) as a fallback.
    if "games_played" not in spine.columns or "team_id" not in spine.columns:
        if {"away_team", "home_team", "season_id", "game_id", "game_date"}.issubset(spine.columns):
            rows = []
            for _, r in spine.iterrows():
                rows.append(
                    {"season_id": int(r["season_id"]), "game_id": int(r["game_id"]),
                    "team_id": int(r["away_team"]), "game_date": pd.to_datetime(r["game_date"], errors="coerce")}
                )
                rows.append(
                    {"season_id": int(r["season_id"]), "game_id": int(r["game_id"]),
                    "team_id": int(r["home_team"]), "game_date": pd.to_datetime(r["game_date"], errors="coerce")}
                )
            spine = pd.DataFrame(rows).sort_values(["season_id", "team_id", "game_date", "game_id"]).reset_index(drop=True)
            spine["games_played"] = spine.groupby(["season_id", "team_id"], sort=False).cumcount() + 1
        else:
            raise KeyError("❌ games_spine must contain team_id + games_played (or be a game_summary-like df with away_team/home_team)")

    # Coalesce common duped cols if present
    for base in ["game_date", "team_id", "games_played"]:
        x, y = f"{base}_x", f"{base}_y"
        if x in spine.columns and y in spine.columns and base not in spine.columns:
            spine[base] = spine[x].combine_first(spine[y])
            spine = spine.drop(columns=[x, y])

    spine = spine.loc[:, [c for c in ["season_id", "game_id", "team_id", "game_date", "games_played"] if c in spine.columns]]
    spine = spine.drop_duplicates(subset=["season_id", "game_id", "team_id"])

    # Merge spine and pp/pk with spine on season, game, and team_id
    if "game_date" not in pp.columns:
        pp = pp.merge(spine[["season_id", "game_id", "team_id", "game_date"]], on=["season_id", "game_id", "team_id"], how="left")
    if "game_date" not in pk.columns:
        pk = pk.merge(spine[["season_id", "game_id", "team_id", "game_date"]], on=["season_id", "game_id", "team_id"], how="left")

    # 
    pp = pp.merge(spine[["season_id", "game_id", "team_id", "games_played"]], on=["season_id", "game_id", "team_id"], how="left")
    pk = pk.merge(spine[["season_id", "game_id", "team_id", "games_played"]], on=["season_id", "game_id", "team_id"], how="left")
    print(pp[pp['team_id']==6])
    # -------------------------------------------------
    # ✅ Ensure game_date is datetime (after any merges)
    # -------------------------------------------------
    pp["game_date"] = pd.to_datetime(pp["game_date"], errors="coerce")
    pk["game_date"] = pd.to_datetime(pk["game_date"], errors="coerce")

    # -------------------------------------------------
    # ✅ Sort BEFORE any cumulative work
    # -------------------------------------------------
    pp = pp.sort_values(["season_id", "team_id", "game_date", "game_id"]).reset_index(drop=True)
    pk = pk.sort_values(["season_id", "team_id", "game_date", "game_id"]).reset_index(drop=True)

    # -------------------------------------------------
    # ✅ Offsets merge (MUST happen before groupby/cumsum)
    # -------------------------------------------------
    print(offsets.columns)
    OFFSET_COLS = [
        "team_id",
        "c_pp_net_g_offset", #cumulative pp net goals
        "c_pp_opps_offset", # cumulative pp opps
        "c_pp_ga_offset", # cumulative pp goals allowed
        "c_times_sh_offset", # cumulative times_sh
        "c_pp_gf_offset", #cumulative ppgf 
        "c_sh_ga_offset", 
        "c_sh_gf_offset",
        "c_pk_net_g_offset",
    ]

    if offsets is not None and not offsets.empty:
        offsets_clean = (
            offsets.loc[:, ~offsets.columns.duplicated()]
            .drop_duplicates(subset=["team_id"])
            .reset_index(drop=True)
        )
        use_cols = [c for c in OFFSET_COLS if c in offsets_clean.columns]
        
        pp = pp.merge(offsets_clean[use_cols], on="team_id", how="left")
        pk = pk.merge(offsets_clean[use_cols], on="team_id", how="left")

    # -------------------------------------------------
    # ✅ Ensure offset columns exist (default = 0)
    # -------------------------------------------------
    for c in OFFSET_COLS:
        if c == "team_id":
            continue
        if c not in pp.columns:
            pp[c] = 0
        if c not in pk.columns:
            pk[c] = 0

    # -------------------------------------------------
    # ✅ Ensure base stat columns exist
    # -------------------------------------------------
    for c in ["pp_net_g", "pp_opportunities"]:
        if c not in pp.columns:
            pp[c] = 0
    for c in ["pp_ga", "times_sh"]:
        if c not in pk.columns:
            pk[c] = 0

    # Numeric safety
    for c in ["pp_net_g", "pp_opportunities", "pp_ga", "times_sh", "games_played"]:
        if c in pp.columns:
            pp[c] = pd.to_numeric(pp[c], errors="coerce")
        if c in pk.columns:
            pk[c] = pd.to_numeric(pk[c], errors="coerce")

    pp = pp.fillna(0)
    pk = pk.fillna(0)

    # -------------------------------------------------
    # ✅ NOW build groupby objects (after all merges)
    # -------------------------------------------------
    g_pp = pp.groupby(["season_id", "team_id"], sort=False)
    g_pk = pk.groupby(["season_id", "team_id"], sort=False)

    # -------------------------------------------------
    # POWER PLAY cumulative
    # -------------------------------------------------
    pp["c_pp_net_g"] = g_pp["pp_net_g"].cumsum() + pp["c_pp_net_g_offset"]
    pp["c_pp_opps"]  = g_pp["pp_opportunities"].cumsum() + pp["c_pp_opps_offset"]
    pp["c_pp_gf"]   = g_pp["pp_gf"].cumsum() + pp["c_pp_gf_offset"]
    pp['c_sh_ga']   = g_pp["sh_ga"].cumsum() + pp["c_sh_ga_offset"]

    
    # Net PP rate (goals per opp)
    pp["pp_net_pct"] = np.where(
        pp["c_pp_opps"] > 0,
        pp["c_pp_net_g"] / pp["c_pp_opps"],
        np.nan,
    )

    # -------------------------------------------------
    # PENALTY KILL cumulative
    # -------------------------------------------------
    pk["c_pp_ga"]    = g_pk["pp_ga"].cumsum() + pk["c_pp_ga_offset"]
    pk["c_times_sh"] = g_pk["times_sh"].cumsum() + pk["c_times_sh_offset"]
    pk['c_sh_gf'] = g_pk["sh_gf"].cumsum() + pk["c_sh_gf_offset"]
    pk['c_pk_net_g'] = g_pk["pk_net_g"].cumsum() + pk["c_pk_net_g_offset"]
    
    # Net PK rate (GA per times SH) — kept for backwards-compat with your existing fields
    pk["pk_net_pct"] = np.where(
        pk["c_times_sh"] > 0,
        pk["c_pp_ga"] / pk["c_times_sh"],
        np.nan,
    )

    # -------------------------------------------------
    # ✅ Per-game special teams rates (use canonical games_played)
    # -------------------------------------------------
    
    # power play opportunities per game at this point in the season
    pp["pp_opp_pg"] = np.where(
        pp["games_played"] > 0,
        pp["c_pp_opps"] / pp["games_played"],
        0.0,
    )

    # power play goals per game, at this point
    pp["pp_gpg"] = np.where(
        pp["games_played"] > 0,
        pp["c_pp_gf"] / pp["games_played"],
        0.0,
    )

    # short handed goals allowed per game, at this point
    pp['sh_ga_pg'] = np.where(
        pp["games_played"] > 0,
        pp["c_sh_ga"] / pp["games_played"],
        0.0, 
    )
    
    pk['pp_ga_pg'] = np.where(
        pk["games_played"] > 0,
        pk["c_pp_ga"] / pk["games_played"],
        0.0, 
    )        

    pp["pp_net_gpg"] = np.where(
        pp["games_played"] > 0,
        pp["c_pp_net_g"] / pp["games_played"],
        0.0,
    )

    pk["sh_gf_pg"] = np.where(
        pk["games_played"] > 0,
        pk["c_sh_gf"] / pp["games_played"],
        0.0,
    )

    
    pk["pk_net_gpg"] = np.where(
        pk["games_played"] > 0,
        pk["c_pk_net_g"] / pp["games_played"],
        0.0,
    )

    pk["times_sh_pg"] = np.where(
        pk["games_played"] > 0,
        pk["c_times_sh"] / pk["games_played"],
        0.0,
    )

    # Traditional PK% = 1 - GA / Times SH
    pk["pk_pct"] = np.where(
        pk["c_times_sh"] > 0,
        1 - (pk["c_pp_ga"] / pk["c_times_sh"]),
        1.0,
    )


    pp["pp_net_pct"] = (pp["pp_net_pct"] * 100).round(1)
    pk["pk_net_pct"] = (pk["pk_net_pct"] * 100).round(1)
    pk["pk_pct"]     = (pk["pk_pct"] * 100).round(1)
    pp["pp_pct"]    =(pp["pp_pct"] * 100).round(1)

    # -------------------------------------------------
    # Daily ranks
    # -------------------------------------------------
    pp["ppng_rank"] = (
        pp.groupby(["season_id", "game_date"])["pp_net_pct"]
        .rank(method="dense", ascending=False)
    )

    pk["pk_rank"] = (
        pk.groupby(["season_id", "game_date"])["pk_net_pct"]
        .rank(method="dense", ascending=True)
    )

    pp = pp[[
            "season_id",
            "game_id",
            "team_id",
            "game_date",
            "pp_gf",            # power play goals for
            "pp_net_pct",       # power play net percentage
            "pp_pct",           # power play percentage
            "pp_net_g",         # power play net goals
            "pp_opportunities", # power plays count
            "pp_toi",           # power plays time on ice
            "sh_ga",            # short handed goal allowed
            "pp_opp_pg",        # power play opportunities per game (avg)
            "pp_gpg",           # power play goals per game (avg)
            "pp_net_gpg",       # power play net goals per game (avg)
            "sh_ga_pg",         # short handed goals allowed per game (avg)
            "c_pp_opps",        # cumulative power play opportunities
            "c_pp_net_g",       # cumulative net power play goals
            "ppng_rank",        # power play net goals rank
            
    ]]
    
    
    
    
    pk = pk[[
        
            "season_id",
            "game_id",
            "team_id",
            "game_date",
            "times_sh",
            "times_sh_pg",
            "pk_net_pct",
            "pk_pct",
            "pk_net_g",
            "pk_net_gpg",
            "pk_toi",               #this needs to be fixed, showing up as nan
            "pp_ga",
            "pp_ga_pg",             
            "sh_gf",
            "sh_gf_pg",
            "pk_rank",
            "c_pp_ga",
            "c_times_sh",
            
            
            ]]

    # -------------------------------------------------
    # Merge PP + PK into final special teams table
    # -------------------------------------------------
    special_teams = (
        spine.merge(
            pp,
            on=["season_id", "game_id", "team_id", "game_date"],
            how="left",
        )
        .rename(columns={"pp_opportunities": "pp_opp"})
        .sort_values(["season_id", "game_date", "game_id", "team_id"])
        .reset_index(drop=True)
    )
    print("special teams pp merge:", special_teams.head())
    
    special_teams = (
        special_teams.merge(
            pk,
            on=["season_id", "game_id", "team_id", "game_date"],
            how="left",
        )
        .rename(columns={"pp_opportunities": "pp_opp"})
        .sort_values(["season_id", "game_date", "game_id", "team_id"])
        .reset_index(drop=True)
    )
    print("special teams pk merge:", special_teams.head())

    return special_teams


# ============================================================
# Build Core Feature DF
# ============================================================

def build_core_df(
    play_by_play: pd.DataFrame,
    backfill_df: pd.DataFrame,
    games_spine: pd.DataFrame,
    games_sos: pd.DataFrame,
    special_teams: pd.DataFrame,
    cumulative_rush_stats: pd.DataFrame,
    cumulative_shooting_stats: pd.DataFrame,
    cumulative_scoring: pd.DataFrame,
    cumulative_shots_on_goal: pd.DataFrame,
    cumulative_goalie_stats: pd.DataFrame,
) -> pd.DataFrame:

    # -------------------------------------------------
    # Canonical game × team spine
    # -------------------------------------------------
    games = (
        games_spine[["season_id", "game_id", "team_id", "game_date", "games_played"]]
        .drop_duplicates()
        .sort_values(["season_id", "game_date", "game_id", "team_id"])
        .reset_index(drop=True)
    )
    # print(games[games['team_id']==6])
    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def coalesce_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        for col in list(df.columns):
            if col.endswith("_x"):
                base = col[:-2]
                other = f"{base}_y"
                if other in df.columns:
                    df[base] = df[col].combine_first(df[other])
                    df = df.drop(columns=[col, other])
        return df

    def drop_structural_cols(df: pd.DataFrame) -> pd.DataFrame:
        STRUCTURAL = {
            "game_date",
            "games_played",
            "games_before",
            "games_played_offset",
        }
        return df.drop(
            columns=[c for c in STRUCTURAL if c in df.columns],
            errors="ignore"
        )


    # -------------------------------------------------
    # Strip structural cols from feature blocks
    # -------------------------------------------------

    cumulative_rush_stats     = drop_structural_cols(cumulative_rush_stats)
    cumulative_shooting_stats = drop_structural_cols(cumulative_shooting_stats)
    cumulative_scoring        = drop_structural_cols(cumulative_scoring)
    cumulative_shots_on_goal  = drop_structural_cols(cumulative_shots_on_goal)
    cumulative_goalie_stats   = drop_structural_cols(cumulative_goalie_stats)
    special_teams             = drop_structural_cols(special_teams)
    games_sos                 = drop_structural_cols(games_sos)

    # print("rush:", cumulative_rush_stats[cumulative_rush_stats['team_id']==6])
    # print("shooting:", cumulative_shooting_stats[cumulative_shooting_stats['team_id']==6])
    # print("scoring:", cumulative_scoring[cumulative_scoring['team_id']==6])
    # print("sog:", cumulative_shots_on_goal[cumulative_shots_on_goal['team_id']==6])
    # print("goalie:", cumulative_goalie_stats[cumulative_goalie_stats['team_id']==6])
    # print("special teams:", special_teams[special_teams['team_id']==6])
    # print("sos:", games_sos[games_sos['team_id']==6])        
    # -------------------------------------------------
    # Merge chain (ALL INNER ON SPINE)
    # -------------------------------------------------
    core_df = (
        games
        .merge(
            games_sos,
            on=["season_id", "game_id", "team_id"],
            how="inner",
        )
        .merge(
            special_teams,
            on=["season_id", "game_id", "team_id"],
            how="inner",
        )
        .merge(
            cumulative_goalie_stats[
                ["season_id", "game_id", "team_id", "game_savePct", "game_GAA"]
            ],
            on=["season_id", "game_id", "team_id"],
            how="inner",
        )
        .merge(
            cumulative_rush_stats,
            on=["season_id", "game_id", "team_id"],
            how="inner",
        )
        .merge(
            cumulative_shooting_stats,
            on=["season_id", "game_id", "team_id"],
            how="inner",
        )
        .merge(
            cumulative_scoring,
            on=["season_id", "game_id", "team_id"],
            how="left",
        )
        .merge(
            cumulative_shots_on_goal,
            on=["season_id", "game_id", "team_id"],
            how="inner",
        )
    )

    core_df = coalesce_duplicates(core_df)
    # print("Core:", core_df[core_df['team_id']==6])
    return (
        core_df
        .sort_values(["season_id", "game_date", "game_id", "team_id"])
        .reset_index(drop=True)
    )

# ============================================================
# Build Final DF (join standings correctly)
# ============================================================

def build_final_df(
    core_df: pd.DataFrame,
    standings: pd.DataFrame,
) -> pd.DataFrame:

    core = core_df.copy()
    std  = standings.copy()

    print("▶ Merging standings → final_df...")

    # -------------------------------------------------
    # 1. Ensure datetime + clean dtypes
    # -------------------------------------------------
    core["game_date"] = pd.to_datetime(core["game_date"], errors="coerce")
    std["game_date"]  = pd.to_datetime(std["game_date"], errors="coerce")

    core["season_id"] = core["season_id"].astype(int)
    std["season_id"]  = std["season_id"].astype(int)

    core["team_id"] = core["team_id"].astype(int)
    std["team_id"]  = std["team_id"].astype(int)

    # -------------------------------------------------
    # 2. Rename standings date → snapshot_date
    # -------------------------------------------------
    std = std.rename(columns={"game_date": "snapshot_date"})

    # -------------------------------------------------
    # 3. Force identical datetime64[ns]
    # -------------------------------------------------
    core["game_date"] = core["game_date"].astype("datetime64[ns]")
    std["snapshot_date"] = std["snapshot_date"].astype("datetime64[ns]")

    # -------------------------------------------------
    # 4. Build GAME-LEVEL spine (1 row per team/game)
    #    ✅ merge_asof requires global monotonic sorting
    # -------------------------------------------------
    core_games = (
        core[["season_id", "team_id", "game_id", "game_date"]]
        .drop_duplicates()
        .sort_values(["game_date", "season_id", "team_id"])  # ✅ DATE FIRST
        .reset_index(drop=True)
    )

    # -------------------------------------------------
    # 5. Deduplicate standings snapshots
    #    (one row per team per snapshot_date)
    # -------------------------------------------------
    std = (
        std.drop_duplicates(subset=["season_id", "team_id", "snapshot_date"])
        .sort_values(["snapshot_date", "season_id", "team_id"])  # ✅ DATE FIRST
        .reset_index(drop=True)
    )


    # -------------------------------------------------
    # 6. Debug safety checks (optional)
    # -------------------------------------------------
    if not core_games["game_date"].is_monotonic_increasing:
        raise ValueError("❌ core_games game_date is not globally sorted!")

    if not std["snapshot_date"].is_monotonic_increasing:
        raise ValueError("❌ standings snapshot_date is not globally sorted!")

    # -------------------------------------------------
    # 7. ✅ Merge standings snapshot BEFORE each game
    #    Each game gets the most recent daily standings row
    # -------------------------------------------------
    standings_game_spine = pd.merge_asof(
        core_games,
        std,
        left_on="game_date",
        right_on="snapshot_date",
        by=["season_id", "team_id"],
        direction="backward"
    )


    # -------------------------------------------------
    # 8. Merge spine back into full core_df
    # -------------------------------------------------
    final_updates = core.merge(
        standings_game_spine.drop(columns=["snapshot_date"]),
        on=["season_id", "team_id", "game_id", "game_date"],
        how="inner",
        suffixes=("", "_std")
    )

    # -------------------------------------------------
    # 9. Clean up duplicate columns (_x/_y merges)
    # -------------------------------------------------
    def coalesce_dupes(df: pd.DataFrame) -> pd.DataFrame:
        for col in list(df.columns):
            if col.endswith("_x"):
                base = col[:-2]
                other = f"{base}_y"
                if other in df.columns:
                    df[base] = df[col].combine_first(df[other])
                    df = df.drop(columns=[col, other])
        return df

    final_updates = coalesce_dupes(final_updates)
    
    # print("final updates df",final_updates[final_updates['team_id']==6])
    
    # -------------------------------------------------
    # 10. Return final sorted dataset
    # -------------------------------------------------
    return (
        final_updates
        .sort_values(["season_id", "game_date", "game_id", "team_id"])
        .reset_index(drop=True)
    )


# ============================================================
# Derived / Interaction Features
# ============================================================

def special_features(final_updates: pd.DataFrame) -> pd.DataFrame:

    df = final_updates.copy()

    df["ot_wins"] = (df["rotw"] - df["rw"]).clip(lower=0)
    df["overtimes"] = df["ot_losses"] + df["ot_wins"]

    df["ot_games_pct"] = np.where(
        df["games_played"] > 0,
        df["overtimes"] / df["games_played"],
        0.0,
    ).round(2)

    df["pt_pct_x_sos"]   = df["points_pct"] * df["sos_future"]
    df["gaa_x_sv"]       = df["game_GAA"] * df["game_savePct"]
    df["gd_x_rushdiff"]  = df["goal_diff"] * df["rush_shot_differential"]
    df["pk_load_x_pk_pct"] = df["times_sh_pg"] * df["pk_pct"]

    return df

# ============================================================
# Execution
# ============================================================
# ============================================================
# 1. Determine backfill window
# ============================================================
def run_prediction_backfill(con:duckdb.DuckDBPyConnection, season_id: int):
    
    print(f"\n🏒 Backfilling prediction data for {season_id}")
    
    summary_df = load_game_summary_df(con, season_id)
    
    if summary_df.empty:
        print("⚠ No completed games.")
        return
    
    existing_prediction_data = load_existing_predictions(con, season_id)

    backfill_df = summary_df[
        ~summary_df["game_id"].isin(existing_prediction_data)
        ] # df of game ids from the game summary df that do not exist in pred data
    

    if backfill_df.empty:
        return
    missing_ids = backfill_df['game_id'].tolist()
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




# ============================================================
# EXECUTION
# ============================================================
if __name__ == "__main__":

    con = duckdb.connect(DB_PATH)

    for season_id in range(20252026, 20252026 +1):
        run_prediction_backfill(con, season_id)

    con.close()
    print("\n🏁 Historical play by plays backfill complete.")