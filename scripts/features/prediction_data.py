import numpy as np
import pandas as pd
import duckdb
from typing import Any

from config import DB_PATH


def load_teams(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:

    return con.execute("""
        SELECT team_id, tri_code
        FROM all_teams
    """).df()
def add_opponent_team_id(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df = df.dropna(
        subset=["team_id", "home_team", "away_team"]
    )

    mask = df["team_id"].eq(df["home_team"])

    df["opponent_team_id"] = np.where(
        mask,
        df["away_team"],
        df["home_team"],
    )

    return df
# ============================================================
# Fetch source data
# ============================================================

def fetch_tables(con: duckdb.DuckDBPyConnection, season_id: int) -> dict[str, pd.DataFrame]:

    print(f"📦 Loading tables for season {season_id}")

    tables: dict[str, pd.DataFrame] = {}

    tables["penalty_kill"] = con.execute("""
        SELECT pk.*, t.tri_code
        FROM all_pk pk
        JOIN all_teams t
            ON t.team_id = pk.team_id
        WHERE pk.season_id = ?
        ORDER BY pk.game_id
    """, [season_id]).df()

    tables["power_play"] = con.execute("""
        SELECT pp.*, t.tri_code
        FROM all_pp pp
        JOIN all_teams t
            ON t.team_id = pp.team_id
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
        ORDER BY game_id
    """, [season_id]).df()

    return tables

# ============================================================
# Load cumulative offsets from team_daily_stats
# Last completed row per team before begins
# ============================================================

def load_prediction_offsets(
    con: duckdb.DuckDBPyConnection,
    season_id: int,
    start_game_id: int,
) -> pd.DataFrame:

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

    df = con.execute(
        q,
        [season_id, start_game_id],
    ).df()

    if df.empty:
        print("⚠️ No offset rows found in team_daily_stats.")

        return pd.DataFrame(
            columns=["team_id", "games_played_offset"]
        )

    if "games_played" not in df.columns:
        raise KeyError(
            "team_daily_stats is missing games_played"
        )

    # ------------------------------------------------------------
    # Canonical games-played offset
    # ------------------------------------------------------------
    df["games_played_offset"] = (
        pd.to_numeric(
            df["games_played"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    # ------------------------------------------------------------
    # Rename cumulative values as offsets
    # ------------------------------------------------------------
    # The previous row's wins_before excludes that row's result.
    # Carry forward wins through the previous completed game.
    if "wins_before" in df.columns and "game_win" in df.columns:
        df["wins_completed_offset"] = (
            pd.to_numeric(df["wins_before"], errors="coerce").fillna(0)
            + pd.to_numeric(df["game_win"], errors="coerce").fillna(0)
        )
    else:
        raise KeyError("Existing team_daily_stats requires wins_before and game_win for safe incremental offsets")

    KEY_COLS = {
        "season_id", "game_id", "team_id", "games_played",
        "games_played_offset", "wins_completed_offset",
    }

    df = df.rename(
        columns={
            c: f"{c}_offset"
            for c in df.columns
            if c not in KEY_COLS
        }
    )

    keep_cols = [
        "team_id",
        "wins_completed_offset",
        *[
            c
            for c in df.columns
            if c.endswith("_offset")
        ],
    ]

    df = df.loc[:, keep_cols]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.drop_duplicates(
        subset=["team_id"]
    ).reset_index(drop=True)

    return df


# ============================================================
# Games played spine
# ============================================================

def build_games_played_spine(
    new_games: pd.DataFrame,
    offsets: pd.DataFrame | None = None,
) -> pd.DataFrame:

    rows = []

    for _, row in new_games.iterrows():

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
        [
            "season_id",
            "team_id",
            "game_date",
            "game_id",
        ]
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Games within this window
    # ------------------------------------------------------------
    spine["games_before_local"] = (
        spine
        .groupby(
            ["season_id", "team_id"],
            sort=False,
        )
        .cumcount()
    )

    spine["games_played_local"] = (
        spine["games_before_local"] + 1
    )

    # ------------------------------------------------------------
    # Add historical games-played offset
    # ------------------------------------------------------------
    spine["games_played_offset"] = 0

    if (
        offsets is not None
        and not offsets.empty
        and "games_played_offset" in offsets.columns
    ):

        off = (
            offsets[
                ["team_id", "games_played_offset"]
            ]
            .drop_duplicates("team_id")
        )

        spine = spine.merge(
            off,
            on="team_id",
            how="left",
            suffixes=("", "_off"),
        )

        spine["games_played_offset"] = (
            spine["games_played_offset_off"]
            .fillna(0)
            .astype(int)
        )

        spine = spine.drop(
            columns=["games_played_offset_off"]
        )

    # ------------------------------------------------------------
    # Canonical games-before / games-played
    # ------------------------------------------------------------
    spine["games_before"] = (
        spine["games_before_local"]
        + spine["games_played_offset"]
    )

    spine["games_played"] = (
        spine["games_played_local"]
        + spine["games_played_offset"]
    )

    return spine[
        [
            "season_id",
            "game_id",
            "team_id",
            "game_date",
            "games_before",
            "games_played",
            "games_played_offset",
        ]
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
    # ✅ STEP 1: FIX game_win
    # ============================================================

    if "win" in df.columns:
        df["game_win"] = pd.to_numeric(df["win"], errors="coerce").fillna(0).astype("Int64")

    elif "game_win" in df.columns:
        df["game_win"] = pd.to_numeric(df["game_win"], errors="coerce").fillna(0).astype("Int64")

    else:
        df["game_win"] = pd.Series(0, index=df.index, dtype="Int64")


    # ============================================================
    # STEP 2: Preserve canonical games_played / games_before
    # These already come from games_spine and include the offset.
    # ============================================================

    if "games_played" not in df.columns:
        raise KeyError("games_played missing from canonical games spine")

    if "games_before" not in df.columns:
        raise KeyError("games_before missing from canonical games spine")

    df["games_played"] = pd.to_numeric(
        df["games_played"], errors="coerce"
    ).astype("Int64")

    df["games_before"] = pd.to_numeric(
        df["games_before"], errors="coerce"
    ).astype("Int64")


    # ============================================================
    # STEP 3: Preserve canonical wins_before
    # ============================================================

    if "wins_before" not in df.columns:
        raise KeyError("wins_before missing from final dataframe")

    df["wins_before"] = pd.to_numeric(
        df["wins_before"],
        errors="coerce",
    ).astype("Int64")


    # ============================================================
    # ✅ STEP 4: win_pct_before recompute
    # ============================================================

    df["win_pct_before"] = pd.NA
    mask = df["games_before"] > 0

    df.loc[mask, "win_pct_before"] = (
        df.loc[mask, "wins_before"] / df.loc[mask, "games_before"]
    )


    # ============================================================
    # STEP 5: Preserve canonical special-teams features
    # Cumulative values and offsets are handled upstream by
    # build_special_teams_stats().
    # ============================================================

    special_teams_required = [
        "times_sh",
        "c_times_sh",
        "times_sh_pg",
        "pp_opp",
        "c_pp_opps",
        "pp_opp_pg",
        "pp_net_g",
        "c_pp_net_g",
        "pp_net_pct",
    ]

    missing_special_teams = [
        col for col in special_teams_required
        if col not in df.columns
    ]

    if missing_special_teams:
        raise KeyError(
            "Missing canonical special-teams columns: "
            f"{missing_special_teams}"
        )

    # ============================================================
    # ✅ STEP 6: FINAL DUCKDB TYPE ENFORCEMENT
    # ============================================================

    # Add all missing schema columns at once to avoid fragmentation
    missing_cols = [
        col for col in DATA_SCHEMA
        if col not in df.columns
    ]

    if missing_cols:
        missing_df = pd.DataFrame(
            pd.NA,
            index=df.index,
            columns=missing_cols,
        )

        df = pd.concat(
            [df, missing_df],
            axis=1,
        )

    df = df.copy()

    # Enforce schema data types
    for col, dtype in DATA_SCHEMA.items():
        if dtype in ("VARCHAR", "STRING"):
            df[col] = df[col].astype("string")

        elif dtype in ("DOUBLE", "FLOAT", "REAL"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        elif dtype in ("TIMESTAMP", "DATETIME"):
            df[col] = pd.to_datetime(df[col], errors="coerce")

        else:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).astype("Int64")

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
        print("⚠️ games_played missing — filling with 0")
        rush["games_played"] = 0

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
        .groupby(["season_id", "game_id"])
        .size()
        .reset_index(name="rows")
        .query("rows != 2")
    )

    if not bad_games.empty:
        bad_keys = bad_games[["season_id", "game_id"]]

        shot_events = shot_events.merge(
            bad_keys,
            on=["season_id", "game_id"],
            how="left",
            indicator=True
        )

        shot_events = (
            shot_events[shot_events["_merge"] == "left_only"]
            .drop(columns=["_merge"])
        )

    # For blocked-shot events (508), team_id identifies the blocking team
    # under the same convention used by build_danger_corsi_and_scf_stats().
    shot_events["shooting_team_id"] = np.where(
        shot_events["type_code"].eq(508),
        shot_events["opponent_team_id"],
        shot_events["team_id"],
    )
    shot_events["defending_team_id"] = np.where(
        shot_events["type_code"].eq(508),
        shot_events["team_id"],
        shot_events["opponent_team_id"],
    )

    shots_for = (
        shot_events.groupby(["season_id", "game_id", "shooting_team_id"], as_index=False)
        .size()
        .rename(columns={"shooting_team_id": "team_id", "size": "shot_attempts_for"})
    )

    shots_against = (
        shot_events[shot_events["defending_team_id"].notna()]
        .groupby(["season_id", "game_id", "defending_team_id"], as_index=False)
        .size()
        .rename(columns={"defending_team_id": "team_id", "size": "shot_attempts_against"})
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
        .groupby(["season_id", "game_id", "shooting_team_id"], as_index=False)
        .size()
        .rename(columns={"shooting_team_id": "team_id", "size": "corsi_for"})
    )

    corsi_against = (
        shot_events[(shot_events["zone_code"] == "O") & (shot_events["defending_team_id"].notna())]
        .groupby(["season_id", "game_id", "defending_team_id"], as_index=False)
        .size()
        .rename(columns={"defending_team_id": "team_id", "size": "corsi_against"})
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

    md = md & ~hd
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

    df["context_bonus"] = (df["is_rebound"] == 1).astype(int) + (df["rush_shot"] == 1).astype(int)
    df["blocked_penalty"] = (df["type_code"] == 508).astype(int)
    
    df["adj_scoring_chance_score"] = df["base_scoring_chance_score"] + df["context_bonus"]- df["blocked_penalty"]
    df["scoring_chance"] = (df["adj_scoring_chance_score"] >= 2).astype(int)

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
    # AGAINST — opponent of the actual shooting team
    # -----------------------------
    d["defending_team_id"] = np.where(
        d["shooting_team_id"].eq(d["team_id"]),
        d["opponent_team_id"],
        d["team_id"],
    )
    sca = (
        d.groupby(["season_id", "game_id", "defending_team_id"], as_index=False)
        .agg(SCA=("scoring_chance", "sum"))
        .rename(columns={"defending_team_id": "team_id"})
    )

    # -----------------------------
    # Danger-split Corsi FOR
    # -----------------------------
    d["hd_corsi"] = d["hd_shot"]
    d["md_corsi"] = d["md_shot"]
    d["ld_corsi"] = d["ld_shot"]

    corsi_for = (
        d.groupby(["season_id", "game_id", "shooting_team_id"], as_index=False)
        .agg(
            HDCF=("hd_corsi", "sum"),
            MDCF=("md_corsi", "sum"),
            LDCF=("ld_corsi", "sum"),
        )
        .rename(columns={"shooting_team_id": "team_id"})
    )

    # -----------------------------
    # Danger-split Corsi AGAINST (opponent-based)
    # -----------------------------
    corsi_against = (
        d.groupby(["season_id", "game_id", "defending_team_id"], as_index=False)
        .agg(
            HDCA=("hd_corsi", "sum"),
            MDCA=("md_corsi", "sum"),
            LDCA=("ld_corsi", "sum"),
        )
        .rename(columns={"defending_team_id": "team_id"})
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

    # Verify opponent reciprocity whenever both team rows are available.
    for _, game in final.groupby(["season_id", "game_id"]):
        if len(game) == 2:
            left, right = game.iloc[0], game.iloc[1]
            if left["SCF"] != right["SCA"] or right["SCF"] != left["SCA"]:
                raise ValueError(f"SCF/SCA opponent mismatch in game {left['game_id']}")

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

    df["shooting_team_id"] = np.where(
        df["type_code"].eq(508), df["opponent_team_id"], df["team_id"]
    )
    df["defending_team_id"] = np.where(
        df["type_code"].eq(508), df["team_id"], df["opponent_team_id"]
    )

    df["goal"] = df["scoring_chance_goal"]
    df["sog"] = df["shot_on_goal"]

    df["hd_sog"] = ((df["sog"] == 1) & (df["hd_shot"] == 1)).astype(int)
    df["md_sog"] = ((df["sog"] == 1) & (df["md_shot"] == 1)).astype(int)
    df["ld_sog"] = ((df["sog"] == 1) & (df["ld_shot"] == 1)).astype(int)

    df["hd_goal"] = ((df["goal"] == 1) & (df["hd_shot"] == 1)).astype(int)
    df["md_goal"] = ((df["goal"] == 1) & (df["md_shot"] == 1)).astype(int)
    df["ld_goal"] = ((df["goal"] == 1) & (df["ld_shot"] == 1)).astype(int)
    
    if "opponent_team_id" not in df.columns:
        raise KeyError(
            "opponent_team_id missing from scoring_chances_events"
        )
        
    stats_for = (
        df.groupby(["season_id", "game_id", "shooting_team_id"], as_index=False)
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
        .rename(columns={"shooting_team_id": "team_id"})
    )
    stats_for["shot_pct"] = 100 * stats_for["goals_for"] / stats_for["shots_for"].replace(0, np.nan)

    stats_against = (
        df.groupby(["season_id", "game_id", "defending_team_id"], as_index=False)
        .agg(
            shots_against=("sog", "sum"),
            goals_against=("goal", "sum"),
            hd_against=("hd_sog", "sum"),
            md_against=("md_sog", "sum"),
            ld_against=("ld_sog", "sum"),
        )
        .rename(columns={"defending_team_id": "team_id"})
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
    rush_stats = games_spine[
            ["season_id", "game_id", "team_id", "games_played", "game_date"]
        ].merge(
            rush_stats,
            on=["season_id", "game_id", "team_id"],
            how="left"
        )

    rush_cols = [
        "rush_shots_for",
        "rush_shots_against",
        "rush_shot_differential",
    ]

    rush_stats[rush_cols] = (
        rush_stats[rush_cols]
        .fillna(0)
        .astype(int)
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
# Strength of schedule At this point in the season, based only on results that have already happened, how difficult is this team's remaining schedule?
# ============================================================
def strength_of_schedule(game_summary: pd.DataFrame, sos_standings: pd.DataFrame) -> pd.DataFrame:

    rows = []

    # ============================================================
    # Build full historical schedule
    # ============================================================
    for _, row in game_summary.iterrows():

        rows.append({
            "season_id": int(row["season_id"]),
            "game_id": int(row["game_id"]),
            "game_date": pd.to_datetime(row["game_date"]),
            "team_id": int(row["away_team"]),
            "tri_code": row["away_tri_code"],
            "opponent": row["home_tri_code"],
        })

        rows.append({
            "season_id": int(row["season_id"]),
            "game_id": int(row["game_id"]),
            "game_date": pd.to_datetime(row["game_date"]),
            "team_id": int(row["home_team"]),
            "tri_code": row["home_tri_code"],
            "opponent": row["away_tri_code"],
        })

    games_sos = (
        pd.DataFrame(rows)
        .sort_values(["season_id", "team_id", "game_date", "game_id"])
        .reset_index(drop=True)
    )

    # ============================================================
    # Games played
    # ============================================================
    g = games_sos.groupby(["season_id", "team_id"], sort=False)

    games_sos["games_before"] = g.cumcount()
    games_sos["games_played"] = games_sos["games_before"] + 1

    # ============================================================
    # Historical standings matrix
    # ============================================================
    standings = sos_standings.copy()

    standings["game_date"] = pd.to_datetime(
        standings["game_date"],
        errors="coerce",
    ).dt.normalize()

    standings["points_pct"] = pd.to_numeric(
        standings["points_pct"],
        errors="coerce",
    )

    standings_matrix = (
        standings
        .pivot_table(
            index="game_date",
            columns="tri_code",
            values="points_pct",
            aggfunc="last",
        )
        .sort_index()
        .ffill()
    )

    # Complete calendar so previous-day lookup always works
    full_dates = pd.date_range(
        standings_matrix.index.min(),
        standings_matrix.index.max(),
        freq="D",
    )

    standings_matrix = (
        standings_matrix
        .reindex(full_dates)
        .ffill()
    )

    first_date = standings_matrix.index.min()
    
    last_date = standings_matrix.index.max()

    # ============================================================
    # Point-in-time SOS
    # ============================================================
    def calc_team_sos(group: pd.DataFrame) -> pd.DataFrame:

        group = (
            group
            .sort_values(["game_date", "game_id"])
            .copy()
        )

        past_values = []
        future_values = []

        for i in range(len(group)):

            current_date = pd.to_datetime(
                group.iloc[i]["game_date"]
            ).normalize()

            # Use only standings known before this game
            lookup_date = current_date - pd.Timedelta(days=1)

            if lookup_date < first_date:
                past_values.append(np.nan)
                future_values.append(np.nan)
                continue

            if lookup_date > last_date:
                lookup_date = last_date
            # ----------------------------------------------------
            # PAST SOS
            #
            # Opponents already played.
            # Evaluate ALL of them using their CURRENT points_pct.
            # ----------------------------------------------------
            past_opponents = (
                group.iloc[:i]["opponent"]
                .dropna()
                .tolist()
            )

            past_opponents = [
                opp for opp in past_opponents
                if opp in standings_matrix.columns
            ]

            if past_opponents:
                past = pd.to_numeric(
                    standings_matrix.loc[
                        lookup_date,
                        past_opponents,
                    ],
                    errors="coerce",
                ).mean()
            else:
                past = np.nan

            # ----------------------------------------------------
            # FUTURE SOS
            #
            # Current opponent + remaining schedule.
            #
            # NO .unique() — repeated opponents count once
            # for every remaining game.
            # ----------------------------------------------------
            future_opponents = (
                group.iloc[i:]["opponent"]
                .dropna()
                .tolist()
            )

            future_opponents = [
                opp for opp in future_opponents
                if opp in standings_matrix.columns
            ]

            if future_opponents:
                future = pd.to_numeric(
                    standings_matrix.loc[
                        lookup_date,
                        future_opponents,
                    ],
                    errors="coerce",
                ).mean()
            else:
                future = np.nan

            past_values.append(past)
            future_values.append(future)

        group["sos_past"] = past_values
        group["sos_future"] = future_values

        return group

    games_sos = (
        games_sos
        .groupby(
            ["season_id", "team_id"],
            group_keys=False,
        )
        .apply(calc_team_sos)
        .reset_index(drop=True)
    )

    return games_sos[
        [
            "season_id",
            "game_id",
            "team_id",
            "games_before",
            "games_played",
            "sos_past",
            "sos_future",
        ]
    ]

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
    
    # If someone accidentally passed new_games/game_summary instead of games_spine,
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
    
    # Canonical games_played comes from games_spine
    pp = pp.drop(columns=["games_played"], errors="ignore")
    pk = pk.drop(columns=["games_played"], errors="ignore")
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
    if offsets is not None:
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
        pk["c_sh_gf"] / pk["games_played"],
        0.0,
    )

    
    pk["pk_net_gpg"] = np.where(
        pk["games_played"] > 0,
        pk["c_pk_net_g"] / pk["games_played"],
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
    games_spine: pd.DataFrame,
    games_sos: pd.DataFrame,
    special_teams: pd.DataFrame,
    cumulative_rush_stats: pd.DataFrame,
    cumulative_shooting_stats: pd.DataFrame,
    cumulative_scoring: pd.DataFrame,
    cumulative_shots_on_goal: pd.DataFrame,
    cumulative_goalie_stats: pd.DataFrame,
    offsets: pd.DataFrame | None = None,
) -> pd.DataFrame:

    # -------------------------------------------------
    # Canonical game × team spine
    # -------------------------------------------------
    games = (
        games_spine[
            [
                "season_id",
                "game_id",
                "team_id",
                "game_date",
                "games_before",
                "games_played",
            ]
        ]
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
                ["season_id", "game_id", "team_id", "game_savePct", "game_GAA", "game_wins"]
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
    core_df["game_win"] = (
        pd.to_numeric(
            core_df["game_wins"],
            errors="coerce",
        )
        .fillna(0)
        .astype("Int64")
    )

    # -------------------------------------------------
    # Canonical wins_before
    # team_daily_stats wins + wins from earlier games
    # -------------------------------------------------

    # wins_offset = (
    #     offsets[["team_id", "wins_completed_offset"]]
    #     .drop_duplicates("team_id")
    #     .copy()
    # )
    if (offsets is not None
        and not offsets.empty
        and "wins_completed_offset" in offsets.columns
    ):
        wins_offset = (
            offsets[["team_id", "wins_completed_offset"]]
            .drop_duplicates("team_id")
            .copy()
        )
    else:
        wins_offset = pd.DataFrame(
            columns=["team_id", "wins_completed_offset"]
        )

    wins_offset["wins_completed_offset"] = pd.to_numeric(
        wins_offset["wins_completed_offset"],
        errors="coerce",
    ).fillna(0)

    core_df = core_df.merge(
        wins_offset,
        on="team_id",
        how="left",
    )

    core_df["wins_completed_offset"] = (
        core_df["wins_completed_offset"]
        .fillna(0)
    )

    core_df = core_df.sort_values(
        ["season_id", "team_id", "game_date", "game_id"]
    )

    core_df["wins_before"] = (
        core_df["wins_completed_offset"]
        + core_df.groupby(
            ["season_id", "team_id"],
            sort=False,
        )["game_wins"].cumsum()
        - core_df["game_wins"]
        )
            
    core_df = core_df.drop(columns=["wins_completed_offset"])
    
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
    std = standings.copy()

    print("▶ Merging standings → final_df...")

    # -------------------------------------------------
    # 1. Ensure datetime + clean dtypes
    # -------------------------------------------------
    core["game_date"] = pd.to_datetime(
        core["game_date"],
        errors="coerce",
    )

    std["game_date"] = pd.to_datetime(
        std["game_date"],
        errors="coerce",
    )

    core["season_id"] = core["season_id"].astype(int)
    std["season_id"] = std["season_id"].astype(int)

    core["team_id"] = core["team_id"].astype(int)
    std["team_id"] = std["team_id"].astype(int)

    # -------------------------------------------------
    # 2. Rename standings date → snapshot_date
    # -------------------------------------------------
    std = std.rename(
        columns={"game_date": "snapshot_date"}
    )

    # -------------------------------------------------
    # 3. Force identical datetime64[ns]
    # -------------------------------------------------
    core["game_date"] = core["game_date"].astype(
        "datetime64[ns]"
    )

    std["snapshot_date"] = std["snapshot_date"].astype(
        "datetime64[ns]"
    )

    # -------------------------------------------------
    # 4. Build GAME-LEVEL spine
    #    1 row per team/game
    #    merge_asof requires global monotonic sorting
    # -------------------------------------------------
    core_games = (
        core[
            [
                "season_id",
                "team_id",
                "game_id",
                "game_date",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "game_date",
                "season_id",
                "team_id",
            ]
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------
    # 5. Deduplicate standings snapshots
    #    One row per team per snapshot_date
    # -------------------------------------------------
    std = (
        std
        .drop_duplicates(
            subset=[
                "season_id",
                "team_id",
                "snapshot_date",
            ]
        )
        .sort_values(
            [
                "snapshot_date",
                "season_id",
                "team_id",
            ]
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------
    # 6. Debug safety checks
    # -------------------------------------------------
    if not core_games["game_date"].is_monotonic_increasing:
        raise ValueError(
            "❌ core_games game_date is not globally sorted!"
        )

    if not std["snapshot_date"].is_monotonic_increasing:
        raise ValueError(
            "❌ standings snapshot_date is not globally sorted!"
        )

    # -------------------------------------------------
    # 7. Merge standings snapshot BEFORE each game
    # -------------------------------------------------
    print("\nCORE GAMES:")

    print(
        core_games[
            [
                "season_id",
                "game_id",
                "game_date",
                "team_id",
            ]
        ]
        .head(1)
        .to_string(index=False)
    )

    print("\nCORE GAMES DTYPES:")

    print(
        core_games[
            [
                "season_id",
                "game_date",
                "team_id",
            ]
        ].dtypes
    )

    print(
        "\nStandings snapshot range:",
        std["snapshot_date"].min(),
        "→",
        std["snapshot_date"].max(),
    )

    print("\nSTANDINGS STD DTYPES:")

    print(
        std[
            [
                c
                for c in [
                    "season_id",
                    "snapshot_date",
                    "team_id",
                    "tri_code",
                ]
                if c in std.columns
            ]
        ].dtypes
    )

    standings_game_spine = pd.merge_asof(
        core_games,
        std,
        left_on="game_date",
        right_on="snapshot_date",
        by=["season_id", "team_id"],
        direction="backward",
        allow_exact_matches=False,
    )

    # -------------------------------------------------
    # Season-opening standings baseline
    # No prior snapshot exists before opening-day games
    # -------------------------------------------------
    opening_mask = (
        standings_game_spine["snapshot_date"].isna()
    )

    zero_start_cols = [
        "wins",
        "wins_pctg",
        "losses",
        "ot_losses",
        "rotw",
        "rotw_pctg",
        "rw",
        "so_losses",
        "so_wins",
        "points",
        "points_pct",
        "projected_points",
        "in_playoff_spot",
        "gf_pctg",
        "goal_diff",
        "goal_diff_pctg",
        "streak_count",
        "last_10_gp",
        "last_10_wins",
        "last_10_losses",
        "last_10_otl",
        "last_10_points",
        "last_10_gf",
        "last_10_ga",
        "last_10_goal_diff",
        "last_10_rotw",
        "last_10_rw",
        "home_gp",
        "home_wins",
        "home_losses",
        "home_otlosses",
        "home_rotw",
        "home_rw",
        "home_points",
        "home_gf",
        "home_ga",
        "home_goal_diff",
        "road_gp",
        "road_wins",
        "road_losses",
        "road_ot_losses",
        "road_rotw",
        "road_rw",
        "road_points",
        "road_gf",
        "road_ga",
        "road_goal_diff",
        "division_rank",
        "division_home_rank",
        "division_road_rank",
        "division_last_10_rank",
        "conference_rank",
        "conference_home_rank",
        "conference_road_rank",
        "conference_last_10_rank",
        "league_rank",
        "league_home_rank",
        "league_road_rank",
        "league_last_10_rank",
        "wildcard_rank",
    ]

    for col in zero_start_cols:
        if col in standings_game_spine.columns:
            standings_game_spine.loc[
                opening_mask,
                col,
            ] = 0

    # -------------------------------------------------
    # Streak baseline
    # -------------------------------------------------
    if "streak" in standings_game_spine.columns:
        standings_game_spine["streak"] = (
            standings_game_spine["streak"]
            .fillna("NONE")
        )

    if "streak_count" in standings_game_spine.columns:
        standings_game_spine["streak_count"] = (
            pd.to_numeric(
                standings_game_spine["streak_count"],
                errors="coerce",
            )
            .fillna(0)
            .astype("Int64")
        )

    # -------------------------------------------------
    # Static team attributes for season-opening rows
    # Division/conference do not depend on game results
    # -------------------------------------------------
    team_alignment = (
        std
        .sort_values("snapshot_date")
        .drop_duplicates(
            subset=[
                "season_id",
                "team_id",
            ],
            keep="first",
        )
        [
            [
                "season_id",
                "team_id",
                "division",
                "conference",
            ]
        ]
        .rename(
            columns={
                "division": "opening_division",
                "conference": "opening_conference",
            }
        )
    )

    standings_game_spine = standings_game_spine.merge(
        team_alignment,
        on=[
            "season_id",
            "team_id",
        ],
        how="left",
    )

    standings_game_spine.loc[
        opening_mask,
        "division",
    ] = standings_game_spine.loc[
        opening_mask,
        "opening_division",
    ]

    standings_game_spine.loc[
        opening_mask,
        "conference",
    ] = standings_game_spine.loc[
        opening_mask,
        "opening_conference",
    ]

    standings_game_spine = (
        standings_game_spine.drop(
            columns=[
                "opening_division",
                "opening_conference",
            ]
        )
    )

    # -------------------------------------------------
    # Canonical pregame wins
    # Latest standings snapshot strictly before game
    # -------------------------------------------------
    standings_game_spine["wins_before"] = (
        pd.to_numeric(
            standings_game_spine["wins"],
            errors="coerce",
        )
        .astype("Int64")
    )

    # -------------------------------------------------
    # Canonical pregame games played
    # Latest standings snapshot strictly before game
    # -------------------------------------------------
    standings_game_spine["games_before"] = (
        pd.to_numeric(
            standings_game_spine["games_played"],
            errors="coerce",
        )
        .fillna(0)
        .astype("Int64")
    )

    # Preserve point-in-time games_remaining from the standings snapshot.
    # Counting games in core_games gives games completed so far, not the
    # scheduled season length. For opening-day rows without prior standings,
    # infer the season total only from an observed standings record.
    if "games_remaining" in standings_game_spine.columns:
        if {"games_played", "games_remaining"}.issubset(std.columns):
            season_totals = std.copy()
            season_totals["season_total"] = (
                pd.to_numeric(season_totals["games_played"], errors="coerce")
                + pd.to_numeric(season_totals["games_remaining"], errors="coerce")
            )
            season_totals = (
                season_totals.groupby(["season_id", "team_id"], as_index=False)
                ["season_total"].max()
            )
            standings_game_spine = standings_game_spine.merge(
                season_totals, on=["season_id", "team_id"], how="left"
            )
            standings_game_spine.loc[opening_mask, "games_remaining"] = (
                standings_game_spine.loc[opening_mask, "season_total"]
            )
            standings_game_spine = standings_game_spine.drop(columns=["season_total"])

    print("\nSTANDINGS GAME COLUMNS:")

    print(
        [
            c
            for c in standings_game_spine.columns
            if "game" in c.lower()
        ]
    )

    print("\nSTANDINGS MERGE CHECK:")

    print(
        standings_game_spine[
            [
                "game_id",
                "game_date",
                "team_id",
                "snapshot_date",
                "games_played",
                "wins",
                "points",
                "points_pct",
            ]
        ]
        .head(1)
        .to_string(index=False)
    )

    # -------------------------------------------------
    # 8. Merge spine back into full core_df
    # -------------------------------------------------
    final_updates = core.merge(
        standings_game_spine.drop(
            columns=["snapshot_date"]
        ),
        on=[
            "season_id",
            "team_id",
            "game_id",
            "game_date",
        ],
        how="inner",
        suffixes=("", "_std"),
    )
    # -------------------------------------------------
    # Resolve canonical pregame values
    # -------------------------------------------------

    # games_before:
    # Prefer the canonical game spine.
    # Fall back to standings only if the spine value is missing.
    if "games_before_std" in final_updates.columns:
        final_updates["games_before"] = (
            final_updates["games_before"]
            .combine_first(final_updates["games_before_std"])
        )

        final_updates = final_updates.drop(
            columns=["games_before_std"]
        )


    # wins_before:
    # Prefer point-in-time standings.
    # Fall back to the existing value if standings is missing.
    if "wins_before_std" in final_updates.columns:
        final_updates["wins_before"] = (
            final_updates["wins_before_std"]
            .combine_first(final_updates["wins_before"])
        )

        final_updates = final_updates.drop(
            columns=["wins_before_std"]
        )
    # -------------------------------------------------
    # Current game number =
    # games completed before this game + 1
    # -------------------------------------------------
    final_updates["games_played"] = (
        pd.to_numeric(
            final_updates["games_before"],
            errors="coerce",
        ) + 1
    ).astype("Int64")

    # -------------------------------------------------
    # 9. Clean up duplicate columns (_x/_y merges)
    # -------------------------------------------------
    def coalesce_dupes(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        for col in list(df.columns):
            if col.endswith("_x"):
                base = col[:-2]
                other = f"{base}_y"

                if other in df.columns:
                    df[base] = (
                        df[col]
                        .combine_first(df[other])
                    )

                    df = df.drop(
                        columns=[
                            col,
                            other,
                        ]
                    )

        return df

    final_updates = coalesce_dupes(
        final_updates
    )

    # -------------------------------------------------
    # 10. Return final sorted dataset
    # -------------------------------------------------
    return (
        final_updates
        .sort_values(
            [
                "season_id",
                "game_date",
                "game_id",
                "team_id",
            ]
        )
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
# Prediction Data Runner Main Entry Point
# ============================================================
def run_prediction_data(*, con: duckdb.DuckDBPyConnection, season_id: int, ) -> None:
    
    print("\n==============================")
    print(f"▶ Running prediction for season {season_id}")
    print("==============================\n")

    # ------------------------------------------------------------
    # 1. Load completed game summary
    # ------------------------------------------------------------
    game_summary = con.execute("""
        SELECT *
        FROM all_game_summary
        WHERE season_id = ?
          AND gameState = 'OFF'
        ORDER BY game_id
    """, [season_id]).df()

    if game_summary.empty:
        print("⚠️ No completed games found.")
        return

    game_summary["game_date"] = pd.to_datetime(game_summary["game_date"])

    # ------------------------------------------------------------
    # 2. Find missing or incomplete games and rebuild from the
    #    earliest affected game onward to protect cumulative stats.
    # ------------------------------------------------------------
    existing = con.execute("""
        SELECT game_id, COUNT(DISTINCT team_id) AS team_rows,
               COUNT(*) AS total_rows
        FROM team_daily_stats
        WHERE season_id = ?
        GROUP BY game_id
    """, [season_id]).df()

    game_summary = game_summary.sort_values(["game_date", "game_id"]).copy()
    complete_ids = set(existing.loc[
        (existing["team_rows"] == 2) & (existing["total_rows"] == 2), "game_id"
    ].astype(int))
    missing = game_summary.loc[~game_summary["game_id"].astype(int).isin(complete_ids)]

    if missing.empty:
        print("✅ No missing or incomplete games detected.")
        return

    earliest = missing.iloc[0]
    new_games = game_summary.loc[
        (game_summary["game_date"] > earliest["game_date"])
        | ((game_summary["game_date"] == earliest["game_date"])
           & (game_summary["game_id"] >= earliest["game_id"]))
    ].copy()
    print(f"🔄 Rebuilding {len(new_games)} games from {int(earliest['game_id'])} onward")
        
    # ------------------------------------------------------------
    # 3. Select completed games AFTER checkpoint
    # ------------------------------------------------------------
    
    if new_games.empty:
        print("✅ No games need updating.")
        return

    game_ids = new_games["game_id"].astype(int).tolist()

    min_game_id = int(new_games["game_id"].min())
    max_game_id = int(new_games["game_id"].max())

    min_day = new_games["game_date"].min().date()
    max_day = new_games["game_date"].max().date()

    print(
        f"New games: {new_games['game_id'].nunique()} "
        f"({min_game_id} → {max_game_id})"
    )
    print(f"Date window: {min_day} → {max_day}")

    # ------------------------------------------------------------
    # 4. Load cumulative offsets from team_daily_stats
    # ------------------------------------------------------------
    offsets = load_prediction_offsets(
        con,
        season_id=season_id,
        start_game_id=min_game_id,
    )

    print(f"Offsets loaded: {len(offsets)} teams")
    # Incremental cumulative features must not silently restart at zero.
    prior_teams = con.execute("""
        SELECT DISTINCT team_id FROM team_daily_stats
        WHERE season_id = ? AND game_id < ?
    """, [season_id, min_game_id]).df()
    missing_offsets = set(prior_teams["team_id"]) - set(offsets["team_id"])
    if missing_offsets:
        raise RuntimeError(f"Missing cumulative offsets for teams: {sorted(missing_offsets)}")

    # ------------------------------------------------------------
    # 5. Build canonical games-played spine
    # ------------------------------------------------------------
    games_spine = build_games_played_spine(
        new_games=new_games,
        offsets=offsets,
    )

    print(f"Games spine rows: {len(games_spine)}")

    # ------------------------------------------------------------
    # 6. Load all source tables
    # ------------------------------------------------------------
    tables = fetch_tables(con, season_id)

    # ------------------------------------------------------------
    # 7. Filter game-based source tables
    # ------------------------------------------------------------
    def filter_to_games(df: pd.DataFrame, game_ids: list[int]) -> pd.DataFrame:
        if "game_id" not in df.columns:
            return df.copy()

        return df[
            pd.to_numeric(df["game_id"], errors="coerce").isin(game_ids)
        ].copy()

    for key in [
        "play_by_play",
        "goalie_stats",
        "power_play",
        "penalty_kill",
    ]:
        tables[key] = filter_to_games(tables[key], game_ids)
    # ------------------------------------------------------------
    # 8. Prepare standings
    # ------------------------------------------------------------
    standings = tables["daily_standings"].copy()

    standings["game_date"] = pd.to_datetime(
        standings["game_date"],
        errors="coerce",
    )

    # Full-season standings history for point-in-time SOS
    sos_standings = standings.copy()

    # Date-window standings for final feature rows
    tables["daily_standings"] = standings[
        standings["game_date"].dt.date <= max_day
    ].copy()
    

    # ------------------------------------------------------------
    # 9. Debug source-table sizes
    # ------------------------------------------------------------
    print("\n========== SOURCE TABLE ROW COUNTS ==========")

    for name, df in tables.items():
        print(f"{name:<20} → {df.shape}")

    print("============================================\n")
    # ------------------------------------------------------------
    # 10. Build strength of schedule
    # ------------------------------------------------------------
    print("▶ Building strength of schedule...")

    games_sos = strength_of_schedule(
        game_summary=game_summary,
        sos_standings=sos_standings,
    )

    games_sos = games_sos[
    games_sos["game_id"].isin(game_ids)
    ].copy()

    print(f"SOS rows: {len(games_sos)}")

    # -------------------------------------------------
    # Bootstrap SOS for teams with no prior games
    # -------------------------------------------------
    games_sos["sos_past"] = pd.to_numeric(
        games_sos["sos_past"],
        errors="coerce",
    )

    games_sos.loc[
        games_sos["games_before"].eq(0),
        "sos_past"
    ] = 0.0


    # Neutral SOS baseline when no opponent strength exists yet
    games_sos["sos_future"] = pd.to_numeric(
        games_sos["sos_future"],
        errors="coerce",
    )

    games_sos.loc[
        games_sos["sos_future"].isna()
        & games_sos["games_before"].eq(0),
        "sos_future"
    ] = 0.500
    
    # ------------------------------------------------------------
    # 11. Build transition / shooting features
    # ------------------------------------------------------------
    print("▶ Building transition offense features...")

    transition = build_transition_offense_features(
        tables["play_by_play"],
        games_spine=games_spine,
        offsets=offsets,
    )

    cumulative_rush_stats = transition["cumulative_rush"]
    cumulative_shooting_stats = transition["cumulative_shooting"]
    cumulative_scoring = transition["cumulative_scoring"]
    cumulative_shots_on_goal = transition["cumulative_sog"]

    # ------------------------------------------------------------incomplete
    # 12. Build goalie features
    # ------------------------------------------------------------
    print("▶ Building goalie features...")

    teams = load_teams(con)

    cumulative_goalie_stats = build_team_goalie_stats(
        goalies=tables["goalie_stats"],
        teams=teams,
        games_spine=games_spine,
    )

    print(f"Goalie rows: {len(cumulative_goalie_stats)}")

    print("\nANA GOALIE SOURCE CHECK:")

    print("Raw goalie_stats:")

    # ------------------------------------------------------------
    # 13. Build special teams features
    # ------------------------------------------------------------
    print("▶ Building special teams features...")

    special_teams = build_special_teams_stats(
        power_play=tables["power_play"],
        penalty_kill=tables["penalty_kill"],
        games_spine=games_spine,
        offsets=offsets,
    )

    print(f"Special teams rows: {len(special_teams)}")
    # ------------------------------------------------------------
    # 14. Build core feature dataframe
    # ------------------------------------------------------------
    print("▶ Building core feature dataframe...")
   
    core_df = build_core_df(
        play_by_play=tables["play_by_play"],
        games_spine=games_spine,
        games_sos=games_sos,
        special_teams=special_teams,
        cumulative_rush_stats=cumulative_rush_stats,
        cumulative_shooting_stats=cumulative_shooting_stats,
        cumulative_scoring=cumulative_scoring,
        cumulative_shots_on_goal=cumulative_shots_on_goal,
        cumulative_goalie_stats=cumulative_goalie_stats,
        offsets=offsets,
    )

    print(f"Core rows: {len(core_df)}")

    print("========================================\n")
    print("\nWINS OFFSET CHECK:")

    check_cols = [
        "game_id",
        "game_date",
        "team_id",
        "games_before",
        "games_played",
        "wins_before",
        "game_wins",
    ]

    print(
        core_df[
            [c for c in check_cols if c in core_df.columns]
        ]
        .sort_values(["team_id", "game_date", "game_id"])
        .head(1)
        .to_string(index=False)
    )
    
    # ------------------------------------------------------------
    # 15. Merge standings
    # ------------------------------------------------------------
    final_df = build_final_df(
        core_df=core_df,
        standings=tables["daily_standings"],
    )
    print(f"Final rows after standings: {len(final_df)}")

    print("\nFINAL PREGAME STATE CHECK:")
    print(
        final_df[
            [
                "game_id",
                "team_id",
                "games_before",
                "games_played",
                "wins_before",
            ]
        ]
        .sort_values(["team_id", "game_id"])
        .head(1)
        .to_string(index=False)
    )
    
    standings_check_cols = [
        "games_remaining",
        "wins",
        "wins_pctg",
        "losses",
        "ot_losses",
        "rotw",
        "rotw_pctg",
        "rw",
        "so_losses",
        "so_wins",
        "points",
        "points_pct",
        "projected_points",
        "in_playoff_spot",
        "gf_pctg",
        "goal_diff",
        "goal_diff_pctg",
        "streak",
        "streak_count",
        "last_10_gp",
        "last_10_wins",
        "last_10_losses",
        "last_10_otl",
        "last_10_points",
        "last_10_gf",
        "last_10_ga",
        "last_10_goal_diff",
        "last_10_rotw",
        "last_10_rw",
        "home_gp",
        "home_wins",
        "home_losses",
        "home_otlosses",
        "home_rotw",
        "home_rw",
        "home_points",
        "home_gf",
        "home_ga",
        "home_goal_diff",
        "road_gp",
        "road_wins",
        "road_losses",
        "road_ot_losses",
        "road_rotw",
        "road_rw",
        "road_points",
        "road_gf",
        "road_ga",
        "road_goal_diff",
        "division",
        "division_rank",
        "division_home_rank",
        "division_road_rank",
        "division_last_10_rank",
        "conference",
        "conference_rank",
        "conference_home_rank",
        "conference_road_rank",
        "conference_last_10_rank",
        "league_rank",
        "league_home_rank",
        "league_road_rank",
        "league_last_10_rank",
        "wildcard_rank",
        
    ]
    missing_standings_cols = [
        c for c in standings_check_cols
        if c not in final_df.columns
    ]

    print("\nMISSING STANDINGS COLUMNS:")
    print(missing_standings_cols)

    existing_standings_cols = [
        c for c in standings_check_cols
        if c in final_df.columns
    ]

    print("\nFINAL STANDINGS NULL CHECK:")
    print(
        final_df[existing_standings_cols]
        .isna()
        .sum()
        .to_string()
    )
    print("\nFINAL STANDINGS NULL CHECK:")
    print(
        final_df[existing_standings_cols]
        .isna()
        .sum()
        .to_string()
    )

    print("\nFINAL SOS NULL CHECK:")
    print(
        final_df[
            [
                "sos_past",
                "sos_future",
            ]
        ]
        .isna()
        .sum()
        .to_string()
    )

    # ------------------------------------------------------------
    # 16. Add derived / interaction features
    # ------------------------------------------------------------
    print("▶ Adding derived features...")

    final_df = special_features(final_df)

    # ------------------------------------------------------------
    # 17. Enforce final schema
    # ------------------------------------------------------------
    print("▶ Enforcing final schema...")

    final_df = enforce_data_schema(final_df)

    print(f"Final schema rows: {len(final_df)}")


    # ---------------------------------------------------------
    # Debug checks
    # ---------------------------------------------------------

    derived_check_cols = [
        "ot_wins",
        "overtimes",
        "ot_games_pct",
        "pt_pct_x_sos",
        "gd_x_rushdiff",
        "pk_load_x_pk_pct",
    ]

    print("\nFINAL DERIVED FEATURE NULL CHECK:")
    print(
        final_df[derived_check_cols]
        .isna()
        .sum()
        .to_string()
    )


    print("\nGAME WIN CHECK:")
    print(final_df["game_win"].value_counts(dropna=False))

    print("\nWINS PER GAME CHECK:")
    print(
        final_df.groupby("game_id")["game_win"]
        .sum()
        .value_counts()
    )


    # ------------------------------------------------------------
    # 18. Validate output
    # ------------------------------------------------------------
    print("▶ Validating output...")

    key_cols = ["season_id", "game_id", "team_id"]

    # No duplicate team/game rows
    dupes = final_df.duplicated(
        subset=key_cols,
        keep=False,
    )

    if dupes.any():
        raise ValueError(
            "Duplicate prediction rows found:\n"
            + final_df.loc[
                dupes,
                key_cols,
            ].to_string(index=False)
        )

    # Every game should have exactly 2 team rows
    rows_per_game = (
        final_df
        .groupby(["season_id", "game_id"])
        .size()
    )

    bad_games = rows_per_game[
        rows_per_game != 2
    ]

    if not bad_games.empty:
        raise ValueError(
            "Games without exactly 2 team rows:\n"
            + bad_games.to_string()
        )
   
    # Every requested game must be present
    expected_games = set(game_ids)
    actual_games = set(
        final_df["game_id"]
        .dropna()
        .astype(int)
    )

    missing_games = expected_games - actual_games

    if missing_games:
        raise ValueError(
            f"Missing games: {sorted(missing_games)}"
        )

    # Keys cannot be null
    if final_df[key_cols].isna().any().any():
        raise ValueError(
            "Null values found in prediction key columns."
        )

    print(
        f"✅ Validation passed: "
        f"{len(final_df)} rows / "
        f"{final_df['game_id'].nunique()} games"
    )
    # ------------------------------------------------------------
    # 19. Write validated to team_daily_stats
    # ------------------------------------------------------------
    print("▶ Writing to team_daily_stats...")
    # ------------------------------------------------------------
    # Verify table schema matches DATA_SCHEMA
    # ------------------------------------------------------------
    daily_stats_cols = [
        row[1]
        for row in con.execute(
            "PRAGMA table_info('team_daily_stats')"
        ).fetchall()
    ]

    expected_cols = list(DATA_SCHEMA.keys())

    missing_in_daily_stats = [
        col for col in expected_cols
        if col not in daily_stats_cols
    ]

    extra_in_daily_stats = [
        col for col in daily_stats_cols
        if col not in expected_cols
    ]

    if missing_in_daily_stats:
        raise ValueError(
            f"Columns missing from team_daily_stats: "
            f"{missing_in_daily_stats}"
        )

    if extra_in_daily_stats:
        print(
            f"⚠️ Extra columns in team_daily_stats "
            f"(not being written): {extra_in_daily_stats}"
        )

    print("✅ Team daily stats table schema validated.")
    con.register("prediction_updates", final_df)

    try:
        con.execute("BEGIN TRANSACTION")

        # Replace the recalculated suffix atomically.
        con.execute("""
            DELETE FROM team_daily_stats AS target
            WHERE EXISTS (
                SELECT 1
                FROM prediction_updates AS src
                WHERE target.season_id = src.season_id
                AND target.game_id = src.game_id
                AND target.team_id = src.team_id
            )
        """)

        columns = list(DATA_SCHEMA.keys())
        column_sql = ", ".join(f'"{col}"' for col in columns)

        con.execute(f"""
            INSERT INTO team_daily_stats ({column_sql})
            SELECT {column_sql}
            FROM prediction_updates
        """)

        con.execute("COMMIT")

    except Exception:
        con.execute("ROLLBACK")
        raise

    finally:
        con.unregister("prediction_updates")

    print(
        f"✅ Wrote {len(final_df)} rows to team_daily_stats."
    )
