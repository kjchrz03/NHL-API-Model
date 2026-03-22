import numpy as np
import pandas as pd
import warnings
import duckdb

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

DB_PATH = r"C:\Users\kjchr\Desktop\hockey\new_hockey_env\API data\duckdb\nhl_data.duckdb"


# ============================================================
# OPPONENT MAP (CANONICAL) — adds opponent_team_id per row
# ============================================================
def add_opponent_team_id(df: pd.DataFrame) -> pd.DataFrame:
    teams = (
        df[["season_id", "game_id", "team_id"]]
        .dropna()
        .drop_duplicates()
    )

    opponents = (
        teams.merge(teams, on=["season_id", "game_id"], suffixes=("", "_opp"))
        .query("team_id != team_id_opp")
        .rename(columns={"team_id_opp": "opponent_team_id"})
    )

    bad_games = (
        opponents.groupby(["season_id", "game_id"]).size()
        .reset_index(name="rows")
        .query("rows != 2")
    )
    if not bad_games.empty:
        raise ValueError(f"Invalid team count in some games:\n{bad_games.head(20)}")

    return df.merge(opponents, on=["season_id", "game_id", "team_id"], how="left")


# ============================================================
# RUSH STATS — CONSISTENT FOR / AGAINST LOGIC
# (No home/away assumptions; opponent inferred from possession)
# ============================================================
def build_rush_stats(play_by_play: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = play_by_play.copy()

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

    # -----------------------------
    # FOR — rush shots
    # -----------------------------
    rush_for = (
        rush_events
        .groupby(["season_id", "game_id", "team_id"], as_index=False)
        .size()
        .rename(columns={"size": "rush_shots_for"})
    )

    # -----------------------------
    # AGAINST — rush shots
    # prev_team_id is the defending team on possession change
    # -----------------------------
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
    rush_stats["rush_shot_differential"] = (
        rush_stats["rush_shots_for"] - rush_stats["rush_shots_against"]
    )

    return rush_events, rush_stats


# ============================================================
# CUMULATIVE RUSH — SEASON → GAME GRAIN SAFE
# ============================================================
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
# ============================================================
# SHOT ATTEMPTS (CORSI) — FIXED FOR/AGAINST USING SHOOTING TEAM
# Key fix: BLOCKED SHOTS (508) belong to the SHOOTING TEAM,
# but NHL PBP team_id is the BLOCKING TEAM on those rows.
# ============================================================
def extract_shot_attempts(play_by_play: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    SHOT_CODES = [505, 506, 507, 508, 537]  # goal, SOG, miss, block, failed attempt
    pbp = play_by_play.sort_values(["game_id", "sortOrder"]).copy()

    shot_events = pbp[pbp["type_code"].isin(SHOT_CODES)].copy()
    shot_events["shot_attempt"] = 1

    # add opponent on the shot_events rows
    shot_events = add_opponent_team_id(shot_events)

    # --- FIXED TEAM ATTRIBUTION ---
    # For blocks: team_id == blocking team, opponent_team_id == shooting team
    is_block = shot_events["type_code"].eq(508)

    shot_events["shooting_team_id"] = np.where(
        is_block,
        shot_events["opponent_team_id"],   # shooter
        shot_events["team_id"],            # shooter
    )

    shot_events["against_team_id"] = np.where(
        is_block,
        shot_events["team_id"],            # defender (blocker)
        shot_events["opponent_team_id"],   # defender
    )

    # --- GAME-LEVEL FOR ---
    shots_for = (
        shot_events.groupby(["season_id", "game_id", "shooting_team_id"], as_index=False)
        .size()
        .rename(columns={"shooting_team_id": "team_id", "size": "shot_attempts_for"})
    )

    # --- GAME-LEVEL AGAINST ---
    shots_against = (
        shot_events[shot_events["against_team_id"].notna()]
        .groupby(["season_id", "game_id", "against_team_id"], as_index=False)
        .size()
        .rename(columns={"against_team_id": "team_id", "size": "shot_attempts_against"})
    )

    shot_stats = (
        shots_for.merge(shots_against, on=["season_id", "game_id", "team_id"], how="outer")
        .fillna(0)
    )

    shot_stats["shot_attempts_for"] = shot_stats["shot_attempts_for"].astype(int)
    shot_stats["shot_attempts_against"] = shot_stats["shot_attempts_against"].astype(int)
    shot_stats["shot_attempts_differential"] = shot_stats["shot_attempts_for"] - shot_stats["shot_attempts_against"]

    # --- CORSI (you had OZ-only) ---
    corsi_for = (
        shot_events[shot_events["zone_code"] == "O"]
        .groupby(["season_id", "game_id", "shooting_team_id"], as_index=False)
        .size()
        .rename(columns={"shooting_team_id": "team_id", "size": "corsi_for"})
    )

    corsi_against = (
        shot_events[(shot_events["zone_code"] == "O") & (shot_events["against_team_id"].notna())]
        .groupby(["season_id", "game_id", "against_team_id"], as_index=False)
        .size()
        .rename(columns={"against_team_id": "team_id", "size": "corsi_against"})
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


# ============================================================
# CUMULATIVE SHOT ATTEMPTS
# ============================================================
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

    shots["shot_attempts_for_per_game"] = np.where(shots["games_played"] > 0, shots["c_shots_for"] / shots["games_played"], 0.0)
    shots["shot_attempts_against_per_game"] = np.where(shots["games_played"] > 0, shots["c_shots_against"] / shots["games_played"], 0.0)
    shots["shots_diff_per_game"] = np.where(shots["games_played"] > 0, shots["c_shots_diff"] / shots["games_played"], 0.0)
    shots["corsi_diff_per_game"] = np.where(shots["games_played"] > 0, shots["c_corsi_diff"] / shots["games_played"], 0.0)

    return shots


# ============================================================
# DANGER ZONES + REBOUNDS
# ============================================================
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

    # ensure opponent exists for later steps
    if "opponent_team_id" not in combined.columns:
        combined = add_opponent_team_id(combined)

    return combined


def add_scoring_chances(combined_chances_data: pd.DataFrame) -> pd.DataFrame:
    df = combined_chances_data.copy().sort_values(["game_id", "sortOrder"])

    df["base_scoring_chance_score"] = 3 * df["hd_shot"] + 2 * df["md_shot"] + 1 * df["ld_shot"]
    df["scoring_chance"] = (df["base_scoring_chance_score"] >= 2).astype(int)

    df["context_bonus"] = (df["is_rebound"] == 1).astype(int) + (df["rush_shot"] == 1).astype(int)
    df["blocked_penalty"] = (df["type_code"] == 508).astype(int)
    df["adj_scoring_chance_score"] = df["base_scoring_chance_score"] + df["context_bonus"] - df["blocked_penalty"]

    df["scoring_chance_goal"] = (df["type_code"] == 505).astype(int)
    df["shot_on_goal"] = (df["type_code"] == 506).astype(int) + df["scoring_chance_goal"]

    df["rush_scoring_chance"] = ((df["scoring_chance"] == 1) & (df["rush_shot"] == 1)).astype(int)
    return df


# ============================================================
# DANGER CORSI + SCF/SCA — FIXED FOR/AGAINST USING SHOOTING/AGAINST TEAM IDS
# ============================================================
def build_danger_corsi_and_scf_stats(scoring_chances_events: pd.DataFrame) -> pd.DataFrame:
    d = scoring_chances_events.copy()

    if "opponent_team_id" not in d.columns:
        d = add_opponent_team_id(d)

    is_block = d["type_code"].eq(508)

    # shooter attribution (same logic as shot attempts)
    d["shooting_team_id"] = np.where(is_block, d["opponent_team_id"], d["team_id"])
    d["against_team_id"]  = np.where(is_block, d["team_id"], d["opponent_team_id"])

    d["shot_on_goal"] = d["type_code"].isin([505, 506]).astype(int)
    d["scoring_chance_goal"] = (d["type_code"] == 505).astype(int)

    scf = (
        d.groupby(["season_id", "game_id", "shooting_team_id"], as_index=False)
        .agg(
            SCF=("scoring_chance", "sum"),
            SCSF=("shot_on_goal", "sum"),
            SCGF=("scoring_chance_goal", "sum"),
        )
        .rename(columns={"shooting_team_id": "team_id"})
    )

    sca = (
        d[d["against_team_id"].notna()]
        .groupby(["season_id", "game_id", "against_team_id"], as_index=False)
        .agg(SCA=("scoring_chance", "sum"))
        .rename(columns={"against_team_id": "team_id"})
    )

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

    corsi_against = (
        d[d["against_team_id"].notna()]
        .groupby(["season_id", "game_id", "against_team_id"], as_index=False)
        .agg(
            HDCA=("hd_corsi", "sum"),
            MDCA=("md_corsi", "sum"),
            LDCA=("ld_corsi", "sum"),
        )
        .rename(columns={"against_team_id": "team_id"})
    )

    final = (
        scf.merge(sca, on=["season_id", "game_id", "team_id"], how="left")
           .merge(corsi_for, on=["season_id", "game_id", "team_id"], how="left")
           .merge(corsi_against, on=["season_id", "game_id", "team_id"], how="left")
           .fillna(0)
    )

    final["HDCF_pct"] = final["HDCF"] / (final["HDCF"] + final["HDCA"]).replace(0, np.nan)
    final["HDCA_pct"] = final["HDCA"] / (final["HDCF"] + final["HDCA"]).replace(0, np.nan)

    final["MDCF_pct"] = final["MDCF"] / (final["MDCF"] + final["MDCA"]).replace(0, np.nan)
    final["MDCA_pct"] = final["MDCA"] / (final["MDCF"] + final["MDCA"]).replace(0, np.nan)

    final["LDCF_pct"] = final["LDCF"] / (final["LDCF"] + final["LDCA"]).replace(0, np.nan)
    final["LDCA_pct"] = final["LDCA"] / (final["LDCF"] + final["LDCA"]).replace(0, np.nan)

    return final


# ============================================================
# CUMULATIVE SCORING CHANCES
# ============================================================
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

    shots["corsi_diff_per_game"] = np.where(shots["games_played"] > 0, shots["c_corsi_diff"] / shots["games_played"], 0.0)

    merge_cols = ["season_id", "team_id", "game_id"]
    if "game_date" in scoring.columns and "game_date" in shots.columns:
        merge_cols.append("game_date")

    scoring = scoring.merge(
        shots[merge_cols + ["c_corsi_for", "c_corsi_against", "c_corsi_diff", "corsi_diff_per_game"]],
        on=merge_cols,
        how="left",
    )

    return scoring


# ============================================================
# TEAM SHOOTING STATS (SOG/GOALS) — opponent map via canonical helper
# ============================================================
def build_team_shooting_stats(scoring_chances_events: pd.DataFrame) -> pd.DataFrame:
    df = scoring_chances_events.copy()

    # ensure opponent_team_id exists exactly once
    if "opponent_team_id" not in df.columns:
        df = add_opponent_team_id(df)
        
    df["goal"] = df["scoring_chance_goal"]
    df["sog"] = df["shot_on_goal"]

    df["hd_sog"] = ((df["sog"] == 1) & (df["hd_shot"] == 1)).astype(int)
    df["md_sog"] = ((df["sog"] == 1) & (df["md_shot"] == 1)).astype(int)
    df["ld_sog"] = ((df["sog"] == 1) & (df["ld_shot"] == 1)).astype(int)

    df["hd_goal"] = ((df["goal"] == 1) & (df["hd_shot"] == 1)).astype(int)
    df["md_goal"] = ((df["goal"] == 1) & (df["md_shot"] == 1)).astype(int)
    df["ld_goal"] = ((df["goal"] == 1) & (df["ld_shot"] == 1)).astype(int)

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
        df[df["opponent_team_id"].notna()]
        .groupby(["season_id", "game_id", "opponent_team_id"], as_index=False)
        .agg(
            shots_against=("sog", "sum"),
            goals_against=("goal", "sum"),
            hd_against=("hd_sog", "sum"),
            md_against=("md_sog", "sum"),
            ld_against=("ld_sog", "sum"),
        )
        .rename(columns={"opponent_team_id": "team_id"})
    )

    shooting_stats = stats_for.merge(stats_against, on=["season_id", "game_id", "team_id"], how="left").fillna(0)
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


# ============================================================
# SEASON-LEVEL REBUILD EXECUTOR
# ============================================================
def rebuild_shooting_for_season(
    con: duckdb.DuckDBPyConnection,
    season_id: int,
    offsets: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:

    print(f"\n🔄 Rebuilding shooting metrics for season {season_id}")

    pbp = con.execute("""
        SELECT *
        FROM all_play_by_play
        WHERE season_id = ?
        ORDER BY game_id, sortOrder
    """, [season_id]).df()

    games_spine = con.execute("""
        SELECT season_id, game_id, team_id, game_date, games_played
        FROM team_daily_stats
        WHERE season_id = ?
    """, [season_id]).df()

    # NOTE: build_rush_stats / cumulative_rush must exist in your environment
    rush_events, _ = build_rush_stats(pbp)
    shot_stats, shot_events = extract_shot_attempts(pbp)

    # attach games spine to shot_stats BEFORE cumulative calcs
    shot_stats = shot_stats.merge(
        games_spine[["season_id", "game_id", "team_id", "games_played", "game_date"]],
        on=["season_id", "game_id", "team_id"],
        how="left"
    )

    danger_events = preprocess_shot_dataframe(shot_events)
    combined = build_enriched_shot_events(shot_events, danger_events, rush_events)
    scoring_events = add_scoring_chances(combined)

    danger_stats = build_danger_corsi_and_scf_stats(scoring_events)
    sog_stats = build_team_shooting_stats(scoring_events)

    # attach games spine to danger/sog BEFORE cumulative calcs
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

    out = {
        "cumulative_shooting": cumulative_shot_attempts(shot_stats, offsets),
        "cumulative_scoring": cumulative_scoring_chances(danger_stats, shot_stats, offsets),
        "cumulative_sog": cumulative_sog(sog_stats, offsets),
    }

    for name, df in out.items():
        assert_game_grain(df, name)

    return out

# ============================================================
# CANONICAL LIST OF RECOMPUTED COLUMNS
# ============================================================
RECOMPUTED_COLS = [
    # keys / spine
    "season_id",
    "game_id",
    "team_id",
    "game_date",
    "games_played",

    # shooting attempts / corsi
    "shot_attempts_for",
    "shot_attempts_against",
    "shot_attempts_differential",
    "corsi_for",
    "corsi_against",
    "corsi_diff",
    "shot_attempts_for_per_game",
    "shot_attempts_against_per_game",
    "shots_diff_per_game",
    "corsi_diff_per_game",

    # scoring chances
    "SCF",
    "SCA",
    "SCSF",
    "SCGF",
    "SCF_pg",
    "SCA_pg",

    # danger corsi
    "HDCF",
    "HDCA",
    "MDCF",
    "MDCA",
    "LDCF",
    "LDCA",
    "HDCF_pct",
    "HDCA_pct",
    "MDCF_pct",
    "MDCA_pct",
    "LDCF_pct",
    "LDCA_pct",

    # cumulative totals
    "c_shots_for",
    "c_shots_against",
    "c_shots_diff",
    "c_corsi_for",
    "c_corsi_against",
    "c_corsi_diff",

    # SOG / PDO
    "shots_for",
    "shots_against",
    "goals_for",
    "goals_against",
    "shot_pct",
    "sv_pct",
    "pdo",
]

# ============================================================
# GRAIN ASSERTION (GAME-LEVEL SAFETY CHECK)
# ============================================================
def assert_game_grain(df: pd.DataFrame, name: str):
    dupes = (
        df.groupby(["season_id", "game_id", "team_id"])
          .size()
          .reset_index(name="n")
          .query("n > 1")
    )
    if not dupes.empty:
        raise ValueError(
            f"{name} violates game-level grain "
            f"(season_id, game_id, team_id). Sample:\n{dupes.head()}"
        )
    print("✅ Game-level grain check passed")

# ============================================================
# STAGING + SAFETY CHECKS + UPSERT
# ============================================================
def write_staging_table(con: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    print("🧪 Writing stg_team_daily_stats")

    missing = set(RECOMPUTED_COLS) - set(df.columns)
    if missing:
        raise KeyError(f"Missing recomputed columns: {missing}")

    staged = df[RECOMPUTED_COLS].copy()

    # 🔑 Register DataFrame with DuckDB
    con.register("staged_df", staged)

    con.execute("DROP TABLE IF EXISTS stg_team_daily_stats")
    con.execute("""
        CREATE TABLE stg_team_daily_stats AS
        SELECT * FROM staged_df
    """)

    # Optional but clean
    con.unregister("staged_df")

def assert_schema_compatible(con: duckdb.DuckDBPyConnection):
    """
    Ensure staging contains exactly the columns we intend to overwrite.
    Do NOT require full team_daily_stats schema.
    """

    staged_cols = set(
        con.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'stg_team_daily_stats'
        """).df()["column_name"]
    )

    expected = set(RECOMPUTED_COLS)

    missing = expected - staged_cols
    extra   = staged_cols - expected

    if missing or extra:
        raise ValueError(
            f"Staging schema mismatch:\n"
            f"Missing recomputed cols: {missing}\n"
            f"Unexpected extra cols: {extra}"
        )

    print("✅ Recomputed-column schema check passed")

def assert_pk_uniqueness(con: duckdb.DuckDBPyConnection):
    dupes = con.execute("""
        SELECT season_id, game_id, team_id, COUNT(*) AS n
        FROM stg_team_daily_stats
        GROUP BY 1,2,3
        HAVING COUNT(*) > 1
    """).df()

    if not dupes.empty:
        raise ValueError(
            f"Duplicate (season_id, game_id, team_id) rows detected:\n{dupes.head()}"
        )

    print("✅ Primary key uniqueness check passed")

def assert_reasonable_row_delta(
    con: duckdb.DuckDBPyConnection,
    season_id: int,
    max_pct_change: float = 0.05,
):
    # Known anomalous seasons (COVID / format changes)
    BUBBLE_SEASONS = {20192020}

    before = con.execute("""
        SELECT COUNT(*) FROM team_daily_stats
        WHERE season_id = ?
    """, [season_id]).fetchone()[0]

    after = con.execute("""
        SELECT COUNT(*) FROM stg_team_daily_stats
        WHERE season_id = ?
    """, [season_id]).fetchone()[0]

    if before == 0:
        print("⚠️ No existing rows for season; skipping delta check")
        return

    pct_change = abs(after - before) / before

    if season_id in BUBBLE_SEASONS:
        print(
            f"⚠️ Skipping strict row-delta check for anomalous season {season_id} "
            f"({before} → {after}, {pct_change:.1%})"
        )
        return

    if pct_change > max_pct_change:
        raise ValueError(
            f"Row count change too large for season {season_id}: "
            f"{before} → {after} ({pct_change:.1%})"
        )

    print(f"✅ Row count delta OK ({before} → {after})")
def assert_metric_sanity(con: duckdb.DuckDBPyConnection):
    bad = con.execute("""
        SELECT *
        FROM stg_team_daily_stats
        WHERE
            HDCF < 0 OR HDCA < 0
            OR MDCF < 0 OR MDCA < 0
            OR LDCF < 0 OR LDCA < 0
            OR HDCF_pct < 0 OR HDCF_pct > 1
            OR MDCF_pct < 0 OR MDCF_pct > 1
            OR LDCF_pct < 0 OR LDCF_pct > 1
    """).df()

    if not bad.empty:
        raise ValueError(
            f"Metric sanity violation detected:\n{bad.head()}"
        )

    print("✅ Metric sanity checks passed")

def upsert_team_daily_stats(con: duckdb.DuckDBPyConnection):
    print("⬆️ Updating recomputed columns in team_daily_stats")

    # build SET clause dynamically
    set_clause = ",\n        ".join(
        f"{col} = stg.{col}" for col in RECOMPUTED_COLS
        if col not in ("season_id", "game_id", "team_id")
    )

    sql = f"""
        UPDATE team_daily_stats AS tgt
        SET
            {set_clause}
        FROM stg_team_daily_stats AS stg
        WHERE
            tgt.season_id = stg.season_id
            AND tgt.game_id = stg.game_id
            AND tgt.team_id = stg.team_id
    """

    con.execute(sql)

    print("✅ Recomputed columns updated without touching other data")
# ============================================================
# FULL BACKFILL RUNNER (SEASON BY SEASON)
# ============================================================
def backfill_all_shooting(
    con: duckdb.DuckDBPyConnection,
    season_ids: list[int],
    offsets: pd.DataFrame | None = None,
):
    for season_id in season_ids:
        print(f"\n🔄 Backfilling season {season_id}")

        dfs = rebuild_shooting_for_season(con, season_id, offsets)

        # ----------------------------------------------------
        # SAFE MERGE (do NOT drop join keys)
        # ----------------------------------------------------
        # ----------------------------------------------------
        # VERIFY JOIN KEYS EXIST BEFORE MERGE
        # ----------------------------------------------------
        JOIN_KEYS = ["season_id", "game_id", "team_id", "game_date", "games_played"]

        for name in ["cumulative_shooting", "cumulative_scoring", "cumulative_sog"]:
            missing = set(JOIN_KEYS) - set(dfs[name].columns)
            if missing:
                raise KeyError(f"{name} missing join keys: {missing}")
        base = dfs["cumulative_shooting"].copy()

        scoring = dfs["cumulative_scoring"].drop(
            columns=[c for c in dfs["cumulative_scoring"].columns
                    if c in base.columns and c not in JOIN_KEYS],
            errors="ignore",
        )

        sog = dfs["cumulative_sog"].drop(
            columns=[c for c in dfs["cumulative_sog"].columns
                    if c in base.columns and c not in JOIN_KEYS],
            errors="ignore",
        )

        final = (
            base
            .merge(scoring, on=JOIN_KEYS, how="left")
            .merge(sog,     on=JOIN_KEYS, how="left")
        )

        # ----------------------------------------------------
        # SAFETY CHECKS
        # ----------------------------------------------------
        assert_game_grain(final, f"final season {season_id}")

        print(
            f"Rows recomputed for season {season_id}: "
            f"{final[['season_id','game_id','team_id']].drop_duplicates().shape[0]}"
        )

        write_staging_table(con, final)

        # ---- SAFETY RAILS ----
        assert_schema_compatible(con)
        assert_pk_uniqueness(con)
        assert_reasonable_row_delta(con, season_id)
        assert_metric_sanity(con)

        # ----------------------------------------------------
        # UPSERT
        # ----------------------------------------------------
        upsert_team_daily_stats(con)

        print(f"✅ Season {season_id} complete")
# ============================================================
# USAGE
# ============================================================
con = duckdb.connect(DB_PATH)
backfill_all_shooting(
    con,
    season_ids=[
        20132014, 20142015, 20152016, 20162017,
        20172018, 20182019, 
        
        20192020, 20202021,
        20212022, 20222023, 20232024, 20242025,
        20252026,
    ],
)

con.close()
print("DuckDB connection closed.")