# ============================================================
# Feature Engineering — Playoff Model (v4)
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
# Wild-card field pressure features
# ============================================================

def add_wc_field_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["wildcard_rank"] = pd.to_numeric(out["wildcard_rank"], errors="coerce")
    out["division_rank"] = pd.to_numeric(out["division_rank"], errors="coerce")

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

    out["in_wc_spot"] = (out["wildcard_rank"] <= 2).fillna(False).astype(int)
    out["in_div_top3"] = (out["division_rank"] <= 3).fillna(False).astype(int)
    out["on_wc_bubble"] = out["wildcard_rank"].between(1, 4).fillna(False).astype(int)

    return out


def add_wc_density_features(df: pd.DataFrame, windows=(2, 4, 6)) -> pd.DataFrame:
    out = df.copy()

    if "wc2_points" not in out.columns:
        raise RuntimeError("Run add_wc_field_pressure_features first")

    wc_mask = out["division_rank"] > 3

    for w in windows:
        col = f"wc_teams_within_{w}pts"

        out[col] = (
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
# OPTIONAL feature pruning (OFF by default)
# ============================================================

# DROP_COLS = [
#     "shots_for", "shots_against",
#     "goals_for", "goals_against",
#     "hd_sog", "md_sog", "ld_sog",
#     "hd_goals", "md_goals", "ld_goals",
#     "hd_against", "md_against", "ld_against",
#     "game_savePct", "game_GAA",
#     "pp_gf", "pp_ga", "sh_gf", "sh_ga", "times_sh",
# ]

# def drop_high_noise_features(df: pd.DataFrame) -> pd.DataFrame:
#     return df.drop(columns=DROP_COLS, errors="ignore")


# ============================================================
# MASTER ENTRY POINT
# ============================================================

def build_features(df: pd.DataFrame, drop_noise: bool = False) -> pd.DataFrame:

    out = df.copy()

    REQUIRED_COLS = [
        "season_id", "game_date", "team_id", "tri_code",
        "conference", "conference_rank", "division_rank",
        "wildcard_rank", "points", "points_pct",
    ]

    missing = [c for c in REQUIRED_COLS if c not in out.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = (
        out
        .sort_values(["season_id", "team_id", "game_date"])
        .reset_index(drop=True)
    )

    out = normalize_made_playoffs(out)
    out = add_conference_baseline(out)
    out = add_streak_features(out)
    out = add_rank_bucket(out)
    out = add_wc_field_pressure_features(out)
    out = add_wc_density_features(out)

    # if drop_noise:
    #     out = drop_high_noise_features(out)

    return out.copy()