

"""Game-level SHAP explanations, incrementally persisted to DuckDB.

Run after update_shap_summary. Game dates are actual game dates.
Missing feature/game values remain NULL; they are not fabricated.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
from joblib import load

from config import MODEL_PATH
from feature_engineering import build_features
from .feature_metadata import (
    FEATURE_GAME_STATS,
    GAME_CONTRIBUTION_VALUES,
    CUMULATIVE_RATE_VALUES,
    GAME_AVERAGE_VALUES,
    GAME_RATE_VALUES,
    PER_GAME_CONTEXT,
    RANK_CONTEXT,
    get_feature_label,
)
EXPLANATION_VERSION = "game-explanations-v6-balanced-context-v4"
DESTINATION = "game_shap_explanations_v6_balanced"
TOP_N = 5
KEYS = ["season_id", "snapshot_date", "team_id", "model_version"]


def _ident(name):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return f'"{name}"'


def _columns(con, table):
    return {row[1] for row in con.execute(f"PRAGMA table_info({_ident(table)})").fetchall()}


def _exists(con, table):
    return bool(con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone())


def _version():
    version = str(load(MODEL_PATH)["version"])
    if not re.fullmatch(r"v[0-9]+", version):
        raise ValueError(f"Invalid model version: {version}")
    return version


def _source_tables(version):
    if version == "v6":
        return "playoff_shap_summary_v6_balanced", "playoff_shap_values_v6_balanced"
    return f"playoff_shap_summary_{version}", f"playoff_shap_values_{version}"


def _ensure_table(con):
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {DESTINATION} (
            season_id BIGINT NOT NULL,
            snapshot_date DATE NOT NULL,
            game_date DATE,
            game_id VARCHAR,
            team_id BIGINT NOT NULL,
            tri_code VARCHAR,
            model_version VARCHAR NOT NULL,
            playoff_prob DOUBLE,
            feature VARCHAR NOT NULL,
            direction VARCHAR NOT NULL,
            rank INTEGER NOT NULL,
            shap_value DOUBLE,
            feature_value DOUBLE,
            game_contribution DOUBLE,
            game_vs_cumulative DOUBLE,
            game_vs_average DOUBLE,
            game_stats JSON,
            model_explanation VARCHAR,
            season_context VARCHAR,
            game_context VARCHAR,
            game_explanation VARCHAR,
            needs_review BOOLEAN,
            explanation_version VARCHAR NOT NULL,
            source_hash VARCHAR NOT NULL,
            generated_at TIMESTAMPTZ NOT NULL
        )
    """)
    required = {"season_id", "snapshot_date", "game_date", "game_id", "team_id",
                "tri_code", "model_version", "playoff_prob", "feature", "direction",
                "rank", "shap_value", "feature_value", "game_contribution",
                "game_vs_cumulative", "game_vs_average", "game_stats",
                "model_explanation", "season_context", "game_context",
                "game_explanation", "needs_review", "explanation_version",
                "source_hash", "generated_at"}
    missing = required - _columns(con, DESTINATION)
    if missing:
        raise ValueError(f"{DESTINATION} incompatible schema: {sorted(missing)}")
    con.execute(f"""CREATE UNIQUE INDEX IF NOT EXISTS uq_game_shap_explanations_v6_balanced
        ON {DESTINATION}(season_id, snapshot_date, team_id, model_version, direction, rank)""")


def _load(con, season_id, version):
    """
    Load ranked SHAP features, model feature values,
    and corresponding game-level statistics.
    """

    # ====================================================
    # 1. IDENTIFY AND VALIDATE SOURCE TABLES
    # ====================================================

    summary, raw = _source_tables(version)

    for table in (summary, raw, "team_daily_stats"):
        if not _exists(con, table):
            raise ValueError(f"Missing required table: {table}")

    summary_cols = _columns(con, summary)

    required = set(
        KEYS + [
            "game_date",
            "tri_code",
            "playoff_prob",
            "feature",
            "shap_value",
            "direction",
            "rank",
        ]
    )

    missing = required - summary_cols

    if missing:
        raise ValueError(
            f"{summary} missing columns: {sorted(missing)}"
        )

    # ====================================================
    # 2. LOAD TOP SHAP FEATURE RANKINGS
    # ====================================================

    ranked = con.execute(
        f"""
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
        FROM {_ident(summary)}
        WHERE season_id = ?
          AND model_version = ?
          AND direction IN ('helpful', 'harmful')
          AND rank BETWEEN 1 AND ?
        """,
        [season_id, version, TOP_N],
    ).df()

    if ranked.empty:
        return ranked

    if ranked.duplicated(
        KEYS + ["direction", "rank"]
    ).any():
        raise ValueError("Duplicate SHAP ranking keys")

    ranked["game_date"] = pd.to_datetime(
        ranked["game_date"]
    ).dt.normalize()

    ranked["snapshot_date"] = pd.to_datetime(
        ranked["snapshot_date"]
    ).dt.normalize()

    # ====================================================
    # 3. IDENTIFY TRAINED MODEL FEATURES
    # ====================================================

    raw_cols = _columns(con, raw)
    if not set(KEYS).issubset(raw_cols):
        raise ValueError(f"{raw} lacks snapshot keys")

    model_features = list(load(MODEL_PATH)["features"])
    ranked_features = set(ranked["feature"].unique())
    unknown = ranked_features - set(model_features)
    if unknown:
        raise ValueError(
            f"SHAP rankings contain features absent from the model: {sorted(unknown)}"
        )

    # If raw SHAP rows contain actual input columns, retain them as the
    # preferred values. SHAP contribution columns are never model inputs.
    available = [feature for feature in model_features if feature in raw_cols]
    if available:
        raw_df = con.execute(
            f"""
            SELECT {', '.join(map(_ident, KEYS + available))}
            FROM {_ident(raw)}
            WHERE season_id = ? AND model_version = ?
            """,
            [season_id, version],
        ).df()
        if raw_df.duplicated(KEYS).any():
            raise ValueError("Duplicate raw SHAP snapshot keys")
        raw_df = raw_df.rename(
            columns={feature: f"raw_model__{feature}" for feature in available}
        )
        ranked = ranked.merge(raw_df, on=KEYS, how="left", validate="many_to_one")

    # ====================================================
    # 4. LOAD ALL HISTORICAL TEAM SNAPSHOTS
    # ====================================================

    daily_cols = _columns(con, "team_daily_stats")
    join_keys = ["season_id", "team_id", "game_date"]
    if not set(join_keys).issubset(daily_cols):
        raise ValueError(
            "team_daily_stats must contain season_id, team_id, game_date"
        )

    daily = con.execute(
        "SELECT * FROM team_daily_stats WHERE season_id = ?",
        [season_id],
    ).df()
    daily["game_date"] = pd.to_datetime(daily["game_date"]).dt.normalize()

    if daily.duplicated(join_keys).any():
        raise ValueError(
            "team_daily_stats contains duplicate season/team/game dates"
        )

    # ====================================================
    # 5. RECONSTRUCT ENGINEERED MODEL FEATURES
    # ====================================================

    # Use the complete daily cross-section: wildcard cutlines and
    # relative league statistics depend on other teams on the same date.
    # build_features only uses rows within each season/game_date for
    # cross-team calculations, so future dates do not enter a snapshot.
    engineered = build_features(daily.copy())

    if len(engineered) != len(daily):
        raise ValueError("build_features changed the number of team snapshots")
    if engineered.duplicated(join_keys).any():
        raise ValueError("build_features produced duplicate team snapshot keys")

    missing_columns = ranked_features - set(engineered.columns) - set(available)
    if missing_columns:
        raise ValueError(
            "Model feature columns cannot be reconstructed: "
            f"{sorted(missing_columns)}"
        )

    engineered_cols = [
        feature for feature in model_features if feature in engineered.columns
    ]
    feature_df = engineered[join_keys + engineered_cols].rename(
        columns={feature: f"engineered__{feature}" for feature in engineered_cols}
    )

    ranked = ranked.merge(
        feature_df,
        on=join_keys,
        how="left",
        validate="many_to_one",
    )

    # ====================================================
    # 6. JOIN ACTUAL GAME STATISTICS
    # ====================================================

    overlapping = (set(ranked.columns) & set(daily.columns)) - set(join_keys)
    daily = daily.rename(
        columns={col: f"daily__{col}" for col in overlapping}
    )
    ranked = ranked.merge(
        daily,
        on=join_keys,
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    missing_games = ranked["_merge"] != "both"
    if missing_games.any():
        examples = ranked.loc[missing_games, join_keys].head(5).to_dict("records")
        raise ValueError(f"Missing game stats for SHAP snapshots: {examples}")
    ranked.drop(columns="_merge", inplace=True)

    if "game_id" not in ranked.columns:
        ranked["game_id"] = None

    # ====================================================
    # 7. EXTRACT MODEL INPUT VALUES (NOT SHAP CONTRIBUTIONS)
    # ====================================================

    def lookup_feature(row):
        feature = row["feature"]
        for col in (f"raw_model__{feature}", f"engineered__{feature}"):
            if col in row.index and pd.notna(row[col]):
                return row[col]
        return np.nan

    ranked["feature_value"] = pd.to_numeric(
        ranked.apply(lookup_feature, axis=1), errors="coerce"
    )

    # Genuine missing source inputs stay NULL; never fabricate zeroes.
    # The explanatory text already handles missing feature values, and
    # these rows are explicitly flagged for review downstream.
    missing_values = ranked["feature_value"].isna()
    if missing_values.any():
        counts = ranked.loc[missing_values, "feature"].value_counts().to_dict()
        print(f"Warning: unavailable model feature values (kept NULL): {counts}")

    return ranked


def _numeric(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    if f"daily__{col}" in df.columns:
        return pd.to_numeric(df[f"daily__{col}"], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _safe_div(a, b):
    return a.div(b.replace(0, np.nan))


def _derive(df):
    specs = {
        "game_goal_diff": ("goals_for", "goals_against"),
        "game_shots_diff": ("shots_for", "shots_against"),
        "game_corsi_diff": ("corsi_for", "corsi_against"),
        "game_sc_net": ("SCF", "SCA"),
        "game_pp_net": ("pp_gf", "sh_ga"),
        "game_pk_net": ("sh_gf", "pp_ga"),
    }
    for target, (left, right) in specs.items():
        if left in df or f"daily__{left}" in df:
            if right in df or f"daily__{right}" in df:
                df[target] = _numeric(df, left) - _numeric(df, right)
    if {"game_pp_net", "game_pk_net"}.issubset(df):
        df["game_special_teams_net"] = df["game_pp_net"] + df["game_pk_net"]
    if "game_pp_net" in df:
        df["game_pp_net_pct"] = _safe_div(df["game_pp_net"], _numeric(df, "pp_opp"))
    if "game_pk_net" in df:
        df["game_pk_net_pct"] = _safe_div(df["game_pk_net"], _numeric(df, "times_sh"))
    if "goals_for" in df or "daily__goals_for" in df:
        df["game_shootingPct"] = _safe_div(_numeric(df, "goals_for"), _numeric(df, "shots_for"))
        df["game_gf_pct"] = _safe_div(_numeric(df, "goals_for"), _numeric(df, "goals_for") + _numeric(df, "goals_against"))
    if "game_shootingPct" in df and ("game_savePct" in df or "daily__game_savePct" in df):
        df["game_pdo"] = df["game_shootingPct"] + _numeric(df, "game_savePct")
    # An OT/SO loss earns one point, so do not infer points from game_win alone.
    if "game_points_earned" not in df:
        df["game_points_earned"] = np.where(_numeric(df, "game_win") == 1, 2.0, np.nan)
    if "game_points_pct" not in df:
        df["game_points_pct"] = df["game_points_earned"] / 2
    comparisons = {
        "rush_for_vs_avg": ("rush_shots_for", "rush_for_per_game"),
        "rush_against_vs_avg": ("rush_shots_against", "rush_against_per_game"),
        "rush_diff_vs_avg": ("rush_shot_differential", "rush_diff_per_game"),
        "corsi_diff_vs_avg": ("game_corsi_diff", "corsi_diff_per_game"),
        "shot_attempts_for_vs_avg": ("shot_attempts_for", "shot_attempts_for_per_game"),
        "shot_attempts_against_vs_avg": ("shot_attempts_against", "shot_attempts_against_per_game"),
        "shots_diff_vs_avg": ("game_shots_diff", "shots_diff_per_game"),
        "scf_vs_avg": ("SCF", "SCF_pg"),
        "sca_vs_avg": ("SCA", "SCA_pg"),
    }
    for target, (left, right) in comparisons.items():
        df[target] = _numeric(df, left) - _numeric(df, right)
    return df


def _context(row):
    feature = row["feature"]
    def value(col):
        if col in row.index and pd.notna(row[col]):
            return row[col]
        alt = f"daily__{col}"
        return row[alt] if alt in row.index else np.nan
    contrib = GAME_CONTRIBUTION_VALUES.get(feature)
    rate = CUMULATIVE_RATE_VALUES.get(feature)
    avg = GAME_AVERAGE_VALUES.get(feature)
    stats = {}
    for col in FEATURE_GAME_STATS.get(feature, []):
        v = value(col)
        if pd.notna(v):
            stats[col] = float(v) if isinstance(v, (np.number, int, float)) else str(v)
    if feature in PER_GAME_CONTEXT:
        game_col = PER_GAME_CONTEXT[feature][0]
        v = value(game_col)
        if pd.notna(v):
            stats[game_col] = float(v)
    # Capture only observed game-level context used by dedicated templates.
    # Do not infer home/away, overtime, or shootout results from season totals.
    context_columns = {
        "game_win", "game_id", "game_type", "home_road", "home_away",
        "is_home", "is_road", "game_ot", "game_so", "went_to_ot",
        "went_to_so", "ot", "shootout", "game_points_earned",
        "goals_for", "goals_against", "game_goal_diff",
        "game_gf_pct", "game_points_pct", "game_sc_net",
        "SCF", "SCA", "game_HDCF", "HDCF", "HDCA",
    }
    for col in context_columns:
        v = value(col)
        if pd.notna(v):
            stats[col] = _json_safe(v)
    game_contribution = value(contrib) if contrib else np.nan
    game_vs_cumulative = value(rate) - row["feature_value"] if rate and pd.notna(value(rate)) else np.nan
    game_vs_average = value(avg) - row["feature_value"] if avg and pd.notna(value(avg)) else np.nan
    return pd.Series({"game_contribution": game_contribution,
                      "game_vs_cumulative": game_vs_cumulative,
                      "game_vs_average": game_vs_average, "game_stats": stats})


def format_hockey_pct(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.3f}".removeprefix("0")


def format_number(value, decimals=0, signed=False):
    if pd.isna(value):
        return "N/A"

    sign = "+" if signed else ""
    return f"{value:{sign}.{decimals}f}"


def format_rank(value):
    if pd.isna(value):
        return "N/A"

    rank = int(value)

    if 10 <= rank % 100 <= 20:
        suffix = "th"
    else:
        suffix = {
            1: "st",
            2: "nd",
            3: "rd"
        }.get(rank % 10, "th")

    return f"{rank}{suffix}"


def stat(stats, key):
    value = stats.get(key, np.nan)
    return value if pd.notna(value) else np.nan


def describe_difference(diff, label, decimals=2):

    if pd.isna(diff):
        return ""

    if abs(diff) < 0.005:
        return f"essentially matching its season average"

    amount = format_number(abs(diff), decimals)

    if diff > 0:
        return f"{amount} {label} above its season average"

    return f"{amount} {label} below its season average"



def generate_game_explanation(row):

    feature = row["feature"]
    shap_value = row["shap_value"]
    feature_value = row["feature_value"]

    stats = row.get("game_stats", {})
    stats = stats if isinstance(stats, dict) else {}

    game_contribution = row.get(
        "game_contribution", np.nan
    )

    game_vs_cumulative = row.get(
        "game_vs_cumulative", np.nan
    )

    label = get_feature_label(feature)

    season_context = ""
    game_context = ""

    # ====================================================
    # 1. MODEL IMPACT
    # ====================================================

    if shap_value > 0:

        model_explanation = (
            f"{label.capitalize()} is helping the team's "
            f"playoff prediction relative to the model baseline."
        )

    elif shap_value < 0:

        model_explanation = (
            f"{label.capitalize()} is hurting the team's "
            f"playoff prediction relative to the model baseline."
        )

    else:

        model_explanation = (
            f"{label.capitalize()} has no contribution "
            f"relative to the model baseline."
        )

    # Context is based on the point-in-time team snapshot. An engineered
    # season feature is not a statistic from one individual game.
    def snapshot_value(name):
        for key in (name, f"daily__{name}"):
            if key in row.index and pd.notna(row[key]):
                return pd.to_numeric(row[key], errors="coerce")
        return np.nan

    # Dedicated context for results and venue splits. Season aggregates
    # are never presented as if they were single-game statistics.
    def observed(name):
        v = stat(stats, name)
        if pd.notna(v):
            return v
        return snapshot_value(name)

    def venue():
        for name in ("home_road", "home_away"):
            v = observed(name)
            if pd.notna(v):
                token = str(v).strip().lower()
                if token in {"h", "home"}:
                    return "home"
                if token in {"a", "away", "r", "road"}:
                    return "road"
        for name, yes, no in (("is_home", "home", "road"),
                              ("is_road", "road", "home")):
            v = observed(name)
            if pd.notna(v) and str(v).strip().lower() in {"0", "1", "true", "false"}:
                return yes if str(v).strip().lower() in {"1", "true"} else no
        return None

    def result_description():
        win = observed("game_win")
        if pd.notna(win) and str(win).strip() in {"0", "1", "0.0", "1.0"}:
            return "won" if float(win) == 1 else "did not win"
        return None

    if feature in {"wins_pctg", "gf_pctg", "ot_games_pct", "projected_points"}:
        descriptions = {
            "wins_pctg": "season winning percentage",
            "gf_pctg": "season share of goals scored",
            "ot_games_pct": "share of games requiring overtime",
            "projected_points": "projected season standings points",
        }
        season_context = (
            f"The team's {descriptions[feature]} at this snapshot is "
            f"{format_number(feature_value, 3)}."
        )
        if feature == "wins_pctg":
            outcome = result_description()
            if outcome:
                game_context = f"The team {outcome} the selected game."
        elif feature == "gf_pctg":
            gf, ga = observed("goals_for"), observed("goals_against")
            if pd.notna(gf) and pd.notna(ga):
                game_context = f"The team scored {gf:.0f} goals and allowed {ga:.0f} in the selected game."
        elif feature == "ot_games_pct":
            ot = observed("went_to_ot")
            if pd.isna(ot):
                ot = observed("game_ot")
            if pd.notna(ot) and str(ot).strip().lower() in {"0", "1", "true", "false"}:
                game_context = (
                    "The selected game required overtime."
                    if str(ot).strip().lower() in {"1", "true"}
                    else "The selected game did not require overtime."
                )

    elif feature in {
        "home_goal_diff", "home_points", "home_gf", "home_ga", "home_rw",
        "home_otlosses", "road_losses", "road_wins", "road_points",
        "road_goal_diff", "road_ot_losses", "road_gf", "road_ga",
    }:
        season_context = (
            f"The team's {label} at this snapshot is "
            f"{format_number(feature_value, 0)}."
        )
        selected_venue = venue()
        relevant_venue = "home" if feature.startswith("home_") else "road"
        if selected_venue is not None and selected_venue != relevant_venue:
            game_context = (
                f"The selected game was played {selected_venue}; this feature "
                f"tracks the team's {relevant_venue} results."
            )
        elif selected_venue == relevant_venue:
            gf, ga = observed("goals_for"), observed("goals_against")
            outcome = result_description()
            if feature.endswith("goal_diff") and pd.notna(gf) and pd.notna(ga):
                game_context = f"In this {relevant_venue} game, the goal differential was {gf - ga:+.0f}."
            elif feature.endswith(("_gf", "_ga")):
                number = gf if feature.endswith("_gf") else ga
                if pd.notna(number):
                    game_context = f"The team {'scored' if feature.endswith('_gf') else 'allowed'} {number:.0f} goals in this {relevant_venue} game."
            elif feature.endswith(("wins", "losses")) and outcome:
                game_context = f"The team {outcome} this {relevant_venue} game."
            elif feature.endswith("points"):
                earned = observed("game_points_earned")
                if pd.notna(earned):
                    game_context = f"The team earned {earned:.0f} standings points in this {relevant_venue} game."

    elif feature in {"sos_future", "sos_past"}:
        qualifier = "remaining" if feature == "sos_future" else "completed"
        season_context = (
            f"The model's strength-of-schedule measure for {qualifier} games "
            f"is {format_number(feature_value, 3)} at this snapshot."
        )

    elif feature in {"urgency", "rush_balance_ratio"}:
        descriptions = {
            "urgency": "conference rank pressure adjusted for season progress",
            "rush_balance_ratio": "rush chances created per rush chance allowed",
        }
        season_context = (
            f"This model feature measures {descriptions[feature]}; "
            f"its snapshot value is {format_number(feature_value, 3)}."
        )

    elif feature in {"so_wins", "so_losses", "ot_wins"}:
        season_context = (
            f"The team's season-to-date {label} is "
            f"{format_number(feature_value, 0)}."
        )
        # A win/loss alone does not establish whether the game was a SO/OT result.

    elif feature == "sc_net":
        season_context = (
            f"The team's scoring-chance differential is "
            f"{format_number(feature_value, 2, signed=True)} chances per game "
            f"(chances created minus chances allowed)."
        )
        scf, sca = snapshot_value("SCF"), snapshot_value("SCA")
        if pd.notna(scf) and pd.notna(sca):
            game_context = (
                f"In the selected game, the team recorded {scf:.0f} scoring "
                f"chances for and {sca:.0f} against."
            )

    elif feature == "special_teams_net":
        season_context = (
            f"The combined power-play and penalty-kill net per game is "
            f"{format_number(feature_value, 3, signed=True)}."
        )
        pp_net, pk_net = snapshot_value("game_pp_net"), snapshot_value("game_pk_net")
        if pd.notna(pp_net) and pd.notna(pk_net):
            game_context = (
                f"The selected game's power-play net was {pp_net:+.0f} "
                f"and penalty-kill net was {pk_net:+.0f}."
            )

    elif feature == "pk_load_x_pk_pct":
        season_context = (
            f"The model's penalty-kill workload and efficiency interaction "
            f"is {format_number(feature_value, 3)}. It combines the team's "
            f"penalty-kill workload with its penalty-kill percentage."
        )
        times_sh = snapshot_value("times_sh")
        if pd.notna(times_sh):
            game_context = f"The team was shorthanded {times_sh:.0f} times in the selected game."

    elif feature == "wc_pressure_x_performance":
        season_context = (
            "The model combines the team's points-percentage margin relative "
            "to the second wild-card position with the fraction of the season "
            f"completed; its interaction value is {format_number(feature_value, 3)}."
        )
        margin = snapshot_value("wc_points_pct_margin")
        if pd.notna(margin):
            game_context = (
                f"At this snapshot, the points-percentage margin to the "
                f"wild-card cutline was {margin:+.3f}."
            )

    elif feature in {"special_teams_x_season", "rush_net_x_season", "corsi_x_special_teams", "pdo_x_corsi", "rush_x_special_teams", "urgency_x_points"}:
        definitions = {
            "special_teams_x_season": "special-teams net per game and season progress",
            "rush_net_x_season": "rush differential per game and season progress",
            "corsi_x_special_teams": "Corsi differential per game and special-teams net per game",
            "pdo_x_corsi": "PDO deviation from 1.0 and Corsi differential per game",
            "rush_x_special_teams": "rush differential per game and special-teams net per game",
            "urgency_x_points": "standings urgency and points percentage",
        }
        season_context = (
            f"This model interaction combines {definitions[feature]}. "
            f"Its value at this snapshot is {format_number(feature_value, 3)}."
        )
        game_context = (
            "This is a season-level interaction, not a direct measure of "
            "the selected game's performance."
        )

    elif feature == "wc_teams_within_2pts":
        season_context = (
            f"There are {format_number(feature_value, 0)} wild-card-eligible "
            "teams within two standings points of the conference cutline "
            "at this snapshot."
        )
        game_context = "This measures the wild-card race at the snapshot, not one game."

    # ====================================================
    # 2. CUMULATIVE FEATURES
    # ====================================================

    elif feature.startswith("c_"):

        # --------------------------------------------
        # Season context
        # --------------------------------------------

        if feature in ["c_sv_pct", "c_shot_pct"]:

            season_context = (
                f"The team's season-to-date {label} "
                f"is {format_hockey_pct(feature_value)}."
            )

        elif feature == "c_pdo":

            season_context = (
                f"The team's season-to-date PDO "
                f"is {format_number(feature_value, 3)}."
            )

        else:

            season_context = (
                f"The team's season-to-date {label} "
                f"is {format_number(feature_value, 0, signed=True)}."
            )

        # --------------------------------------------
        # Game context — cumulative counts
        # --------------------------------------------

        if pd.notna(game_contribution):

            if feature == "c_pp_net_g":

                pp_gf = stat(stats, "pp_gf")
                sh_ga = stat(stats, "sh_ga")

                if pd.notna(pp_gf) and pd.notna(sh_ga):

                    game_context = (
                        f"In this game, the team scored "
                        f"{pp_gf:.0f} power-play goals and "
                        f"allowed {sh_ga:.0f} shorthanded goals, "
                        f"finishing with a power-play net "
                        f"of {game_contribution:+.0f}."
                    )

                else:

                    game_context = (
                        f"The team's power-play net "
                        f"for this game was "
                        f"{game_contribution:+.0f}."
                    )

            elif feature == "c_corsi_diff":

                corsi_for = stat(stats, "corsi_for")
                corsi_against = stat(stats, "corsi_against")

                if pd.notna(corsi_for) and pd.notna(corsi_against):

                    game_context = (
                        f"The team recorded {corsi_for:.0f} "
                        f"Corsi attempts for and "
                        f"{corsi_against:.0f} against, "
                        f"finishing with a differential "
                        f"of {game_contribution:+.0f}."
                    )

            elif feature == "c_shots_diff":

                shots_for = stat(stats, "shots_for")
                shots_against = stat(stats, "shots_against")

                if pd.notna(shots_for) and pd.notna(shots_against):

                    game_context = (
                        f"The team put {shots_for:.0f} shots "
                        f"on goal and allowed "
                        f"{shots_against:.0f}, finishing "
                        f"with a shot differential of "
                        f"{game_contribution:+.0f}."
                    )

            if not game_context:

                game_context = (
                    f"In the selected game, the team recorded "
                    f"{format_number(game_contribution, 0, signed=True)} "
                    f"for this statistic."
                )

        # --------------------------------------------
        # Game context — cumulative rates
        # --------------------------------------------

        elif pd.notna(game_vs_cumulative):

            game_value = (
                feature_value + game_vs_cumulative
            )

            if feature == "c_sv_pct":

                shots_against = stat(stats, "shots_against")

                game_context = (
                    f"The team recorded a "
                    f"{format_hockey_pct(game_value)} "
                    f"save percentage"
                )

                if pd.notna(shots_against):
                    game_context += (
                        f" on {shots_against:.0f} shots against"
                    )

                game_context += (
                    f", compared with its cumulative "
                    f"rate of {format_hockey_pct(feature_value)}."
                )

                if game_vs_cumulative < 0:

                    game_context += (
                        f" That was "
                        f"{abs(game_vs_cumulative):.3f} "
                        f"lower than its season-to-date rate."
                    )

                elif game_vs_cumulative > 0:

                    game_context += (
                        f" That was "
                        f"{game_vs_cumulative:.3f} "
                        f"higher than its season-to-date rate."
                    )

            elif feature == "c_shot_pct":

                game_context = (
                    f"The team recorded a shooting percentage "
                    f"of {format_hockey_pct(game_value)} "
                    f"in this game, compared with its "
                    f"cumulative rate of "
                    f"{format_hockey_pct(feature_value)}."
                )

            else:

                game_context = (
                    f"The game's {label} was "
                    f"{format_number(game_value, 3)}, "
                    f"compared with the cumulative value "
                    f"of {format_number(feature_value, 3)}."
                )

    # ====================================================
    # 3. PER-GAME AVERAGES
    # ====================================================

    elif feature in PER_GAME_CONTEXT:

        game_col, game_label = PER_GAME_CONTEXT[feature]

        game_value = stat(stats, game_col)
        difference = row.get("game_vs_average", np.nan)

        season_context = (
            f"The team averages "
            f"{format_number(feature_value, 2)} "
            f"{game_label} per game."
        )

        if pd.notna(game_value):

            game_context = (
                f"In this game, the team recorded "
                f"{format_number(game_value, 0)} "
                f"{game_label}."
            )

            if pd.notna(difference):

                comparison = describe_difference(
                    difference,
                    "per game"
                )

                game_context += (
                    f" That was {comparison}."
                )

    # ====================================================
    # 4. GOAL DIFFERENTIAL
    # ====================================================

    elif feature == "goal_diff":

        gf = stat(stats, "goals_for")
        ga = stat(stats, "goals_against")

        season_context = (
            f"The team's season goal differential "
            f"stands at {format_number(feature_value, 0, signed=True)}."
        )

        if pd.notna(gf) and pd.notna(ga):

            game_diff = gf - ga

            game_context = (
                f"The team scored {gf:.0f} goals "
                f"and allowed {ga:.0f}, finishing "
                f"the game with a goal differential "
                f"of {game_diff:+.0f}."
            )

    # ====================================================
    # 5. POINTS PERCENTAGE
    # ====================================================

    elif feature == "points_pct":

        points = stat(stats, "points")
        game_win = stat(stats, "game_win")

        season_context = (
            f"The team's points percentage is "
            f"{format_hockey_pct(feature_value)}."
        )

        if pd.notna(points):

            season_context += (
                f" It has earned {points:.0f} "
                f"standings points."
            )

        if pd.notna(game_win):

            if game_win == 1:

                game_context = (
                    "The team won the selected game, "
                    "adding to its standings position."
                )

            else:

                game_context = (
                    "The team did not win the selected game."
                )

    # ====================================================
    # 6. WIN PERCENTAGE BEFORE GAME
    # ====================================================

    elif feature == "win_pct_before":

        wins = stat(stats, "wins_before")
        games = stat(stats, "games_before")

        season_context = (
            f"Entering the game, the team's "
            f"winning percentage was "
            f"{feature_value:.1%}." if pd.notna(feature_value) else "The pregame winning percentage is unavailable."
        )

        if pd.notna(wins) and pd.notna(games):

            game_context = (
                f"The team entered with "
                f"{wins:.0f} wins in "
                f"{games:.0f} games."
            )

    # ====================================================
    # 7. RANKINGS
    # ====================================================

    elif feature in RANK_CONTEXT:

        location = RANK_CONTEXT[feature]

        season_context = (
            f"The team ranks "
            f"{format_rank(feature_value)} "
            f"{location}."
        )

        game_context = (
            "This ranking reflects the team's "
            "standings position at the selected snapshot."
        )

    # ====================================================
    # 8. POWER PLAY
    # ====================================================

    elif feature in [
        "pp_net_gpg",
        "pp_net_pct",
        "pp_gpg",
        "pp_opp_pg",
        "sh_ga_pg"
    ]:

        pp_gf = stat(stats, "pp_gf")
        sh_ga = stat(stats, "sh_ga")
        pp_opp = stat(stats, "pp_opp")
        game_pp_net = stat(stats, "game_pp_net")

        season_context = (
            f"The team's season-to-date {label} is "
            f"{format_number(feature_value, 3)}."
        )

        parts = []

        if pd.notna(pp_gf):
            parts.append(
                f"{pp_gf:.0f} power-play goals"
            )

        if pd.notna(pp_opp):
            parts.append(
                f"{pp_opp:.0f} opportunities"
            )

        if pd.notna(sh_ga):
            parts.append(
                f"{sh_ga:.0f} shorthanded goals allowed"
            )

        if parts:

            game_context = (
                "In this game, the team recorded "
                + ", ".join(parts)
                + "."
            )

        if pd.notna(game_pp_net):

            game_context += (
                f" Its power-play net was "
                f"{game_pp_net:+.0f}."
            )

    # ====================================================
    # 9. PENALTY KILL
    # ====================================================

    elif feature in [
        "pk_net_gpg",
        "pk_net_pct",
        "pp_ga_pg",
        "sh_gf_pg",
        "times_sh_pg"
    ]:

        pp_ga = stat(stats, "pp_ga")
        sh_gf = stat(stats, "sh_gf")
        times_sh = stat(stats, "times_sh")
        game_pk_net = stat(stats, "game_pk_net")

        season_context = (
            f"The team's season-to-date {label} is "
            f"{format_number(feature_value, 3)}."
        )

        parts = []

        if pd.notna(pp_ga):
            parts.append(
                f"{pp_ga:.0f} power-play goals allowed"
            )

        if pd.notna(times_sh):
            parts.append(
                f"{times_sh:.0f} times shorthanded"
            )

        if pd.notna(sh_gf):
            parts.append(
                f"{sh_gf:.0f} shorthanded goals scored"
            )

        if parts:

            game_context = (
                "In this game, the team recorded "
                + ", ".join(parts)
                + "."
            )

        if pd.notna(game_pk_net):

            game_context += (
                f" Its penalty-kill net was "
                f"{game_pk_net:+.0f}."
            )

    # ====================================================
    # 10. GENERAL FEATURE FALLBACK
    # ====================================================

    else:

        season_context = (
            f"At this snapshot, {label} is "
            f"{format_number(feature_value, 3)}."
        )

        if stats:

            details = []

            for key, value in stats.items():

                if (
                    isinstance(value, (int, float, np.number))
                    and pd.notna(value)
                ):

                    details.append(
                        f"{key.replace('_', ' ')}: "
                        f"{format_number(value, 2)}"
                    )

            if details:

                game_context = (
                    "Supporting statistics include "
                    + ", ".join(details)
                    + "."
                )

    # ====================================================
    # 11. COMBINE EXPLANATIONS
    # ====================================================

    game_explanation = " ".join(
        part for part in [
            season_context,
            game_context
        ]
        if part
    )

    return pd.Series({

        "model_explanation": model_explanation,

        "season_context": season_context,

        "game_context": game_context,

        "game_explanation": game_explanation

    })


OUTPUT_COLUMNS = [
    "season_id", "snapshot_date", "game_date", "game_id", "team_id", "tri_code",
    "model_version", "playoff_prob", "feature", "direction", "rank",
    "shap_value", "feature_value", "game_contribution", "game_vs_cumulative",
    "game_vs_average", "game_stats", "model_explanation", "season_context",
    "game_context", "game_explanation", "needs_review", "explanation_version",
    "source_hash", "generated_at",
]


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or (not isinstance(value, (dict, list, tuple)) and pd.isna(value)):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _hash_snapshot(group):
    # Include game stats and model inputs, not only SHAP rankings.
    omit = {"generated_at", "source_hash", "needs_review", "model_explanation",
            "season_context", "game_context", "game_explanation"}
    columns = sorted(c for c in group.columns if c not in omit)
    records = [_json_safe(row) for row in
               group.sort_values(["direction", "rank"])[columns].to_dict("records")]
    payload = json.dumps({"version": EXPLANATION_VERSION, "rows": records},
                         default=str, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()



def run_game_shap_explanations(
    *,
    con: duckdb.DuckDBPyConnection,
    season_id: int,
    refresh_existing: bool = False,
) -> None:
    """
    Generate and persist game-level SHAP explanations.

    Processes the top 5 helpful and harmful features for each
    team/snapshot. Only new or changed snapshots are updated,
    unless refresh_existing=True.
    """

    # ====================================================
    # 1. INITIALIZE MODEL VERSION AND DESTINATION TABLE
    # ====================================================

    version = _version()

    # Create the destination table if it does not exist.
    _ensure_table(con)

    # ====================================================
    # 2. LOAD SHAP RANKINGS AND GAME STATISTICS
    # ====================================================

    source = _load(
        con=con,
        season_id=season_id,
        version=version,
    )

    if source.empty:
        print(
            f"No SHAP rankings for season {season_id}, "
            f"model {version}"
        )
        return

    # ====================================================
    # 3. DERIVE GAME METRICS AND FEATURE CONTEXT
    # ====================================================

    source = _derive(source)

    context = source.apply(
        _context,
        axis=1,
    )

    source = source.join(context)

    source["game_stats"] = source["game_stats"].map(
        lambda stats: json.dumps(stats, default=str)
    )

    # ====================================================
    # 4. LOAD EXISTING EXPLANATION SNAPSHOTS
    # ====================================================

    existing = con.execute(
        f"""
        SELECT
            season_id,
            snapshot_date,
            team_id,
            model_version,
            MIN(source_hash) AS source_hash,
            COUNT(*) AS row_count
        FROM {DESTINATION}
        WHERE season_id = ?
          AND model_version = ?
        GROUP BY
            season_id,
            snapshot_date,
            team_id,
            model_version
        """,
        [season_id, version],
    ).df()

    # Build a lookup of previously processed snapshots.
    old = {
        (
            int(row.season_id),
            pd.Timestamp(row.snapshot_date).normalize(),
            int(row.team_id),
            str(row.model_version),
        ): (
            row.source_hash,
            int(row.row_count),
        )
        for row in existing.itertuples(index=False)
    }

    # ====================================================
    # 5. IDENTIFY NEW OR CHANGED SNAPSHOTS
    # ====================================================

    pending = []

    for key, group in source.groupby(
        KEYS,
        dropna=False,
        sort=False,
    ):

        # Each snapshot should contain:
        # 5 helpful + 5 harmful SHAP features.
        expected_rows = 2 * TOP_N

        if len(group) != expected_rows:
            raise ValueError(
                f"Incomplete SHAP top-{TOP_N} rankings: "
                f"{key} ({len(group)} rows)"
            )

        group = group.copy()

        # Convert JSON strings to dictionaries for hashing.
        group["game_stats"] = group["game_stats"].map(
            json.loads
        )

        digest = _hash_snapshot(group)

        # Restore JSON strings for database insertion.
        group["game_stats"] = group["game_stats"].map(
            json.dumps
        )

        normalized_key = (
            int(key[0]),
            pd.Timestamp(key[1]).normalize(),
            int(key[2]),
            str(key[3]),
        )

        # Skip unchanged snapshots unless a refresh is requested.
        if (
            not refresh_existing
            and old.get(normalized_key) == (digest, len(group))
        ):
            continue

        group["source_hash"] = digest

        pending.append(group)

    # ====================================================
    # 6. EXIT IF EVERYTHING IS CURRENT
    # ====================================================

    if not pending:
        print(
            f"Game SHAP explanations up to date: "
            f"{season_id} {version}"
        )
        return

    # ====================================================
    # 7. GENERATE NATURAL-LANGUAGE EXPLANATIONS
    # ====================================================

    result = pd.concat(
        pending,
        ignore_index=True,
    )

    explanation_input = result.copy()

    explanation_input["game_stats"] = (
        explanation_input["game_stats"].map(json.loads)
    )

    explanations = explanation_input.apply(
        generate_game_explanation,
        axis=1,
    )

    result = pd.concat(
        [result, explanations],
        axis=1,
    )

    # ====================================================
    # 8. FLAG EXPLANATIONS NEEDING REVIEW
    # ====================================================

    # Review missing inputs and genuinely incomplete or generic text.
    # Do not flag a feature solely because metadata lacks a mapping.
    # Season-level interactions can legitimately lack game-specific context.
    season_only = {
        "special_teams_x_season", "rush_net_x_season",
        "corsi_x_special_teams", "pdo_x_corsi",
        "rush_x_special_teams", "urgency_x_points",
        "wc_pressure_x_performance", "wc_teams_within_2pts",
        "sos_future", "sos_past", "projected_points",
        "urgency", "rush_balance_ratio",
    }
    missing_game = result["game_context"].fillna("").str.strip().eq("")
    result["needs_review"] = (
        result["feature_value"].isna()
        | result["season_context"].fillna("").str.strip().eq("")
        | (missing_game & ~result["feature"].isin(season_only))
        | result["season_context"].fillna("").str.contains(
            "feature has a value", regex=False
        )
    )

    # ====================================================
    # 9. ADD EXPLANATION METADATA
    # ====================================================

    result["explanation_version"] = EXPLANATION_VERSION

    result["generated_at"] = datetime.now(timezone.utc)

    # Ensure game statistics contain valid JSON.
    result["game_stats"] = result["game_stats"].map(
        lambda stats: json.dumps(
            json.loads(stats),
            allow_nan=False,
        )
    )

    # ====================================================
    # 10. VALIDATE OUTPUT COLUMNS
    # ====================================================

    for col in OUTPUT_COLUMNS:

        if col not in result.columns:
            raise ValueError(
                f"Missing generated column: {col}"
            )

    output = result[OUTPUT_COLUMNS].copy()

    # ====================================================
    # 11. REGISTER TEMPORARY DATAFRAME
    # ====================================================

    con.register(
        "_game_shap_updates",
        output,
    )

    # ====================================================
    # 12. ATOMICALLY REPLACE CHANGED SNAPSHOTS
    # ====================================================

    try:

        con.execute("BEGIN TRANSACTION")

        # Remove previous records for changed snapshots.
        con.execute(
            f"""
            DELETE FROM {DESTINATION} AS d
            USING (
                SELECT DISTINCT
                    season_id,
                    snapshot_date,
                    team_id,
                    model_version
                FROM _game_shap_updates
            ) AS u
            WHERE d.season_id = u.season_id
              AND d.snapshot_date = u.snapshot_date
              AND d.team_id = u.team_id
              AND d.model_version = u.model_version
            """
        )

        # Insert updated explanations.
        columns = ", ".join(
            map(_ident, OUTPUT_COLUMNS)
        )

        con.execute(
            f"""
            INSERT INTO {DESTINATION} ({columns})
            SELECT {columns}
            FROM _game_shap_updates
            """
        )

        con.execute("COMMIT")

    except Exception:

        con.execute("ROLLBACK")
        raise

    finally:

        con.unregister("_game_shap_updates")

    # ====================================================
    # 13. REPORT RESULTS
    # ====================================================

    print(
        f"Updated {len(pending)} snapshots / "
        f"{len(output)} game SHAP explanations "
        f"({version})"
    )


# Alias for consistency with update_shap_summary naming.
update_game_shap_explanations = run_game_shap_explanations
