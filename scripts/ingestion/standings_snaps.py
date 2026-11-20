
# ============================================================
# standings_snaps.py
# ============================================================

"""
Production NHL daily standings snapshots.

Canonical source:
    Historical standings snapshots backfill.

Production features:
    - Recover missing snapshot dates
    - Recover incomplete snapshots
    - Refresh the latest 3 calendar dates
    - Preserve historical field names and calculations
    - Carry forward valid standings when the API returns
      a legitimately empty standings response
    - Never treat an HTTP failure as an empty standings day
    - Maintain first-known clinch/elimination history
    - Atomically update standings and clinch history
"""

import random
import time

import duckdb
import numpy as np
import pandas as pd
import requests


BASE_URL = "https://api-web.nhle.com/v1/standings"

REFRESH_DAYS = 3
SLEEP_SECONDS = 0.15


# ============================================================
# Canonical Historical Schema
# ============================================================

SNAP_SCHEMA = {
    "game_date": "DATE",
    "season_id": "Int64",
    "tri_code": "string",
    "team_name": "string",
    "games_played": "Int64",
    "road_gp": "Int64",
    "division": "string",
    "division_rank": "Int64",
    "division_home_rank": "Int64",
    "division_road_rank": "Int64",
    "division_last_10_rank": "Int64",
    "conference": "string",
    "conference_rank": "Int64",
    "conference_home_rank": "Int64",
    "conference_road_rank": "Int64",
    "conference_last_10_rank": "Int64",
    "league_rank": "Int64",
    "league_home_rank": "Int64",
    "league_road_rank": "Int64",
    "league_last_10_rank": "Int64",
    "wildcard_rank": "Int64",
    "wins": "Int64",
    "wins_pctg": "float64",
    "rw": "Int64",
    "losses": "Int64",
    "ot_losses": "Int64",
    "rotw": "Int64",
    "rotw_pctg": "float64",
    "road_losses": "Int64",
    "road_ot_losses": "Int64",
    "road_rotw": "Int64",
    "road_rw": "Int64",
    "points": "Int64",
    "points_pct": "float64",
    "home_points": "Int64",
    "road_points": "Int64",
    "goals_for": "Int64",
    "gf_pctg": "float64",
    "home_gf": "Int64",
    "road_gf": "Int64",
    "goals_against": "Int64",
    "home_ga": "Int64",
    "road_ga": "Int64",
    "goal_diff": "Int64",
    "goal_diff_pctg": "float64",
    "home_goal_diff": "Int64",
    "road_goal_diff": "Int64",
    "streak": "string",
    "streak_count": "float64",
    "last_10_gp": "Int64",
    "last_10_wins": "Int64",
    "last_10_goal_diff": "Int64",
    "last_10_ga": "Int64",
    "last_10_gf": "Int64",
    "last_10_losses": "Int64",
    "last_10_otl": "Int64",
    "last_10_points": "Int64",
    "last_10_rotw": "Int64",
    "last_10_rw": "Int64",
    "home_gp": "Int64",
    "home_wins": "Int64",
    "home_losses": "Int64",
    "home_otlosses": "Int64",
    "home_rotw": "Int64",
    "home_rw": "Int64",
    "road_wins": "Int64",
    "so_losses": "Int64",
    "so_wins": "Int64",
    "in_playoff_spot": "Int64",
    "games_remaining": "Int64",
    "ppg": "float64",
    "projected_points": "Int64",
    "team_id": "Int64",
}


CLINCH_SCHEMA = {
    "clinch_indicator": "boolean",
    "tri_code": "string",
    "clinch_type": "string",
    "clinch_date": "DATE",
    "season_id": "Int64",
}


# ============================================================
# Season Length
# ============================================================

def get_regular_season_games(season_id: int) -> int:

    if season_id == 20202021:
        return 56

    if season_id >= 20262027:
        return 84

    return 82


# ============================================================
# Schema Enforcement
# ============================================================

def enforce_schema(
    df: pd.DataFrame,
    schema: dict,
) -> pd.DataFrame:

    df = df.copy()

    for col in schema:
        if col not in df.columns:
            df[col] = pd.NA

    for col, dtype in schema.items():

        if dtype == "string":
            df[col] = df[col].astype("string")

        elif dtype == "boolean":
            df[col] = df[col].astype("boolean")

        elif dtype == "float64":
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            ).astype("float64")

        elif dtype == "DATE":
            df[col] = pd.to_datetime(
                df[col],
                errors="coerce",
            ).dt.date

        else:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            ).astype("Int64")

    return df[list(schema.keys())]


def enforce_snap_schema(df):
    return enforce_schema(df, SNAP_SCHEMA)


def enforce_clinch_schema(df):
    return enforce_schema(df, CLINCH_SCHEMA)


# ============================================================
# Safe NHL Request
# ============================================================

def safe_request(url: str, max_retries: int = 5):
    """Retry only transient NHL API failures; never silently skip a date."""
    retryable_statuses = {429, 500, 502, 503, 504}

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=(10, 60))
            response.raise_for_status()
            return response

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            error = exc

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in retryable_statuses:
                raise RuntimeError(f"Non-retryable NHL API HTTP {status}: {url}") from exc
            error = exc

        if attempt == max_retries:
            raise RuntimeError(
                f"NHL API request failed after {max_retries} attempts: {url}"
            ) from error

        retry_after = None
        if isinstance(error, requests.exceptions.HTTPError) and error.response is not None:
            header = error.response.headers.get("Retry-After")
            if header:
                try:
                    retry_after = max(0.0, float(header))
                except ValueError:
                    pass

        delay = max(min(2 ** (attempt - 1), 30) + random.uniform(0, 0.5), retry_after or 0)
        print(f"⚠️ NHL API request failed ({attempt}/{max_retries}): {error}")
        print(f"🔄 Retrying in {delay:.1f}s: {url}")
        time.sleep(delay)

    raise RuntimeError(f"NHL API request failed: {url}")


# ============================================================
# Season Dates
# ============================================================

def load_season_dates(con, season_id):

    row = con.execute(
        """
        SELECT
            regular_start,
            regular_end
        FROM important_dates
        WHERE season_id = ?
        """,
        [season_id],
    ).fetchone()

    if row is None or row[0] is None or row[1] is None:
        raise RuntimeError(
            f"Missing regular-season dates for {season_id}"
        )

    return (
        pd.to_datetime(row[0]).normalize(),
        pd.to_datetime(row[1]).normalize(),
    )


# ============================================================
# Existing Snapshot Coverage
# ============================================================

def load_snapshot_counts(con, season_id, target_end):

    return con.execute(
        """
        SELECT
            game_date,
            COUNT(*) AS row_count,
            COUNT(DISTINCT tri_code) AS team_count
        FROM all_standings_daily
        WHERE season_id = ?
          AND game_date BETWEEN (
              SELECT regular_start FROM important_dates WHERE season_id = ?
          ) AND ?
        GROUP BY game_date
        """,
        [season_id, season_id, target_end.date()],
    ).df()


def determine_dates_to_process(
    con,
    season_id,
    regular_start,
    target_end,
):

    expected = pd.date_range(
        regular_start,
        target_end,
        freq="D",
    )

    counts = load_snapshot_counts(
        con,
        season_id,
        target_end,
    )

    coverage = {}

    for row in counts.itertuples():
        coverage[pd.to_datetime(row.game_date).date()] = (
            int(row.row_count),
            int(row.team_count),
        )

    # Use league membership rather than inferring a target from possibly
    # incomplete stored snapshots. Vegas joined in 2017-18 and Seattle
    # in 2021-22. Earlier historical seasons may need separate handling.
    expected_teams = 32 if season_id >= 20212022 else (31 if season_id >= 20172018 else 30)

    missing_dates = []
    incomplete_dates = []

    for dt in expected:

        day = dt.date()

        if day not in coverage:
            missing_dates.append(dt)
            continue

        rows, teams = coverage[day]

        if (
            rows != teams
            or (
                expected_teams is not None
                and teams < expected_teams
            )
        ):
            incomplete_dates.append(dt)

    refresh_start = max(
        regular_start,
        target_end - pd.Timedelta(
            days=REFRESH_DAYS - 1
        ),
    )

    refresh_dates = list(
        pd.date_range(
            refresh_start,
            target_end,
            freq="D",
        )
    )

    dates = sorted(
        set(missing_dates)
        | set(incomplete_dates)
        | set(refresh_dates)
    )

    print(
        f"🔎 Missing dates: {len(missing_dates):,}"
    )

    print(
        f"🩹 Incomplete dates: {len(incomplete_dates):,}"
    )

    print(
        f"🔄 Refresh dates: {len(refresh_dates):,}"
    )

    return dates, expected_teams


# ============================================================
# Fetch NHL Standings
# ============================================================

def fetch_standings(snapshot_date):

    date_str = snapshot_date.strftime("%Y-%m-%d")

    response = safe_request(
        f"{BASE_URL}/{date_str}"
    )

    try:
        payload = response.json()
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Invalid standings JSON for {date_str}"
        ) from exc

    if (
        not isinstance(payload, dict)
        or "standings" not in payload
        or not isinstance(payload["standings"], list)
    ):
        raise RuntimeError(
            f"Invalid standings payload for {date_str}"
        )

    return payload["standings"]


# ============================================================
# Canonical Standings Mapping
# ============================================================

def normalize_standings(
    rows,
    snapshot_date,
    season_id,
):

    date_str = snapshot_date.strftime("%Y-%m-%d")

    season_games = get_regular_season_games(
        season_id
    )

    output = []
    clinch_rows = []

    for team in rows:

        abbrev = team.get("teamAbbrev") or {}
        name = team.get("teamName") or {}

        tri_code = (
            abbrev.get("default")
            if isinstance(abbrev, dict)
            else abbrev
        )

        team_name = (
            name.get("default")
            if isinstance(name, dict)
            else name
        )

        row = {
            "game_date": date_str,
            "season_id": season_id,
            "tri_code": tri_code,
            "team_name": team_name,

            "games_played": team.get("gamesPlayed", 0),
            "road_gp": team.get("roadGamesPlayed", 0),

            "division": team.get("divisionName"),
            "division_rank": team.get("divisionSequence", 0),
            "division_home_rank": team.get("divisionHomeSequence", 0),
            "division_road_rank": team.get("divisionRoadSequence", 0),
            "division_last_10_rank": team.get("divisionL10Sequence", 0),

            "conference": team.get("conferenceName"),
            "conference_rank": team.get("conferenceSequence", 0),
            "conference_home_rank": team.get("conferenceHomeSequence", 0),
            "conference_road_rank": team.get("conferenceRoadSequence", 0),
            "conference_last_10_rank": team.get("conferenceL10Sequence", 0),

            "league_rank": team.get("leagueSequence", 0),
            "league_home_rank": team.get("leagueHomeSequence", 0),
            "league_road_rank": team.get("leagueRoadSequence", 0),
            "league_last_10_rank": team.get("leagueL10Sequence", 0),

            "wildcard_rank": team.get("wildcardSequence", 0),

            "wins": team.get("wins", 0),
            "wins_pctg": team.get("winPctg", 0.0),
            "rw": team.get("regulationWins", 0),
            "losses": team.get("losses", 0),
            "ot_losses": team.get("otLosses", 0),
            "rotw": team.get("regulationPlusOtWins", 0),
            "rotw_pctg": team.get("regulationPlusOtWinPctg", 0.0),

            "road_losses": team.get("roadLosses", 0),
            "road_ot_losses": team.get("roadOtLosses", 0),
            "road_rotw": team.get("roadRegulationPlusOtWins", 0),
            "road_rw": team.get("roadRegulationWins", 0),

            "points": team.get("points", 0),
            "points_pct": team.get("pointPctg", 0.0),
            "home_points": team.get("homePoints", 0),
            "road_points": team.get("roadPoints", 0),

            "goals_for": team.get("goalFor", 0),
            "gf_pctg": team.get("goalsForPctg", 0.0),
            "home_gf": team.get("homeGoalsFor", 0),
            "road_gf": team.get("roadGoalsFor", 0),

            "goals_against": team.get("goalAgainst", 0),
            "home_ga": team.get("homeGoalsAgainst", 0),
            "road_ga": team.get("roadGoalsAgainst", 0),

            "goal_diff": team.get("goalDifferential", 0),
            "goal_diff_pctg": team.get("goalDifferentialPctg", 0.0),
            "home_goal_diff": team.get("homeGoalDifferential", 0),
            "road_goal_diff": team.get("roadGoalDifferential", 0),

            "streak": team.get("streakCode"),
            "streak_count": team.get("streakCount"),

            "last_10_gp": team.get("l10GamesPlayed", 0),
            "last_10_wins": team.get("l10Wins", 0),
            "last_10_goal_diff": team.get("l10GoalDifferential", 0),
            "last_10_ga": team.get("l10GoalsAgainst", 0),
            "last_10_gf": team.get("l10GoalsFor", 0),
            "last_10_losses": team.get("l10Losses", 0),
            "last_10_otl": team.get("l10OtLosses", 0),
            "last_10_points": team.get("l10Points", 0),
            "last_10_rotw": team.get("l10RegulationPlusOtWins", 0),
            "last_10_rw": team.get("l10RegulationWins", 0),

            "home_gp": team.get("homeGamesPlayed", 0),
            "home_wins": team.get("homeWins", 0),
            "home_losses": team.get("homeLosses", 0),
            "home_otlosses": team.get("homeOtLosses", 0),
            "home_rotw": team.get("homeRegulationPlusOtWins", 0),
            "home_rw": team.get("homeRegulationWins", 0),

            "road_wins": team.get("roadWins", 0),

            "so_losses": team.get("shootoutLosses", 0),
            "so_wins": team.get("shootoutWins", 0),

            "in_playoff_spot": (
                1
                if (
                    (
                        team.get("divisionRank") is not None
                        and team["divisionRank"] <= 3
                    )
                    or (
                        team.get("wildCardRank") is not None
                        and team["wildCardRank"] <= 2
                    )
                )
                else 0
            ),

            "team_id": team.get("teamId", 0),
        }

        output.append(row)

        indicator = team.get("clinchIndicator")

        if indicator in ("x", "e"):
            clinch_rows.append({
                "clinch_indicator": True,
                "tri_code": tri_code,
                "clinch_type": indicator,
                "clinch_date": snapshot_date.date(),
                "season_id": season_id,
            })

    df = pd.DataFrame(output)

    if df.empty:
        return enforce_snap_schema(df), clinch_rows

    gp = pd.to_numeric(
        df["games_played"],
        errors="coerce",
    ).fillna(0)

    points = pd.to_numeric(
        df["points"],
        errors="coerce",
    ).fillna(0)

    # Canonical historical calculations.
    df["games_remaining"] = (
        season_games - gp
    ).clip(lower=0).astype("Int64")

    df["ppg"] = np.where(
        gp > 0,
        points / gp.replace(0, np.nan),
        0.0,
    )

    df["projected_points"] = (
        df["ppg"] * season_games
    ).round().astype("Int64")

    # Canonical Utah identity handling.
    if season_id == 20242025:
        df.loc[
            df["tri_code"] == "UTA",
            "team_id",
        ] = 59

    elif season_id >= 20252026:
        df.loc[
            df["tri_code"] == "UTA",
            "team_id",
        ] = 68

    return enforce_snap_schema(df), clinch_rows


# ============================================================
# Load Previous Snapshot
# ============================================================

def load_previous_snapshot(
    con,
    season_id,
    snapshot_date,
):

    previous_date = con.execute(
        """
        SELECT MAX(game_date)
        FROM all_standings_daily
        WHERE season_id = ?
          AND game_date < ?
        """,
        [season_id, snapshot_date.date()],
    ).fetchone()[0]

    if previous_date is None:
        return pd.DataFrame()

    df = con.execute(
        """
        SELECT *
        FROM all_standings_daily
        WHERE season_id = ?
          AND game_date = ?
        ORDER BY tri_code
        """,
        [season_id, previous_date],
    ).df()

    if df.empty:
        return df

    df["game_date"] = snapshot_date.date()

    return enforce_snap_schema(df)


# ============================================================
# Snapshot Validation
# ============================================================

def validate_snapshot(
    df,
    snapshot_date,
    expected_teams=None,
):

    if df.empty:
        raise ValueError(
            f"Empty standings snapshot: {snapshot_date.date()}"
        )

    if df["tri_code"].isna().any():
        raise ValueError(
            f"Missing team abbreviations: {snapshot_date.date()}"
        )

    if df["team_id"].isna().any() or (df["team_id"] <= 0).any():
        raise ValueError(
            f"Missing team IDs: {snapshot_date.date()}"
        )

    if df["tri_code"].duplicated().any():
        raise ValueError(
            f"Duplicate teams: {snapshot_date.date()}"
        )

    if df["team_id"].duplicated().any():
        raise ValueError(
            f"Duplicate team IDs: {snapshot_date.date()}"
        )

    if (
        expected_teams is not None
        and len(df) != expected_teams
    ):
        raise ValueError(
            f"Incomplete standings snapshot "
            f"for {snapshot_date.date()}: "
            f"{len(df)} teams; expected exactly "
            f"{expected_teams}"
        )


# ============================================================
# Clinch History Normalization
# ============================================================

def prepare_clinch_history(records):

    if not records:
        return enforce_clinch_schema(
            pd.DataFrame()
        )

    df = enforce_clinch_schema(
        pd.DataFrame(records)
    )

    # Historical canonical behavior:
    # First x/e indicator per team and season.
    df = (
        df.sort_values(
            ["season_id", "tri_code", "clinch_date"]
        )
        .drop_duplicates(
            subset=["season_id", "tri_code"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# Main Runner
# ============================================================

def run_standings_snaps(
    *,
    con: duckdb.DuckDBPyConnection,
    season_id: int,
) -> None:

    print(
        f"▶ Running standings snapshots "
        f"for season {season_id}"
    )

    # ========================================================
    # 1. Determine season range
    # ========================================================

    regular_start, regular_end = load_season_dates(
        con,
        season_id,
    )

    today = pd.Timestamp.today().normalize()

    if today < regular_start:
        print("ℹ️ Season has not started.")
        return

    target_end = min(
        today,
        regular_end,
    )

    print(
        f"📅 Standings range: "
        f"{regular_start.date()} → {target_end.date()}"
    )

    print(
        f"🏒 Regular-season length: "
        f"{get_regular_season_games(season_id)} games"
    )

    # ========================================================
    # 2. Identify dates to process
    # ========================================================

    dates_to_process, expected_teams = (
        determine_dates_to_process(
            con,
            season_id,
            regular_start,
            target_end,
        )
    )

    if not dates_to_process:
        print("✅ Standings already in sync.")
        return

    print(
        f"📋 Processing {len(dates_to_process):,} "
        "snapshot date(s)"
    )

    # ========================================================
    # 3. Download and normalize snapshots
    # ========================================================

    snapshot_frames = []
    all_clinch_records = []

    # Keep snapshots from this run by date.
    # This avoids using a later snapshot to fill an
    # earlier historical gap.
    processed_snapshots = {}

    for snapshot_date in dates_to_process:

        date_str = snapshot_date.strftime("%Y-%m-%d")

        print(f"📊 Loading standings for {date_str}")

        # API failures raise an exception.
        # They must not trigger carry-forward.
        rows = fetch_standings(
            snapshot_date
        )

        if rows:

            snap_df, clinch_records = (
                normalize_standings(
                    rows,
                    snapshot_date,
                    season_id,
                )
            )
            # Populate team IDs from the canonical all_teams table
            team_ids = con.execute("""
                SELECT tri_code, team_id
                FROM all_teams
            """).df()

            # Preserve the existing season-specific Utah identity
            team_ids = team_ids.loc[team_ids["tri_code"] != "UTA"]

            snap_df = snap_df.merge(
                team_ids,
                on="tri_code",
                how="left",
                suffixes=("", "_lookup"),
                validate="many_to_one"
            )

            snap_df["team_id"] = snap_df["team_id_lookup"].combine_first(
                snap_df["team_id"].replace(0, pd.NA)
            )

            snap_df = snap_df.drop(columns="team_id_lookup")
            snap_df = enforce_snap_schema(snap_df)

                        # Diagnose missing team IDs before validation
            invalid_ids = snap_df.loc[
                snap_df["team_id"].isna() | (snap_df["team_id"] <= 0),
                ["tri_code", "team_name", "team_id"]
            ]

            if not invalid_ids.empty:
                print(f"⚠️ Invalid team IDs for {snapshot_date.date()}:")
                print(invalid_ids.to_string(index=False))


            validate_snapshot(
                snap_df,
                snapshot_date,
                expected_teams,
            )

            all_clinch_records.extend(
                clinch_records
            )

        else:

            print(
                f"↪ Empty NHL standings response "
                f"for {date_str}; using prior snapshot"
            )

            # Find the most recent snapshot from this run
            # that precedes the current date.
            earlier_dates = [
                day
                for day in processed_snapshots
                if day < snapshot_date.date()
            ]

            # Only carry forward standings if no regular-season games
            # finished between the source snapshot and target date.
            # A complete 32-team snapshot can still be factually stale.
            if earlier_dates:
                prior_day = max(earlier_dates)
                snap_df = processed_snapshots[prior_day].copy()
            else:
                prior_day = con.execute(
                    """SELECT MAX(game_date) FROM all_standings_daily
                       WHERE season_id = ? AND game_date < ?""",
                    [season_id, snapshot_date.date()],
                ).fetchone()[0]
                snap_df = load_previous_snapshot(con, season_id, snapshot_date)

            if prior_day is not None:
                prior_day = pd.to_datetime(prior_day).date()
                completed_between = con.execute(
                    """SELECT COUNT(DISTINCT game_id)
                       FROM all_game_summary
                       WHERE season_id = ? AND gameState = 'OFF'
                         AND game_date > ? AND game_date <= ?
                         AND SUBSTR(CAST(game_id AS VARCHAR), 5, 2) = '02'""",
                    [season_id, prior_day, snapshot_date.date()],
                ).fetchone()[0]
                if completed_between:
                    raise RuntimeError(
                        f"Cannot carry forward standings for {date_str}: "
                        f"{completed_between} completed game(s) since {prior_day}. "
                        "NHL standings API returned an empty response."
                    )

            if not snap_df.empty:
                snap_df["game_date"] = snapshot_date.date()
                snap_df = enforce_snap_schema(snap_df)

            if snap_df.empty:

                raise RuntimeError(
                    f"No prior standings snapshot available for {date_str}; "
                    "cannot safely carry forward an empty API response"
                )

            validate_snapshot(
                snap_df,
                snapshot_date,
                expected_teams,
            )

        processed_snapshots[
            snapshot_date.date()
        ] = snap_df.copy()

        snapshot_frames.append(
            snap_df
        )

        time.sleep(SLEEP_SECONDS)

    # ========================================================
    # 4. Combine updates
    # ========================================================

    if not snapshot_frames:
        raise RuntimeError("Standings dates were queued but no snapshots were produced")

    standings_updates = pd.concat(
        snapshot_frames,
        ignore_index=True,
    )

    standings_updates = (
        standings_updates
        .sort_values(
            ["game_date", "tri_code"]
        )
        .drop_duplicates(
            subset=[
                "season_id",
                "game_date",
                "tri_code",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    standings_updates = enforce_snap_schema(
        standings_updates
    )

    clinch_updates = prepare_clinch_history(
        all_clinch_records
    )

    # ========================================================
    # 5. Atomic DuckDB write
    # ========================================================

    con.register(
        "standings_updates",
        standings_updates,
    )

    try:

        con.register(
            "clinch_updates",
            clinch_updates,
        )

        try:

            print(
                f"🧪 Writing "
                f"{len(standings_updates):,} standings rows"
            )

            con.execute("BEGIN TRANSACTION")

            try:

                # ---------------------------------------------
                # Replace processed snapshot dates
                # ---------------------------------------------

                con.execute(
                    """
                    DELETE FROM all_standings_daily
                    WHERE season_id = ?
                      AND game_date IN (
                          SELECT DISTINCT game_date
                          FROM standings_updates
                      )
                    """,
                    [season_id],
                )

                con.execute(
                    """
                    INSERT INTO all_standings_daily BY NAME
                    SELECT *
                    FROM standings_updates
                    """
                )

                # ---------------------------------------------
                # Historical team ID normalization
                # ---------------------------------------------

                con.execute(
                    """
                    UPDATE all_standings_daily AS s
                    SET team_id = t.team_id
                    FROM all_teams AS t
                    WHERE s.season_id = ?
                      AND s.game_date IN (
                          SELECT DISTINCT game_date
                          FROM standings_updates
                      )
                      AND s.tri_code = t.tri_code
                      AND s.tri_code <> 'UTA'
                    """,
                    [season_id],
                )

                con.execute(
                    """
                    UPDATE all_standings_daily
                    SET team_id =
                        CASE
                            WHEN season_id = 20242025
                                THEN 59
                            WHEN season_id >= 20252026
                                THEN 68
                            ELSE team_id
                        END
                    WHERE season_id = ?
                      AND tri_code = 'UTA'
                      AND game_date IN (
                          SELECT DISTINCT game_date
                          FROM standings_updates
                      )
                    """,
                    [season_id],
                )

                # ---------------------------------------------
                # First-known clinch history
                # ---------------------------------------------

                if not clinch_updates.empty:

                    # Remove a later recorded first event
                    # when an earlier event is discovered.
                    con.execute(
                        """
                        DELETE FROM all_clinch_history AS h
                        USING clinch_updates AS c
                        WHERE h.season_id = c.season_id
                          AND h.tri_code = c.tri_code
                          AND h.clinch_date > c.clinch_date
                        """
                    )

                    # Insert only teams without an existing
                    # equal or earlier clinch event.
                    con.execute(
                        """
                        INSERT INTO all_clinch_history BY NAME
                        SELECT c.*
                        FROM clinch_updates AS c
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM all_clinch_history AS h
                            WHERE h.season_id = c.season_id
                              AND h.tri_code = c.tri_code
                              AND h.clinch_date <= c.clinch_date
                        )
                        """
                    )

                con.execute("COMMIT")

            except Exception:

                con.execute("ROLLBACK")
                raise

        finally:
            con.unregister(
                "clinch_updates"
            )

    finally:
        con.unregister(
            "standings_updates"
        )

    # ========================================================
    # 6. Post-write validation
    # ========================================================

    # Recent dates are deliberately reselected, so only
    # evaluate whether any expected calendar date is absent.
    existing = con.execute(
        """
        SELECT DISTINCT game_date
        FROM all_standings_daily
        WHERE season_id = ?
          AND game_date BETWEEN ? AND ?
        """,
        [
            season_id,
            regular_start.date(),
            target_end.date(),
        ],
    ).df()

    existing_dates = set(
        pd.to_datetime(
            existing["game_date"]
        ).dt.date
    )

    expected_dates = {
        day.date()
        for day in pd.date_range(
            regular_start,
            target_end,
            freq="D",
        )
    }

    missing_dates = sorted(
        expected_dates - existing_dates
    )

    if missing_dates:

        print(
            f"⚠️ {len(missing_dates):,} "
            "standings date(s) still missing"
        )

        print(missing_dates[:20])
        raise RuntimeError(
            f"Standings ingestion incomplete: {len(missing_dates)} snapshot date(s) missing"
        )

    else:
        print(
            "✅ All expected standings dates exist."
        )

    # Verify row count and unique team coverage for every date, not merely
    # the existence of a row for that date. This catches partial writes and
    # previously stored incomplete snapshots before downstream modeling.
    coverage = load_snapshot_counts(con, season_id, target_end)
    bad_coverage = coverage.loc[
        (coverage["row_count"] != expected_teams)
        | (coverage["team_count"] != expected_teams)
    ]
    if not bad_coverage.empty:
        examples = bad_coverage[["game_date", "row_count", "team_count"]].head(10).to_dict("records")
        raise RuntimeError(
            f"Standings ingestion incomplete: {len(bad_coverage)} date(s) have "
            f"incorrect team coverage (expected {expected_teams}): {examples}"
        )

    print(
        f"✅ Updated "
        f"{standings_updates['game_date'].nunique():,} "
        "standings date(s)"
    )

    print(
        f"📅 Updated range: "
        f"{standings_updates['game_date'].min()} "
        f"→ {standings_updates['game_date'].max()}"
    )

    print(
        f"🏁 Clinch candidates: "
        f"{len(clinch_updates):,}"
    )
