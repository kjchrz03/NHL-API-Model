# ============================================================
# game_summary.py
# ============================================================

"""
Live NHL game-summary ingestion.

Canonical calculation / normalization logic is shared with the
historical backfill implementation.

Responsibilities:
- Fetch regular-season NHL games from the schedule API.
- Discover games missing from all_game_summary.
- Refresh existing FUT / unresolved games.
- Catch up automatically if the pipeline was not run for
    several days.
- Replace stored rows with the latest API representation.
- Return only games that newly became completed ("OFF") so
    downstream game-dependent pipeline steps know what changed.
"""

# ============================================================
# Imports
# ============================================================

import time
import random
from datetime import date, datetime, timezone

import duckdb
import pandas as pd
import pytz
import requests


# ============================================================
# Config
# ============================================================

BASE_URL = "https://api-web.nhle.com/v1/schedule/"
GAMECENTER_URL = "https://api-web.nhle.com/v1/gamecenter/"

SLEEP_SECONDS = 0.15
MAX_API_ATTEMPTS = 4
RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}

# Always re-check a small overlap before the last stored date.
# This allows recent schedule/game-state changes to be refreshed.
REFRESH_OVERLAP_DAYS = 2


# ============================================================
# Game Summary Schema
# ============================================================

GS_SCHEMA = {
    "game_id": "Int64",
    "season_id": "Int64",
    "away_team": "Int64",
    "away_tri_code": "string",
    "home_team": "Int64",
    "home_tri_code": "string",
    "awayTeam.score": "Int64",
    "homeTeam.score": "Int64",
    "gameState": "string",
    "link": "string",
    "game_date": "DATE",
    "api_snapshot_date": "DATE",
    "api_fetched_at": "TIMESTAMPTZ",
    "pipeline_run_at": "TIMESTAMPTZ",  # Last time this game row was refreshed; see pipeline_run_log for all runs.
}


# ============================================================
# Schema Enforcement
# ============================================================

def enforce_gs_schema(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------------------------
    # Keep only known columns
    # --------------------------------------------------------

    df = df[
        [c for c in GS_SCHEMA if c in df.columns]
    ]

    # --------------------------------------------------------
    # Add any missing columns
    # --------------------------------------------------------

    for col in GS_SCHEMA:

        if col not in df.columns:
            df[col] = pd.NA

    # --------------------------------------------------------
    # Enforce dtypes
    # --------------------------------------------------------

    for col, dtype in GS_SCHEMA.items():

        if dtype == "string":

            df[col] = df[col].astype("string")

        elif dtype == "TIMESTAMPTZ":
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        elif dtype == "DATE":

            df[col] = (
                pd.to_datetime(
                    df[col],
                    errors="coerce",
                )
                .dt.date
            )

        else:

            df[col] = (
                pd.to_numeric(
                    df[col],
                    errors="coerce",
                )
                .astype("Int64")
            )

    return df[list(GS_SCHEMA.keys())]


# ============================================================
# Determine Fetch Window
# ============================================================

def determine_fetch_window(
    *,
    con: duckdb.DuckDBPyConnection,
    season_id: int,
    season_start: pd.Timestamp,
    season_end: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:

    """
    Determine the date range that needs to be checked against
    the NHL schedule API.

    The live process must:
        1. discover games that were never inserted,
        2. catch up after missed pipeline days,
        3. refresh recent stored games,
        4. remain within the regular-season boundaries.
    """

    season_start = pd.Timestamp(season_start).normalize()
    season_end = pd.Timestamp(season_end).normalize()

    today = pd.Timestamp(date.today())

    # Never fetch beyond the regular season.
    fetch_end = min(today, season_end)

    # --------------------------------------------------------
    # Season has not started yet
    # --------------------------------------------------------

    if fetch_end < season_start:

        print(
            f"ℹ️ Regular season has not started yet "
            f"({season_start.date()})."
        )

        return season_start, fetch_end

    # --------------------------------------------------------
    # Inspect stored season
    # --------------------------------------------------------

    status = con.execute(
        """
        SELECT
            MIN(game_date) AS first_game_date,
            MAX(game_date) AS last_game_date,
            MAX(
                CASE WHEN gameState = 'OFF' THEN game_date END
            ) AS last_completed_date,
            MIN(
                CASE WHEN gameState IS NULL OR gameState <> 'OFF'
                    THEN game_date END
            ) AS earliest_unresolved_date
        FROM all_game_summary
        WHERE season_id = ?
        """,
        [season_id],
    ).df()

    last_game_date = status["last_game_date"].iloc[0]
    last_completed_date = status["last_completed_date"].iloc[0]
    earliest_unresolved_date = status["earliest_unresolved_date"].iloc[0]

    # --------------------------------------------------------
    # Nothing stored yet
    #
    # Start from the beginning of the regular season.
    # --------------------------------------------------------

    if pd.isna(last_game_date):

        print("🌱 No stored games found for season.")

        return season_start, fetch_end

    last_game_date = pd.Timestamp(last_game_date).normalize()

    if not pd.isna(last_completed_date):
        last_completed_date = (
            pd.Timestamp(last_completed_date)
            .normalize()
        )

    # --------------------------------------------------------
    # Catch-up point
    #
    # Use the last completed date rather than simply the last
    # scheduled game in the table.
    #
    # This matters because all_game_summary may already contain
    # FUT games well into the future.
    # --------------------------------------------------------

    if last_completed_date is not None and not pd.isna(
        last_completed_date
    ):

        fetch_start = (
            last_completed_date
            - pd.Timedelta(days=REFRESH_OVERLAP_DAYS)
        )

    else:

        # Games may exist as FUT, but none have been completed.
        fetch_start = season_start

    # Revisit any older unresolved game whose scheduled date has passed.
    # A postponed or unfinalized game may predate the normal overlap window.
    if pd.notna(earliest_unresolved_date):
        unresolved_date = pd.Timestamp(earliest_unresolved_date).normalize()
        if unresolved_date <= fetch_end:
            fetch_start = min(fetch_start, unresolved_date)

    # Never move before the season.
    fetch_start = max(fetch_start, season_start)

    print(
        f"📅 API catch-up window: "
        f"{fetch_start.date()} → {fetch_end.date()}"
    )

    return fetch_start, fetch_end


# ============================================================
# NHL API Request with Bounded Retries
# ============================================================

def get_nhl_schedule(url: str) -> dict:
    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            response = requests.get(url, timeout=(10, 60))
            response.raise_for_status()
            return response.json()

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            error = exc
            retryable = True

        except requests.exceptions.HTTPError as exc:
            error = exc
            status = exc.response.status_code if exc.response is not None else None
            retryable = status in RETRYABLE_HTTP_STATUSES

        if not retryable or attempt == MAX_API_ATTEMPTS:
            print(f"❌ NHL API request failed after {attempt} attempt(s): {url}: {error}")
            raise error

        delay = min(5 * 2 ** (attempt - 1), 60) + random.uniform(0, 1)
        print(f"⚠️ NHL API request failed ({attempt}/{MAX_API_ATTEMPTS}): {error}")
        print(f"🔄 Retrying in {delay:.1f} seconds...")
        time.sleep(delay)


# ============================================================
# Fetch Games from NHL API
# ============================================================

def fetch_games(
    *,
    season_id: int,
    fetch_start: pd.Timestamp,
    fetch_end: pd.Timestamp,
    pipeline_run_at: datetime | None = None,
) -> pd.DataFrame:

    """
    Fetch NHL schedule data for every date in the requested
    range and normalize it to all_game_summary format.
    """

    if pipeline_run_at is None:
        pipeline_run_at = datetime.now(timezone.utc)

    if fetch_end < fetch_start:
        return pd.DataFrame(columns=GS_SCHEMA.keys())

    all_games = []
    seen_game_ids = set()

    print(
        f"🌐 Checking NHL schedule "
        f"{fetch_start.date()} → {fetch_end.date()}"
    )

    # ========================================================
    # Fetch each date
    # ========================================================

    for d in pd.date_range(
        fetch_start,
        fetch_end,
        freq="D",
    ):

        url = f"{BASE_URL}{d.strftime('%Y-%m-%d')}"

        data = get_nhl_schedule(url)

        # ------------------------------------------------
        # Extract games
        # ------------------------------------------------

        for week in data.get("gameWeek", []):

            for game in week.get("games", []):

                # ----------------------------------------
                # REGULAR SEASON ONLY
                # ----------------------------------------

                if game.get("gameType") != 2:
                    continue

                # Schedule responses overlap adjacent weeks/seasons.
                # Only accept games belonging to the requested season.
                game_season = game.get("season")
                if game_season is not None and str(game_season) != str(season_id):
                    continue

                game_id = game.get("id")

                if game_id is None:
                    continue

                # ----------------------------------------
                # Validate NHL regular-season game ID
                #
                # YYYY02#### = regular season
                # ----------------------------------------

                game_id_str = str(game_id)

                if (
                    len(game_id_str) < 6
                    or game_id_str[4:6] != "02"
                    or game_id_str[:4] != str(season_id)[:4]
                ):

                    raise RuntimeError(
                        "❌ NHL API returned gameType=2 "
                        f"with unexpected game ID: {game_id}"
                    )

                # ----------------------------------------
                # Schedule endpoint can repeat games
                # across neighboring gameWeek responses.
                # ----------------------------------------

                if game_id in seen_game_ids:
                    continue

                seen_game_ids.add(game_id)
                all_games.append(game)


        time.sleep(SLEEP_SECONDS)

    # ========================================================
    # No games
    # ========================================================

    if not all_games:

        print("ℹ️ No regular-season games returned.")

        return pd.DataFrame(
            columns=GS_SCHEMA.keys()
        )

    # ========================================================
    # Normalize API Data
    # ========================================================

    df = pd.json_normalize(all_games)

    # --------------------------------------------------------
    # Game ID
    # --------------------------------------------------------

    df["game_id"] = (
        pd.to_numeric(
            df["id"],
            errors="raise",
        )
        .astype("int64")
    )

    df = df.drop_duplicates(
        subset="game_id"
    )

    # --------------------------------------------------------
    # Team fields
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "awayTeam.id": "away_team",
            "awayTeam.abbrev": "away_tri_code",
            "homeTeam.id": "home_team",
            "homeTeam.abbrev": "home_tri_code",
        }
    )

    # --------------------------------------------------------
    # Season
    # --------------------------------------------------------

    df["season_id"] = season_id

    # --------------------------------------------------------
    # Play-by-play link
    # --------------------------------------------------------

    df["link"] = (
        GAMECENTER_URL
        + df["game_id"].astype(str)
        + "/play-by-play"
    )

    # --------------------------------------------------------
    # Canonical game date
    #
    # NHL supplies UTC start time.
    # Store the Eastern calendar date.
    # --------------------------------------------------------

    eastern = pytz.timezone(
        "America/New_York"
    )

    df["startTimeUTC"] = pd.to_datetime(
        df["startTimeUTC"],
        utc=True,
    )

    df["game_date"] = (
        df["startTimeUTC"]
        .dt.tz_convert(eastern)
        .dt.date
    )

    # --------------------------------------------------------
    # API snapshot and retrieval timestamps
    # --------------------------------------------------------

    fetched_at = datetime.now(timezone.utc)
    df["api_snapshot_date"] = fetched_at.date()
    df["api_fetched_at"] = fetched_at
    df["pipeline_run_at"] = pd.Timestamp(pipeline_run_at)

    # --------------------------------------------------------
    # Final table columns
    # --------------------------------------------------------

    df = df[
        [
            "game_id",
            "season_id",
            "away_team",
            "away_tri_code",
            "home_team",
            "home_tri_code",
            "awayTeam.score",
            "homeTeam.score",
            "gameState",
            "link",
            "game_date",
            "api_snapshot_date",
            "api_fetched_at",
            "pipeline_run_at",
        ]
    ].copy()

    df = enforce_gs_schema(df)

    df = (
        df
        .sort_values("game_id")
        .reset_index(drop=True)
    )

    print(
        f"📥 Retrieved "
        f"{len(df):,} regular-season games"
    )

    return df


# ============================================================
# Load Existing Stored Games
# ============================================================

def load_existing_games(
    *,
    con: duckdb.DuckDBPyConnection,
    season_id: int,
    game_ids: list[int],
) -> pd.DataFrame:

    """
    Load the existing table rows corresponding to the games
    returned by the API.
    """

    if not game_ids:
        return pd.DataFrame(
            columns=GS_SCHEMA.keys()
        )

    ids_df = pd.DataFrame(
        {"game_id": game_ids}
    )

    con.register(
        "api_game_ids",
        ids_df,
    )

    try:

        existing = con.execute(
            """
            SELECT gs.*
            FROM all_game_summary gs
            INNER JOIN api_game_ids ids
                ON gs.game_id = ids.game_id
            WHERE gs.season_id = ?
            """,
            [season_id],
        ).df()

    finally:

        con.unregister("api_game_ids")

    if existing.empty:
        return existing

    return enforce_gs_schema(existing)


# ============================================================
# Determine Newly Completed or Corrected Games
# ============================================================


def determine_changed_games(
    *,
    api_games: pd.DataFrame,
    existing_games: pd.DataFrame,
) -> pd.DataFrame:

    """
    Return completed games requiring downstream processing.

    Includes:
    - Newly discovered completed games
    - Games transitioning to OFF
    - Previously completed games with corrected data
    """

    if api_games.empty:
        return pd.DataFrame(columns=GS_SCHEMA.keys())

    if existing_games.empty:
        return (
            api_games[
                api_games["gameState"].eq("OFF")
            ]
            .copy()
            .reset_index(drop=True)
        )

    compare_cols = [
        "away_team",
        "away_tri_code",
        "home_team",
        "home_tri_code",
        "awayTeam.score",
        "homeTeam.score",
        "game_date",
    ]

    existing_lookup = (
        existing_games
        .drop_duplicates(subset=["game_id"])
        .set_index("game_id")
    )

    changed_ids = []

    for _, game in api_games.iterrows():

        if game["gameState"] != "OFF":
            continue

        game_id = game["game_id"]

        if game_id not in existing_lookup.index:
            changed_ids.append(game_id)
            continue

        stored = existing_lookup.loc[game_id]

        if stored["gameState"] != "OFF":
            changed_ids.append(game_id)
            continue

        for col in compare_cols:

            old_value = stored[col]
            new_value = game[col]

            if pd.isna(old_value) and pd.isna(new_value):
                continue

            if pd.isna(old_value) != pd.isna(new_value):
                changed_ids.append(game_id)
                break

            if old_value != new_value:
                changed_ids.append(game_id)
                break

    return (
        api_games[
            api_games["game_id"].isin(changed_ids)
        ]
        .copy()
        .reset_index(drop=True)
    )

# ============================================================
# Upsert API Games
# ============================================================


def upsert_games(
    *,
    con: duckdb.DuckDBPyConnection,
    api_games: pd.DataFrame,
) -> None:

    """
    Replace stored game summaries with the latest API values.

    DELETE and INSERT execute in a single transaction.
    """

    if api_games.empty:
        return

    df = (
        api_games
        .drop_duplicates(subset=["game_id"])
        .copy()
    )

    df = enforce_gs_schema(df)

    con.register("game_summary_updates", df)

    try:
        cols = ", ".join(
            f'"{c}"'
            for c in df.columns
        )

        con.execute("BEGIN TRANSACTION")

        try:
            con.execute("""
                DELETE FROM all_game_summary
                WHERE EXISTS (
                    SELECT 1 FROM game_summary_updates AS updates
                    WHERE updates.game_id = all_game_summary.game_id
                      AND updates.season_id = all_game_summary.season_id
                )
            """)

            con.execute(f"""
                INSERT INTO all_game_summary ({cols})
                SELECT {cols}
                FROM game_summary_updates
            """)

            con.execute("COMMIT")

        except Exception:
            con.execute("ROLLBACK")
            raise

    finally:
        con.unregister("game_summary_updates")

    print(
        f"💾 Refreshed {len(df):,} game-summary rows"
    )

# ============================================================
# Timestamp Column Migration
# ============================================================

def ensure_game_summary_timestamps(con: duckdb.DuckDBPyConnection) -> None:
    # Existing databases may predate these columns.
    cols = {row[1] for row in con.execute("PRAGMA table_info('all_game_summary')").fetchall()}
    for col in ("api_fetched_at", "pipeline_run_at"):
        if col not in cols:
            con.execute(f'ALTER TABLE all_game_summary ADD COLUMN "{col}" TIMESTAMPTZ')


# ============================================================
# Main Entry Point
# ============================================================

def run_game_summary(
    *,
    con: duckdb.DuckDBPyConnection,
    season_id: int,
    season_start: pd.Timestamp,
    season_end: pd.Timestamp,
) -> pd.DataFrame:

    """
    Incrementally update all_game_summary for the requested
    NHL regular season.

    Returns
    -------
    pd.DataFrame
        Games that newly became completed since the stored
        version of all_game_summary.

        pipeline.py uses this as changed_games for downstream
        play-by-play, goalie, special-teams, etc. processing.
    """

    pipeline_run_at = datetime.now(timezone.utc)
    ensure_game_summary_timestamps(con)

    print(
        f"▶ Running game summary update "
        f"for season {season_id}"
    )

    # ========================================================
    # 1. Determine catch-up window
    # ========================================================

    fetch_start, fetch_end = determine_fetch_window(
        con=con,
        season_id=season_id,
        season_start=season_start,
        season_end=season_end,
    )

    if fetch_end < fetch_start:

        print(
            "ℹ️ No regular-season dates "
            "available to process."
        )

        return pd.DataFrame(
            columns=GS_SCHEMA.keys()
        )

    # ========================================================
    # 2. Fetch latest NHL schedule data
    # ========================================================

    api_games = fetch_games(
        season_id=season_id,
        fetch_start=fetch_start,
        fetch_end=fetch_end,
        pipeline_run_at=pipeline_run_at,
    )

    if api_games.empty:

        print(
            "ℹ️ No game-summary updates "
            "returned from NHL API."
        )

        return pd.DataFrame(
            columns=GS_SCHEMA.keys()
        )

    # ========================================================
    # 3. Load current stored versions
    #
    # IMPORTANT:
    # Do this BEFORE upserting so we can determine which
    # games actually changed to OFF.
    # ========================================================

    existing_games = load_existing_games(
        con=con,
        season_id=season_id,
        game_ids=(
            api_games["game_id"]
            .astype(int)
            .tolist()
        ),
    )

    # ========================================================
    # 4. Determine newly completed or corrected completed games
    # ========================================================

    changed_games = determine_changed_games(
        api_games=api_games,
        existing_games=existing_games,
    )

    # ========================================================
    # 5. Refresh all fetched games
    #
    # This inserts missing games and updates existing FUT/OFF
    # representations using the current API response.
    # ========================================================

    upsert_games(
        con=con,
        api_games=api_games,
    )

    # ========================================================
    # 6. Summary
    # ========================================================

    new_game_ids = set(
        api_games["game_id"]
    ) - set(
        existing_games["game_id"]
        if not existing_games.empty
        else []
    )

    print(
        f"🆕 Newly discovered games: "
        f"{len(new_game_ids):,}"
    )

    print(
        f"🏁 Completed games requiring downstream updates: "
        f"{len(changed_games):,}"
    )

    if changed_games.empty:

        print(
            "✅ Game summary is up to date; "
            "no newly completed games."
        )

    else:

        print(
            "✅ Game summary updated with "
            f"{len(changed_games):,} newly completed game(s)."
        )

        print(
            changed_games[
                [
                    "game_id",
                    "game_date",
                    "away_tri_code",
                    "home_tri_code",
                    "awayTeam.score",
                    "homeTeam.score",
                ]
            ].to_string(
                index=False
            )
        )

    return changed_games