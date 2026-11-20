
# ============================================================
# goalie_stats.py
# ============================================================

"""
Live NHL goalie-stat ingestion.

Canonical behavior follows the historical goalie backfill:
    - Same NHL API endpoint and parameters
    - Same field mappings and schema
    - Regular-season games only
    - One row per goalie per game
    - Historical duplicate behavior (keep first)
    - Same team-ID mapping, including Utah

Production behavior:
    - Process newly completed or corrected games
    - Recover completed games with no goalie records
    - Refresh the most recent 7 days of completed games
    - Fetch only the required date range
    - Filter results to required game IDs
    - Reject incomplete API pagination
    - Reject incomplete game coverage
    - Atomically replace affected games
"""

# ============================================================
# Imports
# ============================================================

import random
import time

import duckdb
import pandas as pd
import requests


# ============================================================
# Canonical Schema
# ============================================================

GOALIE_SCHEMA = {
    "game_id": "Int64",
    "games_played": "Int64",
    "games_started": "Int64",
    "goalie_name": "string",
    "player_id": "Int64",
    "tri_code": "string",
    "toi": "Int64",
    "wins": "Int64",
    "losses": "Int64",
    "ot_losses": "Int64",
    "gaa": "float64",
    "saves": "Int64",
    "shutouts": "Int64",
    "save_pct": "float64",
    "shots_against": "Int64",
    "goals_against": "Int64",
    "season_id": "Int64",
    "team_id": "Int64",
}


# ============================================================
# Schema Enforcement
# ============================================================

def enforce_goalie_schema(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    for col in GOALIE_SCHEMA:
        if col not in df.columns:
            df[col] = pd.NA

    for col, dtype in GOALIE_SCHEMA.items():

        if dtype == "string":
            df[col] = df[col].astype("string")

        elif dtype == "float64":
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        else:
            df[col] = (
                pd.to_numeric(
                    df[col],
                    errors="coerce",
                )
                .astype("Int64")
            )

    return df[list(GOALIE_SCHEMA.keys())]


# ============================================================
# Safe NHL API Request
# ============================================================

def safe_request(url, params=None, max_retries=5):
    """Retry transient NHL API failures; never silently accept failed requests."""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=(10, 60))
            response.raise_for_status()
            return response

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            error = exc

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in {429, 500, 502, 503, 504}:
                raise
            error = exc

        if attempt == max_retries:
            raise RuntimeError(
                f"NHL goalie API failed after {max_retries} attempts: {url}"
            ) from error

        delay = min(5 * 2 ** (attempt - 1), 60) + random.uniform(0, 1)
        if isinstance(error, requests.exceptions.HTTPError) and error.response is not None:
            retry_after = error.response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = max(delay, min(int(retry_after), 120))
        print(f"⚠️ NHL goalie API attempt {attempt}/{max_retries} failed: {error}; retrying in {delay:.1f}s")
        time.sleep(delay)


# ============================================================
# Load Completed Regular-Season Games
# ============================================================

def load_game_summary_df(
    con: duckdb.DuckDBPyConnection,
    season_id: int,
) -> pd.DataFrame:

    return con.execute(
        """
        SELECT
            game_id,
            game_date
        FROM all_game_summary
        WHERE season_id = ?
          AND gameState = 'OFF'
          AND SUBSTR(
              CAST(game_id AS VARCHAR),
              5,
              2
          ) = '02'
        ORDER BY game_id
        """,
        [season_id],
    ).df()


# ============================================================
# Missing-Game Recovery
# ============================================================

def load_missing_goalie_ids(
    con: duckdb.DuckDBPyConnection,
    season_id: int,
) -> list[int]:

    """
    Recover completed regular-season games that have
    no rows in all_goalie_games.
    """

    df = con.execute(
        """
        SELECT DISTINCT
            gs.game_id

        FROM all_game_summary AS gs

        WHERE gs.season_id = ?
          AND gs.gameState = 'OFF'

          AND SUBSTR(
              CAST(gs.game_id AS VARCHAR),
              5,
              2
          ) = '02'

          AND NOT EXISTS (
              SELECT 1
              FROM all_goalie_games AS gg
              WHERE gg.season_id = gs.season_id
                AND gg.game_id = gs.game_id
          )

        ORDER BY gs.game_id
        """,
        [season_id],
    ).df()

    if df.empty:
        return []

    return (
        df["game_id"]
        .dropna()
        .astype("int64")
        .tolist()
    )


# ============================================================
# Seven-Day Rolling Refresh
# ============================================================

def load_recent_goalie_ids(
    con: duckdb.DuckDBPyConnection,
    season_id: int,
    lookback_days: int = 7,
) -> list[int]:

    """
    Refresh recently completed games even if goalie
    records already exist.

    This repairs partial ingestion and captures
    corrections to NHL goalie statistics.
    """

    df = con.execute(
        """
        SELECT DISTINCT
            game_id

        FROM all_game_summary

        WHERE season_id = ?
          AND gameState = 'OFF'

          AND SUBSTR(
              CAST(game_id AS VARCHAR),
              5,
              2
          ) = '02'

          AND game_date >= (
              CURRENT_DATE - (? * INTERVAL '1 day')
          )

          AND game_date <= CURRENT_DATE

        ORDER BY game_id
        """,
        [season_id, lookback_days],
    ).df()

    if df.empty:
        return []

    return (
        df["game_id"]
        .dropna()
        .astype("int64")
        .tolist()
    )


# ============================================================
# Download NHL Goalie Games
# ============================================================

def get_goalie_games(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:

    url = (
        "https://api.nhle.com/"
        "stats/rest/en/goalie/summary"
    )

    limit = 100

    cayenne = (
        f'gameDate>="{start_date} 00:00:00" '
        f'and gameDate<="{end_date} 23:59:59"'
    )

    params = {
        "isAggregate": "false",
        "isGame": "true",
        "sort": (
            '[{"property":"gameId","direction":"ASC"},'
            '{"property":"playerId","direction":"ASC"}]'
        ),
        "cayenneExp": cayenne,
        "start": 0,
        "limit": limit,
    }

    # Fetch complete pagination without an extra initial request.
    rows = []
    start = 0
    total = None

    while total is None or start < total:
        params["start"] = start
        response = safe_request(url, params=params)
        payload = response.json()

        if not isinstance(payload, dict) or "total" not in payload:
            raise RuntimeError("NHL goalie API response missing total count; existing data preserved")

        page_total = int(payload["total"])
        if total is None:
            total = page_total
        elif page_total != total:
            raise RuntimeError("NHL goalie API total changed during pagination; existing data preserved")

        page = payload.get("data")
        if not isinstance(page, list) or (total > 0 and not page):
            raise RuntimeError(f"NHL goalie API returned invalid page at offset {start}; existing data preserved")

        rows.extend(page)
        start += len(page)
        if start < total:
            time.sleep(0.75 + random.uniform(0.1, 0.3))

    if len(rows) != total:
        raise RuntimeError(f"NHL goalie API incomplete pagination: expected {total}, received {len(rows)}")

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # --------------------------------------------------------
    # Canonical field names
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "gameId": "game_id",
            "gamesPlayed": "games_played",
            "gamesStarted": "games_started",
            "goalieFullName": "goalie_name",
            "playerId": "player_id",
            "teamAbbrev": "tri_code",
            "timeOnIce": "toi",
            "otLosses": "ot_losses",
            "goalsAgainstAverage": "gaa",
            "savePct": "save_pct",
            "shotsAgainst": "shots_against",
            "goalsAgainst": "goals_against",
            "season": "season_id",
        }
    )

    # --------------------------------------------------------
    # Canonical regular-season filtering
    # --------------------------------------------------------

    df["game_id"] = (
        pd.to_numeric(
            df["game_id"],
            errors="coerce",
        )
        .astype("Int64")
    )

    df = df[
        df["game_id"]
        .astype(str)
        .str[4:6]
        .eq("02")
    ].copy()

    return df


# ============================================================
# Main Runner
# ============================================================

def run_goalie_stats(
    *,
    con: duckdb.DuckDBPyConnection,
    season_id: int,
    changed_games: pd.DataFrame | None = None,
) -> None:

    print(
        f"▶ Running goalie stats "
        f"for season {season_id}"
    )

    # ========================================================
    # 1. Find missing games
    # ========================================================

    missing_ids = load_missing_goalie_ids(
        con,
        season_id,
    )

    # ========================================================
    # 2. Find recently completed games
    # ========================================================

    recent_ids = load_recent_goalie_ids(
        con,
        season_id,
        lookback_days=7,
    )

    # ========================================================
    # 3. Identify changed games
    # ========================================================

    changed_ids = []

    if (
        changed_games is not None
        and not changed_games.empty
    ):

        changed_ids = (
            changed_games["game_id"]
            .dropna()
            .astype("int64")
            .unique()
            .tolist()
        )

    # ========================================================
    # 4. Combine work
    # ========================================================

    candidate_ids = sorted(
        set(changed_ids)
        | set(missing_ids)
        | set(recent_ids)
    )

    print(
        f"🆕 Changed games: "
        f"{len(changed_ids):,}"
    )

    print(
        f"🩹 Missing goalie games: "
        f"{len(missing_ids):,}"
    )

    print(
        f"🔄 Recent games: "
        f"{len(recent_ids):,}"
    )

    if not candidate_ids:

        print(
            "✅ No goalie games require processing."
        )

        return

    # ========================================================
    # 5. Load completed regular-season summary
    # ========================================================

    summary_df = load_game_summary_df(
        con,
        season_id,
    )

    if summary_df.empty:
        raise RuntimeError(
            "Goalie games were queued, but no completed regular-season games "
            "exist in all_game_summary"
        )

    # Only valid completed regular-season games
    backfill_df = summary_df[
        summary_df["game_id"].isin(candidate_ids)
    ].copy()

    missing_summary_ids = set(candidate_ids) - set(backfill_df["game_id"].dropna().astype("int64"))
    if missing_summary_ids:
        raise RuntimeError(
            f"Queued goalie game IDs absent from completed summary: {sorted(missing_summary_ids)}"
        )

    if backfill_df.empty:
        raise RuntimeError("Queued goalie games have no matching completed summary rows")

    game_ids = (
        backfill_df["game_id"]
        .astype("int64")
        .tolist()
    )

    # ========================================================
    # 6. Determine API date range
    # ========================================================

    # Game summary uses Eastern dates; include one day on either side
    # to avoid excluding games near the NHL stats API date boundary.
    # Results are still filtered to the exact requested game IDs below.
    if backfill_df["game_date"].isna().any():
        raise RuntimeError("Queued goalie games contain missing game_date values")

    start_date = (pd.to_datetime(backfill_df["game_date"].min()) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (pd.to_datetime(backfill_df["game_date"].max()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print(
        f"📅 Fetching goalie stats for "
        f"{len(game_ids):,} game(s): "
        f"{start_date} → {end_date}"
    )

    # ========================================================
    # 7. Download goalie data
    # ========================================================

    goalie_updates = get_goalie_games(
        start_date,
        end_date,
    )

    if goalie_updates.empty:

        raise RuntimeError("NHL goalie API returned no data for requested games; existing records preserved")

    # ========================================================
    # 8. Filter to required games
    # ========================================================

    goalie_updates = goalie_updates[
        goalie_updates["game_id"].isin(game_ids)
    ].copy()

    if goalie_updates.empty:

        raise RuntimeError("No NHL goalie rows matched requested games; existing records preserved")

    # ========================================================
    # 9. Validate game coverage
    # ========================================================

    returned_ids = set(
        goalie_updates["game_id"]
        .dropna()
        .astype("int64")
        .tolist()
    )

    missing_api_ids = (
        set(game_ids) - returned_ids
    )

    if missing_api_ids:

        raise RuntimeError(
            "NHL goalie API did not return all requested games; existing records preserved. "
            f"Missing game IDs: {sorted(missing_api_ids)}"
        )

    # Validate identifiers before deduplication can conceal bad rows.
    for key in ("game_id", "player_id"):
        if key not in goalie_updates.columns:
            raise RuntimeError(f"NHL goalie API omitted required field: {key}")
        if pd.to_numeric(goalie_updates[key], errors="coerce").isna().any():
            raise RuntimeError(f"NHL goalie API returned missing/invalid {key}; existing data preserved")

    # ========================================================
    # 10. Canonical season assignment
    # ========================================================

    goalie_updates["season_id"] = season_id

    # ========================================================
    # 11. Canonical uniqueness
    # ========================================================

    goalie_updates = (
        goalie_updates
        .drop_duplicates(
            subset=[
                "game_id",
                "player_id",
            ],
            keep="first",
        )
        .copy()
    )

    # ========================================================
    # 12. Enforce canonical schema
    # ========================================================

    goalie_updates = enforce_goalie_schema(
        goalie_updates
    )

    # Reject rows without a valid game or goalie ID
    if (
        goalie_updates["game_id"].isna().any()
        or goalie_updates["player_id"].isna().any()
    ):
        raise ValueError(
            "Goalie API returned invalid game_id "
            "or player_id. No database changes made."
        )

    # ========================================================
    # 13. Atomic database write
    # ========================================================

    print(
        "🧪 Writing goalie stats to DuckDB"
    )

    con.register(
        "goalie_updates",
        goalie_updates,
    )

    try:

        con.execute("BEGIN TRANSACTION")

        try:

            # ------------------------------------------------
            # Delete existing affected games
            # ------------------------------------------------

            con.execute(
                """
                DELETE FROM all_goalie_games
                WHERE season_id = ?
                  AND game_id IN (
                      SELECT DISTINCT game_id
                      FROM goalie_updates
                  )
                """,
                [season_id],
            )

            # ------------------------------------------------
            # Insert canonical goalie rows
            # ------------------------------------------------

            con.execute(
                """
                INSERT INTO all_goalie_games BY NAME
                SELECT *
                FROM goalie_updates
                """
            )

            # ------------------------------------------------
            # Canonical team mapping
            # ------------------------------------------------

            con.execute(
                """
                UPDATE all_goalie_games AS g

                SET team_id = t.team_id

                FROM all_teams AS t

                WHERE g.tri_code = t.tri_code
                  AND g.tri_code NOT IN ('UTA')
                  AND g.season_id = ?

                  AND g.game_id IN (
                      SELECT DISTINCT game_id
                      FROM goalie_updates
                  )

                  AND (
                      g.team_id IS NULL
                      OR g.team_id <> t.team_id
                  )
                """,
                [season_id],
            )

            # ------------------------------------------------
            # Utah special handling
            # ------------------------------------------------

            con.execute(
                """
                UPDATE all_goalie_games

                SET team_id =
                    CASE
                        WHEN season_id = 20242025
                            THEN 59

                        WHEN season_id >= 20252026
                            THEN 68

                        ELSE team_id
                    END

                WHERE tri_code = 'UTA'
                  AND season_id = ?

                  AND game_id IN (
                      SELECT DISTINCT game_id
                      FROM goalie_updates
                  )
                """,
                [season_id],
            )

            con.execute("COMMIT")

        except Exception:

            con.execute("ROLLBACK")
            raise

    finally:

        con.unregister("goalie_updates")

    # ========================================================
    # 14. Summary
    # ========================================================

    print(
        f"✅ Inserted "
        f"{len(goalie_updates):,} goalie rows "
        f"for {goalie_updates['game_id'].nunique():,} "
        "game(s)."
    )
