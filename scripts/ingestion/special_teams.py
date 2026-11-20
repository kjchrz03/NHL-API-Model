
# ============================================================
# special_teams.py
# ============================================================

"""
Live NHL special-teams ingestion.

Canonical behavior follows the historical backfill:
    - Same NHL penalty-kill and power-play API endpoints
    - Same field mappings
    - Same PK and PP schemas
    - Same NHL statistical values
    - Regular-season games only

Production behavior:
    - Process newly completed or corrected games
    - Recover missing or incomplete PP/PK games
    - Refresh completed games from the last 7 days
    - Reject incomplete API pagination
    - Validate two team rows per game in both datasets
    - Reject conflicting duplicate team records
    - Replace PK and PP together in one transaction
"""

import random
import time

import duckdb
import pandas as pd
import requests


# ============================================================
# Canonical Schemas
# ============================================================

PK_SCHEMA = {
    "season_id": "Int64",
    "game_id": "Int64",
    "team_id": "Int64",
    "game_date": "DATE",
    "games_played": "Int64",
    "home_road": "string",
    "losses": "Int64",
    "opponent": "string",
    "ot_losses": "Int64",
    "pk_net_pct": "float64",
    "pk_pct": "float64",
    "pk_net_g": "Int64",
    "pk_net_gpg": "float64",
    "pk_toi": "float64",
    "points": "Int64",
    "points_pct": "float64",
    "pp_ga": "Int64",
    "pp_ga_pg": "float64",
    "sh_gf": "Int64",
    "sh_gf_pg": "float64",
    "team_name": "string",
    "ties": "string",
    "times_sh": "Int64",
    "times_sh_pg": "float64",
    "wins": "Int64",
}

PP_SCHEMA = {
    "season_id": "Int64",
    "game_id": "Int64",
    "team_id": "Int64",
    "game_date": "DATE",
    "games_played": "Int64",
    "home_road": "string",
    "losses": "Int64",
    "opponent": "string",
    "ot_losses": "Int64",
    "point_pct": "float64",
    "points": "Int64",
    "pp_gf": "Int64",
    "pp_net_pct": "float64",
    "pp_pct": "float64",
    "pp_gpg": "float64",
    "pp_net_g": "Int64",
    "pp_net_gpg": "float64",
    "pp_opportunities": "Int64",
    "pp_opp_pg": "float64",
    "pp_toi": "float64",
    "sh_ga": "Int64",
    "sh_ga_pg": "float64",
    "team_name": "string",
    "ties": "string",
    "wins": "Int64",
}


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

        elif dtype == "float64":
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        elif dtype == "DATE":
            df[col] = (
                pd.to_datetime(
                    df[col],
                    errors="coerce",
                ).dt.date
            )

        else:
            df[col] = (
                pd.to_numeric(
                    df[col],
                    errors="coerce",
                ).astype("Int64")
            )

    return df[list(schema.keys())]


# ============================================================
# Safe NHL API Request
# ============================================================

def safe_request(url: str, params: dict | None = None, max_retries: int = 5):
    """Retry transient NHL API failures; raise on permanent or exhausted errors."""
    retryable_statuses = {429, 500, 502, 503, 504}
    delay = 1.0

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=(10, 60))

            if response.status_code not in retryable_statuses:
                response.raise_for_status()
                return response

            if attempt == max_retries:
                response.raise_for_status()

            retry_after = response.headers.get("Retry-After")
            try:
                wait = min(float(retry_after), 60) if retry_after else delay
            except (ValueError, TypeError):
                wait = delay

            print(
                f"⚠️ NHL API HTTP {response.status_code} "
                f"(attempt {attempt}/{max_retries}); retrying in {wait:.1f}s"
            )
            time.sleep(wait + random.uniform(0, 0.25))

        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"NHL API request failed after {max_retries} attempts: {url}"
                ) from exc

            print(f"⚠️ NHL API connection/timeout error ({attempt}/{max_retries}): {exc}")
            time.sleep(delay + random.uniform(0, 0.25))

        delay = min(delay * 2, 16)

    raise RuntimeError(f"NHL API retries exhausted: {url}")


# ============================================================
# Completed Regular-Season Games
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
# Missing / Incomplete Game Recovery
# ============================================================

def load_missing_special_teams_ids(
    con: duckdb.DuckDBPyConnection,
    season_id: int,
) -> list[int]:

    """
    A complete game must have exactly two rows,
    representing two distinct teams, in BOTH tables.

    Detects:
        - Missing games
        - One-team games
        - Duplicate team records
        - Games with unexpected extra rows
    """

    df = con.execute(
        """
        SELECT gs.game_id

        FROM all_game_summary AS gs

        LEFT JOIN (
            SELECT
                game_id,
                COUNT(*) AS row_count,
                COUNT(DISTINCT team_id) AS team_count
            FROM all_pp
            WHERE season_id = ?
            GROUP BY game_id
        ) AS pp
            ON gs.game_id = pp.game_id

        LEFT JOIN (
            SELECT
                game_id,
                COUNT(*) AS row_count,
                COUNT(DISTINCT team_id) AS team_count
            FROM all_pk
            WHERE season_id = ?
            GROUP BY game_id
        ) AS pk
            ON gs.game_id = pk.game_id

        WHERE gs.season_id = ?
          AND gs.gameState = 'OFF'

          AND SUBSTR(
              CAST(gs.game_id AS VARCHAR),
              5,
              2
          ) = '02'

          AND (
              COALESCE(pp.row_count, 0) <> 2
              OR COALESCE(pp.team_count, 0) <> 2
              OR COALESCE(pk.row_count, 0) <> 2
              OR COALESCE(pk.team_count, 0) <> 2
          )

        ORDER BY gs.game_id
        """,
        [season_id, season_id, season_id],
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

def load_recent_special_teams_ids(
    con: duckdb.DuckDBPyConnection,
    season_id: int,
    lookback_days: int = 7,
) -> list[int]:

    df = con.execute(
        """
        SELECT DISTINCT game_id

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
# NHL API Pagination
# ============================================================

def get_special_teams_by_game(
    start_date: str,
    end_date: str,
    stat_type: str,
) -> pd.DataFrame:

    if stat_type not in ("penaltykill", "powerplay"):
        raise ValueError(
            f"Unsupported special teams type: {stat_type}"
        )

    url = (
        "https://api.nhle.com/"
        f"stats/rest/en/team/{stat_type}"
    )

    limit = 100

    cayenne = (
        f'gameDate>="{start_date} 00:00:00" '
        f'and gameDate<="{end_date} 23:59:59" '
        f'and gameTypeId=2'
    )

    params = {
        "isAggregate": "false",
        "isGame": "true",
        "sort": (
            '[{"property":"gameId","direction":"ASC"},'
            '{"property":"teamId","direction":"ASC"}]'
        ),
        "cayenneExp": cayenne,
        "start": 0,
        "limit": limit,
    }

    # --------------------------------------------------------
    # Initial request
    # --------------------------------------------------------

    response = safe_request(
        url,
        params=params,
    )

    if response is None:
        raise RuntimeError(
            f"{stat_type} API initial request failed. "
            "Existing data was not modified."
        )

    payload = response.json()

    if not isinstance(payload, dict) or "total" not in payload:
        raise RuntimeError(f"{stat_type} API response missing pagination total.")

    total = int(payload["total"])

    if total == 0:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Fetch all pages, starting with the initial response
    # --------------------------------------------------------

    rows = []
    first_page = payload.get("data", [])
    if not isinstance(first_page, list) or not first_page:
        raise RuntimeError(f"{stat_type} API returned an invalid initial page.")

    if len(first_page) > limit or len(first_page) > total:
        raise RuntimeError(f"{stat_type} API returned an oversized initial page.")
    rows.extend(first_page)
    start = len(first_page)

    while start < total:
        params["start"] = start
        time.sleep(0.75 + random.uniform(0.2, 0.4))
        response = safe_request(url, params=params)
        payload = response.json()
        if not isinstance(payload, dict) or "total" not in payload:
            raise RuntimeError(f"{stat_type} API returned invalid pagination metadata at offset {start}.")
        if int(payload["total"]) != total:
            raise RuntimeError(f"{stat_type} API pagination total changed during retrieval at offset {start}.")
        page = payload.get("data", [])

        if not isinstance(page, list) or not page:
            raise RuntimeError(
                f"{stat_type} API returned an empty or invalid page at offset {start}."
            )

        if len(page) > limit or start + len(page) > total:
            raise RuntimeError(f"{stat_type} API returned an oversized page at offset {start}.")
        rows.extend(page)
        start += len(page)

    if len(rows) != total:
        raise RuntimeError(
            f"{stat_type} API row count mismatch: "
            f"expected {total}, received {len(rows)}."
        )

    result = pd.DataFrame(rows)
    if not {"gameId", "teamId"}.issubset(result.columns):
        raise RuntimeError(f"{stat_type} API response missing gameId/teamId identifiers.")
    keys = result[["gameId", "teamId"]].copy()
    if keys.isna().any().any() or keys.duplicated().any():
        raise RuntimeError(f"{stat_type} API pagination returned null or repeated game/team identifiers.")
    return result


def get_pk_by_game(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:

    return get_special_teams_by_game(
        start_date,
        end_date,
        "penaltykill",
    )


def get_pp_by_game(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:

    return get_special_teams_by_game(
        start_date,
        end_date,
        "powerplay",
    )


# ============================================================
# Canonical API Field Mappings
# ============================================================

PK_RENAME_MAP = {
    "season": "season_id",
    "gameId": "game_id",
    "teamId": "team_id",
    "gameDate": "game_date",
    "teamFullName": "team_name",
    "gamesPlayed": "games_played",
    "homeRoad": "home_road",
    "opponentTeamAbbrev": "opponent",
    "otLosses": "ot_losses",
    "penaltyKillNetPct": "pk_net_pct",
    "penaltyKillPct": "pk_pct",
    "pkNetGoals": "pk_net_g",
    "pkNetGoalsPerGame": "pk_net_gpg",
    "pkTimeOnIce": "pk_toi",
    "pointsPct": "points_pct",
    "ppGoalsAgainst": "pp_ga",
    "ppGoalsAgainstPerGame": "pp_ga_pg",
    "shGoalsFor": "sh_gf",
    "shGoalsForPerGame": "sh_gf_pg",
    "timesShorthanded": "times_sh",
    "timesShorthandedPerGame": "times_sh_pg",
}

PP_RENAME_MAP = {
    "season": "season_id",
    "gameId": "game_id",
    "teamId": "team_id",
    "gameDate": "game_date",
    "teamFullName": "team_name",
    "gamesPlayed": "games_played",
    "homeRoad": "home_road",
    "opponentTeamAbbrev": "opponent",
    "otLosses": "ot_losses",
    "pointPct": "point_pct",
    "powerPlayGoalsFor": "pp_gf",
    "powerPlayNetPct": "pp_net_pct",
    "powerPlayPct": "pp_pct",
    "ppGoalsPerGame": "pp_gpg",
    "ppNetGoals": "pp_net_g",
    "ppNetGoalsPerGame": "pp_net_gpg",
    "ppOpportunities": "pp_opportunities",
    "ppOpportunitiesPerGame": "pp_opp_pg",
    "ppTimeOnIce": "pp_toi",
    "shGoalsAgainst": "sh_ga",
    "shGoalsAgainstPerGame": "sh_ga_pg",
}


# ============================================================
# Normalize API Data
# ============================================================

def normalize_special_teams(
    df: pd.DataFrame,
    rename_map: dict,
    schema: dict,
    season_id: int,
    game_ids: list[int],
) -> pd.DataFrame:

    if df.empty:
        return enforce_schema(df, schema)

    df = df.rename(
        columns=rename_map
    ).copy()

    if "game_id" not in df.columns:
        raise ValueError(
            "Special teams API response missing game_id."
        )

    if "team_id" not in df.columns:
        raise ValueError(
            "Special teams API response missing team_id."
        )

    # Reject malformed API identifiers before filtering can silently discard them.
    for key in ("game_id", "team_id"):
        numeric = pd.to_numeric(df[key], errors="coerce")
        invalid = numeric.isna() | (numeric <= 0) | (numeric % 1 != 0)
        if invalid.any():
            raise ValueError(
                f"Special teams API returned {int(invalid.sum())} invalid {key} "
                "value(s) before game filtering."
            )
        df[key] = numeric.astype("Int64")

    # --------------------------------------------------------
    # Required games only
    # --------------------------------------------------------

    df = df[
        df["game_id"].isin(game_ids)
    ].copy()

    # --------------------------------------------------------
    # Regular season only
    # --------------------------------------------------------

    df = df[
        df["game_id"]
        .astype("string")
        .str[4:6]
        .eq("02")
    ].copy()

    if df.empty:
        return enforce_schema(df, schema)

    df["season_id"] = season_id

    df = enforce_schema(
        df,
        schema,
    )

    # --------------------------------------------------------
    # Reject invalid identifiers
    # --------------------------------------------------------

    if (
        df["game_id"].isna().any()
        or df["team_id"].isna().any()
    ):
        raise ValueError(
            "Special teams API returned invalid "
            "game_id or team_id."
        )

    # --------------------------------------------------------
    # Exact duplicate rows are harmless
    # --------------------------------------------------------

    df = df.drop_duplicates(
        keep="first"
    ).copy()

    # --------------------------------------------------------
    # Conflicting duplicate team rows are not harmless
    # --------------------------------------------------------

    duplicate_teams = df.duplicated(
        subset=["game_id", "team_id"],
        keep=False,
    )

    if duplicate_teams.any():
        raise ValueError(
            "Special teams API returned conflicting "
            "duplicate rows for the same game/team."
        )

    return df.reset_index(drop=True)


# ============================================================
# Validate Complete Game Coverage
# ============================================================

def validate_special_teams_coverage(
    pk_df: pd.DataFrame,
    pp_df: pd.DataFrame,
    game_ids: list[int],
) -> None:

    required_ids = set(game_ids)

    for name, df in (
        ("PK", pk_df),
        ("PP", pp_df),
    ):

        returned_ids = set(
            df["game_id"]
            .dropna()
            .astype("int64")
            .tolist()
        )

        missing_ids = (
            required_ids - returned_ids
        )

        if missing_ids:
            raise ValueError(
                f"{name} API missing games: "
                f"{sorted(missing_ids)}"
            )

        counts = df.groupby(
            "game_id"
        )["team_id"].agg(
            rows="size",
            teams="nunique",
        )

        invalid = counts[
            (counts["rows"] != 2)
            | (counts["teams"] != 2)
        ]

        if not invalid.empty:
            raise ValueError(
                f"{name} API returned incomplete "
                "or invalid team rows for games: "
                f"{invalid.index.tolist()}"
            )

    # --------------------------------------------------------
    # PK and PP must contain the same teams for each game
    # --------------------------------------------------------

    pk_teams = (
        pk_df.groupby("game_id")["team_id"]
        .apply(lambda x: set(x.astype(int)))
        .to_dict()
    )

    pp_teams = (
        pp_df.groupby("game_id")["team_id"]
        .apply(lambda x: set(x.astype(int)))
        .to_dict()
    )

    mismatched = [
        game_id
        for game_id in game_ids
        if pk_teams[game_id] != pp_teams[game_id]
    ]

    if mismatched:
        raise ValueError(
            "PK and PP returned different teams "
            f"for games: {mismatched}"
        )


# ============================================================
# Main Runner
# ============================================================

def run_special_teams(
    *,
    con: duckdb.DuckDBPyConnection,
    season_id: int,
    changed_games: pd.DataFrame | None = None,
) -> None:

    print(
        f"▶ Running special teams "
        f"for season {season_id}"
    )

    # ========================================================
    # 1. Find missing/incomplete games
    # ========================================================

    missing_ids = load_missing_special_teams_ids(
        con,
        season_id,
    )

    # ========================================================
    # 2. Find recent games
    # ========================================================

    recent_ids = load_recent_special_teams_ids(
        con,
        season_id,
        lookback_days=7,
    )

    # ========================================================
    # 3. Find changed games
    # ========================================================

    changed_ids = []

    if (
        changed_games is not None
        and not changed_games.empty
    ):

        changed = changed_games.copy()

        changed["game_id"] = (
            pd.to_numeric(
                changed["game_id"],
                errors="coerce",
            ).astype("Int64")
        )

        changed = changed[
            changed["game_id"]
            .astype("string")
            .str[4:6]
            .eq("02")
        ]

        changed_ids = (
            changed["game_id"]
            .dropna()
            .astype("int64")
            .unique()
            .tolist()
        )

    # ========================================================
    # 4. Combine selected games
    # ========================================================

    candidate_ids = sorted(
        set(changed_ids)
        | set(missing_ids)
        | set(recent_ids)
    )

    print(
        f"🆕 Changed games: {len(changed_ids):,}"
    )

    print(
        f"🩹 Missing/incomplete games: "
        f"{len(missing_ids):,}"
    )

    print(
        f"🔄 Recent games: {len(recent_ids):,}"
    )

    if not candidate_ids:
        print(
            "✅ No special teams games require processing."
        )
        return

    # ========================================================
    # 5. Load eligible completed games
    # ========================================================

    summary_df = load_game_summary_df(
        con,
        season_id,
    )

    if summary_df.empty:
        if missing_ids or recent_ids:
            raise RuntimeError("Special teams games were queued but no completed regular-season games were found")
        print("ℹ️ No completed regular-season games eligible for special teams.")
        return

    backfill_df = summary_df[
        summary_df["game_id"].isin(
            candidate_ids
        )
    ].copy()

    if backfill_df.empty:
        if missing_ids or recent_ids:
            raise RuntimeError("Queued missing/recent special teams games are absent from completed game summary")
        print("ℹ️ Changed games contain no completed regular-season games; skipping.")
        return

    # Every missing/recent ID must be present in the eligible completed-game summary.
    recovery_ids = set(missing_ids) | set(recent_ids)
    unresolved = recovery_ids - set(backfill_df["game_id"].astype("int64"))
    if unresolved:
        raise RuntimeError(f"Queued special teams recovery games missing from completed summary: {sorted(unresolved)}")

    game_ids = (
        backfill_df["game_id"]
        .astype("int64")
        .tolist()
    )

    # ========================================================
    # 6. Determine API date range
    # ========================================================

    # The stats API may use a different calendar-date boundary from game summary.
    # Fetch a one-day buffer; normalization still retains only requested game IDs.
    start_date = (pd.to_datetime(backfill_df["game_date"].min()) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (pd.to_datetime(backfill_df["game_date"].max()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print(
        f"📅 Fetching special teams for "
        f"{len(game_ids):,} game(s): "
        f"{start_date} → {end_date}"
    )

    # ========================================================
    # 7. Download both datasets
    # ========================================================

    pk_raw = get_pk_by_game(
        start_date,
        end_date,
    )

    pp_raw = get_pp_by_game(
        start_date,
        end_date,
    )

    # ========================================================
    # 8. Normalize both datasets
    # ========================================================

    pk_df = normalize_special_teams(
        pk_raw,
        PK_RENAME_MAP,
        PK_SCHEMA,
        season_id,
        game_ids,
    )

    pp_df = normalize_special_teams(
        pp_raw,
        PP_RENAME_MAP,
        PP_SCHEMA,
        season_id,
        game_ids,
    )

    # ========================================================
    # 9. Validate both before modifying DuckDB
    # ========================================================

    validate_special_teams_coverage(pk_df, pp_df, game_ids)

    # Verify both API datasets identify the actual home/away teams, not merely
    # the same two teams as each other.
    summary_columns = {row[1] for row in con.execute("PRAGMA table_info('all_game_summary')").fetchall()}
    home_col = next((c for c in ("home_team", "home_team_id", "homeTeamId") if c in summary_columns), None)
    away_col = next((c for c in ("away_team", "away_team_id", "awayTeamId") if c in summary_columns), None)
    if home_col is None or away_col is None:
        raise RuntimeError("Cannot validate special teams teams: game summary lacks known home/away team ID columns")
    expected_teams = con.execute(f"""
        SELECT game_id, {home_col} AS home_team_id, {away_col} AS away_team_id
        FROM all_game_summary
        WHERE season_id = ? AND game_id IN (SELECT UNNEST(?::BIGINT[]))
    """, [season_id, game_ids]).df()
    if len(expected_teams) != len(set(game_ids)):
        raise RuntimeError("Missing game-summary team identities for requested special teams games")
    for row in expected_teams.itertuples(index=False):
        if pd.isna(row.home_team_id) or pd.isna(row.away_team_id):
            raise RuntimeError(f"Missing home/away team IDs for game {row.game_id}")
        expected = {int(row.home_team_id), int(row.away_team_id)}
        for name, frame in (("PK", pk_df), ("PP", pp_df)):
            actual = set(frame.loc[frame["game_id"] == row.game_id, "team_id"].astype(int))
            if actual != expected:
                raise RuntimeError(f"{name} team IDs do not match game summary for game {row.game_id}: {actual} != {expected}")

    # ========================================================
    # 10. Register temporary DataFrames
    # ========================================================

    con.register(
        "pk_updates",
        pk_df,
    )

    try:

        con.register(
            "pp_updates",
            pp_df,
        )

        try:

            # =================================================
            # 11. Atomic PK + PP write
            # =================================================

            print(
                "🧪 Writing PK and PP to DuckDB"
            )

            con.execute(
                "BEGIN TRANSACTION"
            )

            try:

                # ---------------------------------------------
                # Delete affected PK games
                # ---------------------------------------------

                con.execute(
                    """
                    DELETE FROM all_pk
                    WHERE season_id = ?
                      AND game_id IN (
                          SELECT DISTINCT game_id
                          FROM pk_updates
                      )
                    """,
                    [season_id],
                )

                # ---------------------------------------------
                # Delete affected PP games
                # ---------------------------------------------

                con.execute(
                    """
                    DELETE FROM all_pp
                    WHERE season_id = ?
                      AND game_id IN (
                          SELECT DISTINCT game_id
                          FROM pp_updates
                      )
                    """,
                    [season_id],
                )
                # ---------------------------------------------
                # Insert canonical PK rows
                # ---------------------------------------------

                con.execute("""
                    INSERT INTO all_pk BY NAME
                    SELECT *
                    FROM pk_updates
                """)

                # ---------------------------------------------
                # Insert canonical PP rows
                # ---------------------------------------------

                con.execute("""
                    INSERT INTO all_pp BY NAME
                    SELECT *
                    FROM pp_updates
                """)

                # ========================================================
                # 12. Verify persisted PK/PP coverage
                # ========================================================

                for table_name, expected_df in (("all_pk", pk_df), ("all_pp", pp_df)):
                    stored = con.execute(f"""
                        SELECT game_id, team_id, COUNT(*) AS row_count
                        FROM {table_name}
                        WHERE season_id = ? AND game_id IN (SELECT UNNEST(?::BIGINT[]))
                        GROUP BY game_id, team_id
                    """, [season_id, game_ids]).df()

                    actual = {(int(row.game_id), int(row.team_id)): int(row.row_count)
                              for row in stored.itertuples(index=False)}
                    expected = {(int(row.game_id), int(row.team_id)): 1
                                for row in expected_df[["game_id", "team_id"]].itertuples(index=False)}
                    if actual != expected:
                        missing = sorted(set(expected) - set(actual))
                        extra = sorted(set(actual) - set(expected))
                        duplicates = sorted(key for key, count in actual.items() if count != 1)
                        raise RuntimeError(
                            f"Post-write {table_name} coverage failed: "
                            f"missing={missing[:10]}, extra={extra[:10]}, duplicates={duplicates[:10]}"
                        )


                con.execute(
                    "COMMIT"
                )

            except Exception:
                con.execute(
                    "ROLLBACK"
                )
                raise

        finally:
            con.unregister(
                "pp_updates"
            )

    finally:
        con.unregister(
            "pk_updates"
        )

    # ========================================================
    # 13. Summary
    # ========================================================

    print(
        f"✅ Inserted {len(pk_df):,} PK rows "
        f"and {len(pp_df):,} PP rows "
        f"for {len(game_ids):,} game(s)."
    )
