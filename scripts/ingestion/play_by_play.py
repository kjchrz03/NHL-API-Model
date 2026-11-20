# ============================================================
# play_by_play.py
# ============================================================
import requests
import numpy as np
import pandas as pd
import time
import random
import duckdb
# ============================================================
# Play-by-Play Runner
# ============================================================
def run_play_by_play(*, con: duckdb.DuckDBPyConnection, season_id: int, changed_games: pd.DataFrame | None = None):
    """
    Update play-by-play data for a given season.
    Assumes game summary has already been run.
    Normal live behavior:
        - Process games returned by game_summary as changed_games.
    Recovery behavior:
        - Also detect completed regular-season games in
          all_game_summary that do not yet exist in all_play_by_play.
    This allows the pipeline to recover automatically if it has
    not been run for one or more days.
    """
    print(f"▶ Running play-by-play for season {season_id}")

    # ============================================================
    # HELPERS

    # ============================================================

    # ------------------------------------------------------------
    # Recovery protection

    # ------------------------------------------------------------

    def load_missing_pbp_ids(season_id: int) -> list[int]:
        """
        Find completed regular-season games that exist in
        all_game_summary but do not yet exist in all_play_by_play.
        """
        query = """
            SELECT DISTINCT gs.game_id
            FROM all_game_summary AS gs
            WHERE gs.season_id = ?
              AND gs.gameState = 'OFF'
              AND SUBSTR(CAST(gs.game_id AS VARCHAR), 5, 2) = '02'
              AND NOT EXISTS (
                  SELECT 1
                  FROM all_play_by_play AS pbp
                  WHERE pbp.season_id = gs.season_id
                    AND pbp.game_id = gs.game_id
              )
            ORDER BY gs.game_id
        """
        df = con.execute(
            query,
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

    def load_recent_pbp_ids(
        season_id: int,
        lookback_days: int = 7,
    ) -> list[int]:
        """
        Return completed regular-season games from the
        last N days for automatic PBP refresh.
        """
        query = """
            SELECT DISTINCT game_id
            FROM all_game_summary
            WHERE season_id = ?
              AND gameState = 'OFF'
              AND SUBSTR(CAST(game_id AS VARCHAR), 5, 2) = '02'
              AND game_date >= CURRENT_DATE - (? * INTERVAL '1 day')
              AND game_date <= CURRENT_DATE
            ORDER BY game_id
        """
        df = con.execute(
            query,
            [season_id, lookback_days],
        ).df()
        return df["game_id"].astype("int64").tolist()

    # ------------------------------------------------------------
    # Load completed regular-season games

    # ------------------------------------------------------------

    def load_game_summary_df(season_id: int) -> pd.DataFrame:
        query = """
            SELECT *
            FROM all_game_summary
            WHERE season_id = ?
              AND gameState = 'OFF'
              AND SUBSTR(CAST(game_id AS VARCHAR), 5, 2) = '02'
            ORDER BY game_id
        """
        df = con.execute(
            query,
            [season_id],
        ).df()
        print(
            f"Loaded {len(df):,} completed "
            f"regular-season games from summary"
        )
        return df

    # ============================================================
    # SAFE REQUEST WITH BACKOFF

    # ============================================================

    def safe_request(url: str, max_retries: int = 4):
        """Retry transient NHL API failures; never silently skip a failed game."""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=(10, 60))
                response.raise_for_status()

                if attempt > 0:
                    print(f"✅ PBP API recovered after {attempt + 1} attempts: {url}")

                return response
   
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status not in {429, 500, 502, 503, 504}:
                    raise
                error = exc
                retry_after = exc.response.headers.get("Retry-After") if exc.response is not None else None
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                error = exc
                retry_after = None
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"NHL PBP API failed after {max_retries} attempts: {url}"
                ) from error
            delay = min(5 * (2 ** attempt), 60) + random.uniform(0, 1)
            if retry_after:
                try:
                    delay = max(delay, min(float(retry_after), 120))
                except ValueError:
                    pass
            print(f"⚠️ PBP API request failed ({error}); retrying in {delay:.1f}s")
            time.sleep(delay)

    # ============================================================
    # PBP SCHEMA

    # ============================================================
    PBP_SCHEMA = {
        "season_id": "Int64",
        "game_id": "Int64",
        "eventId": "Int64",
        "timeInPeriod": "string",
        "timeRemaining": "string",
        "situationCode": "Int64",
        "homeTeamDefendingSide": "string",
        "type_code": "Int64",
        "typeDescKey": "string",
        "sortOrder": "Int64",
        "period_number": "Int64",
        "period_type": "string",
        "max_regulation_periods": "Int64",
        "team_id": "Int64",
        "losing_player_id": "Int64",
        "winning_player_id": "Int64",
        "x_coord": "float64",
        "y_coord": "float64",
        "zone_code": "string",
        "hitter": "Int64",
        "hittee": "Int64",
        "shot_type": "string",
        "shooting_player": "Int64",
        "goalie": "Int64",
        "away_sog": "Int64",
        "home_sog": "Int64",
        "player_id": "Int64",
        "reason": "string",
        "blocking_player": "Int64",
        "desc_key": "string",
        "duration": "Int64",
        "committed_by": "Int64",
        "drawn_by": "Int64",
        "secondary_reason": "string",
        "scoring_player": "Int64",
        "scoring_player_total": "Int64",
        "assist_1": "Int64",
        "assist1_total": "Int64",
        "away_score": "Int64",
        "home_score": "Int64",
        "assist_2": "Int64",
        "assist2_total": "Int64",
        "served_by": "Int64",
        "situation": "string",
        "goalie_situation": "string",
        "game_in_seconds": "Int64",
        "home_team": "Int64",
        "away_team": "Int64",
    }

    # ============================================================
    # SCHEMA ENFORCEMENT

    # ============================================================

    def enforce_pbp_schema(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # -------- DUPLICATE COLUMN PROTECTION --------
        if df.columns.duplicated().any():
            df = df.loc[
                :,
                ~df.columns.duplicated()
            ]
        # -------- KEEP KNOWN SCHEMA COLUMNS --------
        df = df[
            [
                col
                for col in PBP_SCHEMA
                if col in df.columns
            ]
        ]
        # -------- ADD MISSING SCHEMA COLUMNS --------
        for col in PBP_SCHEMA:
            if col not in df.columns:
                df[col] = pd.NA
        # -------- ENFORCE DATA TYPES --------
        for col, dtype in PBP_SCHEMA.items():
            if dtype == "string":
                df[col] = df[col].astype(
                    "string"
                )
            elif dtype == "float64":
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )
            else:
                # Guard against accidental duplicate-column
                # DataFrame objects.
                if isinstance(
                    df[col],
                    pd.DataFrame,
                ):
                    df[col] = df[col].iloc[:, 0]
                df[col] = (
                    pd.to_numeric(
                        df[col],
                        errors="coerce",
                    )
                    .astype("Int64")
                )
        return df[
            list(PBP_SCHEMA.keys())
        ]

    # ============================================================
    # PLAY-BY-PLAY FETCH

    # ============================================================

    def play_by_play_events(
        game_summary: pd.DataFrame,
    ) -> pd.DataFrame:
        all_game_ids = (
            game_summary["game_id"]
            .dropna()
            .astype("int64")
            .unique()
            .tolist()
        )
        results = []
        # ------------------------------------------------------------
        # Process individual game
        # ------------------------------------------------------------
        def process_game(
            game_id: int,
        ) -> pd.DataFrame:
            match = game_summary.loc[
                game_summary["game_id"] == game_id,
                "link",
            ]
            if match.empty:
                print(
                    f"⚠️ No PBP link found for "
                    f"game {game_id}"
                )
                raise ValueError(f"No PBP link found for game {game_id}")
            link = str(
                match.iloc[0]
            ).strip()
            if not link.startswith("http"):
                print(
                    f"⚠️ Invalid PBP link for "
                    f"game {game_id}"
                )
                raise ValueError(f"Invalid PBP link for game {game_id}: {link}")
            response = safe_request(
                link
            )
            data = response.json()
            if not isinstance(data, dict) or "plays" not in data or not isinstance(data["plays"], list):
                print(
                    f"⚠️ No plays returned for "
                    f"game {game_id}"
                )
                raise ValueError(f"NHL API response missing plays for game {game_id}")
            df = pd.json_normalize(
                data["plays"]
            )
            if df.empty:
                raise ValueError(f"NHL API returned no PBP events for completed game {game_id}")
            df["game_id"] = game_id
            return df
        # ------------------------------------------------------------
        # Fetch requested games
        # ------------------------------------------------------------
        for i, game_id in enumerate(
            all_game_ids,
            start=1,
        ):
            print(
                f"🎯 Processing game "
                f"{i}/{len(all_game_ids)} — "
                f"game_id={game_id}"
            )
            game_df = process_game(game_id)
            results.append(game_df)
        if not results:
            return pd.DataFrame()
        out = pd.concat(
            results,
            ignore_index=True,
        )
        out["season_id"] = season_id
        return out

    # ============================================================
    # PLAY-BY-PLAY NORMALIZATION

    # ============================================================

    def normalize_play_by_play(df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            # -------- TOP-LEVEL EVENT FIELDS (CRITICAL) --------
            "typeCode": "type_code",
            # -------- PERIOD INFO --------
            "periodDescriptor.number": "period_number",
            "periodDescriptor.periodType": "period_type",
            "periodDescriptor.maxRegulationPeriods": "max_regulation_periods",
            # -------- TEAM / PLAYER --------
            "details.eventOwnerTeamId": "team_id",
            "details.losingPlayerId": "losing_player_id",
            "details.winningPlayerId": "winning_player_id",
            # -------- LOCATION --------
            "details.xCoord": "x_coord",
            "details.yCoord": "y_coord",
            "details.zoneCode": "zone_code",
            # -------- SHOOTING / SCORING --------
            "details.shotType": "shot_type",
            "details.shootingPlayerId": "shooting_player",
            "details.goalieInNetId": "goalie",
            "details.scoringPlayerId": "scoring_player",
            "details.scoringPlayerTotal": "scoring_player_total",
            "details.assist1PlayerId": "assist_1",
            "details.assist1PlayerTotal": "assist1_total",
            "details.assist2PlayerId": "assist_2",
            "details.assist2PlayerTotal": "assist2_total",
            # -------- SCORE --------
            "details.awayScore": "away_score",
            "details.homeScore": "home_score",
            # -------- MISC --------
            "details.blockingPlayerId": "blocking_player",
            "details.playerId": "player_id",
            "details.reason": "reason",
            "details.hittingPlayerId": "hitter",
            "details.hitteePlayerId": "hittee",
            "details.awaySOG": "away_sog",
            "details.homeSOG": "home_sog",
            "details.secondaryReason": "secondary_reason",
            "details.duration": "duration",
            "details.committedByPlayerId": "committed_by",
            "details.drawnByPlayerId": "drawn_by",
            "details.servedByPlayerId": "served_by",
        }
        df = df.rename(columns=rename_map)
        # -------- SITUATION / GOALIE --------
        # Default to "in net" so early-season / incomplete API
        # situationCode data does not incorrectly create null goalie status.
        df["goalie_situation"] = "in net"
        df["situation"] = pd.NA
        # Only evaluate situationCode if the field exists AND
        # the API actually returned at least one non-null value.
        if "situationCode" in df.columns and df["situationCode"].notna().any():
            # NHL may encode situationCode numerically, dropping leading zeros.
            # Restore four digits before situation mapping / pulled-goalie checks.
            sc = df["situationCode"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(4)
            SITUATION_DICT = {
                "1551": "5 on 5", "1451": "5 on 4", "1541": "5 on 4",
                "0651": "6 on 5", "1560": "6 on 5", "1441": "4 on 4",
                "1331": "3 on 3", "1460": "6 on 4", "1351": "5 on 3",
                "0641": "6 on 4", "1341": "4 on 3"
            }
            df["situation"] = sc.map(SITUATION_DICT)
            # situationCode encodes skater/goalie state.
            # A leading 0 or a 0 in the goalie-position portion
            # indicates a pulled goalie.
            pulled_mask = (
                sc.str.startswith("0", na=False)
                | ((sc.str.len() >= 4) & (sc.str[3] == "0"))
            )
            df.loc[
                pulled_mask.fillna(False),
                "goalie_situation"
            ] = "pulled"
        # -------- TIME IN SECONDS --------
        df["game_in_seconds"] = df["timeInPeriod"].apply(
            lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1])
            if isinstance(x, str) and ":" in x else np.nan
        )
        # -------- HARD GUARD (NO SILENT FAILURES) --------
        if "typeDescKey" not in df.columns:
            raise RuntimeError("Normalization failed: typeDescKey missing")
        if "eventId" not in df.columns:
            raise RuntimeError("Normalization failed: eventId missing")
        return df

    # ============================================================
    # INGESTION / BACKFILL EXECUTION

    # ============================================================

    # ------------------------------------------------------------
    # Recovery protection:
    # find completed games that are missing from all_play_by_play.

    # ------------------------------------------------------------
    missing_ids = load_missing_pbp_ids(
        season_id
    )
    if missing_ids:
        print(
            f"🔄 Found {len(missing_ids):,} "
            f"completed game(s) missing PBP"
        )

    # ------------------------------------------------------------
    # Games changed during the current pipeline run

    # ------------------------------------------------------------
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
            )
            .astype("Int64")
        )
        # -------- REGULAR SEASON ONLY (02) --------
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
        if changed_ids:
            print(
                f"🆕 {len(changed_ids):,} changed "
                f"regular-season game(s)"
            )

    # ------------------------------------------------------------
    # Combine normal live updates + recovery games

    # ------------------------------------------------------------
    recent_ids = load_recent_pbp_ids(
        season_id,
        lookback_days=7,
    )
    game_ids = sorted(
        set(changed_ids)
        | set(missing_ids)
        | set(recent_ids)
    )
    if not game_ids:
        print(
            "✅ PBP and game summary are in sync."
        )
        return
    print(
        f"📋 {len(game_ids):,} total game(s) "
        f"need PBP processing"
    )

    # ============================================================
    # LOAD GAME SUMMARY

    # ============================================================
    summary_df = load_game_summary_df(
        season_id
    )
    if summary_df.empty:
        if missing_ids or recent_ids:
            raise RuntimeError("PBP recovery games queued but completed game summary is empty")
        print("ℹ️ No completed regular-season games eligible for PBP; skipping.")
        return

    # ------------------------------------------------------------
    # Only process games identified above

    # ------------------------------------------------------------
    backfill_df = summary_df[
        summary_df["game_id"].isin(
            game_ids
        )
    ].copy()
    # Changed games can be scheduled or in progress; only recovery games
    # must already exist in the completed regular-season summary.
    eligible_ids = set(backfill_df["game_id"].dropna().astype("int64"))
    absent_recovery_ids = (set(missing_ids) | set(recent_ids)) - eligible_ids
    if absent_recovery_ids:
        raise RuntimeError(f"Recovery PBP games absent from completed summary: {sorted(absent_recovery_ids)}")
    skipped_changed_ids = set(changed_ids) - eligible_ids
    if skipped_changed_ids:
        print(f"ℹ️ Skipping {len(skipped_changed_ids)} changed game(s) not yet completed")
    game_ids = sorted(eligible_ids)
    if backfill_df.empty:
        print("ℹ️ No completed regular-season games eligible for PBP; skipping.")
        return
    print(
        f"📥 Fetching play-by-play for "
        f"{len(backfill_df):,} game(s)"
    )

    # ============================================================
    # FETCH RAW PLAY-BY-PLAY

    # ============================================================
    pbp_updates = play_by_play_events(
        backfill_df
    )
    if pbp_updates.empty:
        raise RuntimeError("Requested PBP games but no play-by-play data was returned")

    # ============================================================
    # NORMALIZE PLAY-BY-PLAY

    # ============================================================
    print(
        "🧹 Normalizing play-by-play data"
    )
    pbp_updates = normalize_play_by_play(
        pbp_updates
    )

    # ============================================================
    # ENFORCE SCHEMA

    # ============================================================
    print(
        "📐 Enforcing play-by-play schema"
    )
    pbp_updates = enforce_pbp_schema(
        pbp_updates
    )

    # ============================================================
    # FINAL REGULAR-SEASON SAFETY CHECK

    # ============================================================
    pbp_updates = pbp_updates[
        pbp_updates["game_id"]
        .astype("string")
        .str[4:6]
        .eq("02")
    ].copy()
    if pbp_updates.empty:
        raise RuntimeError("All fetched PBP rows failed regular-season validation")
    # Fail before writing if the API omitted event IDs or entire games.
    if pbp_updates["eventId"].isna().any():
        raise RuntimeError("PBP contains missing eventId values; refusing to replace stored games")
    received_ids = set(pbp_updates["game_id"].dropna().astype("int64"))
    missing_results = set(game_ids) - received_ids
    if missing_results:
        raise RuntimeError(f"PBP response missing requested games: {sorted(missing_results)}")

    # ============================================================
    # DEDUPLICATE EVENTS

    # ============================================================
    # Exact duplicate events are safe to collapse; conflicting versions
    # indicate an inconsistent API response and must not be chosen arbitrarily.
    event_keys = ["game_id", "eventId"]
    conflicting = pbp_updates.groupby(event_keys, dropna=False).filter(
        lambda group: len(group.drop_duplicates()) > 1
    )
    if not conflicting.empty:
        keys = conflicting[event_keys].drop_duplicates().head(10).values.tolist()
        raise RuntimeError(f"Conflicting duplicate PBP events returned: {keys}")
    pbp_updates = (
        pbp_updates
        .sort_values(
            [
                "game_id",
                "eventId",
            ]
        )
        .drop_duplicates(
            subset=[
                "game_id",
                "eventId",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # ============================================================
    # EXECUTION — ATOMIC DUCKDB UPSERT

    # ============================================================
    # Validate summary identities before any changes to persisted PBP.
    summary_teams = con.execute("""
        SELECT game_id, home_team, away_team
        FROM all_game_summary
        WHERE season_id = ?
          AND game_id IN (SELECT UNNEST(?::BIGINT[]))
    """, [season_id, game_ids]).df()
    summary_counts = summary_teams.groupby("game_id").size()
    if (set(summary_counts.index.astype("int64")) != set(game_ids)
            or (summary_counts != 1).any()):
        raise RuntimeError("PBP game summary has missing or duplicate game identities")
    if (summary_teams[["home_team", "away_team"]].isna().any().any()
            or (summary_teams["home_team"] == summary_teams["away_team"]).any()):
        raise RuntimeError("PBP game summary contains missing or identical home/away team IDs")

    print("🧪 Writing play-by-play to DuckDB")
    con.register("pbp_updates", pbp_updates)
    try:
        con.execute("BEGIN TRANSACTION")
        try:
            # --------------------------------------------
            # 1. Delete existing events for refreshed games
            # --------------------------------------------
            con.execute(
                """
                DELETE FROM all_play_by_play
                WHERE season_id = ?
                  AND game_id IN (
                      SELECT DISTINCT game_id
                      FROM pbp_updates
                  )
                """,
                [season_id],
            )
            # --------------------------------------------
            # 2. Insert refreshed play-by-play events
            # --------------------------------------------
            con.execute(
                """
                INSERT INTO all_play_by_play BY NAME
                SELECT *
                FROM pbp_updates
                """
            )
            # --------------------------------------------
            # 3. Populate home / away team IDs
            # --------------------------------------------
            con.execute(
                """
                UPDATE all_play_by_play AS pbp
                SET
                    home_team = gs.home_team,
                    away_team = gs.away_team
                FROM all_game_summary AS gs
                WHERE pbp.game_id = gs.game_id
                  AND pbp.season_id = gs.season_id
                  AND pbp.season_id = ?
                  AND pbp.game_id IN (
                      SELECT DISTINCT game_id
                      FROM pbp_updates
                  )
                """,
                [season_id],
            )
            # Verify persisted events and team IDs before committing.
            expected_counts = pbp_updates.groupby("game_id").size().to_dict()
            stored = con.execute("""
                SELECT pbp.game_id,
                       COUNT(*) AS event_count,
                       COUNT(DISTINCT pbp.eventId) AS distinct_events,
                       COUNT(*) FILTER (
                           WHERE pbp.home_team IS NULL OR pbp.away_team IS NULL
                       ) AS missing_teams,
                       COUNT(*) FILTER (
                           WHERE pbp.home_team IS DISTINCT FROM gs.home_team
                              OR pbp.away_team IS DISTINCT FROM gs.away_team
                       ) AS incorrect_teams
                FROM all_play_by_play AS pbp
                JOIN all_game_summary AS gs
                  ON pbp.game_id = gs.game_id AND pbp.season_id = gs.season_id
                WHERE pbp.season_id = ?
                  AND pbp.game_id IN (SELECT DISTINCT game_id FROM pbp_updates)
                GROUP BY pbp.game_id
            """, [season_id]).df()
            stored_ids = set(stored["game_id"].astype("int64"))
            if stored_ids != set(expected_counts):
                raise RuntimeError(
                    f"PBP post-write game coverage mismatch: "
                    f"missing={sorted(set(expected_counts) - stored_ids)}, "
                    f"unexpected={sorted(stored_ids - set(expected_counts))}"
                )
            for row in stored.itertuples(index=False):
                expected_count = expected_counts[int(row.game_id)]
                if (int(row.event_count) != expected_count
                        or int(row.distinct_events) != expected_count
                        or int(row.missing_teams) != 0
                        or int(row.incorrect_teams) != 0):
                    raise RuntimeError(
                        f"PBP post-write validation failed for game {row.game_id}: "
                        f"expected={expected_count}, stored={row.event_count}, "
                        f"distinct={row.distinct_events}, "
                        f"missing_teams={row.missing_teams}, "
                        f"incorrect_teams={row.incorrect_teams}"
                    )

            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
    finally:
        con.unregister("pbp_updates")

    # ============================================================
    # SUMMARY

    # ============================================================
    games_inserted = (
        pbp_updates["game_id"]
        .nunique()
    )
    print(
        f"✅ Inserted "
        f"{len(pbp_updates):,} "
        f"play-by-play rows "
        f"for {games_inserted:,} game(s)."
    )
