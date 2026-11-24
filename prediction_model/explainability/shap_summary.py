
"""Incremental SHAP summary for NHL playoff predictions.

Reads historical daily SHAP explanations and stores the top
helpful, harmful, and absolute feature contributions per team
and snapshot.

Safeguards:
- Model-version-specific source and destination tables
- Saved model feature validation
- Source and destination schema validation
- Unique snapshot identity validation
- Team coverage validation
- Finite SHAP and probability validation
- Incremental processing using source hashes
- Detection of missing or incomplete summaries
- Atomic snapshot replacement
- Database uniqueness constraints
- Explicit full-season refresh support

The existing v6 playoff_shap_summary table is preserved.
Balanced v6 summaries are written to separate tables.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
from joblib import load

from config import MODEL_PATH


# ============================================================
# CONFIGURATION
# ============================================================

TOP_N = 10

FEATURE_SOURCE_TABLE = "team_daily_stats"

SUMMARY_COLUMNS = [
    "season_id",
    "snapshot_date",
    "game_date",
    "team_id",
    "tri_code",
    "model_version",
    "playoff_prob",
    "feature",
    "shap_value",
    "direction",
    "rank",
]

SNAPSHOT_KEYS = [
    "season_id",
    "snapshot_date",
    "team_id",
    "model_version",
]

RANKING_KEYS = SNAPSHOT_KEYS + [
    "direction",
    "rank",
]

DIRECTIONS = (
    "helpful",
    "harmful",
    "absolute",
)

# Changing the ranking algorithm or TOP_N will
# invalidate previous processing hashes.
HASH_VERSION = "shap-summary-v6-balanced-v1"


# ============================================================
# MODEL VERSION ROUTING
# ============================================================

def _validate_version(version: str) -> str:

    version = str(version)

    if not re.fullmatch(r"v[0-9]+", version):
        raise ValueError(
            f"Unsupported model version: {version}"
        )

    return version


def _tables_for_version(version: str) -> dict:

    version = _validate_version(version)

    if version == "v6":
        return {
            "source": "playoff_shap_values_v6_balanced",
            "summary": "playoff_shap_summary_v6_balanced",
            "state": "playoff_shap_summary_state_v6_balanced",
        }

    return {
        "source": f"playoff_shap_values_{version}",
        "summary": f"playoff_shap_summary_{version}",
        "state": f"playoff_shap_summary_state_{version}",
    }


def _quote(name: str) -> str:

    return '"' + name.replace('"', '""') + '"'


# ============================================================
# LOAD MODEL
# ============================================================

def _load_model_config():

    bundle = load(MODEL_PATH)

    features = list(bundle["features"])

    version = _validate_version(
        str(bundle["version"])
    )

    if not features:
        raise ValueError(
            "Saved model contains no features"
        )

    if len(features) != len(set(features)):
        raise ValueError(
            "Saved model contains duplicate features"
        )

    shap_columns = [
        f"shap_{feature}"
        for feature in features
    ]

    if len(shap_columns) != len(set(shap_columns)):
        raise ValueError(
            "SHAP output column collision"
        )

    return version, features, shap_columns


# ============================================================
# DATABASE HELPERS
# ============================================================

def _table_exists(con, table: str) -> bool:

    return bool(
        con.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = current_schema()
              AND table_name = ?
        """, [table]).fetchone()[0]
    )


def _table_schema(con, table: str) -> dict:

    rows = con.execute(
        f"DESCRIBE {_quote(table)}"
    ).fetchall()

    return {
        row[0]: str(row[1]).upper()
        for row in rows
    }


def _type_family(dtype: str) -> str:

    dtype = dtype.upper()

    if dtype.startswith("TIMESTAMP"):
        return "datetime"

    if dtype == "DATE":
        return "datetime"

    if dtype in {
        "BIGINT", "INTEGER", "SMALLINT",
        "TINYINT", "UBIGINT", "UINTEGER",
        "USMALLINT", "UTINYINT",
    }:
        return "integer"

    if (
        dtype in {"FLOAT", "DOUBLE", "REAL"}
        or dtype.startswith("DECIMAL")
    ):
        return "numeric"

    if dtype in {"VARCHAR", "TEXT", "STRING"}:
        return "text"

    return dtype


def _validate_schema(
    con,
    table: str,
    expected: dict,
    *,
    allow_extra: bool = False,
):

    if not _table_exists(con, table):
        raise RuntimeError(
            f"Required table does not exist: {table}"
        )

    actual = _table_schema(con, table)

    missing = sorted(
        set(expected) - set(actual)
    )

    extra = sorted(
        set(actual) - set(expected)
    )

    incompatible = {}

    for col, expected_family in expected.items():

        if col not in actual:
            continue

        actual_family = _type_family(
            actual[col]
        )

        compatible = (
            actual_family == expected_family
            or (
                expected_family == "numeric"
                and actual_family == "integer"
            )
        )

        if not compatible:
            incompatible[col] = {
                "expected": expected_family,
                "actual": actual[col],
            }

    if missing or incompatible or (
        extra and not allow_extra
    ):
        raise RuntimeError(
            f"Incompatible schema for {table}. "
            f"Missing: {missing}; "
            f"Extra: {extra}; "
            f"Incompatible: {incompatible}. "
            "No summary rows have been modified."
        )


# ============================================================
# EXPECTED TABLE SCHEMAS
# ============================================================

SUMMARY_SCHEMA = {
    "season_id": "integer",
    "snapshot_date": "datetime",
    "game_date": "datetime",
    "team_id": "integer",
    "tri_code": "text",
    "model_version": "text",
    "playoff_prob": "numeric",
    "feature": "text",
    "shap_value": "numeric",
    "direction": "text",
    "rank": "integer",
}

STATE_SCHEMA = {
    "season_id": "integer",
    "snapshot_date": "datetime",
    "team_id": "integer",
    "model_version": "text",
    "source_hash": "text",
    "processed_at": "datetime",
}


# ============================================================
# INITIALIZE DESTINATION TABLES
# ============================================================

def _ensure_tables(con, tables: dict):

    summary = _quote(tables["summary"])
    state = _quote(tables["state"])

    # DDL and index creation are transactional.
    # Existing tables are not dropped or recreated.

    started = False

    try:
        con.execute("BEGIN TRANSACTION")
        started = True

        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {summary} (
                season_id BIGINT,
                snapshot_date DATE,
                game_date DATE,
                team_id BIGINT,
                tri_code VARCHAR,
                model_version VARCHAR,
                playoff_prob DOUBLE,
                feature VARCHAR,
                shap_value DOUBLE,
                direction VARCHAR,
                rank BIGINT
            )
        """)

        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {state} (
                season_id BIGINT,
                snapshot_date DATE,
                team_id BIGINT,
                model_version VARCHAR,
                source_hash VARCHAR,
                processed_at TIMESTAMPTZ
            )
        """)

        _validate_schema(
            con,
            tables["summary"],
            SUMMARY_SCHEMA,
        )

        _validate_schema(
            con,
            tables["state"],
            STATE_SCHEMA,
        )

        # Validate existing data before creating indexes.

        duplicate_summary = con.execute(f"""
            SELECT COUNT(*)
            FROM (
                SELECT
                    season_id,
                    snapshot_date,
                    team_id,
                    model_version,
                    direction,
                    rank
                FROM {summary}
                GROUP BY ALL
                HAVING COUNT(*) > 1
            )
        """).fetchone()[0]

        if duplicate_summary:
            raise RuntimeError(
                f"{duplicate_summary} duplicate ranking "
                "keys found in existing summary table"
            )

        duplicate_state = con.execute(f"""
            SELECT COUNT(*)
            FROM (
                SELECT
                    season_id,
                    snapshot_date,
                    team_id,
                    model_version
                FROM {state}
                GROUP BY ALL
                HAVING COUNT(*) > 1
            )
        """).fetchone()[0]

        if duplicate_state:
            raise RuntimeError(
                f"{duplicate_state} duplicate snapshot "
                "keys found in processing state"
            )

        # Existing v6 records remain untouched.

        summary_index = _quote(
            f"{tables['summary']}_unique_rank"
        )

        state_index = _quote(
            f"{tables['state']}_unique_snapshot"
        )

        con.execute(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS
                {summary_index}
            ON {summary} (
                season_id,
                snapshot_date,
                team_id,
                model_version,
                direction,
                rank
            )
        """)

        con.execute(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS
                {state_index}
            ON {state} (
                season_id,
                snapshot_date,
                team_id,
                model_version
            )
        """)

        con.execute("COMMIT")
        started = False

    except Exception:

        if started:
            con.execute("ROLLBACK")

        raise


# ============================================================
# LOAD SOURCE SHAP SNAPSHOTS
# ============================================================

def _load_source(
    con,
    season_id: int,
    version: str,
    tables: dict,
    shap_columns: list[str],
) -> pd.DataFrame:

    source = tables["source"]

    if not _table_exists(con, source):
        raise RuntimeError(
            f"SHAP source table missing: {source}"
        )

    required = {
        "season_id",
        "snapshot_date",
        "game_date",
        "team_id",
        "tri_code",
        "model_version",
        "xgb_probability",
        *shap_columns,
    }

    existing = set(
        _table_schema(con, source)
    )

    missing = sorted(
        required - existing
    )

    if missing:
        raise RuntimeError(
            f"Missing SHAP source columns: {missing}"
        )

    df = con.execute(f"""
        SELECT *
        FROM {_quote(source)}
        WHERE season_id = ?
        ORDER BY snapshot_date, team_id
    """, [season_id]).df()

    if df.empty:
        raise RuntimeError(
            f"No SHAP snapshots for season {season_id}"
        )

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"]
    ).dt.date

    df["game_date"] = pd.to_datetime(
        df["game_date"]
    ).dt.date

    # --------------------------------------------------------
    # Validate model version
    # --------------------------------------------------------

    versions = set(
        df["model_version"]
        .dropna()
        .astype(str)
    )

    if (
        versions != {version}
        or df["model_version"].isna().any()
    ):
        raise RuntimeError(
            "SHAP source model version mismatch. "
            f"Expected {version}; found {versions}"
        )

    # --------------------------------------------------------
    # Validate snapshot identities
    # --------------------------------------------------------

    if df[SNAPSHOT_KEYS].isna().any().any():
        raise RuntimeError(
            "Null SHAP snapshot identifiers"
        )

    if df.duplicated(
        subset=SNAPSHOT_KEYS
    ).any():
        raise RuntimeError(
            "Duplicate source SHAP snapshot keys"
        )

    # --------------------------------------------------------
    # Validate metadata
    # --------------------------------------------------------

    if df[
        ["game_date", "tri_code"]
    ].isna().any().any():
        raise RuntimeError(
            "Missing SHAP snapshot metadata"
        )

    if (
        df["game_date"] > df["snapshot_date"]
    ).any():
        raise RuntimeError(
            "Future game date found in SHAP snapshot"
        )

    # --------------------------------------------------------
    # Validate probabilities
    # --------------------------------------------------------

    probabilities = pd.to_numeric(
        df["xgb_probability"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "Nonfinite playoff probabilities found"
        )

    if (
        (probabilities < 0)
        | (probabilities > 1)
    ).any():
        raise RuntimeError(
            "Playoff probabilities outside [0, 1]"
        )

    # --------------------------------------------------------
    # Validate SHAP values
    # --------------------------------------------------------

    shap_values = df[shap_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(shap_values).all():
        raise RuntimeError(
            "Missing, invalid, or nonfinite SHAP values"
        )

    print(f"   Source snapshots: {len(df):,}")
    print(f"   Source teams: {df['team_id'].nunique()}")
    print(f"   SHAP features: {len(shap_columns)}")

    return df


# ============================================================
# VALIDATE TEAM COVERAGE
# ============================================================

def _validate_team_coverage(
    con,
    df: pd.DataFrame,
    season_id: int,
):

    if not _table_exists(
        con,
        FEATURE_SOURCE_TABLE
    ):
        raise RuntimeError(
            "Team coverage validation source missing: "
            f"{FEATURE_SOURCE_TABLE}"
        )

    # Match the eligibility rule in daily_feature_predictions:
    # A team is expected once it has completed its first game.

    expected = con.execute(f"""
        WITH first_games AS (
            SELECT
                team_id,
                MIN(CAST(game_date AS DATE)) AS first_date
            FROM {_quote(FEATURE_SOURCE_TABLE)}
            WHERE season_id = ?
            GROUP BY team_id
        )
        SELECT
            snapshot_date,
            COUNT(*) AS expected_teams
        FROM (
            SELECT DISTINCT snapshot_date
            FROM (
                SELECT
                    CAST(game_date AS DATE)
                        AS snapshot_date
                FROM {_quote(FEATURE_SOURCE_TABLE)}
                WHERE season_id = ?
            )
        ) dates
        JOIN first_games
          ON first_games.first_date
             <= dates.snapshot_date
        GROUP BY snapshot_date
    """, [season_id, season_id]).df()

    actual = (
        df.groupby("snapshot_date")
        .agg(
            actual_teams=("team_id", "nunique")
        )
        .reset_index()
    )

    if not expected.empty:
        expected["snapshot_date"] = pd.to_datetime(
            expected["snapshot_date"]
        ).dt.date

    coverage = actual.merge(
        expected,
        on="snapshot_date",
        how="left",
        validate="one_to_one",
    )

    invalid = coverage[
        coverage["expected_teams"].isna()
        | (
            coverage["actual_teams"]
            != coverage["expected_teams"]
        )
    ]

    if not invalid.empty:
        raise RuntimeError(
            "Incomplete SHAP source team coverage:\n"
            + invalid.to_string(index=False)
        )

    print("   ✓ Source team coverage validated")


# ============================================================
# CALCULATE SOURCE HASHES
# ============================================================

def _calculate_hashes(
    df: pd.DataFrame,
    shap_columns: list[str],
) -> pd.DataFrame:

    df = df.copy()

    hash_columns = [
        "game_date",
        "tri_code",
        "model_version",
        "xgb_probability",
        *shap_columns,
    ]

    def hash_row(row):

        values = [
            HASH_VERSION,
            str(TOP_N),
        ]

        for value in row:

            if pd.isna(value):
                values.append("<NULL>")

            elif isinstance(
                value,
                (float, np.floating),
            ):
                values.append(
                    float(value).hex()
                )

            elif isinstance(
                value,
                (datetime, pd.Timestamp),
            ):
                values.append(
                    value.isoformat()
                )

            else:
                values.append(str(value))

        payload = "\x1f".join(values)

        return hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()

    df["source_hash"] = [
        hash_row(row)
        for row in df[
            hash_columns
        ].itertuples(index=False, name=None)
    ]

    return df


# ============================================================
# IDENTIFY SNAPSHOTS TO PROCESS
# ============================================================

def _identify_updates(
    con,
    df: pd.DataFrame,
    tables: dict,
    season_id: int,
    version: str,
    refresh_existing: bool,
) -> pd.DataFrame:

    if refresh_existing:
        print("   Full season refresh requested")
        return df.copy()

    state = con.execute(f"""
        SELECT
            season_id,
            snapshot_date,
            team_id,
            model_version,
            source_hash
        FROM {_quote(tables["state"])}
        WHERE season_id = ?
          AND model_version = ?
    """, [season_id, version]).df()

    if not state.empty:
        state["snapshot_date"] = pd.to_datetime(
            state["snapshot_date"]
        ).dt.date

    merged = df.merge(
        state.rename(
            columns={
                "source_hash": "previous_hash"
            }
        ),
        on=SNAPSHOT_KEYS,
        how="left",
        validate="one_to_one",
    )

    # Existing summary coverage.
    counts = con.execute(f"""
        SELECT
            season_id,
            snapshot_date,
            team_id,
            model_version,
            direction,
            COUNT(*) AS row_count
        FROM {_quote(tables["summary"])}
        WHERE season_id = ?
          AND model_version = ?
        GROUP BY
            season_id,
            snapshot_date,
            team_id,
            model_version,
            direction
    """, [season_id, version]).df()

    if not counts.empty:
        counts["snapshot_date"] = pd.to_datetime(
            counts["snapshot_date"]
        ).dt.date

        pivot = counts.pivot_table(
            index=SNAPSHOT_KEYS,
            columns="direction",
            values="row_count",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()

    else:
        pivot = pd.DataFrame(
            columns=SNAPSHOT_KEYS
        )

    merged = merged.merge(
        pivot,
        on=SNAPSHOT_KEYS,
        how="left",
        validate="one_to_one",
    )

    for direction in DIRECTIONS:
        if direction not in merged.columns:
            merged[direction] = 0

        merged[direction] = (
            merged[direction]
            .fillna(0)
            .astype(int)
        )

    # Expected helpful/harmful counts depend on
    # the signs of the available SHAP values.

    shap_columns = [
        col for col in df.columns
        if col.startswith("shap_")
    ]

    values = merged[
        shap_columns
    ].to_numpy(dtype=float)

    expected_helpful = np.minimum(
        (values > 0).sum(axis=1),
        TOP_N,
    )

    expected_harmful = np.minimum(
        (values < 0).sum(axis=1),
        TOP_N,
    )

    expected_absolute = np.minimum(
        values.shape[1],
        TOP_N,
    )

    new_mask = merged[
        "previous_hash"
    ].isna()

    changed_mask = (
        merged["previous_hash"].notna()
        & (
            merged["source_hash"]
            != merged["previous_hash"]
        )
    )

    incomplete_mask = (
        (merged["helpful"] != expected_helpful)
        | (merged["harmful"] != expected_harmful)
        | (merged["absolute"] != expected_absolute)
    )

    process_mask = (
        new_mask
        | changed_mask
        | incomplete_mask
    )

    print(f"   New snapshots: {new_mask.sum()}")
    print(f"   Changed snapshots: {changed_mask.sum()}")
    print(
        "   Missing/incomplete summaries:",
        incomplete_mask.sum()
    )
    print(
        f"   Snapshots to process: {process_mask.sum()}"
    )

    return merged.loc[
        process_mask,
        df.columns,
    ].copy()


# ============================================================
# GENERATE SHAP RANKINGS
# ============================================================

def explain_all_rows(
    df_present: pd.DataFrame,
    shap_cols: list[str],
    n: int = TOP_N,
) -> pd.DataFrame:

    results = []

    for _, row in df_present.iterrows():

        shap_values = pd.Series(
            row[shap_cols].to_numpy(dtype=float),
            index=[
                col.removeprefix("shap_")
                for col in shap_cols
            ],
        )

        helpful = (
            shap_values[shap_values > 0]
            .nlargest(n)
        )

        harmful = (
            shap_values[shap_values < 0]
            .nsmallest(n)
        )

        absolute = shap_values.reindex(
            shap_values.abs()
            .nlargest(n)
            .index
        )

        base_info = {
            "season_id": row["season_id"],
            "snapshot_date": row["snapshot_date"],
            "game_date": row["game_date"],
            "team_id": row["team_id"],
            "tri_code": row["tri_code"],
            "model_version": row["model_version"],
            "playoff_prob": row["xgb_probability"],
        }

        for direction, values in [
            ("helpful", helpful),
            ("harmful", harmful),
            ("absolute", absolute),
        ]:

            for rank, (feature, value) in enumerate(
                values.items(),
                start=1,
            ):

                results.append({
                    **base_info,
                    "feature": feature,
                    "shap_value": float(value),
                    "direction": direction,
                    "rank": rank,
                })

    return pd.DataFrame(
        results,
        columns=SUMMARY_COLUMNS,
    )


# ============================================================
# VALIDATE GENERATED SUMMARY
# ============================================================

def _validate_generated(
    updates: pd.DataFrame,
    summary: pd.DataFrame,
):

    if summary.empty:
        raise RuntimeError(
            "SHAP summary generation returned no rows"
        )

    if summary[
        RANKING_KEYS
    ].isna().any().any():
        raise RuntimeError(
            "Null SHAP ranking identifiers"
        )

    if summary.duplicated(
        subset=RANKING_KEYS
    ).any():
        raise RuntimeError(
            "Duplicate generated SHAP rankings"
        )

    if not summary[
        "direction"
    ].isin(DIRECTIONS).all():
        raise RuntimeError(
            "Invalid SHAP direction"
        )

    if not summary[
        "rank"
    ].between(1, TOP_N).all():
        raise RuntimeError(
            "Invalid SHAP ranking"
        )

    if not np.isfinite(
        summary["shap_value"].to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Nonfinite generated SHAP values"
        )

    unique_snapshots = (
        summary[SNAPSHOT_KEYS]
        .drop_duplicates()
        .shape[0]
    )

    if unique_snapshots != len(updates):
        raise RuntimeError(
            "Generated summary snapshot mismatch: "
            f"expected {len(updates)}, "
            f"got {unique_snapshots}"
        )

    # Verify direction and rank sequences.
    groups = summary.groupby(
        SNAPSHOT_KEYS + ["direction"]
    )

    for _, group in groups:

        ranks = sorted(
            group["rank"].tolist()
        )

        expected = list(
            range(1, len(group) + 1)
        )

        if ranks != expected:
            raise RuntimeError(
                "Nonsequential SHAP rankings"
            )

    print("   ✓ Generated rankings validated")


# ============================================================
# WRITE SNAPSHOT UPDATES
# ============================================================

def _write_updates(
    con,
    tables: dict,
    updates: pd.DataFrame,
    summary: pd.DataFrame,
    season_id: int,
    version: str,
    refresh_existing: bool,
):

    state_updates = updates[
        SNAPSHOT_KEYS + ["source_hash"]
    ].copy()

    state_updates["processed_at"] = (
        datetime.now(timezone.utc)
    )

    con.register(
        "shap_summary_updates",
        summary,
    )

    con.register(
        "shap_state_updates",
        state_updates,
    )

    summary_table = _quote(
        tables["summary"]
    )

    state_table = _quote(
        tables["state"]
    )

    started = False

    try:

        con.execute("BEGIN TRANSACTION")
        started = True

        # Revalidate destination schemas immediately
        # before modifying existing records.

        _validate_schema(
            con,
            tables["summary"],
            SUMMARY_SCHEMA,
        )

        _validate_schema(
            con,
            tables["state"],
            STATE_SCHEMA,
        )

        if refresh_existing:

            con.execute(f"""
                DELETE FROM {summary_table}
                WHERE season_id = ?
                  AND model_version = ?
            """, [season_id, version])

            con.execute(f"""
                DELETE FROM {state_table}
                WHERE season_id = ?
                  AND model_version = ?
            """, [season_id, version])

        else:

            # Replace only affected snapshots.

            for table in (
                summary_table,
                state_table,
            ):

                con.execute(f"""
                    DELETE FROM {table} AS t
                    WHERE EXISTS (
                        SELECT 1
                        FROM shap_state_updates AS u
                        WHERE
                            t.season_id = u.season_id
                            AND t.snapshot_date =
                                u.snapshot_date
                            AND t.team_id = u.team_id
                            AND t.model_version =
                                u.model_version
                    )
                """)

        # Insert rankings using explicit column names.

        columns = ", ".join(
            _quote(col)
            for col in SUMMARY_COLUMNS
        )

        con.execute(f"""
            INSERT INTO {summary_table}
                ({columns})
            SELECT {columns}
            FROM shap_summary_updates
        """)

        state_columns = [
            *SNAPSHOT_KEYS,
            "source_hash",
            "processed_at",
        ]

        state_sql = ", ".join(
            _quote(col)
            for col in state_columns
        )

        con.execute(f"""
            INSERT INTO {state_table}
                ({state_sql})
            SELECT {state_sql}
            FROM shap_state_updates
        """)

        # Verify the number of inserted rankings.

        saved_rows = con.execute(f"""
            SELECT COUNT(*)
            FROM {summary_table} AS s
            INNER JOIN shap_state_updates AS u
                ON s.season_id = u.season_id
                AND s.snapshot_date = u.snapshot_date
                AND s.team_id = u.team_id
                AND s.model_version = u.model_version
        """).fetchone()[0]

        if saved_rows != len(summary):
            raise RuntimeError(
                "Saved SHAP row count mismatch: "
                f"expected {len(summary)}, "
                f"got {saved_rows}"
            )

        saved_state = con.execute(f"""
            SELECT COUNT(*)
            FROM {state_table} AS s
            INNER JOIN shap_state_updates AS u
                ON s.season_id = u.season_id
                AND s.snapshot_date = u.snapshot_date
                AND s.team_id = u.team_id
                AND s.model_version = u.model_version
        """).fetchone()[0]

        if saved_state != len(updates):
            raise RuntimeError(
                "Saved processing state mismatch"
            )

        con.execute("COMMIT")
        started = False

    except Exception:

        if started:
            con.execute("ROLLBACK")

        raise

    finally:

        con.unregister(
            "shap_summary_updates"
        )

        con.unregister(
            "shap_state_updates"
        )


# ============================================================
# MAIN PIPELINE ENTRY POINT
# ============================================================

def update_shap_summary(
    *,
    con: duckdb.DuckDBPyConnection,
    season_id: int,
    refresh_existing: bool = False,
) -> None:

    print("\n" + "=" * 60)
    print("📊 PLAYOFF SHAP SUMMARY")
    print(f"   Season: {season_id}")
    print(
        "   Mode:",
        "FULL REFRESH"
        if refresh_existing
        else "INCREMENTAL",
    )
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load saved model configuration
    # --------------------------------------------------------

    version, features, shap_cols = (
        _load_model_config()
    )

    tables = _tables_for_version(
        version
    )

    print(f"   Model version: {version}")
    print(f"   Model features: {len(features)}")
    print(f"   Source: {tables['source']}")
    print(f"   Target: {tables['summary']}")

    # --------------------------------------------------------
    # 2. Load and validate source
    # --------------------------------------------------------

    df = _load_source(
        con,
        season_id,
        version,
        tables,
        shap_cols,
    )

    # --------------------------------------------------------
    # 3. Validate source team coverage
    # --------------------------------------------------------

    _validate_team_coverage(
        con,
        df,
        season_id,
    )

    # --------------------------------------------------------
    # 4. Calculate source hashes
    # --------------------------------------------------------

    df = _calculate_hashes(
        df,
        shap_cols,
    )

    # --------------------------------------------------------
    # 5. Initialize and validate destination tables
    # --------------------------------------------------------

    _ensure_tables(
        con,
        tables,
    )

    # --------------------------------------------------------
    # 6. Identify new or changed snapshots
    # --------------------------------------------------------

    updates = _identify_updates(
        con,
        df,
        tables,
        season_id,
        version,
        refresh_existing,
    )

    if updates.empty:

        print("\n✅ SHAP SUMMARY ALREADY CURRENT")
        print("   No snapshots require processing.")
        print("=" * 60)

        return

    # --------------------------------------------------------
    # 7. Generate SHAP rankings
    # --------------------------------------------------------

    summary = explain_all_rows(
        updates,
        shap_cols,
        n=TOP_N,
    )

    # --------------------------------------------------------
    # 8. Validate generated rankings
    # --------------------------------------------------------

    _validate_generated(
        updates,
        summary,
    )

    print(
        f"   Generated rankings: {len(summary):,}"
    )

    # --------------------------------------------------------
    # 9. Save transactionally
    # --------------------------------------------------------

    _write_updates(
        con,
        tables,
        updates,
        summary,
        season_id,
        version,
        refresh_existing,
    )

    # --------------------------------------------------------
    # 10. Final reporting
    # --------------------------------------------------------

    totals = con.execute(f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT team_id) AS teams,
            COUNT(DISTINCT (
                team_id,
                snapshot_date
            )) AS snapshots
        FROM {_quote(tables["summary"])}
        WHERE season_id = ?
          AND model_version = ?
    """, [season_id, version]).fetchone()

    total_rows, teams, snapshots = totals

    print("\n✅ PLAYOFF SHAP SUMMARY COMPLETE")
    print(f"   Model: {version}")
    print(f"   Updated snapshots: {len(updates):,}")
    print(f"   Rankings written: {len(summary):,}")
    print(f"   Season teams: {teams}")
    print(f"   Season snapshots: {snapshots:,}")
    print(f"   Total stored rankings: {total_rows:,}")
    print("=" * 60)
