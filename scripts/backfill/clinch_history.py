# ============================================================
# Rebuild 2020-2021 Clinch History
# Validation only — does NOT update DuckDB
# ============================================================

import duckdb
import requests
import pandas as pd
import time


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

DB_PATH = (
    r"C:\Users\kjchr\Desktop\hockey\new_hockey_env\API data\data\duckdb\nhl_data.duckdb"

)

BASE_URL = "https://api-web.nhle.com/v1/standings/"

SEASON_ID = 20202021
TOTAL_GAMES = 56


# ============================================================
# Helper: Get 4th-place division cutoff
# ============================================================

def get_division_cutoff_points(standings, division):

    division_teams = [
        team
        for team in standings
        if team.get("divisionName") == division
    ]

    cutoff = next(
        (
            team
            for team in division_teams
            if team.get("divisionSequence") == 4
        ),
        None,
    )

    return cutoff.get("points") if cutoff else None


# ============================================================
# Rebuild 2020-2021 Clinch History
# ============================================================

def rebuild_20202021_clinch_history(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:

    print("\n==============================")
    print("▶ REBUILDING 2020-2021")
    print("==============================")

    # --------------------------------------------------------
    # 1. Get season dates
    # --------------------------------------------------------

    dates = con.execute("""
        SELECT
            regular_start,
            regular_end
        FROM important_dates
        WHERE season_id = ?
    """, [SEASON_ID]).fetchone()

    if not dates:
        raise RuntimeError(
            f"Season {SEASON_ID} not found in important_dates."
        )

    start_date = pd.to_datetime(dates[0])
    end_date = pd.to_datetime(dates[1])

    print(
        f"📅 Season dates: "
        f"{start_date.date()} → {end_date.date()}"
    )

    team_status = {}

    # --------------------------------------------------------
    # 2. Walk through every day of the season
    # --------------------------------------------------------

    for check_date in pd.date_range(
        start=start_date,
        end=end_date,
        freq="D",
    ):

        date_string = check_date.strftime("%Y-%m-%d")
        url = f"{BASE_URL}{date_string}"

        try:
            response = requests.get(
                url,
                timeout=30,
            )

        except requests.RequestException as e:
            print(
                f"⚠️ {date_string}: request failed: {e}"
            )
            continue

        if response.status_code != 200:
            print(
                f"⚠️ {date_string}: "
                f"HTTP {response.status_code}"
            )
            continue

        standings = response.json().get(
            "standings",
            [],
        )

        if not standings:
            continue

        # ----------------------------------------------------
        # 3. Determine the 4th-place cutoff in each division
        # ----------------------------------------------------

        divisions = {
            team.get("divisionName")
            for team in standings
            if team.get("divisionName")
        }

        division_cutoffs = {
            division: get_division_cutoff_points(
                standings,
                division,
            )
            for division in divisions
        }

        # ----------------------------------------------------
        # 4. Evaluate each team
        # ----------------------------------------------------

        for team in standings:

            tri_code = (
                team.get("teamAbbrev", {})
                .get("default")
            )

            if not tri_code:
                continue

            games_played = team.get(
                "gamesPlayed",
                0,
            )

            points = team.get(
                "points",
                0,
            )

            division = team.get(
                "divisionName"
            )

            clinch_flag = team.get(
                "clinchIndicator"
            )

            # ------------------------------------------------
            # Initialize team
            # ------------------------------------------------

            if tri_code not in team_status:

                team_status[tri_code] = {
                    "clinch_indicator": False,
                    "tri_code": tri_code,
                    "clinch_type": None,
                    "clinch_date": None,
                    "season_id": SEASON_ID,
                }

            # ------------------------------------------------
            # Already resolved
            # ------------------------------------------------

            if (
                team_status[tri_code]["clinch_type"]
                is not None
            ):
                continue

            # ------------------------------------------------
            # 5. NHL API clinch indicator
            # ------------------------------------------------

            if clinch_flag:

                team_status[tri_code][
                    "clinch_indicator"
                ] = True

                team_status[tri_code][
                    "clinch_type"
                ] = clinch_flag

                team_status[tri_code][
                    "clinch_date"
                ] = check_date.date()

                print(
                    f"🏒 {check_date.date()} | "
                    f"{tri_code} | "
                    f"CLINCHED ({clinch_flag})"
                )

                continue

            # ------------------------------------------------
            # 6. Mathematical elimination
            # ------------------------------------------------

            remaining_games = (
                TOTAL_GAMES - games_played
            )

            max_possible_points = (
                points
                + remaining_games * 2
            )

            cutoff = division_cutoffs.get(
                division
            )

            if (
                cutoff is not None
                and max_possible_points < cutoff
            ):

                team_status[tri_code][
                    "clinch_indicator"
                ] = True

                team_status[tri_code][
                    "clinch_type"
                ] = "e"

                team_status[tri_code][
                    "clinch_date"
                ] = check_date.date()

                print(
                    f"❌ {check_date.date()} | "
                    f"{tri_code} | "
                    f"ELIMINATED | "
                    f"max={max_possible_points}, "
                    f"cutoff={cutoff}"
                )

        time.sleep(0.15)

    # --------------------------------------------------------
    # 7. Build DataFrame
    # --------------------------------------------------------

    clinch_20202021 = pd.DataFrame(
        team_status.values()
    )

    if clinch_20202021.empty:
        raise RuntimeError(
            "No clinch data was generated."
        )

    clinch_20202021 = (
        clinch_20202021
        .sort_values("tri_code")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 8. Validate
    # --------------------------------------------------------

    print("\n==============================")
    print("VALIDATION")
    print("==============================")

    total_teams = (
        clinch_20202021[
            "tri_code"
        ].nunique()
    )

    playoff_count = (
        clinch_20202021[
            "clinch_type"
        ]
        .isin(["x", "y"])
        .sum()
    )

    eliminated_count = (
        clinch_20202021[
            "clinch_type"
        ]
        .eq("e")
        .sum()
    )

    unresolved = (
        clinch_20202021[
            clinch_20202021[
                "clinch_type"
            ].isna()
        ]
    )

    print(f"Total teams: {total_teams}")
    print(f"Playoff teams: {playoff_count}")
    print(
        f"Eliminated teams: "
        f"{eliminated_count}"
    )
    print(
        f"Unresolved teams: "
        f"{len(unresolved)}"
    )

    print("\nClinch types:")

    print(
        clinch_20202021[
            "clinch_type"
        ].value_counts(dropna=False)
    )

    # --------------------------------------------------------
    # 9. Show complete result
    # --------------------------------------------------------

    print("\n==============================")
    print("2020-2021 CLINCH HISTORY")
    print("==============================\n")

    print(
        clinch_20202021[
            [
                "clinch_indicator",
                "tri_code",
                "clinch_type",
                "clinch_date",
                "season_id",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # 10. Warn if result is not what we expect
    # --------------------------------------------------------

    if total_teams != 31:

        print(
            f"\n⚠️ Expected 31 teams, "
            f"but found {total_teams}."
        )

    if playoff_count != 16:

        print(
            f"\n⚠️ Expected 16 playoff teams, "
            f"but found {playoff_count}."
        )

    if eliminated_count != 15:

        print(
            f"\n⚠️ Expected 15 eliminated teams, "
            f"but found {eliminated_count}."
        )

    if not unresolved.empty:

        print("\n⚠️ UNRESOLVED TEAMS:")

        print(
            unresolved[
                [
                    "tri_code",
                    "clinch_type",
                    "clinch_date",
                ]
            ].to_string(index=False)
        )

    return clinch_20202021


# ============================================================
# Execute
# ============================================================
# ============================================================
# Execute
# ============================================================

if __name__ == "__main__":

    print("🔌 Opening DuckDB connection...")

    con = duckdb.connect(DB_PATH)

    try:

        # ----------------------------------------------------
        # Rebuild
        # ----------------------------------------------------

        clinch_20202021 = (
            rebuild_20202021_clinch_history(con)
        )

        # ----------------------------------------------------
        # Final validation before database update
        # ----------------------------------------------------

        total_teams = clinch_20202021["tri_code"].nunique()

        playoff_count = (
            clinch_20202021["clinch_type"]
            .isin(["x", "y"])
            .sum()
        )

        eliminated_count = (
            clinch_20202021["clinch_type"]
            .eq("e")
            .sum()
        )

        unresolved_count = (
            clinch_20202021["clinch_type"]
            .isna()
            .sum()
        )

        if not (
            total_teams == 31
            and playoff_count == 16
            and eliminated_count == 15
            and unresolved_count == 0
        ):
            raise RuntimeError(
                "Validation failed. "
                "DuckDB was NOT modified."
            )

        print(
            "\n✅ Validation passed. "
            "Replacing 2020-2021 in all_clinch_history..."
        )

        # ----------------------------------------------------
        # Register rebuilt DataFrame
        # ----------------------------------------------------

        con.register(
            "clinch_20202021_update",
            clinch_20202021,
        )

        # ----------------------------------------------------
        # Replace season inside a transaction
        # ----------------------------------------------------

        con.execute("BEGIN TRANSACTION")

        try:

            con.execute("""
                DELETE FROM all_clinch_history
                WHERE season_id = 20202021
            """)

            con.execute("""
                INSERT INTO all_clinch_history (
                    clinch_indicator,
                    tri_code,
                    clinch_type,
                    clinch_date,
                    season_id
                )
                SELECT
                    clinch_indicator,
                    tri_code,
                    clinch_type,
                    clinch_date,
                    season_id
                FROM clinch_20202021_update
            """)

            con.execute("COMMIT")

        except Exception:

            con.execute("ROLLBACK")
            raise

        # ----------------------------------------------------
        # Verify actual DuckDB table
        # ----------------------------------------------------

        check = con.execute("""
            SELECT
                clinch_type,
                COUNT(*) AS teams
            FROM all_clinch_history
            WHERE season_id = 20202021
            GROUP BY clinch_type
            ORDER BY clinch_type
        """).df()

        print("\n==============================")
        print("DUCKDB CHECK")
        print("==============================")

        print(check.to_string(index=False))

        print(
            "\n✅ 2020-2021 successfully replaced "
            "in all_clinch_history."
        )

    finally:

        con.close()

        print(
            "🔒 DuckDB connection closed."
        )