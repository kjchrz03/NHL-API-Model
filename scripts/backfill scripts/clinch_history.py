# ============================================================
# Rebuild all_clinch_history using important_dates (DuckDB)
# Seasons: 2000–2001 → 2012–2013
# ============================================================

import requests
import pandas as pd
import time
import duckdb

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DB_PATH = r"C:\Users\kjchr\Desktop\hockey\new_hockey_env\API data\duckdb\nhl_data.duckdb"
BASE_URL = "https://api-web.nhle.com/v1/standings/"
TOTAL_GAMES = 82

# ------------------------------------------------------------
# Helper: get conference cutoff points
# ------------------------------------------------------------
def get_cutoff_points(standings, conference):
    conf_teams = [t for t in standings if t["conferenceName"] == conference]
    cutoff = next((t for t in conf_teams if t["conferenceSequence"] == 8), None)
    return cutoff["points"] if cutoff else None


# ------------------------------------------------------------
# Main rebuild function
# ------------------------------------------------------------
def rebuild_all_clinch_history():

    con = duckdb.connect(DB_PATH)

    # --------------------------------------------------------
    # Pull seasons from important_dates
    # --------------------------------------------------------
    seasons = con.execute("""
        SELECT
            season_id,
            regular_start,
            regular_end
        FROM important_dates
        WHERE season_id BETWEEN 20002001 AND 20122013
        ORDER BY season_id
    """).df()

    if seasons.empty:
        raise RuntimeError("No seasons returned from important_dates")

    print(f"📅 Rebuilding clinch history for {len(seasons)} seasons")

    all_rows = []

    # --------------------------------------------------------
    # Loop seasons
    # --------------------------------------------------------
    for _, row in seasons.iterrows():
        season_id = int(row["season_id"])
        start_date = pd.to_datetime(row["regular_start"])
        end_date = pd.to_datetime(row["regular_end"])

        print(f"\n▶ Season {season_id}: {start_date.date()} → {end_date.date()}")

        team_status = {}

        for check_date in pd.date_range(start=start_date, end=end_date, freq="D"):
            url = f"{BASE_URL}{check_date.strftime('%Y-%m-%d')}"

            r = requests.get(url)
            if r.status_code != 200:
                continue

            standings = r.json().get("standings", [])
            if not standings:
                continue

            # Compute conference cutlines
            east_cutoff = get_cutoff_points(standings, "Eastern")
            west_cutoff = get_cutoff_points(standings, "Western")

            for team in standings:
                tri_code = team.get("teamAbbrev", {}).get("default")
                if not tri_code:
                    continue

                games_played = team.get("gamesPlayed", 0)
                points = team.get("points", 0)
                conference = team.get("conferenceName")
                clinch_flag = team.get("clinchIndicator")

                if tri_code not in team_status:
                    team_status[tri_code] = {
                        "tri_code": tri_code,
                        "season_id": season_id,
                        "clinch_indicator": False,
                        "clinch_type": None,
                        "clinch_date": None,
                    }

                # Already resolved → skip
                if team_status[tri_code]["clinch_type"] is not None:
                    continue

                # ----------------------------------
                # API clinch
                # ----------------------------------
                if clinch_flag:
                    team_status[tri_code]["clinch_indicator"] = True
                    team_status[tri_code]["clinch_type"] = clinch_flag
                    team_status[tri_code]["clinch_date"] = check_date.date()
                    continue

                # ----------------------------------
                # Elimination logic
                # ----------------------------------
                remaining_games = TOTAL_GAMES - games_played
                max_possible_points = points + (remaining_games * 2)

                cutoff = east_cutoff if conference == "Eastern" else west_cutoff
                if cutoff is not None and max_possible_points < cutoff:
                    team_status[tri_code]["clinch_indicator"] = True
                    team_status[tri_code]["clinch_type"] = "e"
                    team_status[tri_code]["clinch_date"] = check_date.date()

            time.sleep(0.15)

        # --------------------------------------
        # Force unresolved teams → eliminated at season end
        # --------------------------------------
        for tri_code, data in team_status.items():
            if data["clinch_type"] is None:
                data["clinch_indicator"] = True
                data["clinch_type"] = "e"
                data["clinch_date"] = end_date.date()

            all_rows.append(data)

    clinch_df = pd.DataFrame(all_rows)
    print(clinch_df.head())
    # --------------------------------------------------------
    # Write to DuckDB (replace / append as you prefer)
    # --------------------------------------------------------
    con.execute("DELETE FROM all_clinch_history WHERE season_id BETWEEN 20002001 AND 20122013")
    con.register("clinch_updates", clinch_df)

    con.execute("""
        INSERT INTO all_clinch_history
        SELECT
            clinch_indicator,
            tri_code,
            clinch_type,
            clinch_date,
            season_id
        FROM clinch_updates
    """)

    print(f"\n✅ Inserted {len(clinch_df):,} clinch history rows")

    con.close()


# ------------------------------------------------------------
# Execute
# ------------------------------------------------------------
if __name__ == "__main__":
    rebuild_all_clinch_history()
