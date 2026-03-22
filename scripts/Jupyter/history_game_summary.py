import requests
import pandas as pd
import pytz
import time
import random
import duckdb
from datetime import date

pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)
DB_PATH = r"C:\Users\kjchr\Desktop\hockey\new_hockey_env\API data\duckdb\nhl_data.duckdb"
con = duckdb.connect(DB_PATH)
print("Connected:", DB_PATH)
def safe_get(url, retries=5, sleep_base=0.15):
    wait = 1.0
    for _ in range(retries):
        r = requests.get(url)
        if r.status_code == 200:
            time.sleep(sleep_base + random.uniform(0, sleep_base))
            return r
        if r.status_code == 429:
            time.sleep(wait)
            wait *= 2
            continue
        return None
    return None
season_start_dates = {
    "20132014": "2013-10-01",
    "20142015": "2014-10-08",
    "20152016": "2015-10-07",
    "20162017": "2016-10-12",
    "20172018": "2017-10-04",
    "20182019": "2018-10-04",
    "20192020": "2019-10-02",
    "20202021": "2021-01-13",
    "20212022": "2021-10-12",
    "20222023": "2022-10-11",
    "20232024": "2023-10-11",
    "20242025": "2024-10-04",
    "20252026": "2025-10-07"
}

season_end_dates = {
    "20132014": "2014-04-13",
    "20142015": "2015-04-11",
    "20152016": "2016-04-10",
    "20162017": "2017-04-09",
    "20172018": "2018-04-08",
    "20182019": "2019-04-06",
    "20192020": "2020-03-12",
    "20202021": "2021-05-19",
    "20212022": "2022-04-29",
    "20222023": "2023-04-13",
    "20232024": "2024-04-18",
    "20242025": "2025-04-17",
    "20252026": "2026-04-16"
}


def fetch_all_season_games(season_start_dates, season_end_dates):

    base_url = "https://api-web.nhle.com/v1/schedule/"
    eastern = pytz.timezone("America/New_York")
    all_games_raw = []

    for season_id in season_start_dates:
        start_date = pd.to_datetime(season_start_dates[season_id])
        end_date = pd.to_datetime(season_end_dates[season_id])

        print(f"\n📅 Processing season {season_id}")

        for d in pd.date_range(start=start_date, end=end_date, freq="D"):
            d_str = d.strftime("%Y-%m-%d")
            r = safe_get(f"{base_url}{d_str}")

            if r is None:
                continue

            data = r.json()
            if "gameWeek" not in data:
                continue

            for week in data["gameWeek"]:
                if week.get("numberOfGames", 0) == 0:
                    continue
                for game in week.get("games", []):
                    all_games_raw.append(game)

        print(f"✔ Completed season {season_id}")

    # --------------------------------------------------
    # Normalize
    # --------------------------------------------------
    df = pd.json_normalize(all_games_raw)

    # ✅ regular season only
    df = df[df["gameType"] == 2]

    df["game_id"] = df["id"].astype(str)
    df["season_id"] = df["season"].astype(str)

    df["link"] = (
        "https://api-web.nhle.com/v1/gamecenter/" + df["game_id"] + "/play-by-play"
    )

    df["startTimeUTC"] = pd.to_datetime(df["startTimeUTC"], utc=True)
    df["game_date"] = (
        df["startTimeUTC"]
        .dt.tz_convert(eastern)
        .dt.strftime("%Y-%m-%d")
    )

    df = df[
        [
            "game_id",
            "season_id",
            "awayTeam.id",
            "awayTeam.abbrev",
            "homeTeam.id",
            "homeTeam.abbrev",
            "awayTeam.score",
            "homeTeam.score",
            "gameState",
            "link",
            "game_date",
        ]
    ].drop_duplicates(subset=["game_id"])

    df = df.rename(columns={
        "awayTeam.id": "away_team",
        "awayTeam.abbrev": "away_tri_code",
        "homeTeam.id": "home_team",
        "homeTeam.abbrev": "home_tri_code",
    })

    df.sort_values("game_id", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"\n📦 Total games collected: {len(df):,}")
    return df
game_summary_df = fetch_all_season_games(
    season_start_dates,
    season_end_dates
)
con.execute("DROP TABLE IF EXISTS all_game_summary")
con.execute("""
CREATE TABLE IF NOT EXISTS all_game_summary AS
SELECT * FROM game_summary_df LIMIT 0
""")

con.execute("""
INSERT INTO all_game_summary
SELECT *
FROM game_summary_df g
WHERE NOT EXISTS (
    SELECT 1
    FROM all_game_summary a
    WHERE a.game_id = g.game_id
)
""")

print("✅ all_game_summary populated correctly.")
