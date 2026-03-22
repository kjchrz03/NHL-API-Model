# ============================================================
# Historical Game Summary Backfill
# Seasons: 2000–2001 → 2012–2013
# Source: api-web.nhle.com (daily schedule)
# Strategy: date loop + hard dedupe
# ============================================================

import requests
import pandas as pd
import pytz
import duckdb
import time

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DB_PATH = r"C:\Users\kjchr\Desktop\hockey\new_hockey_env\API data\duckdb\nhl_data.duckdb"
BASE_URL = "https://api-web.nhle.com/v1/schedule/"
SLEEP_SECONDS = 0.15  # API-friendly

# ------------------------------------------------------------
# Fetch full season safely
# ------------------------------------------------------------
def fetch_season_games(season_id, season_start, season_end):

    all_games = []
    seen_game_ids = set()

    for d in pd.date_range(season_start, season_end, freq="D"):
        url = f"{BASE_URL}{d.strftime('%Y-%m-%d')}"

        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                continue

            data = r.json()

            for week in data.get("gameWeek", []):
                for game in week.get("games", []):

                    if game.get("gameType") != 2:
                        continue

                    game_id = game.get("id")
                    if game_id in seen_game_ids:
                        continue

                    seen_game_ids.add(game_id)
                    all_games.append(game)

        except Exception as e:
            print(f"⚠ {season_id} {d.date()} failed: {e}")

        time.sleep(SLEEP_SECONDS)

    if not all_games:
        return pd.DataFrame()

    df = pd.json_normalize(all_games)

    # --------------------------------------------------------
    # Normalize schema
    # --------------------------------------------------------
    df["game_id"] = pd.to_numeric(df["id"], errors="raise").astype("int64")
    df = df.drop_duplicates(subset="game_id")

    df = df.rename(columns={
        "awayTeam.id": "away_team",
        "awayTeam.abbrev": "away_tri_code",
        "homeTeam.id": "home_team",
        "homeTeam.abbrev": "home_tri_code",
    })

    df["season_id"] = season_id

    df["link"] = (
        "https://api-web.nhle.com/v1/gamecenter/"
        + df["game_id"].astype(str)
        + "/play-by-play"
    )

    eastern = pytz.timezone("America/New_York")
    df["startTimeUTC"] = pd.to_datetime(df["startTimeUTC"], utc=True)
    df["game_date"] = df["startTimeUTC"].dt.tz_convert(eastern).dt.date

    df["api_snapshot_date"] = pd.NaT

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
        ]
    ]

    return df.sort_values("game_id").reset_index(drop=True)


# ------------------------------------------------------------
# Main Driver
# ------------------------------------------------------------
def backfill_game_summary_history():

    con = duckdb.connect(DB_PATH)

    seasons = con.execute("""
        SELECT season_id, regular_start, regular_end
        FROM important_dates
        WHERE season_id BETWEEN 20002001 AND 20122013
        ORDER BY season_id
    """).df()

    print(f"📅 Backfilling {len(seasons)} seasons")

    for _, row in seasons.iterrows():
        season_id = int(row["season_id"])
        season_start = pd.to_datetime(row["regular_start"])
        season_end = pd.to_datetime(row["regular_end"])

        print(f"\n▶ Season {season_id}")

        df = fetch_season_games(season_id, season_start, season_end)

        if df.empty:
            print("⚠ No games found — skipping")
            continue

        # Replace season atomically
        con.execute(
            "DELETE FROM all_game_summary WHERE season_id = ?",
            [season_id]
        )

        con.register("season_games", df)

        cols = ", ".join(f'"{c}"' for c in df.columns)
        con.execute(f"""
            INSERT INTO all_game_summary ({cols})
            SELECT {cols}
            FROM season_games
        """)

        print(f"✅ Inserted {len(df):,} games")

    con.close()
    print("\n🏁 Historical backfill complete")


# ------------------------------------------------------------
# Execute
# ------------------------------------------------------------
if __name__ == "__main__":
    backfill_game_summary_history()
