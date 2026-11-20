import requests
import pandas as pd
import pytz
import duckdb
import time

DB_PATH = r"C:\Users\kjchr\Desktop\hockey\new_hockey_env\API data\data\duckdb\nhl_data.duckdb"
BASE_URL = "https://api-web.nhle.com/v1/schedule/"
SLEEP_SECONDS = 0.15

def load_important_dates(con, season_id: int) -> pd.DataFrame:
    return con.execute(
        """
        SELECT regular_start, regular_end
        FROM important_dates
        WHERE season_id = ?
        """,
        [season_id]
    ).df()

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

    df["game_id"] = pd.to_numeric(
        df["id"],
        errors="raise"
    ).astype("int64")

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

    df["startTimeUTC"] = pd.to_datetime(
        df["startTimeUTC"],
        utc=True
    )

    df["game_date"] = (
        df["startTimeUTC"]
        .dt.tz_convert(eastern)
        .dt.date
    )

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

    return (
        df
        .sort_values("game_id")
        .reset_index(drop=True)
    )
if __name__ == "__main__":

    con = duckdb.connect(DB_PATH)

    seasons = con.execute("""
        SELECT season_id
        FROM important_dates
        WHERE season_id == 20262027
        ORDER BY season_id
    """).fetch_df()["season_id"].tolist()

    for season_id in seasons:

        important_dates = load_important_dates(con, season_id)

        season_start = pd.to_datetime(
            important_dates["regular_start"].iloc[0]
        )

        season_end = pd.to_datetime(
            important_dates["regular_end"].iloc[0]
        )

        print(f"\n▶ Rebuilding missing games for {season_id}")
        print(f"📅 {season_start.date()} → {season_end.date()}")

        df = fetch_season_games(
            season_id,
            season_start,
            season_end,
        )

        if df.empty:
            print("⚠ No games returned from API")
            continue

        print(
            con.execute("""
                SELECT COUNT(*)
                FROM all_game_summary
                WHERE season_id = ?
            """, [season_id]).fetchall()
        )

        con.register("missing_games_df", df)

        cols = ", ".join(f'"{c}"' for c in df.columns)

        con.execute("""
            DELETE FROM all_game_summary
            WHERE game_id IN (
                SELECT DISTINCT game_id
                FROM missing_games_df
            )
        """)

        con.execute(f"""
            INSERT INTO all_game_summary ({cols})
            SELECT {cols}
            FROM missing_games_df
        """)

        print(
            con.execute("""
                SELECT COUNT(*)
                FROM all_game_summary
                WHERE season_id = ?
            """, [season_id]).fetchall()
        )

        print(f"✅ Inserted {len(df)} games")

    con.close()

    print("\n🏁 Done")