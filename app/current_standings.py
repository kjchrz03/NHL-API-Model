import requests
import pandas as pd
from datetime import date

pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

#Standings as of TODAY

def fetch_current_standings():
    today = pd.to_datetime(date.today()).strftime("%Y-%m-%d")
    api_url = f"https://api-web.nhle.com/v1/standings/{today}"

    response = requests.get(api_url)

    if response.status_code != 200:
        raise Exception(f"❌ NHL API request failed: {response.status_code}")

    json_data = response.json()

    if "standings" not in json_data:
        raise ValueError("❌ Missing 'standings' in API response")

    standings_df = pd.DataFrame(json_data["standings"])

    # Extract nested fields
    standings_df["team_name"] = standings_df["teamName"].str["default"]
    standings_df["tri_code"] = standings_df["teamAbbrev"].str["default"]

    standings_df["season_id"] = standings_df["seasonId"]
    standings_df["num_teams"] = 32

    # Derived metrics
    points_per_game = standings_df["points"] / standings_df["gamesPlayed"]
    standings_df["gf_g"] = standings_df["goalFor"] / standings_df["gamesPlayed"]
    standings_df["ga_g"] = standings_df["goalAgainst"] / standings_df["gamesPlayed"]

    TOTAL_GAMES = 82
    games_remaining = TOTAL_GAMES - standings_df["gamesPlayed"]

    standings_df["projected_points"] = round(
        standings_df["points"] + (games_remaining * points_per_game)
    )
    standings_df['home_win_pctg'] =   standings_df['homeWins']/standings_df['wins']
    standings_df['road_win_pctg'] =standings_df['roadWins']/standings_df['wins']
    
    desired_order = [
        'clinchIndicator',
        'conferenceAbbrev',
        'conferenceHomeSequence',
        'conferenceL10Sequence',
        'conferenceName',
        'conferenceRoadSequence',
        'conferenceSequence',
        'date',
        'divisionAbbrev',
        'divisionHomeSequence',
        'divisionL10Sequence',
        'divisionName',
        'divisionRoadSequence',
        'divisionSequence',
        'gameTypeId',
        'gamesPlayed',
        'goalDifferential',
        'goalDifferentialPctg',
        'goalAgainst',
        'goalFor',
        'goalsForPctg',
        'homeGamesPlayed',
        'homeGoalDifferential',
        'homeGoalsAgainst',
        'homeGoalsFor',
        'homeLosses',
        'homeOtLosses',
        'homePoints',
        'homeRegulationPlusOtWins',
        'homeRegulationWins',
        'homeTies',
        'homeWins',
        'l10GamesPlayed',
        'l10GoalDifferential',
        'l10GoalsAgainst',
        'l10GoalsFor',
        'l10Losses',
        'l10OtLosses',
        'l10Points',
        'l10RegulationPlusOtWins',
        'l10RegulationWins',
        'l10Ties',
        'l10Wins',
        'leagueHomeSequence',
        'leagueL10Sequence',
        'leagueRoadSequence',
        'leagueSequence',
        'losses',
        'otLosses',
        'pointPctg',
        'points',
        'regulationPlusOtWinPctg',
        'regulationPlusOtWins',
        'regulationWinPctg',
        'regulationWins',
        'roadGamesPlayed',
        'roadGoalDifferential',
        'roadGoalsAgainst',
        'roadGoalsFor',
        'roadLosses',
        'roadOtLosses',	
        'roadPoints',
        'roadRegulationPlusOtWins',
        'roadRegulationWins',
        'roadTies',	
        'roadWins',	
        'shootoutLosses',
        'shootoutWins',	
        'streakCode',	
        'streakCount',
        'teamLogo',
        'waiversSequence',
        'wildcardSequence',
        'winPctg',
        'wins',
        'team_name',
        'tri_code',
        'season_id',
        'home_win_pctg',
        'road_win_pctg',
        'gf_g',
        'ga_g',
        'projected_points',
       
    ]
    
    standings_df = standings_df.reindex(columns=desired_order)
    return standings_df


