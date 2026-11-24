# Import the external library
# Import the function from season_data.py
from current_standings import fetch_current_standings
from current_predictions import fetch_current_predictions, fetch_shap_values, fetch_shap_summary, calculate_shap_consistency, get_feature_label, get_feature_description, fetch_shap_context
import streamlit as st
import streamlit.components.v1 as components
from streamlit_echarts import st_echarts
import traceback 
import html
import altair as alt
import pandas as pd
import numpy as np
import squarify
import textwrap
from datetime import datetime, date
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend

import logging
logger = logging.getLogger(__name__)

try:
    # Your database queries and merge code here
    pass

except Exception:
    logger.exception("Failed to load prediction and game data")
    st.error("Unable to load the dashboard data.")
    st.stop()


st.set_page_config(page_title="Check This Data", 
                   page_icon="🏒", 
                   initial_sidebar_state="expanded",
                   layout="wide",)

# image = Image.open('logo.png')
# st.image(image)

# primaryColor="#fafaff"
# backgroundColor="#e1dee9"
# secondaryBackgroundColor="#d5cfe1"
# textColor="#48484E"
# font="Garamond"

option = {
    "tooltip": {"trigger": "axis"},
    "legend": {"data": ["Eastern", "Western"]},
    "xAxis": {
        "type": "category",
        "data": ["2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
                    "2021-22", "2022-23", "2023-24", "2024-25", "2026-27"]
    },
    "yAxis": {"type": "value"},
    "series": [
        {
            "name": "Eastern",
            "type": "line",
            "smooth": True,
            "data": [0.750, 0.625, 0.750, 0.750, 0.625, 0.875, 0.875, 0.875, 0.750, 0.750]
        },
        {
            "name": "Western",
            "type": "line",
            "smooth": True,
            "data": [0.875, 0.875, 0.875, 0.625, 0.750, 0.625, 0.750, 0.750, 0.750, 0.750]
        }
    ]
}


  
##########################################
##  Tab Styling                         ##
##########################################


st.markdown("""
<style>

    /* Inactive tab styling */
    button[data-baseweb="tab"] {
        border-bottom: 3px solid transparent !important;
        color: #cccccc !important; /* Silver Line */
        font-size: 75px !important;       /* <<< increase tab font size */
        padding: 10px 20px !important;     /* <<< optional: more spacing */
    }

    /* ACTIVE tab */
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid #2c7da0 !important; /* Blue active tab underline */
        color: white !important;
        font-size: 75px !important;       /* <<< increase tab font size */
        padding: 10px 20px !important;     /* <<< optional: more spacing */
        background-color: #222222 !important; 
    }

    /* Hover effect */
    button[data-baseweb="tab"]:hover {
        color: #FFC300 !important; /* Yellow */
    }

    /* 🔴 Remove / override the moving red underline */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;  /* or same as page bg */
        height: 0px !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)



# Expander Styling

st.markdown(
        """
    <style>
    .streamlit-expanderHeader {
    #   font-weight: bold;
        background: #e1dee9;
        font-size: 180px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    
custom_css = """
<style>
    .streamlit-tabs-label {
        font-size: 300px;  /* You can adjust the font size as needed */
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
  
##########################################
##  Title, Tabs, and Sidebar            ##
##########################################
#Explore NHL Advanced Stats, Simply
st.title("Check This Data")
st.markdown('''##### <span style="color: #aaaaaa">Explore NHL Advanced Stats, Simply</span>
            ''', unsafe_allow_html=True)
                
tab_txgiving, tab_teams, tab_whatif = st.tabs(["Thanksgiving Predictor", "Explore Teams", "What Ifs"])


##########################################
##  Standings Sidebar                   ##
##########################################
@st.cache_data(show_spinner=True)

def display_standings():
    try:
        standings =  fetch_current_standings()
        return standings
    except Exception:
        st.error("Unable to load data. Please try again later.")
        st.stop()
        return None
    

def load_standings(standings):
    standings['Conference'] = standings['conferenceName']
    standings['Conference Rank'] = standings['conferenceSequence']
    standings['Division'] = standings['divisionName']
    standings['Division Rank'] = standings['divisionSequence']
    standings['Team'] = standings['team_name']
    standings['Games Played'] = standings['gamesPlayed']
    standings['logo'] = standings['teamLogo']  # Assuming this contains the SVG URL
    standings['Win Pct'] = standings['winPctg']
    standings['Points'] = standings['points']
    standings['Points Pct'] = standings['pointPctg']
    standings['Wild Card Rank'] = standings['wildcardSequence']
    
    # Select specific columns to return
    selected_columns = ['Conference', 'Conference Rank', 'Division', 'Division Rank', 'Team', 'Games Played', 'logo', 'Win Pct', 
                        'Points', 'Points Pct', 'Wild Card Rank']
    league_standings_df = standings[selected_columns]
    return league_standings_df


##########################################
## Standings Side Tab                   ##
##########################################

def todays_standings():
    try:
        standings = display_standings()  # Fetch the data
        if standings is None:
            return  # Exit if no standings were fetched
            
        league_standings_df = load_standings(standings)
        
        # Create league rank using Points → PointsPct (pointPctg)
        league_standings_df = (
            league_standings_df
                .sort_values(['Points', 'Points Pct'], ascending=[False, False])
                .reset_index(drop=True)
        )

        league_standings_df['League Rank'] = league_standings_df.index + 1

        # ---------------------------------------------------------------------
        # 1. Get top 3 teams in each division (automatic qualifiers)
        # ---------------------------------------------------------------------
        #create a wild card view
        playoff_standings_df = load_standings(standings)
        playoff_standings_df = (
            playoff_standings_df
            .sort_values(['Conference', 'Division','Division Rank'], ascending=[False, False, False])
            .reset_index(drop=True)
        )
        
        division_top3 = (
            playoff_standings_df
            .groupby(['Conference', 'Division'])
            .head(3)                 # top 3 per division
            .reset_index(drop=True)
        )

        # ---------------------------------------------------------------------
        # 2. Collect remaining teams (Division Rank >= 4)
        # ---------------------------------------------------------------------
        division_remaining = (
            playoff_standings_df[
                playoff_standings_df['Division Rank'] > 3
            ].copy()
        )

        # ---------------------------------------------------------------------
        # 3. Wild Card Race
        # ---------------------------------------------------------------------
        # wildcardSequence
        wild_card_east = (
            division_remaining[division_remaining['Conference']== 'Eastern']
            .sort_values(['Points', 'Points Pct'], ascending=[False, False])
            .reset_index(drop=True)
        )
        wild_card_east['Wild Card Rank'] = wild_card_east.index + 1
 
        
        wild_card_west = (
            division_remaining[division_remaining['Conference']== 'Western']
            .sort_values(['Points', 'Points Pct'], ascending=[False, False])
            .reset_index(drop=True)
        )
        wild_card_west['Wild Card Rank'] = wild_card_west.index + 1
        
        
        # ---------------------------------------------------------------------
        # 4. Playoffs Race
        # ---------------------------------------------------------------------
        eastern_conference = (
                    league_standings_df[league_standings_df['Conference']== 'Eastern']
                    .sort_values(['Wild Card Rank'])
                    .reset_index(drop=True)
                ).head(8)
        eastern_conference['Wild Card Rank'] = eastern_conference.index + 1
        
        western_conference = (
                    league_standings_df[league_standings_df['Conference']== 'Western']
                    .sort_values(['Wild Card Rank'])
                    .reset_index(drop=True)
                ).head(8)
        western_conference['Wild Card Rank'] = western_conference.index + 1
        
        
        #Define colors for each division
        division_colors = {
            'Atlantic': '#087e8b', 
            'Metropolitan': '#ff5a5f',  
            'Central': '#dcdcdd',  
            'Pacific': '#3c3c3c',  
        }

        # Dropdown for standings selection
        divisions = league_standings_df['Division'].unique().tolist()
        divisions.sort()
        divisions.append("Wild Card Race - East")
        divisions.append("Wild Card Race - West")
        divisions.append("League-Wide")
        divisions.append("Eastern Conference")
        divisions.append("Western Conference")


        # Select division from the sidebar
        selected_division = st.selectbox("Select View:", divisions)
        
        # ------------------------------------------------------------
        # **1. LEAGUE-WIDE VIEW**
        # ------------------------------------------------------------
        if selected_division == "League-Wide":

            # already sorted by Points → Points%
            sorted_df = league_standings_df

            # top 8 per conference
            conference_teams = (
                sorted_df
                .groupby("Conference")
                .head(32)
                .reset_index(drop=True)
            )

            filtered_standings = (
                conference_teams
                .sort_values(['Points', 'Points Pct'], ascending=[False, False])
                .reset_index(drop=True)
            )

            ranking = filtered_standings['League Rank']


        # ------------------------------------------------------------
        # **2. Conference View
        # ------------------------------------------------------------
        elif selected_division == "Eastern Conference":
            
            eastern_conference_view = (
                eastern_conference
                .sort_values([
                    'Wild Card Rank'
                ], ascending=[True])
                .reset_index(drop=True)
            )

            filtered_standings = eastern_conference_view
            ranking = None

        elif selected_division == "Western Conference":
           
            
            western_conference_view = (
                western_conference
                .sort_values([
                    'Wild Card Rank'
                ], ascending=[True])
                .reset_index(drop=True)
            )

            filtered_standings = western_conference_view
            ranking = None
            
        # ------------------------------------------------------------
        # **2. Wild Card East
        # ------------------------------------------------------------
        elif selected_division == "Wild Card Race - East":

            east_playoff_view = (
                wild_card_east
                .sort_values([
                    'Wild Card Rank'
                ], ascending=[True])
                .reset_index(drop=True)
            )

            filtered_standings = east_playoff_view
            ranking = None

        elif selected_division == "Wild Card Race - West":
        # Eastern Conference only
            # west_top3 = division_top3[division_top3['Conference']=='Western']
            

            # west_playoff_view = pd.concat([west_top3, wild_card_west], ignore_index=True)

            west_playoff_view = (
                wild_card_west
                .sort_values([
                    'Wild Card Rank'
                ], ascending=[True])
                .reset_index(drop=True)
            )

            filtered_standings = west_playoff_view
            ranking = None

        # ------------------------------------------------------------
        # **3. DIVISION VIEW**
        # ------------------------------------------------------------
        else:
            filtered_standings = (
                league_standings_df[
                    league_standings_df['Division'] == selected_division
                ]
                .sort_values(['Division Rank'], ascending=[True])
                .reset_index(drop=True)
            )

            ranking = filtered_standings['Division Rank']
        
        def create_dataframe(n):
                div_standings = []

                for index, row in filtered_standings.iterrows():
                    team = row['Team']
                    points = row['Points']
                    division = row['Division']
                    games_played = row['Games Played']
                    logo_url = row['logo']  # SVG logo link
                    div_standings.append(row)

                    # Create DataFrame from the collected data
                return pd.DataFrame(div_standings)

            # Call the function to create the DataFrame
        
        div_standings = create_dataframe(8)
        
        # #Create visual representation for each team
        team = div_standings['Team']
        points = div_standings['Points']
        division = div_standings['Division']
        logo_url = div_standings['logo']
        games_played = div_standings['Games Played']

        max_points = points.max()
        min_points = points.min()

        # ----- HEIGHT & SPACING SETUP -----
        BASE_HEIGHT = 300          # this will now be the TRUE visible range
        SPACING = 3
        scale_height = BASE_HEIGHT * SPACING
        

        # Add some padding to top/bottom of the line only (NOT the whole container)
        ## change the left/right of the bar
        LINE_EXTRA = 50  # pixels above + below
        html_content = f"""
        <div style="display: flex; 
                flex-direction: column; 
                align-items: flex-start;
                position: relative; 
                height: {scale_height}px;
                margin: 30px 0;">

            <div style="
                position: absolute;
                left: 55px;
                width: 10px;
                height: {scale_height + LINE_EXTRA}px;
                background: linear-gradient(to bottom,  #d9d9d9, #ffffff,#3c6e71,#284b63, #353535 );
                transform: translateY(-{LINE_EXTRA/2}px);
            "></div>
        """

        # Group teams by point total
        standing_points = {}

        for i, point in enumerate(points):
            logo_url = div_standings['logo'].iloc[i]
            team_name = div_standings['Team'].iloc[i]
            division_name = div_standings['Division'].iloc[i]

            standing_points.setdefault(point, []).append({
                'team': team_name,
                'logo': logo_url,
                'division': division_name,
            })

        # Sort points from high to low
        sorted_standing_points = {
            k: standing_points[k] for k in sorted(standing_points.keys(), reverse=True)
        }

        spread = max(max_points - min_points, 2)
        
        # Draw each point level
        for point, teams in sorted_standing_points.items():

            # NEW → scale relative to actual standings
            normalized = (point - min_points) / spread
            position = scale_height - (normalized * scale_height)

            # Label left of the line
            html_content += f"""
            <div style="position: absolute; 
                        left: 0%; top: {position}px;
                        color: white; 
                        font-size: 35px; 
                        line-height: 10px;">
                {point}
            </div>

            <div style="position: absolute; 
                        left: 60px;
                        top: {position-25}px;
                        display: flex; 
                        gap: 5px;">               
        
                        """
            # Team logo circles
            for i, team_info in enumerate(teams):
                tooltip = f"{team_info['team']} - {team_info['division']} Division"

                html_content += f"""
                <div title="{tooltip}"
                    style="background-color: white;
                            border: 8px solid {division_colors.get(team_info['division'], 'grey')};
                            border-radius: 50%;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            margin-left: {10 if i == 0 else 0}px;">
                    <img src="{team_info['logo']}"
                        alt="{team_info['team']} Logo"
                        width="50" height="50"
                        style="border-radius: 50%;">
                </div>
                """

            # Close logos container
            html_content += "</div>"

        # Close outer container
        html_content += "</div>"
        centered_html = f"""
        <div style="
            width: 100%;
            display: flex;
            justify-content: center;
            margin-top: 20px;
        ">
            <div style="
                position: relative;
                width: 600px;      /* ← adjust — this defines the width of the whole graphic */
            ">
                {html_content}
            </div>
        </div>
        """

        st.html(centered_html)

    except Exception as e:
        st.error(f"Error loading standings data: {e}")
        return None


with st.sidebar:

    st.subheader("Current Standings")

    today = datetime.now().strftime("%B %d, %Y")
    st.markdown(f"{today}")
    todays_standings()
    


##########################################
## Thanksgiving Tab                     
##########################################
with tab_txgiving:

    cols = st.columns([1, 3])
    @st.cache_data(ttl="6h")
    def load_full_data(season_id, model_version="v6"):
        try:
            df = fetch_current_predictions(
                season_id=season_id,
                model_version=model_version
            )

            # Normalize fields
            df["label"] = df["tri_code"].str.upper().str.strip()
            df["team_name"] = df["team_name"].str.strip()
            df["season"] = df["season_id"].astype(int)

            return df

        except Exception:
            st.error("Unable to load data. Please try again later.")
            st.stop()
            return None
        
    @st.cache_data(ttl="6h")
    def load_shap_data(season_id, model_version="v6"):
        
        return fetch_shap_values(
            season_id=season_id,
            model_version=model_version
        )

    
    
    @st.cache_data(ttl="6h")
    def load_shap_summary(season_id, model_version="v6"):
        return fetch_shap_summary(
                    season_id=season_id,
                    model_version=model_version
                )

    
        
    full_df = load_full_data(
        season_id=20262027,
        model_version="v6"
    )

    shap_df = load_shap_data(
        season_id=20262027,
        model_version="v6"
    )
    
    shap_summary = load_shap_summary(
        season_id=20262027,
        model_version="v6"
    )
    
    # =========================================================
    #  Division Colors
    # =========================================================
    division_colors = {
        "Atlantic": "#087e8b",
        "Metropolitan": "#ff5a5f",
        "Central": "#dcdcdd",
        "Pacific": "#3c3c3c",
        "Unknown": "#808080",
    }

    team_to_division = {
        # Atlantic
        "BOS": "Atlantic", "MTL": "Atlantic", "OTT": "Atlantic", "TOR": "Atlantic",
        "DET": "Atlantic", "BUF": "Atlantic", "TBL": "Atlantic", "FLA": "Atlantic",

        # Metropolitan
        "NYR": "Metropolitan", "NYI": "Metropolitan", "NJD": "Metropolitan",
        "WSH": "Metropolitan", "PIT": "Metropolitan", "PHI": "Metropolitan",
        "CAR": "Metropolitan", "CBJ": "Metropolitan",

        # Central
        "NSH": "Central", "DAL": "Central", "STL": "Central", "CHI": "Central",
        "MIN": "Central", "WPG": "Central", "UTA": "Central", "COL": "Central",

        # Pacific
        "CGY": "Pacific", "EDM": "Pacific", "VAN": "Pacific",
        "SEA": "Pacific", "SJS": "Pacific", "VGK": "Pacific",
        "LAK": "Pacific", "ANA": "Pacific",

        # ARI – not used anymore
        "ARI": "Unknown",
    }

    # =========================================================
    #  Inject JS to tag each multiselect chip with class "tag-XXX"
    # =========================================================
    st.markdown("""
    <script>
    const observer = new MutationObserver(() => {
        document.querySelectorAll('span[data-baseweb="tag"]').forEach(tag => {
            const text = tag.innerText.replace("×","").trim();
            if (text && !tag.classList.contains("tag-" + text)) {
                tag.classList.add("tag-" + text);
            }
        });
    });
    observer.observe(document.body, { subtree: true, childList: true });
    </script>
    """, unsafe_allow_html=True)

    # =========================================================
    #  CSS to color the chips based on division
    # =========================================================
    css_rules = "<style>"

    for team, div in team_to_division.items():
        color = division_colors.get(div, "#808080")
        css_rules += f"""
        .tag-{team} {{
            background-color: {color} !important;
            color: white !important;
            border-radius: 6px !important;
        }}
        """

    css_rules += "</style>"

    st.markdown(css_rules, unsafe_allow_html=True)
    
    
    # =========================================================
    #  TEAM SELECTION UI
    # =========================================================
    TRI_CODES = sorted(full_df["tri_code"].unique())
    available_labels = sorted(TRI_CODES)
    SHAP_FEATURE = sorted(
        shap_summary["feature"].dropna().unique(),
        key=lambda feature: get_feature_label(feature).casefold()
    )

    shap_labels = sorted(
        get_feature_label(feature)
        for feature in SHAP_FEATURE
    )
    # =========================================================
    # DATE SELECTION
    # =========================================================
    predictions_df = full_df.copy()

    predictions_df["game_date"] = pd.to_datetime(
        predictions_df["game_date"]
    ).dt.normalize()

    current_date = pd.Timestamp.today().normalize()
    thanksgiving = pd.Timestamp("2026-11-25")

    available_dates = {
        "Today": current_date,
        "Thanksgiving Day": thanksgiving,
        "Cumulative To Date": current_date,
        "Cumulative to Thanksgiving": thanksgiving,
    }
    date_labels = list(available_dates.keys())

    top_left_cell = cols[0].container(border=True)

    with top_left_cell:
        st.subheader("Date Selection")

        date_mode = st.pills(
            "View",
            options=["Today", "Thanksgiving Day"],
            default="Today",
        )
             
    
    predictions_df['game_date'] = pd.to_datetime(predictions_df['game_date']).dt.normalize() 

    # =========================================================
    # HANDLE DATE MODES
    # =========================================================
    if date_mode == "Today":

        selected_date = ["Today"]

        # Find the most recent game date, excluding future games
        valid_dates = predictions_df.loc[
            predictions_df['game_date'] <= current_date,
            'game_date'
        ].dropna()

        if valid_dates.empty:
            st.warning("No completed game dates available.")
            st.stop()

        last_game_date = valid_dates.max()

        # Filter to only games from the latest game day
        predictions_df = predictions_df[
            predictions_df['game_date'] == last_game_date
        ].copy()

        selected_date_ids = [last_game_date]
        current_date_id = last_game_date

        if predictions_df.empty:
            st.warning("No games available for the latest game date.")
            st.stop()

        # Reduce space above the date range box
        st.markdown(
            """
            <style>
            .st-key-date_range_box {
                margin-top: -3rem;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
                
    # ----------------------------------------------------
    # HANDLE "Thanksgiving Day"
    # ----------------------------------------------------
    elif date_mode == "Thanksgiving Day":

        selected_date = ["Thanksgiving Day"]
        selected_date_ids = [thanksgiving]
        thanksgiving_date_id = thanksgiving

        # Filter to Thanksgiving games only
        predictions_df = predictions_df[
            predictions_df['game_date'] == thanksgiving
        ].copy()

        if predictions_df.empty:
            st.warning("Probabilities will be available on November 26, 2026.")
            st.stop()   

        # Reduce space above the date range box
        st.markdown(
            """
            <style>
            .st-key-date_range_box {
                margin-top: -3rem;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    elif date_mode == "Cumulative To Date":

        selected_date = ["Cumulative To Date"]

        predictions_df = predictions_df[
            predictions_df["game_date"] <= current_date
        ].copy()

        selected_date_ids = sorted(
            predictions_df["game_date"].dropna().unique()
        )

    elif date_mode == "Cumulative to Thanksgiving":

        selected_date = ["Cumulative to Thanksgiving"]

        predictions_df = predictions_df[
            predictions_df["game_date"] <= thanksgiving
        ].copy()

        selected_date_ids = sorted(
            predictions_df["game_date"].dropna().unique()
        )
    # ---------------------------------------------------------
    # CHECK FOR AVAILABLE DATA
    # ---------------------------------------------------------
    if predictions_df.empty:
        st.warning("No predictions available for this date selection.")
        st.stop()

    # =========================================================
    # FILTER DATA
    # =========================================================
    if selected_date_ids:
        predictions_df = predictions_df[predictions_df["game_date"].isin(selected_date_ids)]

    # =========================================================
    # RIGHT PANEL — ALTair Visualization
    # =========================================================
    right_cell = cols[1].container(border=True, height="stretch", vertical_alignment="top")

    team_colors = {
    "ANA": "#F47A38",
    "ARI": "#8C2633",
    "BOS": "#FFB81C",
    "BUF": "#002654",
    "CAR": "#CC0000",
    "CBJ": "#002654",
    "CGY": "#C8102E",
    "CHI": "#CF0A2C",
    "COL": "#6F263D",
    "DAL": "#006847",
    "DET": "#CE1126",
    "EDM": "#FF4C00",
    "FLA": "#B9975B",
    "LAK": "#F7F3F3",
    "MIN": "#154734",
    "MTL": "#AF1E2D",
    "NJD": "#CE1126",
    "NSH": "#FFB81C",
    "NYI": "#00529B",
    "NYR": "#0038A8",
    "OTT": "#DA1A32",
    "PHI": "#F74902",
    "PIT": "#FCB514",
    "SEA": "#99D9D9",
    "SJS": "#006D75",
    "STL": "#002F87",
    "TBL": "#002868",
    "TOR": "#00205B",
    "UTA": "#6F2DA8",
    "VAN": "#00205B",
    "VGK": "#B4975A",
    "WPG": "#041E42",
    "WSH": "#C8102E",
    }


    with right_cell:
        st.subheader("Playoff Probabilities")

        if predictions_df.empty:
            st.warning("No data to plot.")
        else:

            # ---------------------------------------------------------------
            # Build dynamic domain & range for selected teams ONLY
            # ---------------------------------------------------------------
            color_domain = [t for t in predictions_df["tri_code"].unique() if t in team_colors]
            color_range = [team_colors[t] for t in color_domain]

                        
            # ---------------------------------------------------------------
            # TODAY / THANKSGIVING DAY → Bar Chart with Model Probability Ranges
            # ---------------------------------------------------------------
            if date_mode in ["Today", "Thanksgiving Day"]:

                date_label = selected_date[0]

                # Already filtered by date_mode
                chart_prediction = predictions_df.copy()

                # Calculate lower and upper bounds between models
                chart_prediction["prob_lower"] = chart_prediction[
                    ["playoff_prob_xgb", "playoff_prob_rf"]
                ].min(axis=1)

                chart_prediction["prob_upper"] = chart_prediction[
                    ["playoff_prob_xgb", "playoff_prob_rf"]
                ].max(axis=1)

                # Sort teams by displayed playoff probability
                chart_prediction = chart_prediction.sort_values(
                    by="playoff_prob_display",
                    ascending=False
                )

                team_order = chart_prediction["tri_code"].tolist()
                
                # ----------------------------------------------------
                # BASE CHART
                # ----------------------------------------------------
                base = alt.Chart(chart_prediction).encode(
                    x=alt.X(
                        "tri_code:N",
                        sort=team_order,
                        title="Team",
                        axis=alt.Axis(
                            labelAngle=-45,
                            labelOverlap=False,
                            labelLimit=100,
                            labelPadding=8,
                            labelFontSize=10,
                            labelExpr="""
                                indexof(
                                    [
                                        'ANA','BOS','BUF','CAR','CBJ','CGY','CHI','COL',
                                        'DAL','DET','EDM','FLA','LAK','MIN','MTL','NJD',
                                        'NSH','NYI','NYR','OTT','PHI','PIT','SEA','SJS',
                                        'STL','TBL','TOR','UTA','VAN','VGK','WPG','WSH'
                                    ],
                                    datum.label
                                ) % 2 == 0
                                ? datum.label
                                : '  ' + datum.label
                            """
                        )
                    )
                         )

                # ----------------------------------------------------
                # BARS → Displayed Playoff Probability Today
                # ----------------------------------------------------
                bars = base.mark_bar().encode(
                    y=alt.Y(
                        "playoff_prob_display:Q",
                        title="Playoff Probability",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format=".0%")
                    ),
                    color=alt.Color(
                        "tri_code:N",
                        title="Teams",
                        scale=alt.Scale(
                            domain=color_domain,
                            range=color_range
                        ),
                        legend=None
                    ),
                    tooltip=[
                        alt.Tooltip("team_name:N", title="Team"),
                        alt.Tooltip(
                            "playoff_prob_display:Q",
                            title="Displayed Probability",
                            format=".1%"
                        ),

                        alt.Tooltip(
                            "playoff_tier:N",
                            title="Playoff Prediction"
                        )
                    ]
                )

                # ----------------------------------------------------
                # ERROR BARS → Range Between XGP and RF Models
                # ----------------------------------------------------
                error_bars = base.mark_errorbar(
                    ticks=False,
                    color="white"
                ).encode(
                    y=alt.Y(
                        "prob_lower:Q",
                        title="Playoff Probability",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format=".0%")
                    ),
                    y2="prob_upper:Q"
                )
                # ----------------------------------------------------
                # COMBINE CHARTS
                # ----------------------------------------------------
                combined_chart = (
                    alt.layer(bars, error_bars)
                    .properties(
                        title=f"Playoff Probability — {date_label}",
                        height=500
                    )
                    .configure_axisY(
                        title="Playoff Probability"
                    )
                    .configure_legend(
                        orient="bottom",
                        padding=0,
                        labelLimit=600,
                        title=None
                    )
                )
                st.altair_chart(combined_chart, width="stretch")
                
        with right_cell:
            st.divider()
                            
            # ====================================================
            # COLLECTIVE PLAYOFF DRIVERS — CUMULATIVE SHAP
            # ====================================================
            if date_mode in ["Today", "Thanksgiving Day"]:

                st.subheader("Collective Playoff Drivers")
                # ------------------------------------------------
                # 1. FILTER SHAP DATA TO THE SELECTED PERIOD
                # ------------------------------------------------
                filtered_shap = shap_df.copy()

                filtered_shap["game_date"] = pd.to_datetime(
                    filtered_shap["game_date"]
                ).dt.normalize()

                # Start of available season data
                start_date = filtered_shap["game_date"].min()

                # End date based on existing selection
                end_date = (
                    current_date
                    if date_mode == "Today"
                    else thanksgiving
                )

                # Cumulative SHAP data
                filtered_shap = filtered_shap[
                    filtered_shap["game_date"].between(
                        start_date,
                        end_date
                    )
                ].copy()
                # ------------------------------------------------
                # 2. IDENTIFY SHAP FEATURE COLUMNS
                # ------------------------------------------------
                shap_cols = [
                    c for c in filtered_shap.columns
                    if c.startswith("shap_")
                    and c != "shap_base_value"
                ]

                # ------------------------------------------------
                # 3. CALCULATE PER-TEAM FEATURE IMPACT
                # ------------------------------------------------
                # Each team receives equal weight, regardless
                # of how many snapshots it has.
                #
                # Replace "tri_code" if your SHAP dataframe
                # uses another team identifier.

                if filtered_shap.empty or not shap_cols:
                    st.info("No SHAP data available for this period.")

                else:

                    filtered_shap[shap_cols] = (
                        filtered_shap[shap_cols]
                        .apply(pd.to_numeric, errors="coerce")
                    )

                    # Absolute impact averaged over time per team
                    team_abs = (
                        filtered_shap
                        .assign(
                            **{
                                c: filtered_shap[c].abs()
                                for c in shap_cols
                            }
                        )
                        .groupby("tri_code")[shap_cols]
                        .mean()
                    )

                    # Signed impact averaged over time per team
                    team_signed = (
                        filtered_shap
                        .groupby("tri_code")[shap_cols]
                        .mean()
                    )

                    # ------------------------------------------------
                    # 4. RANK FEATURES BY LEAGUE-WIDE ABSOLUTE IMPACT
                    # ------------------------------------------------
                    feature_impact = (
                        team_abs.mean()
                        .sort_values(ascending=False)
                        .head(10)
                    )

                    # ------------------------------------------------
                    # 5. HELPING / HARMING PERCENTAGES
                    # ------------------------------------------------
                    records = []

                    for feature, impact in feature_impact.items():

                        values = team_signed[feature].dropna()

                        # Zero impact is neutral
                        total_teams = len(values)

                        if total_teams == 0:
                            continue

                        helping = (values > 0).sum()
                        harming = (values < 0).sum()
                        neutral = (values == 0).sum()

                        records.append({
                            "feature": feature,
                            "impact": float(impact),
                            "helping_pct": helping / total_teams * 100,
                            "harming_pct": harming / total_teams * 100,
                            "neutral_pct": neutral / total_teams * 100,
                            "teams": total_teams
                        })

                    mosaic_df = pd.DataFrame(records)

                    if mosaic_df.empty or mosaic_df["impact"].sum() <= 0:
                        st.info("No measurable SHAP impact available.")

                    else:

                        # --------------------------------------------
                        # 6. CALCULATE RELATIVE IMPACT
                        # --------------------------------------------
                        mosaic_df["impact_share"] = (
                            mosaic_df["impact"]
                            / mosaic_df["impact"].sum()
                            * 100
                        )

                        # --------------------------------------------
                        # 7. GENERATE MOSAIC RECTANGLES
                        # --------------------------------------------
                        # Coordinates use a 100 x 100 layout.
                        MOSAIC_WIDTH = 350
                        MOSAIC_HEIGHT = 500

                        sizes = squarify.normalize_sizes(
                            mosaic_df["impact"].tolist(),
                            MOSAIC_WIDTH,
                            MOSAIC_HEIGHT
                        )

                        rectangles = squarify.squarify(
                            sizes,
                            0,
                            0,
                            MOSAIC_WIDTH,
                            MOSAIC_HEIGHT
                        )

                        for key in ["x", "y", "dx", "dy"]:
                            mosaic_df[key] = [
                                rect[key] for rect in rectangles
                            ]

                        mosaic_df["x2"] = (
                            mosaic_df["x"] + mosaic_df["dx"]
                        )

                        mosaic_df["y2"] = (
                            mosaic_df["y"] + mosaic_df["dy"]
                        )

                        mosaic_df["center_x"] = (
                            mosaic_df["x"] + mosaic_df["dx"] / 2
                        )

                        mosaic_df["center_y"] = (
                            mosaic_df["y"] + mosaic_df["dy"] / 2
                        )

                                                
                        # --------------------------------------------
                        # 8. PREPARE LABELS
                        # --------------------------------------------

                        mosaic_df["feature_label"] = (
                            mosaic_df["feature"]
                            .str.removeprefix("shap_")
                            .apply(get_feature_label)
                        )
                        
                        # Get existing feature descriptions for tooltips
                        mosaic_df["feature_description"] = (
                            mosaic_df["feature"]
                            .str.removeprefix("shap_")
                            .apply(get_feature_description)
                        )
                        

                        # Wrap long feature names
                        mosaic_df["feature_label_wrapped"] = (
                            mosaic_df["feature_label"]
                            .apply(
                                lambda label: "\n".join(
                                    textwrap.wrap(
                                        str(label),
                                        width=10,
                                        break_long_words=False
                                    )
                                )
                            )
                        )

                        # Count wrapped lines
                        mosaic_df["feature_lines"] = (
                            mosaic_df["feature_label_wrapped"]
                            .str.count("\n") + 1
                        )

                        # --------------------------------------------
                        # CREATE TEXT LABELS
                        # --------------------------------------------

                        mosaic_df["impact_label"] = (
                            "League Impact\n"
                            + mosaic_df["impact_share"].round(1).astype(str)
                            + "%"
                        )
                        mosaic_df["impact_pct"] = mosaic_df["impact"] * 100

                        mosaic_df["help_label"] = (
                            "Helped: "
                            + mosaic_df["helping_pct"].round(1).astype(str)
                            + "%"
                        )

                        mosaic_df["harm_label"] = (
                            "Hurt: "
                            + mosaic_df["harming_pct"].round(1).astype(str)
                            + "%"
                        )

                        # --------------------------------------------
                        # DYNAMIC FONT SIZES
                        # --------------------------------------------

                        # Use the smaller rectangle dimension
                        mosaic_df["box_size"] = (
                            mosaic_df["dx"] * mosaic_df["dy"]
                        ) ** 0.5

                        min_size = mosaic_df["box_size"].min()
                        max_size = mosaic_df["box_size"].max()

                        # Normalize rectangle sizes between 0 and 1
                        if max_size > min_size:
                            size_factor = (
                                (mosaic_df["box_size"] - min_size)
                                / (max_size - min_size)
                            )
                        else:
                            size_factor = pd.Series(
                                0.5,
                                index=mosaic_df.index
                            )

                        # Initial font sizes
                        # Feature names: 12–24px
                        mosaic_df["feature_font_size"] = (
                            12 + size_factor * 12
                        )

                        # League Impact: 10–17px
                        mosaic_df["impact_font_size"] = (
                            10 + size_factor * 7
                        )

                        # Helped / Hurt: 9–14px
                        mosaic_df["detail_font_size"] = (
                            9 + size_factor * 5
                        )

                        # --------------------------------------------
                        # FIT TEXT TO RECTANGLE HEIGHT
                        # --------------------------------------------

                        CHART_HEIGHT = 650

                        # Convert Squarify coordinates to rendered pixels
                        mosaic_df["box_height_px"] = (
                            mosaic_df["dy"]
                            / MOSAIC_HEIGHT
                            * CHART_HEIGHT
                        )

                        # Leave vertical padding inside rectangles
                        mosaic_df["available_height"] = (
                            mosaic_df["box_height_px"] - 20
                        ).clip(lower=1)

                        # --------------------------------------------
                        # ESTIMATE INITIAL TEXT HEIGHTS
                        # --------------------------------------------

                        mosaic_df["feature_line_height"] = (
                            mosaic_df["feature_font_size"] * 1.4
                        )

                        mosaic_df["impact_line_height"] = (
                            mosaic_df["impact_font_size"] * 1.5
                        )

                        mosaic_df["feature_height"] = (
                            mosaic_df["feature_lines"]
                            * mosaic_df["feature_line_height"]
                        )

                        # League Impact always contains two lines
                        mosaic_df["impact_height"] = (
                            2 * mosaic_df["impact_line_height"]
                        )

                        # Helped and Hurt are single lines
                        mosaic_df["detail_height"] = (
                            mosaic_df["detail_font_size"] * 1.3
                        )

                        # Initial spacing between sections
                        GAP_FEATURE = -10
                        GAP_IMPACT = 20
                        GAP_DETAIL = 2

                        # Total estimated height before scaling
                        mosaic_df["total_text_height"] = (
                            mosaic_df["feature_height"]
                            + mosaic_df["impact_height"]
                            + mosaic_df["detail_height"] * 2
                            + GAP_FEATURE
                            + GAP_IMPACT
                            + GAP_DETAIL
                        )

                        # --------------------------------------------
                        # SCALE TEXT TO FIT SMALL RECTANGLES
                        # --------------------------------------------

                        mosaic_df["text_scale"] = (
                            mosaic_df["available_height"]
                            / mosaic_df["total_text_height"]
                        ).clip(upper=1.0)

                        # Apply scaling to all font sizes
                        for col in [
                            "feature_font_size",
                            "impact_font_size",
                            "detail_font_size"
                        ]:
                            mosaic_df[col] *= mosaic_df["text_scale"]

                        # --------------------------------------------
                        # RECALCULATE HEIGHTS AFTER SCALING
                        # --------------------------------------------

                        mosaic_df["feature_line_height"] = (
                        mosaic_df["feature_font_size"] * 1.2
                        )

                        mosaic_df["impact_line_height"] = (
                            mosaic_df["impact_font_size"] * 1.5
                        )

                        mosaic_df["feature_height"] = (
                            mosaic_df["feature_lines"]
                            * mosaic_df["feature_line_height"]
                        )

                        mosaic_df["impact_height"] = (
                            2 * mosaic_df["impact_line_height"]
                        )

                        mosaic_df["detail_height"] = (
                            mosaic_df["detail_font_size"] * 1.3
                        )

                        # Scale spacing proportionally
                        mosaic_df["gap_feature"] = (
                            GAP_FEATURE * mosaic_df["text_scale"]
                        )
                        mosaic_df["gap_impact"] = (
                            GAP_IMPACT * mosaic_df["text_scale"]
                        )

                        mosaic_df["gap_detail"] = (
                            GAP_DETAIL * mosaic_df["text_scale"]
                        )

                        # --------------------------------------------
                        # CENTER COMPLETE TEXT GROUP VERTICALLY
                        # --------------------------------------------

                        # Total height after scaling
                        mosaic_df["total_text_height"] = (
                            mosaic_df["feature_height"]
                            + mosaic_df["gap_feature"]
                            + mosaic_df["impact_height"]
                            + mosaic_df["gap_impact"]
                            + mosaic_df["detail_height"] * 2
                            + mosaic_df["gap_detail"]
                        )

                        # Top of the complete text group,
                        # relative to the rectangle center
                        mosaic_df["text_top"] = (
                            -mosaic_df["total_text_height"] / 2
                        )

                        # Feature label center
                        mosaic_df["feature_dy"] = mosaic_df["text_top"]

                        # League Impact center
                        # Always begins after the full feature label,
                        # regardless of the number of wrapped lines
                        mosaic_df["impact_dy"] = (
                            mosaic_df["text_top"]
                            + mosaic_df["feature_height"]
                            + mosaic_df["gap_feature"]
                            + mosaic_df["impact_height"] / 2
                        )

                        # Helped center
                        mosaic_df["help_dy"] = (
                            mosaic_df["text_top"]
                            + mosaic_df["feature_height"]
                            + mosaic_df["gap_feature"]
                            + mosaic_df["impact_height"]
                            + mosaic_df["gap_impact"]
                            + mosaic_df["detail_height"] / 2
                        )

                        # Hurt center
                        mosaic_df["harm_dy"] = (
                            mosaic_df["help_dy"]
                            + mosaic_df["detail_height"]
                            + mosaic_df["gap_detail"]
                        )


                        # --------------------------------------------
                        # 9. BASE MOSAIC
                        # --------------------------------------------
                        mosaic_base = alt.Chart(mosaic_df)

                        x_scale = alt.Scale(domain=[0, 350])
                        y_scale = alt.Scale(
                            domain=[0, 500],
                            reverse=True
                        )

                        rectangles_chart = (
                            mosaic_base
                            .mark_rect(
                                stroke="#111111",
                                strokeWidth=3,
                                cornerRadius=4
                            )
                            .encode(
                                x=alt.X(
                                    "x:Q",
                                    scale=x_scale,
                                    axis=None
                                ),
                                x2="x2:Q",
                                y=alt.Y(
                                    "y:Q",
                                    scale=y_scale,
                                    axis=None
                                ),
                                y2="y2:Q",
                                color=alt.Color(
                                    "impact_share:Q",
                                    scale=alt.Scale(
                                        domain=[
                                            mosaic_df["impact_share"].min(),
                                            mosaic_df["impact_share"].max()
                                        ],
                                        range=[
                                            "#aec3b0",
                                            "#81b29a",
                                            "#3c6e71",
                                            "#427aa1",
                                            "#284b63",
                                            "#353535"
                                        ],
                                        interpolate="rgb"
                                    ),
                                    legend=None
                                ),
                                
                                tooltip=[
                                    alt.Tooltip(
                                        "feature_description:N",
                                        title="What This Measures: "
                                    ),
                
                                ]
                            )
                        )

                        # --------------------------------------------
                        # 10. TEXT LABELS INSIDE RECTANGLES
                        # --------------------------------------------

                        label_base = mosaic_base.encode(
                            x=alt.X(
                                "center_x:Q",
                                scale=x_scale,
                                axis=None
                            ),
                            y=alt.Y(
                                "center_y:Q",
                                scale=y_scale,
                                axis=None
                            ),
                            tooltip=[]
                        )

                        # --------------------------------------------
                        # FEATURE NAME
                        # --------------------------------------------

                        feature_text = (
                            label_base
                            .mark_text(
                                fontWeight="bold",
                                color="white",
                                dy=alt.ExprRef(expr="datum.feature_dy"),
                                lineBreak="\n",
                                lineHeight=alt.ExprRef(expr="datum.feature_line_height"),
                                align="center",
                                baseline="top"
                            )
                            .encode(
                                text="feature_label_wrapped:N",
                                size=alt.Size(
                                    "feature_font_size:Q",
                                    scale=None,
                                    legend=None
                                )
                            )
                        )

                        # --------------------------------------------
                        # LEAGUE IMPACT
                        # --------------------------------------------

                        impact_text = (
                            label_base
                            .mark_text(
                                fontWeight="bold",
                                color="white",
                                dy=alt.ExprRef(expr="datum.impact_dy"),
                                lineBreak="\n",
                                lineHeight=alt.ExprRef(
                                    expr="datum.impact_line_height"),
                                align="center",
                                baseline="middle"
                            )
                            .encode(
                                text="impact_label:N",
                                size=alt.Size(
                                    "impact_font_size:Q",
                                    scale=None,
                                    legend=None
                                )
                            )
                        )

                        # --------------------------------------------
                        # TEAMS HELPED
                        # --------------------------------------------

                        help_text = (
                            label_base
                            .mark_text(
                                color="white",
                                dy=alt.ExprRef(
                                    expr="datum.help_dy"),
                                align="center",
                                baseline="middle"
                            )
                            .encode(
                                text="help_label:N",
                                size=alt.Size(
                                    "detail_font_size:Q",
                                    scale=None,
                                    legend=None
                                )
                            )
                        )

                        # --------------------------------------------
                        # TEAMS HURT
                        # --------------------------------------------

                        harm_text = (
                            label_base
                            .mark_text(
                                color="white",
                                dy=alt.ExprRef(
                                    expr="datum.harm_dy"
                                ),
                                align="center",
                                baseline="middle"
                            )
                            .encode(
                                text="harm_label:N",
                                size=alt.Size(
                                    "detail_font_size:Q",
                                    scale=None,
                                    legend=None
                                )
                            )
                        )
                        # --------------------------------------------
                        # 11. COMBINE MOSAIC
                        # --------------------------------------------
                        mosaic_chart = (
                            alt.layer(
                                rectangles_chart,
                                feature_text,
                                impact_text,
                                help_text,
                                harm_text
                            )
                            .properties(
                                title=(
                                    "Top 10 League-Wide Playoff Drivers"
                                    f" — {date_label}"
                                ),
                                width="container",
                                height=650
                            )
                            .configure_view(
                                stroke=None
                            )
                        )
                        st.caption(
                            "**League Impact** shows how much each factor influences the model's playoff predictions compared with the other top 10 factors. **Helping and Harming** show the percentage of teams whose playoff chances are positively or negatively influenced by that factor this season."
                        )
                        st.altair_chart(
                            mosaic_chart,
                            width="stretch"
                        )


    if not predictions_df.empty:

        # Keep team names and probabilities for each tier
        columns = ["team_name", "playoff_prob_display"]

        lock_teams = predictions_df.loc[predictions_df["playoff_tier"] == "Lock", columns].sort_values("playoff_prob_display", ascending=False)

        likely_teams = predictions_df.loc[predictions_df["playoff_tier"] == "Likely",columns].sort_values("playoff_prob_display", ascending=False)

        bubble_teams = predictions_df.loc[predictions_df["playoff_tier"] == "Bubble", columns].sort_values("playoff_prob_display", ascending=False)

        danger_teams = predictions_df.loc[predictions_df["playoff_tier"] == "Danger",columns].sort_values("playoff_prob_display", ascending=False)

        long_shot_teams = predictions_df.loc[predictions_df["playoff_tier"] == "Long Shot",columns].sort_values("playoff_prob_display", ascending=False)

        bottom_left_cell = cols[0].container(border=True, height="stretch", vertical_alignment="center")
           
        with bottom_left_cell:
            st.subheader("Playoff Tiers")
            st.badge(
                "Lock",
                icon ="🔒",
                color = "green",
                width = "stretch",
                help = "These teams have a probability range of 85%–100%")
            
            teams_list = "\n".join(
                f"- {row['team_name']}: **{row['playoff_prob_display']:.1%}**"
                for _, row in lock_teams.iterrows()
            )

            st.markdown(teams_list)
            
            st.badge(
                "Likely",
                icon ="✅",
                color = "blue",
                width = "stretch",
                help = "These teams have a probability range of 65%-<85%")
            
            teams_list = "\n".join(
                f"- {row['team_name']}: **{row['playoff_prob_display']:.1%}**"
                for _, row in likely_teams.iterrows()
            )

            st.markdown(teams_list)
            
            st.badge(
                "Bubble",
                icon ="🫧",
                color = "yellow",
                width = "stretch",
                help = "These teams have a probability range of 45%–<65%")
            
            teams_list = "\n".join(
                f"- {row['team_name']}: **{row['playoff_prob_display']:.1%}**"
                for _, row in bubble_teams.iterrows()
            )

            st.markdown(teams_list)
            
            st.badge(
                "Danger",
                icon ="⚠️",
                color = "orange",
                width = "stretch",
                help = "These teams have a probability range of 25%–<45%")
            
            teams_list = "\n".join(
                f"- {row['team_name']}: **{row['playoff_prob_display']:.1%}**"
                for _, row in danger_teams.iterrows()
            )

            st.markdown(teams_list)
            
            
            st.badge(
                "Long Shot",
                icon ="❌",
                color = "red",
                width = "stretch",
                help = "These teams have a probability range of 0%–<25%")
            
            teams_list = "\n".join(
                f"- {row['team_name']}: **{row['playoff_prob_display']:.1%}**"
                for _, row in long_shot_teams.iterrows()
            )

            st.markdown(teams_list)
            

##########################################
## Teams Tab                            ##
##########################################
with tab_teams:

    cols = st.columns([1, 3])

    @st.cache_data(ttl="6h")
    def load_full_data(season_id, model_version="v6"):
        try:
            df = fetch_current_predictions(
                season_id=season_id,
                model_version=model_version
            )

            # Normalize fields
            df["label"] = df["tri_code"].str.upper().str.strip()
            df["team_name"] = df["team_name"].str.strip()
            df["season"] = df["season_id"].astype(int)

            return df

        except Exception:
            st.error("Unable to load data. Please try again later.")
            st.stop()
            return None
        
    @st.cache_data(ttl="6h")
    def load_shap_data(season_id, model_version="v6"):
        
        return fetch_shap_values(
            season_id=season_id,
            model_version=model_version
        )

    
    
    @st.cache_data(ttl="6h")
    def load_shap_summary(season_id, model_version="v6"):
        return fetch_shap_summary(
                    season_id=season_id,
                    model_version=model_version
                )
    @st.cache_data(ttl="6h")
    def load_shap_context(season_id, model_version="v6"):
        return fetch_shap_context(
            season_id=season_id,
            model_version=model_version
        )
    
    shap_context = load_shap_context(
        season_id=20262027,
        model_version="v6"
        )
    full_df = load_full_data(
        season_id=20262027,
        model_version="v6"
    )

    shap_df = load_shap_data(
        season_id=20262027,
        model_version="v6"
    )
    
    shap_summary = load_shap_summary(
        season_id=20262027,
        model_version="v6"
    )
    
    # =========================================================
    #  Division Colors
    # =========================================================
    division_colors = {
        "Atlantic": "#087e8b",
        "Metropolitan": "#ff5a5f",
        "Central": "#dcdcdd",
        "Pacific": "#3c3c3c",
        "Unknown": "#808080",
    }

    team_to_division = {
        # Atlantic
        "BOS": "Atlantic", "MTL": "Atlantic", "OTT": "Atlantic", "TOR": "Atlantic",
        "DET": "Atlantic", "BUF": "Atlantic", "TBL": "Atlantic", "FLA": "Atlantic",

        # Metropolitan
        "NYR": "Metropolitan", "NYI": "Metropolitan", "NJD": "Metropolitan",
        "WSH": "Metropolitan", "PIT": "Metropolitan", "PHI": "Metropolitan",
        "CAR": "Metropolitan", "CBJ": "Metropolitan",

        # Central
        "NSH": "Central", "DAL": "Central", "STL": "Central", "CHI": "Central",
        "MIN": "Central", "WPG": "Central", "UTA": "Central", "COL": "Central",

        # Pacific
        "CGY": "Pacific", "EDM": "Pacific", "VAN": "Pacific",
        "SEA": "Pacific", "SJS": "Pacific", "VGK": "Pacific",
        "LAK": "Pacific", "ANA": "Pacific",

        # ARI – not used anymore
        "ARI": "Unknown",
    }

    # =========================================================
    #  Inject JS to tag each multiselect chip with class "tag-XXX"
    # =========================================================
    st.markdown("""
    <script>
    const observer = new MutationObserver(() => {
        document.querySelectorAll('span[data-baseweb="tag"]').forEach(tag => {
            const text = tag.innerText.replace("×","").trim();
            if (text && !tag.classList.contains("tag-" + text)) {
                tag.classList.add("tag-" + text);
            }
        });
    });
    observer.observe(document.body, { subtree: true, childList: true });
    </script>
    """, unsafe_allow_html=True)

    # =========================================================
    #  CSS to color the chips based on division
    # =========================================================
    css_rules = "<style>"

    for team, div in team_to_division.items():
        color = division_colors.get(div, "#808080")
        css_rules += f"""
        .tag-{team} {{
            background-color: {color} !important;
            color: white !important;
            border-radius: 6px !important;
        }}
        """

    css_rules += "</style>"

    st.markdown(css_rules, unsafe_allow_html=True)
        
    # =========================================================
    # FEATURE SELECTION UI
    # =========================================================
    SHAP_FEATURE = sorted(
        shap_summary["feature"].dropna().unique(),
        key=lambda feature: get_feature_label(feature).casefold()
    )
    shap_labels = sorted(
        get_feature_label(feature)
        for feature in SHAP_FEATURE
    )
    # =========================================================
    #  TEAM SELECTION UI
    # =========================================================
    TRI_CODES = sorted(full_df["tri_code"].unique())
    available_labels = sorted(TRI_CODES)

    top_left_cell = cols[0].container(
    border=True
)

    with top_left_cell:
        st.subheader("Team Selection")

        selected_labels = st.selectbox(
        "Select Team",
        options=available_labels,
        index=available_labels.index("BOS") if "BOS" in available_labels else 0,
    )

        # A selectbox returns one team code as a string.
        selected_teams = selected_labels


    # =========================================================
    #  SEASON SELECTION — All / Custom / Range
    # =========================================================
    season_map = {
        "2013-14": 20132014,
        "2014-15": 20142015,
        "2015-16": 20152016,
        "2016-17": 20162017,
        "2017-18": 20172018,
        "2018-19": 20182019,
        "2019-20": 20192020,
        "2021-22": 20212022,
        "2022-23": 20222023,
        "2023-24": 20232024,
        "2024-25": 20242025,
        "2025-26": 20252026,
        "2026-27": 20262027,
    }
    season_labels = list(season_map.keys())


    with top_left_cell:
        st.subheader("Season Selection")

        season_mode = st.pills(
            "View",
            # options=["Current Season", "All Seasons", "Custom Selection"],
            options=["Current Season"],
            default="Current Season",
        )
        
    
    pred = full_df[full_df["tri_code"] == selected_teams].copy()
    

    # ----------------------------------------------------
    # HANDLE "Current Season"
    # ----------------------------------------------------
    if season_mode == "Current Season":

        selected_seasons = ["2026-27"]
        selected_season_ids = [season_map["2026-27"]]

        current_season_id = selected_season_ids[0]

        # Only dates from the current season
        season_dates = pd.to_datetime(
            pred.loc[
                pred["season_id"] == current_season_id,
                "game_date"
            ]
        ).dropna()
        
        if season_dates.empty:
            st.warning("No current-season data available for this team.")
            st.stop()

        
                # Reduce space above the date range box
        st.markdown(
            """
            <style>
            .st-key-date_range_box {
                margin-top: -3rem;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        with st.container(border=True, key="date_range_box"):
            start_date, end_date = st.slider(
                "Select Date Range",
                min_value=season_dates.min().date(),
                max_value=season_dates.max().date(),
                value=(
                    season_dates.min().date(),
                    season_dates.max().date()
                ),
                format="MMM DD, YYYY"
            )

        # ============================================================
        # SEASON OVERVIEW — SHAP CONSISTENCY
        # ============================================================

        if shap_df is not None and not shap_df.empty:

            filtered_shap = shap_df.copy()

            filtered_shap["game_date"] = pd.to_datetime(
                filtered_shap["game_date"]
            )

            filtered_shap = filtered_shap[
                filtered_shap["game_date"].dt.date.between(
                    start_date,
                    end_date
                )
            ]

            shap_cols = [
                c for c in filtered_shap.columns
                if c.startswith("shap_") and c != "shap_base_value"
            ]

            # Calculate using the FULL season, before slider filtering
            shap_consistency = calculate_shap_consistency(shap_df)
            # Filter selected teams
            team_shap = shap_consistency[
                shap_consistency["tri_code"] == selected_teams
            ].copy()

            # Filter displayed dates using existing slider
            team_shap = team_shap[
                team_shap["game_date"].dt.date.between(
                    start_date,
                    end_date
                )
            ].copy()

            if team_shap.empty:
                st.info("No SHAP data available for the selected dates.")

            else:

                # Display a separate chart for each selected team
                for team in [selected_teams]:

                    selected_team_shap = team_shap[
                        team_shap["tri_code"] == team
                    ].copy()

                    if selected_team_shap.empty:
                        continue

                    # Latest snapshot in selected range
                    latest_date = selected_team_shap["game_date"].max()

                    latest_shap = selected_team_shap[
                        selected_team_shap["game_date"] == latest_date
                    ].copy()

                    # ====================================================
                    # TOP FIVE CONSISTENTLY HELPFUL FEATURES
                    # ====================================================

                    # Top five strongest average positive contributors
                    positive = (
                        latest_shap[latest_shap["avg_shap"] > 0]
                        .sort_values(
                            ["avg_shap", "pct_positive"],
                            ascending=[False, False]
                        )
                        .head(5)
                        .copy()
                    )

                    positive["impact"] = "Helping"
                    positive["consistency"] = positive["pct_positive"]

                    # Top five strongest average negative contributors
                    negative = (
                        latest_shap[latest_shap["avg_shap"] < 0]
                        .sort_values(
                            ["avg_shap", "pct_negative"],
                            ascending=[True, False]
                        )
                        .head(5)
                        .copy()
                    )

                    negative["impact"] = "Hurting"
                    negative["consistency"] = negative["pct_negative"]
                    # Combine rankings
                    chart_shap = pd.concat(
                        [positive, negative],
                        ignore_index=True
                    )

                    if chart_shap.empty:
                        continue

                    # Explicit display order
                    # Sort helping first, hurting second
                    # Within each group, highest frequency first

                    chart_shap["impact_order"] = np.where(
                        chart_shap["impact"] == "Helping",
                        0,
                        1
                    )


                    # Sort ALL features by consistency percentage, highest first
                    chart_shap = chart_shap.sort_values(
                        by=["consistency", "avg_shap"],
                        ascending=[False, False]
                    ).reset_index(drop=True)

                    chart_shap["sort_order"] = np.arange(len(chart_shap))
                    chart_shap["feature_label"] = (chart_shap["feature"].apply(get_feature_label)) 
                    # Full background bar represents 100%
                    chart_shap["full_bar"] = 1.0

                    # ====================================================
                    # CHART 1 — MOST CONSISTENT FEATURES
                    # ====================================================

                    st.subheader(f"{team} — Most Consistent Playoff Drivers")

                    # ====================================================
                    # POSTGAME HOME / ROAD CONTEXT (EXPLANATORY ONLY)
                    # ====================================================
                    # Show cumulative results as of the latest SHAP snapshot
                    # in the selected range. These fields do not change SHAP.
                    if shap_context is not None and not shap_context.empty:
                        team_context = shap_context.loc[
                            (shap_context["tri_code"] == team)
                            & (
                                pd.to_datetime(
                                    shap_context["snapshot_date"],
                                    errors="coerce"
                                ) <= pd.Timestamp(latest_date)
                            )
                        ].copy()

                        if not team_context.empty:
                            team_context["snapshot_date"] = pd.to_datetime(
                                team_context["snapshot_date"]
                            )
                            context = team_context.sort_values(
                                "snapshot_date"
                            ).iloc[-1]

                            context_fields = [
                                "postgame_home_gp", "postgame_road_gp",
                                "postgame_home_points", "postgame_road_points",
                                "postgame_home_goal_diff", "postgame_road_goal_diff"
                            ]
                            if context[context_fields].notna().all():
                                home_gp = int(context["postgame_home_gp"])
                                road_gp = int(context["postgame_road_gp"])
                                home_pts = int(context["postgame_home_points"])
                                road_pts = int(context["postgame_road_points"])
                                home_gd = int(context["postgame_home_goal_diff"])
                                road_gd = int(context["postgame_road_goal_diff"])

                                st.caption(
                                    f"Completed-game results through "
                                    f"{context['snapshot_date']:%B %d, %Y} — "
                                    f"Home: {home_pts} points in {home_gp} games "
                                    f"(goal differential {home_gd:+d}) · "
                                    f"Road: {road_pts} points in {road_gp} games "
                                    f"(goal differential {road_gd:+d}). "
                                    "These are descriptive statistics, not SHAP contributions."
                                )

                    st.caption(
                        f"Through {latest_date:%B %d, %Y} — "
                        "percentage of daily prediction snapshots "
                        "each feature helped or hurt the model."
                    )

                    # Shared axis configuration
                    x_axis = alt.X(
                        "full_bar:Q",
                        title="Percentage of Snapshots",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format=".0%")
                    )

                    y_axis = alt.Y(
                        "feature_label:N",
                        title=None,
                        sort=chart_shap["feature_label"].tolist(),
                        axis=alt.Axis(
                            labelLimit=350,
                            labelFontSize=12,
                            labelPadding=10
                        )
                    )

                    # ----------------------------------------------------
                    # BACKGROUND — 100% BAR
                    # ----------------------------------------------------

                    background = (
                        alt.Chart(chart_shap)
                        .mark_bar(
                            color="#343A40",
                            cornerRadius=5,
                            size=24
                        )
                        .encode(
                            x=x_axis,
                            y=y_axis
                        )
                    )

                    # ----------------------------------------------------
                    # FOREGROUND — ACTUAL CONSISTENCY
                    # ----------------------------------------------------

                    foreground = (
                        alt.Chart(chart_shap)
                        .mark_bar(
                            cornerRadius=5,
                            size=24
                        )
                        .encode(
                            x=alt.X(
                                "consistency:Q",
                                title="Percentage of Snapshots",
                                scale=alt.Scale(domain=[0, 1]),
                                axis=alt.Axis(format=".0%")
                            ),
                            y=y_axis,
                            color=alt.Color(
                                "impact:N",
                                scale=alt.Scale(
                                    domain=["Helping", "Hurting"],
                                    range=["#46E75E", "#EA0D02"]
                                ),
                                legend=alt.Legend(title=None)
                            ),
                            tooltip=[
                                alt.Tooltip(
                                    "feature_label:N",
                                    title="Feature"
                                ),
                                alt.Tooltip(
                                    "impact:N",
                                    title="Direction of Impact"
                                ),
                                alt.Tooltip(
                                    "consistency:Q",
                                    title="Snapshot Frequency",
                                    format=".1%"
                                ),
                                alt.Tooltip(
                                    "avg_shap:Q",
                                    title="Average Impact on Playoff Chances",
                                    format="+.4f"
                                ),
                                alt.Tooltip(
                                    "snapshots:Q",
                                    title="Snapshots"
                                )
                            ]
                        )
                    )

                    # ----------------------------------------------------
                    # COMBINE CHARTS
                    # ----------------------------------------------------

                    consistency_chart = (
                            alt.layer(background, foreground)
                            .properties(
                                height=400,
                                width=650
                            )
                        )

                    st.altair_chart(
                        consistency_chart,
                        width="stretch"
                    )

                                        
                    # ====================================================
                    # DIVIDER BETWEEN CHARTS
                    # ====================================================

                    st.divider()
                    
                    # ====================================================
                    # CHART 2 — SHAP IMPACT OVER TIME
                    # ====================================================

                    st.subheader("Specific Impact Over Time")

                    # Sort features alphabetically by display label
                    SHAP_FEATURE = sorted(
                        shap_summary["feature"].dropna().unique(),
                        key=lambda feature: get_feature_label(feature).casefold()
                    )

                    # ====================================================
                    # FEATURE SELECTION
                    # ====================================================

                    with st.container(border=True):

                        selected_feature = st.selectbox(
                            "Select a Feature",
                            options=SHAP_FEATURE,
                            format_func=get_feature_label,
                            index=0,
                            key="shap_feature_selection"
                        )

                    # ====================================================
                    # PREPARE DATA
                    # ====================================================

                    shap_column = f"shap_{selected_feature}"

                    if shap_column not in shap_df.columns:

                        st.warning(
                            f"Data is unavailable for {get_feature_label(selected_feature)}."
                        )

                    else:

                        # Filter to selected team
                        plot_df = shap_df.loc[
                            shap_df["tri_code"] == selected_teams,
                            ["snapshot_date", shap_column]
                        ].copy()

                        # Normalize dates and values
                        plot_df["snapshot_date"] = pd.to_datetime(
                            plot_df["snapshot_date"]
                        )

                        plot_df[shap_column] = pd.to_numeric(
                            plot_df[shap_column],
                            errors="coerce"
                        )

                        # Remove missing values
                        plot_df = (
                            plot_df
                            .dropna(subset=["snapshot_date", shap_column])
                            .sort_values("snapshot_date")
                            .rename(columns={shap_column: "shap_value"})
                        )

                        # Apply existing date slider
                        plot_df = plot_df[
                            plot_df["snapshot_date"].dt.date.between(
                                start_date,
                                end_date
                            )
                        ].copy()

                        # Prevent duplicate snapshots
                        plot_df = plot_df.drop_duplicates(
                            subset=["snapshot_date"],
                            keep="last"
                        )

                        # ====================================================
                        # CREATE CHART
                        # ====================================================

                        if plot_df.empty:

                            st.info(
                                "No data available for the selected "
                                "team, feature, and date range."
                            )

                        else:

                            # ------------------------------------------------
                            # SHARED CHART ENCODING
                            # ------------------------------------------------

                            base = alt.Chart(plot_df).encode(
                                x=alt.X(
                                    "snapshot_date:T",
                                    title="Snapshot Date",
                                    axis=alt.Axis(
                                        format="%b %d",
                                        labelAngle=-45
                                    )
                                )
                            )

                            # ------------------------------------------------
                            # POSITIVE SHAP AREA
                            # ------------------------------------------------

                            positive_area = (
                                base
                                .mark_area(
                                    color="#46E75E",
                                    opacity=0.20
                                )
                                .encode(
                                    y=alt.Y(
                                        "shap_value:Q"
                                    ),
                                    y2=alt.Y2(datum=0)
                                )
                                .transform_calculate(
                                    shap_value="max(datum.shap_value, 0)"
                                )
                            )

                            # ------------------------------------------------
                            # NEGATIVE SHAP AREA
                            # ------------------------------------------------

                            negative_area = (
                                base
                                .mark_area(
                                    color="#EA0D02",
                                    opacity=0.20
                                )
                                .encode(
                                    y=alt.Y(
                                        "shap_value:Q"
                                    ),
                                    y2=alt.Y2(datum=0)
                                )
                                .transform_calculate(
                                    shap_value="min(datum.shap_value, 0)"
                                )
                            )

                            # ------------------------------------------------
                            # SHAP IMPACT LINE
                            # ------------------------------------------------

                            shap_line = (
                                base
                                .mark_line(
                                    color="#4DA6FF",
                                    strokeWidth=2.5,
                                    point=alt.OverlayMarkDef(
                                        filled=True,
                                        size=65
                                    )
                                )
                                .encode(
                                    y=alt.Y(
                                        "shap_value:Q"
                                    ),
                                    tooltip=[
                                        alt.Tooltip(
                                            "snapshot_date:T",
                                            title="Snapshot Date",
                                            format="%b %d, %Y"
                                        ),
                                        alt.Tooltip(
                                            "shap_value:Q",
                                            title="Feature Impact",
                                            format="+.4f"
                                        )
                                    ]
                                )
                            )

                            # ------------------------------------------------
                            # ZERO REFERENCE LINE
                            # ------------------------------------------------

                            zero_line = (
                                alt.Chart(
                                    pd.DataFrame({"zero": [0]})
                                )
                                .mark_rule(
                                    color="#AAAAAA",
                                    strokeDash=[5, 5],
                                    opacity=0.7
                                )
                                .encode(
                                    y="zero:Q"
                                )
                            )

                            # ------------------------------------------------
                            # COMBINE CHART
                            # ------------------------------------------------

                            shap_impact_chart = (
                                alt.layer(
                                    positive_area,
                                    negative_area,
                                    zero_line,
                                    shap_line
                                )
                                .encode(
                                    y=alt.Y(
                                        "shap_value:Q",
                                        title="Impact on Playoff Chances"
                                    )
                                )
                                .properties(
                                    height=350
                                )
                                .interactive()
                            )

                            # ------------------------------------------------
                            # TEAM, FEATURE TITLE AND EXPLANATION
                            # ------------------------------------------------

                            st.markdown(
                                f"#### {selected_teams} — "
                                f"{get_feature_label(selected_feature)}"
                            )

                            st.markdown(
                                get_feature_description(selected_feature)
                            )
                            
                            # ====================================================
                            # HOME / ROAD PERFORMANCE CONTEXT
                            # ====================================================

                            context_feature_map = {
                                "home_points": ("postgame_home_points", "points", "Home"),
                                "road_points": ("postgame_road_points", "points", "Road"),
                                "home_goal_diff": ("postgame_home_goal_diff", "goal differential", "Home"),
                                "road_goal_diff": ("postgame_road_goal_diff", "goal differential", "Road"),
                                "home_gp": ("postgame_home_gp", "games played", "Home"),
                                "road_gp": ("postgame_road_gp", "games played", "Road"),
                            }

                            if selected_feature in context_feature_map:

                                context_column, unit, venue = context_feature_map[selected_feature]

                                feature_context = shap_context[
                                    shap_context["tri_code"] == selected_teams
                                ].copy()

                                feature_context["snapshot_date"] = pd.to_datetime(
                                    feature_context["snapshot_date"]
                                )

                                if not plot_df.empty:

                                    displayed_snapshot = plot_df["snapshot_date"].max()

                                    feature_context = feature_context[
                                        feature_context["snapshot_date"] == displayed_snapshot
                                    ]

                                    if not feature_context.empty:

                                        context_row = feature_context.iloc[-1]
                                        actual_value = context_row[context_column]

                                        if pd.notna(actual_value):

                                            formatted_value = (
                                                f"{int(actual_value):+d}"
                                                if "goal_diff" in selected_feature
                                                else f"{int(actual_value)}"
                                            )

                                            st.caption(
                                                f"Actual {venue.lower()} performance through "
                                                f"{displayed_snapshot:%B %d, %Y}: "
                                                f"{formatted_value} {unit}. "
                                                "This is postgame context, not the "
                                                "pregame value used by the model."
                                            )

                            # ------------------------------------------------
                            # DISPLAY CHART
                            # ------------------------------------------------

                            st.altair_chart(
                                shap_impact_chart,
                                width="stretch"
                            )

                            # ====================================================
                            # CHART EXPLANATION
                            # ====================================================

                            st.caption(
                                "Green means this feature is helping the team's "
                                "playoff chances. Red means it's hurting them. "
                                "The values show the strength of the impact, "
                                "not the percentage change."
                            )

                                                        
                    # ====================================================
                    # DIVIDER BETWEEN CHARTS
                    # ====================================================

                    st.divider()
       
                    # ====================================================
                    # CHART 3 — SEASON PROGRESSION
                    # ====================================================

                    st.subheader("How Playoff Drivers Evolve")

                    # Keep only features from the current top-five rankings
                    progression_parts = []

                    for _, row in chart_shap.iterrows():

                        feature_data = team_shap[
                            team_shap["feature"] == row["feature"]
                        ].copy()

                        if feature_data.empty:
                            continue

                        feature_data["impact"] = row["impact"]

                        feature_data["frequency"] = (
                            feature_data["pct_positive"]
                            if row["impact"] == "Helping"
                            else feature_data["pct_negative"]
                        )

                        # Convert feature name to plain English
                        feature_data["feature_label"] = get_feature_label(
                            row["feature"]
                        )

                        # Unique series name for chart colors
                        feature_data["series"] = (
                            feature_data["feature_label"]
                            + " ("
                            + row["impact"]
                            + ")"
                        )

                        progression_parts.append(feature_data)


                    if progression_parts:

                        progression_df = pd.concat(
                            progression_parts,
                            ignore_index=True
                        )

                        # ----------------------------------------------------
                        # CONSISTENT COLORS FOR CHART AND LEGEND
                        # ----------------------------------------------------

                        series_order = (
                            progression_df["series"]
                            .drop_duplicates()
                            .tolist()
                        )

                        legend_colors = [
                            "#72C4F7",
                            "#0074D9",
                            "#FFB3BA",
                            "#FF3038",
                            "#75E8A5",
                            "#20B2AA",
                            "#FFD166",
                            "#FF9500",
                            "#753BBD",
                            "#D9DDE5"
                        ]

                        color_map = {
                            series: legend_colors[i % len(legend_colors)]
                            for i, series in enumerate(series_order)
                        }

                        # ----------------------------------------------------
                        # SEASON PROGRESSION CHART
                        # ----------------------------------------------------

                        progression_chart = (
                            alt.Chart(progression_df)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X(
                                    "game_date:T",
                                    title="Snapshot Date"
                                ),
                                y=alt.Y(
                                    "frequency:Q",
                                    title="Season-to-Date Snapshot Frequency",
                                    scale=alt.Scale(domain=[0, 1]),
                                    axis=alt.Axis(format=".0%")
                                ),
                                color=alt.Color(
                                    "series:N",
                                    scale=alt.Scale(
                                        domain=series_order,
                                        range=[
                                            color_map[s]
                                            for s in series_order
                                        ]
                                    ),
                                    legend=None
                                ),
                                strokeDash=alt.StrokeDash(
                                    "impact:N",
                                    scale=alt.Scale(
                                        domain=["Helping", "Hurting"],
                                        range=[[1, 0], [5, 4]]
                                    ),
                                    legend=None
                                ),
                                tooltip=[
                                    alt.Tooltip(
                                        "feature_label:N",
                                        title="Feature"
                                    ),
                                    alt.Tooltip(
                                        "impact:N",
                                        title="Direction of Impact"
                                    ),
                                    alt.Tooltip(
                                        "game_date:T",
                                        title="Date",
                                        format="%b %d, %Y"
                                    ),
                                    alt.Tooltip(
                                        "frequency:Q",
                                        title="Season-to-Date Frequency",
                                        format=".1%"
                                    ),
                                    alt.Tooltip(
                                        "avg_shap:Q",
                                        title="Average Impact on Playoff Prediction",
                                        format="+.4f"
                                    ),
                                    alt.Tooltip(
                                        "snapshots:Q",
                                        title="Snapshots Analyzed"
                                    )
                                ]
                            )
                            .properties(height=800)
                            .interactive()
                        )

                        st.altair_chart(
                            progression_chart,
                            width="stretch"
                        )

                        # ----------------------------------------------------
                        # CUSTOM TWO-COLUMN LEGEND
                        # ----------------------------------------------------

                        st.markdown("#### Feature Legend")

                        legend_df = (
                            progression_df[
                                ["series", "feature_label", "impact"]
                            ]
                            .drop_duplicates()
                        )

                        col_helping, col_hurting = st.columns(
                            2,
                            gap="small"
                        )

                        # HELPING FEATURES — LEFT COLUMN
                        with col_helping:

                            st.markdown(
                                """
                                    <div style="
                                        font-size:20px;
                                        font-weight:700;
                                        color:#D1D5DB;
                                        margin-bottom: 2px;
                                    ">
                                        Helping Features are Solid Lines
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            helping_features = legend_df[
                                legend_df["impact"] == "Helping"
                            ]

                            for _, feature in helping_features.iterrows():

                                color = color_map[feature["series"]]

                                st.markdown(
                                    f'<div style="'
                                    f'display:flex;'
                                    f'align-items:baseline;'
                                    f'gap:1px;'
                                    f'margin-bottom:0px;'
                                    f'">'
                                    f'<span style="'
                                    f'color:{color};'
                                    f'font-size:20px;'
                                    f'flex-shrink:0;'
                                    f'">●</span>'
                                    f'<span style="'
                                    f'color:#D1D5DB;'
                                    f'font-size:14px;'
                                    f'line-height:1;'
                                    f'overflow-wrap:anywhere;'
                                    f'">{html.escape(str(feature["feature_label"]))}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )

                        # HURTING FEATURES — RIGHT COLUMN
                        with col_hurting:
                            st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)
                            st.markdown(
                                """
                                <div style="
                                    font-size:20px;
                                    font-weight:700;
                                    color:#D1D5DB;
                                    margin-bottom: 2px;
                                ">
                                    Hurting Features are Dashed Lines
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            hurting_features = legend_df[
                                legend_df["impact"] == "Hurting"
                            ]

                            for _, feature in hurting_features.iterrows():

                                color = color_map[feature["series"]]

                                st.markdown(
                                    f'<div style="'
                                    f'display:flex;'
                                    f'align-items:baseline;'
                                    f'gap:10px;'
                                    f'margin-bottom:0px;'
                                    f'">'
                                    f'<span style="'
                                    f'color:{color};'
                                    f'font-size:20px;'
                                    f'flex-shrink:0;'
                                    f'">●</span>'
                                    f'<span style="'
                                    f'color:#D1D5DB;'
                                    f'font-size:14px;'
                                    f'line-height:1;'
                                    f'overflow-wrap:anywhere;'
                                    f'">{html.escape(str(feature["feature_label"]))}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )

                    else:
                        st.info("No feature progression data available.")

                                   
    # ===============================================
    # CUSTOM SELECTION — MULTI-PICK PILL BUTTONS
    # ===============================================
    elif season_mode == "Custom Selection":

        # Initialize session state on first load
        if "selected_custom_seasons" not in st.session_state:
            st.session_state.selected_custom_seasons = []

        st.write("Choose seasons:")

        # Layout pills in rows
        pill_cols = st.columns(3)

        for i, season in enumerate(season_labels):
            col = pill_cols[i % 3]

            # Whether this pill is ON
            active = season in st.session_state.selected_custom_seasons

            # Button label styling
            label = f"✔️ {season}" if active else season

            # Clicking toggles ON/OFF
            if col.button(label, key=f"pill_{season}"):
                if active:
                    st.session_state.selected_custom_seasons.remove(season)
                else:
                    st.session_state.selected_custom_seasons.append(season)

        selected_seasons = st.session_state.selected_custom_seasons[:]

    # ===============================================
    # ALL SEASONS
    # ===============================================
    elif season_mode == "All Seasons":
        selected_seasons = season_labels[:]

    # Convert selected season labels → numeric IDs
    selected_season_ids = [season_map[s] for s in selected_seasons]


    # =========================================================
    # FILTER DATA
    # =========================================================


    if selected_season_ids:
        pred = pred[pred["season"].isin(selected_season_ids)]

    # pred["made_playoffs_label"] = pred["made_playoffs"].map({1: "Yes", 0: "No"}).fillna("TBD")
    # pred['playoff_pred_label'] = pred['playoff_prob_display'].map({1: "Yes", 0: "No"})  

    # =========================================================
    # RIGHT PANEL — ALTair Visualization
    # =========================================================
    right_cell = cols[1].container(border=True, height="stretch", vertical_alignment="center")

    team_colors = {
    "ANA": "#F47A38",
    "ARI": "#8C2633",
    "BOS": "#FFB81C",
    "BUF": "#A0C5F1",
    "CAR": "#CC0000",
    "CBJ": "#5F89BC",
    "CGY": "#C8102E",
    "CHI": "#CF0A2C",
    "COL": "#6F263D",
    "DAL": "#006847",
    "DET": "#CE1126",
    "EDM": "#FF4C00",
    "FLA": "#B9975B",
    "LAK": "#C64EF2",
    "MIN": "#154734",
    "MTL": "#AF1E2D",
    "NJD": "#CE1126",
    "NSH": "#FFB81C",
    "NYI": "#F68427",
    "NYR": "#0038A8",
    "OTT": "#DA1A32",
    "PHI": "#F74902",
    "PIT": "#FCB514",
    "SEA": "#1CD1D1",
    "SJS": "#006D75",
    "STL": "#3174F2",
    "TBL": "#A6BBDD",
    "TOR": "#0651DC",
    "UTA": "#05F9F9",
    "VAN": "#C6E8F8",
    "VGK": "#B4975A",
    "WPG": "#99B8E2",
    "WSH": "#C8102E",
    }


    with right_cell:
        st.subheader("Playoff Probability at Thanksgiving")

        if pred.empty:
            st.warning("No data to plot.")
        else:

            # ---------------------------------------------------------------
            # Build dynamic domain & range for selected teams ONLY
            # ---------------------------------------------------------------
            color_domain = [t for t in pred["tri_code"].unique() if t in team_colors]
            color_range = [team_colors[t] for t in color_domain]

            # ---------------------------------------------------------------
            # 1. SINGLE SEASON → Line Chart
            # ---------------------------------------------------------------
            if season_mode == "Current Season":

                season_label = selected_seasons[0]

                chart_pred = pred.copy()
                chart_pred["game_date"] = pd.to_datetime(
                    chart_pred["game_date"]
                )

                # Filter to the selected date range
                chart_pred = chart_pred[
                    chart_pred["game_date"].dt.date.between(
                        start_date,
                        end_date
                    )
                ].copy()

                chart_pred = chart_pred.sort_values(
                    ["tri_code", "game_date"]
                )

                chart = (
                    alt.Chart(chart_pred)
                    .mark_line(
                        strokeWidth=2,
                        point=True
                    )
                    .encode(
                        x=alt.X(
                            "game_date:T",
                            title="Game Date",
                            axis=alt.Axis(
                                format="%b %d",
                                labelAngle=0
                            )
                        ),
                        y=alt.Y(
                            "playoff_prob_display:Q",
                            title="Playoff Probability",
                            scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(format=".0%")
                        ),
                        color=alt.Color(
                            "tri_code:N",
                            title="Team",
                            scale=alt.Scale(
                                domain=color_domain,
                                range=color_range
                            ),
                            legend=alt.Legend(
                                orient="bottom",
                                direction="horizontal",
                                columns=16,
                                padding=0,
                                titleAnchor="middle",
                                symbolSize=120,
                                labelLimit=600
                            )
                        ),
                        tooltip=[
                            alt.Tooltip("team_name:N", title="Team"),
                            alt.Tooltip("game_date:T", title="Date", format="%b %d, %Y"),
                            alt.Tooltip("playoff_prob_display:Q", title="Probability", format=".1%"),
                            alt.Tooltip("playoff_tier:N", title="Playoff Prediction"),
                        ],
                        order=alt.Order("game_date:T")
                    )
                    .properties(
                        title=f"Playoff Probability Over Time — {season_label}",
                        height=500
                    )
                    .configure_legend(
                        orient="bottom",
                        padding=0,
                        labelLimit=600,
                        title=None
                    )
                )

                st.altair_chart(chart, width="stretch")

            # ---------------------------------------------------------------
            # 2. MULTIPLE SEASONS → LINE CHART
            # ---------------------------------------------------------------

            elif season_mode in ["All Seasons", "Custom Selection"]:
                chart = (
                    alt.Chart(pred)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(
                            "season:O",
                            title="Season",
                            sort="ascending",
                            axis=alt.Axis(labelAngle=0)
                        ),
                        y=alt.Y(
                            "playoff_prob:Q",
                            title="Probability",
                            scale=alt.Scale(domain=[0, 1])
                        ),
                        color=alt.Color(
                            "tri_code:N",
                            title="Team",
                            scale=alt.Scale(
                                domain=color_domain,
                                range=color_range
                            ),
                            legend=alt.Legend(
                                orient="bottom",
                                direction="horizontal",
                                columns=16,
                                padding=0,        # ← tighten spacing
                                titleAnchor="middle",
                                symbolSize=120,
                                labelLimit=600
                            )
                        ),
                        tooltip=[
                            # alt.Tooltip("tri_code", title="Team"),
                            alt.Tooltip("team_name", title="Team"),
                            alt.Tooltip("season:O", title="Season"),
                            alt.Tooltip("playoff_prob:Q", title="Probability", format=".1%"),
                            alt.Tooltip("playoff_tier:N", title="Playoff Prediction"),
                            alt.Tooltip("made_playoffs_label:N", title="Made Playoffs"),
                        ],
                        
                    )
                    .properties(height=500)
                    .configure_legend(
                        orient="bottom",
                        padding=0,             # ← eliminates huge bottom gap
                        labelLimit=600,
                        title=None
                    )
                )

                st.altair_chart(chart, width='stretch')
  
                

    # =========================================================
    # BEST + WORST PLAYOFF PROBABILITY -- FIX THIS 
    # =========================================================


    team_pred = pred[
        pred["tri_code"] == selected_teams
    ].copy()

    if not team_pred.empty:

        # Highest and lowest probabilities for selected team
        best_row = team_pred.loc[team_pred["playoff_prob"].idxmax()]
        worst_row = team_pred.loc[team_pred["playoff_prob"].idxmin()]

        best_prob = best_row["playoff_prob"]
        worst_prob = worst_row["playoff_prob"]

        best_against = best_row["opponent_name"]
        worst_against = worst_row["opponent_name"]

        best_date = pd.to_datetime(best_row["game_date"])
        worst_date = pd.to_datetime(worst_row["game_date"])

        bottom_left_cell = cols[0].container(
            border=True,
            height="stretch",
            horizontal_alignment="center",
            vertical_alignment="top"
        )

        with bottom_left_cell:
            st.subheader("Season Highlights")

            mcols = st.columns(2, 
                               gap="small",
                               vertical_alignment="top")
            
            st.markdown("""
                <style>
                /* Center the entire metric */
                div[data-testid="stMetric"] {
                    width: 100%;
                    text-align: center !important;
                }

                /* Center the metric label container */
                div[data-testid="stMetricLabel"] {
                    display: flex !important;
                    width: 100% !important;
                    justify-content: center !important;
                    align-items: center !important;
                    text-align: center !important;
                }

                /* Center the actual label text */
                div[data-testid="stMetricLabel"] p {
                    width: 100% !important;
                    text-align: center !important;
                }

                /* Center the metric value */
                div[data-testid="stMetricValue"] {
                    display: flex !important;
                    width: 100% !important;
                    justify-content: center !important;
                    text-align: center !important;
                }
                </style>
                """, unsafe_allow_html=True)

            with mcols[0]:
                
                st.markdown(
                    "<p style='text-align:center; margin-bottom:0;'>Best Probability</p>",
                    unsafe_allow_html=True
                )
                    
                st.metric(
                    label="Best Probability",
                    value=f"{best_prob*100:.1f}%",
                    label_visibility="collapsed",
                    width = "stretch",
                    
                )

                st.markdown(
                    f"""
                    <p style="
                        text-align: center;
                        color: gray;
                        margin-top: -20px;
                        margin-bottom: 0;
                    ">
                    <b>{best_date:%b %d}</b>

                    </p>
                    """,
                    unsafe_allow_html=True
                )

            with mcols[1]:
                st.markdown(
                        "<p style='text-align:center; margin-bottom:0;'>Worst Probability</p>",
                        unsafe_allow_html=True
                    )

                
                                
                st.metric(
                    label="Worst Probability",
                    value=f"{worst_prob*100:.1f}%",
                    label_visibility="collapsed",
                    width="stretch"
                )

                st.markdown(
                    f"""
                    <p style="
                        text-align: center;
                        color: gray;
                        margin-top: -20px;
                        margin-bottom: 0;
                    ">
                    <b>{worst_date:%b %d}</b>           
                    </p>
                    """,
                    unsafe_allow_html=True
                )
    
        # chart_shap = pd.concat(
        #     [positive, negative],
        #     ignore_index=True
        # )

        # chart_shap["impact"] = np.where(
        #     chart_shap["avg_shap"] > 0,
        #     "Positive",
        #     "Negative"
        # )
        

        # # # Explicit position for the chart
        # # Sort positives descending, negatives ascending
        # positive = chart_shap[chart_shap["avg_shap"] > 0].sort_values(
        #     "avg_shap", ascending=False
        # )

        # negative = chart_shap[chart_shap["avg_shap"] < 0].sort_values(
        #     "avg_shap", ascending=True
        # )

        # chart_shap = pd.concat(
        #     [positive, negative],
        #     ignore_index=True
        # )

        # chart_shap["sort_order"] = np.arange(len(chart_shap))

        # # Consistency for the direction of each feature
        # chart_shap["consistency"] = np.where(
        #     chart_shap["avg_shap"] > 0,
        #     chart_shap["days_positive"],
        #     chart_shap["days_negative"]
        # )
        # #=======================================
        # # SHAP IMPACTS
        # #=======================================
        # chart = (
        #     alt.Chart(chart_shap)
        #     .mark_bar()
        #     .encode(
        #         x=alt.X(
        #             "avg_shap:Q",
        #             title="Average SHAP Contribution (Log-Odds)"
        #         ),
        #         y=alt.Y(
        #             "feature:N",
        #             title=None,
        #             sort=alt.SortField(
        #                 field="sort_order",
        #                 order="descending"
        #             )
        #         ),
        #         color=alt.Color(
        #             "impact:N",
        #             scale=alt.Scale(
        #                 domain=["Positive", "Negative"],
        #                 range=["#46E75E", "#EA0D02"]
        #             ),
        #             legend=None
        #         ),
        #         tooltip=[
        #             alt.Tooltip("tri_code:N", title="Team"),
        #             alt.Tooltip("feature:N", title="Feature"),
        #             alt.Tooltip(
        #                 "avg_shap:Q",
        #                 title="Average Contribution",
        #                 format="+.4f"
        #             ),
        #             alt.Tooltip(
        #                 "consistency:Q",
        #                 title="Days Helping/Hurting",
        #                 format=".1%"
        #             ),
        #             alt.Tooltip(
        #                 "observations:Q",
        #                 title="Observations"
        #             )
        #         ]
        #     )
        #     .properties(height=400)
        # )

        # st.altair_chart(chart, width="stretch")
    
            # top_shap = get_top_shap_features(
            #     filtered_shap,
            #     top_n=5
            # ).copy()
    
            # # Sort positive values descending, then negative values ascending
            # top_shap["sort_group"] = np.where(
            #     top_shap["shap_value"] > 0,
            #     0,
            #     1
            # )
    
            # top_shap = top_shap.sort_values(
            #     by=["sort_group", "shap_value"],
            #     ascending=[True, False]
            # ).copy()
    
    
            # # Sum each feature's contributions across the selected date range
            # cumulative_shap = (
            #     get_top_shap_features(filtered_shap, top_n=5)
            #     .groupby("feature", as_index=False)["shap_value"]
            #     .sum()
            # )
    
            # # Sort positive descending, negative ascending
            # positive = cumulative_shap[
            #     cumulative_shap["shap_value"] > 0
            # ].sort_values("shap_value", ascending=False).head(5)
    
            # negative = cumulative_shap[
            #     cumulative_shap["shap_value"] < 0
            # ].sort_values("shap_value", ascending=True).head(5)
    
            # chart_shap = pd.concat(
            #     [positive, negative],
            #     ignore_index=True
            # )
    
            # chart_shap["impact"] = np.where(
            #     chart_shap["shap_value"] > 0,
            #     "Positive",
            #     "Negative"
            # )
    
            # chart_shap["sort_order"] = np.arange(len(chart_shap))
    
    
            # # Negatives need ascending order (most negative first)
            # positive = cumulative_shap[cumulative_shap["shap_value"] > 0].sort_values(
            #     "shap_value", ascending=False
            # )
    
            # negative = cumulative_shap[cumulative_shap["shap_value"] < 0].sort_values(
            #     "shap_value", ascending=True
            # )
    
            # chart_shap = pd.concat(
            #     [positive, negative],
            #     ignore_index=True
            # )
            
        # base = alt.Chart(chart_shap)
        
        # left = base.transform_filter(
        #     alt.datum.impact =='Negative')
        # .encode(
        #     alt.Y('feature:N'),
        #         title= "Negative",
        #         sort=alt.SortField(
        #             field="sort_order",
        #             order="descending")
        #     alt.X()  
        # )
        # st.altair_chart(chart, width="stretch")
        
        

        

# ##########################################
# ## What Ifs Tab                            ##
# ##########################################
with tab_whatif:
    st.markdown("### Coming Soon")