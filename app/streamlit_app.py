import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Premier League Forecasting Dashboard",
    layout="wide"
)

DATA_DIR = Path("data/processed")

@st.cache_data
def load_data():
    season_table = pd.read_csv(DATA_DIR / "season_table.csv")
    team_matches = pd.read_csv(DATA_DIR / "team_matches.csv")
    model_data = pd.read_csv(DATA_DIR / "model_data.csv")
    return season_table, team_matches, model_data

season_table, team_matches, model_data = load_data()

st.title("⚽ Premier League Forecasting Dashboard")

st.write(
    """
    This dashboard analyzes Premier League team performance and shows how prediction
    accuracy improves as the season progresses.
    """
)

# Sidebar
seasons = sorted(season_table["Season"].unique())
selected_season = st.sidebar.selectbox("Select Season", seasons, index=len(seasons)-1)

# League table
st.header("League Table")

table = (
    season_table[season_table["Season"] == selected_season]
    .assign(GD=lambda d: d["GoalsFor"] - d["GoalsAgainst"])
    .sort_values(["Points", "GD", "GoalsFor"], ascending=False)
    .reset_index(drop=True)
)

table.insert(0, "Rank", table.index + 1)

st.dataframe(
    table[["Rank", "Team", "Matches", "Points", "GD", "GoalsFor", "GoalsAgainst"]],
    use_container_width=True
)

# Bar chart
st.header("Points by Team")

fig = px.bar(
    table,
    x="Team",
    y="Points",
    title=f"Points by Team — {selected_season}",
    text="Points"
)

fig.update_layout(xaxis_tickangle=-45)

st.plotly_chart(fig, use_container_width=True)

# Team view
st.header("Team Form")

teams = sorted(table["Team"].unique())
selected_team = st.selectbox("Select Team", teams)

team_data = (
    team_matches[
        (team_matches["Season"] == selected_season) &
        (team_matches["Team"] == selected_team)
    ]
    .sort_values("MatchNumber")
)

fig2 = px.line(
    team_data,
    x="MatchNumber",
    y="Points",
    markers=True,
    title=f"{selected_team} Points by Match"
)

st.plotly_chart(fig2, use_container_width=True)

# Rolling form
fig3 = px.line(
    team_data,
    x="MatchNumber",
    y="Points_Last5",
    markers=True,
    title=f"{selected_team} Rolling Last-5 Points"
)

st.plotly_chart(fig3, use_container_width=True)