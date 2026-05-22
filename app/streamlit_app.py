import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
import sys
sys.path.append(".")
from src.evaluation import evaluate_model_by_cutoff, predict_final_table

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
    Dashboard analyzes Prem team performance and shows how prediction
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


## Forecast evaluation section
st.header("Forecast Evaluation Across Matchweeks")

st.write(
    """
    This section evaluates how accurately the model predicts final season points
    when only the first N matches of a season are available.
    """
)

test_season = st.selectbox(
    "Select test season for forecasting evaluation",
    sorted(season_table["Season"].unique()),
    index=max(0, len(sorted(season_table["Season"].unique())) - 2)
)

cutoffs = [5, 10, 15, 20, 25, 30]

linear_results = evaluate_model_by_cutoff(
    team_matches=team_matches,
    season_table=season_table,
    cutoffs=cutoffs,
    test_season=test_season,
    model_type="linear"
)

rf_results = evaluate_model_by_cutoff(
    team_matches=team_matches,
    season_table=season_table,
    cutoffs=cutoffs,
    test_season=test_season,
    model_type="random_forest"
)

linear_results["Model"] = "Linear Regression"
rf_results["Model"] = "Random Forest"

eval_results = pd.concat(
    [linear_results, rf_results],
    ignore_index=True
)

st.subheader("MAE by Matchweek")

fig_mae = px.line(
    eval_results,
    x="Cutoff",
    y="MAE",
    color="Model",
    markers=True,
    title="Prediction Error vs Season Progress"
)

fig_mae.update_layout(
    xaxis_title="Matchweek Cutoff",
    yaxis_title="MAE (Final Points Prediction)"
)

st.plotly_chart(fig_mae, width="stretch")

st.subheader("R² by Matchweek")

fig_r2 = px.line(
    eval_results,
    x="Cutoff",
    y="R2",
    color="Model",
    markers=True,
    title="Explained Variance vs Season Progress"
)

fig_r2.update_layout(
    xaxis_title="Matchweek Cutoff",
    yaxis_title="R²"
)

st.plotly_chart(fig_r2, width="stretch")

st.subheader("Evaluation Results")

st.dataframe(
    eval_results.sort_values(["Cutoff", "Model"]),
    width="stretch"
)

st.info(
    """
    As more matches are played, prediction error generally decreases.
    This shows that the league table becomes more stable and predictable
    as the season progresses.
    """
)
st.write(team_matches.columns)

#######
# Predicted final table section
st.header("Predicted Final Table")

st.write(
    """
    Select a season, matchweek cutoff, and model type to simulate what the
    final league table would have looked like based only on information
    available up to that point.
    """
)

pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    pred_season = st.selectbox(
        "Prediction season",
        sorted(season_table["Season"].unique()),
        index=max(0, len(sorted(season_table["Season"].unique())) - 2),
        key="pred_season"
    )

with pred_col2:
    pred_cutoff = st.selectbox(
        "Matchweek cutoff",
        [5, 10, 15, 20, 25, 30],
        index=2
    )

with pred_col3:
    pred_model = st.selectbox(
        "Model type",
        ["linear", "random_forest"]
    )

pred_table = predict_final_table(
    team_matches=team_matches,
    season_table=season_table,
    cutoff=pred_cutoff,
    test_season=pred_season,
    model_type=pred_model
)

display_pred_table = pred_table[[
    "PredictedRank",
    "Team",
    "PointsN",
    "GFN",
    "GAN",
    "PredictedFinalPoints",
    "Points",
    "PredictionError"
]].copy()

display_pred_table["PredictedFinalPoints"] = (
    display_pred_table["PredictedFinalPoints"].round(1)
)

display_pred_table["PredictionError"] = (
    display_pred_table["PredictionError"].round(1)
)

display_pred_table = display_pred_table.rename(columns={
    "PointsN": f"Points After MW{pred_cutoff}",
    "GFN": f"Goals For After MW{pred_cutoff}",
    "GAN": f"Goals Against After MW{pred_cutoff}",
    "Points": "Actual Final Points"
})

st.dataframe(
    display_pred_table,
    width="stretch"
)

st.info(
    """
    This table ranks teams by projected final points. The actual final points
    column is included for completed seasons so you can compare predictions
    against reality.
    """
)