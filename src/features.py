import pandas as pd
import glob


def load_all_seasons(path="../data/raw/epl_*.csv"):
    """
    Load and combine all EPL season CSVs.
    """

    files = glob.glob(path)

    dfs = []

    for f in files:
        df = pd.read_csv(f)
        df["Season"] = f.split("/")[-1].replace(".csv", "")
        dfs.append(df)

    all_seasons = pd.concat(dfs, ignore_index=True)

    return all_seasons


def add_match_points(df):
    """
    Convert match results into numeric points.
    """

    df = df.copy()

    df["HomePoints"] = df["FTR"].map({
        "H": 3,
        "D": 1,
        "A": 0
    })

    df["AwayPoints"] = df["FTR"].map({
        "A": 3,
        "D": 1,
        "H": 0
    })

    return df


def build_season_table(all_seasons):
    """
    Build season-level team table.
    """

    home = (
        all_seasons
        .groupby(["Season", "HomeTeam"])
        .agg(
            matches_home=("HomeTeam", "count"),
            goals_for_home=("FTHG", "sum"),
            goals_against_home=("FTAG", "sum"),
            points_home=("HomePoints", "sum"),
        )
        .reset_index()
        .rename(columns={"HomeTeam": "Team"})
    )

    away = (
        all_seasons
        .groupby(["Season", "AwayTeam"])
        .agg(
            matches_away=("AwayTeam", "count"),
            goals_for_away=("FTAG", "sum"),
            goals_against_away=("FTHG", "sum"),
            points_away=("AwayPoints", "sum"),
        )
        .reset_index()
        .rename(columns={"AwayTeam": "Team"})
    )

    season_table = home.merge(
        away,
        on=["Season", "Team"],
        how="outer"
    )

    season_table["Matches"] = (
        season_table["matches_home"]
        + season_table["matches_away"]
    )

    season_table["GoalsFor"] = (
        season_table["goals_for_home"]
        + season_table["goals_for_away"]
    )

    season_table["GoalsAgainst"] = (
        season_table["goals_against_home"]
        + season_table["goals_against_away"]
    )

    season_table["Points"] = (
        season_table["points_home"]
        + season_table["points_away"]
    )

    return season_table


def create_team_match_table(all_seasons):
    """
    Create long-format team match table.
    """

    matches = all_seasons.copy()

    matches["Date"] = pd.to_datetime(
        matches["Date"],
        dayfirst=True
    )

    matches = matches.sort_values(["Season", "Date"])

    home_rows = matches[[
        "Season",
        "Date",
        "HomeTeam",
        "FTHG",
        "FTAG",
        "HomePoints"
    ]].rename(columns={
        "HomeTeam": "Team",
        "FTHG": "GoalsFor",
        "FTAG": "GoalsAgainst",
        "HomePoints": "Points"
    })

    away_rows = matches[[
        "Season",
        "Date",
        "AwayTeam",
        "FTAG",
        "FTHG",
        "AwayPoints"
    ]].rename(columns={
        "AwayTeam": "Team",
        "FTAG": "GoalsFor",
        "FTHG": "GoalsAgainst",
        "AwayPoints": "Points"
    })

    team_matches = pd.concat(
        [home_rows, away_rows],
        ignore_index=True
    )

    team_matches = team_matches.sort_values(
        ["Season", "Team", "Date"]
    )

    return team_matches


def create_rolling_features(team_matches):
    """
    Create leakage-safe rolling features.
    """

    team_matches = team_matches.copy()

    team_matches["Points_Last5"] = (
        team_matches
        .groupby(["Season", "Team"])["Points"]
        .transform(
            lambda x: x.shift().rolling(
                5,
                min_periods=1
            ).sum()
        )
    )

    team_matches["GF_Last5"] = (
        team_matches
        .groupby(["Season", "Team"])["GoalsFor"]
        .transform(
            lambda x: x.shift().rolling(
                5,
                min_periods=1
            ).sum()
        )
    )

    team_matches["GA_Last5"] = (
        team_matches
        .groupby(["Season", "Team"])["GoalsAgainst"]
        .transform(
            lambda x: x.shift().rolling(
                5,
                min_periods=1
            ).sum()
        )
    )

    return team_matches