import pandas as pd
from src.modeling import train_linear_model, train_random_forest


def build_cutoff_dataset(team_matches, season_table, cutoff):
    """
    Build a dataset using only matches up to a given matchweek cutoff.
    """

    midseason = team_matches[
        team_matches["MatchNumber"] <= cutoff
    ]

    stats = (
        midseason
        .groupby(["Season", "Team"])
        .agg(
            PointsN=("Points", "sum"),
            GFN=("GoalsFor", "sum"),
            GAN=("GoalsAgainst", "sum"),
        )
        .reset_index()
    )

    data = stats.merge(
        season_table[["Season", "Team", "Points"]],
        on=["Season", "Team"],
        how="inner"
    )

    return data


def evaluate_model_by_cutoff(
    team_matches,
    season_table,
    cutoffs,
    test_season,
    model_type="linear"
):
    """
    Evaluate model performance across multiple matchweek cutoffs.
    """

    results = []

    features = ["PointsN", "GFN", "GAN"]

    for cutoff in cutoffs:
        data = build_cutoff_dataset(
            team_matches=team_matches,
            season_table=season_table,
            cutoff=cutoff
        )

        train = data[data["Season"] != test_season]
        test = data[data["Season"] == test_season]

        if model_type == "linear":
            output = train_linear_model(
                train_df=train,
                test_df=test,
                features=features
            )

        elif model_type == "random_forest":
            output = train_random_forest(
                train_df=train,
                test_df=test,
                features=features
            )

        else:
            raise ValueError("model_type must be 'linear' or 'random_forest'")

        results.append({
            "Cutoff": cutoff,
            "MAE": output["mae"],
            "R2": output["r2"]
        })

    return pd.DataFrame(results)