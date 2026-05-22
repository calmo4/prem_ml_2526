# Takes raw match CSVs and creates:

# rolling features
# aggregated season tables
# processed datasets

# Outputs to:

# data/processed/
# Your future Streamlit app should NOT:

# recompute all modeling logic every refresh
# rebuild huge rolling features every page load

# Instead:

# preprocessing happens once
# app loads ready-made processed datasets

# This is MUCH better architecture.
import sys
from pathlib import Path

# Allows script to import from src/
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.features import (
    load_all_seasons,
    add_match_points,
    build_season_table,
    create_team_match_table,
    create_rolling_features,
)


def main():
    print("Building processed datasets...")

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    all_seasons = load_all_seasons("data/raw/epl_*.csv")
    all_seasons = add_match_points(all_seasons)

    # 2. Build season table
    season_table = build_season_table(all_seasons)

    # 3. Build team-match table
    team_matches = create_team_match_table(all_seasons)
    team_matches = create_rolling_features(team_matches)

    team_matches = team_matches.sort_values(["Season", "Team", "Date"])
    team_matches["MatchNumber"] = (
        team_matches
        .groupby(["Season", "Team"])
        .cumcount() + 1
    )

    # 4. Build rolling season features
    rolling_season = (
        team_matches
        .groupby(["Season", "Team"])
        .agg(
            AvgPointsLast5=("Points_Last5", "mean"),
            AvgGFLast5=("GF_Last5", "mean"),
            AvgGALast5=("GA_Last5", "mean"),
        )
        .reset_index()
    )

    model_data = season_table.merge(
        rolling_season,
        on=["Season", "Team"],
        how="left"
    )

    # 5. Save processed outputs
    season_table.to_csv(processed_dir / "season_table.csv", index=False)
    team_matches.to_csv(processed_dir / "team_matches.csv", index=False)
    model_data.to_csv(processed_dir / "model_data.csv", index=False)

    print("Done.")
    print(f"Saved: {processed_dir / 'season_table.csv'}")
    print(f"Saved: {processed_dir / 'team_matches.csv'}")
    print(f"Saved: {processed_dir / 'model_data.csv'}")


if __name__ == "__main__":
    main()