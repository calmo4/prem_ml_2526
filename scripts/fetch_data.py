import pandas as pd
from pathlib import Path
from datetime import datetime

# -----------------------------
# Config
# -----------------------------
BASE_URL = "https://www.football-data.co.uk/mmz4281"
LEAGUE_CODE = "E0"  # Premier League
RAW_DIR = Path("data/raw")

# Seasons to fetch
PAST_SEASONS = [
    "1920", "2021", "2122", "2223", "2324", "2425"
]
CURRENT_SEASON = "2526"  # current season

RAW_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Helper function
# -----------------------------
def fetch_season(season_code, filename):
    url = f"{BASE_URL}/{season_code}/{LEAGUE_CODE}.csv"
    print(f"Fetching {url}")
    df = pd.read_csv(url)
    df.to_csv(RAW_DIR / filename, index=False)
    print(f"Saved → {filename} ({len(df)} rows)")

# -----------------------------
# Fetch past seasons
# -----------------------------
for season in PAST_SEASONS:
    start = season[:2]
    end = season[2:]
    filename = f"epl_20{start}_20{end}.csv"
    fetch_season(season, filename)

# -----------------------------
# Fetch current season (always refreshed)
# -----------------------------
fetch_season(CURRENT_SEASON, "epl_2025_26_curr.csv")

print("\n✅ Data fetch complete")
print("Last updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
