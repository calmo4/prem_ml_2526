# ⚽ Premier League Forecasting Project

## 📌 Objective
Forecast final Premier League points using historical match data and evaluate how prediction accuracy evolves throughout the season.

---

## 📊 Data Source
Match-level data from:
https://www.football-data.co.uk/

Seasons used:
2019/20 – 2025/26 (current season auto-updated via script)

---

## 🛠 Project Pipeline

### Data Fetching
`scripts/fetch_data.py`
- Automatically downloads latest season data
- Keeps dataset current on every run

Run with:
```bash
source .venv/bin/activate
python3 scripts/fetch_data.py
