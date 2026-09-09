"""
Centralized configuration for football-model.
Single source of truth for paths, leagues, features, and split dates.
"""

from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"
CONFIG_DIR = ROOT / "config"

for d in [RAW_DIR, PROC_DIR, MODEL_DIR, REPORTS_DIR, LOGS_DIR, CONFIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Leagues
# ------------------------------------------------------------------
LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]

# ------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------
WINDOWS = [3, 5, 10]

# ------------------------------------------------------------------
# Train / val / test splits
# ------------------------------------------------------------------
SPLIT_DATES = {
    "train_end": "2024-08-01",
    "val_end": "2025-01-15",
}

# ------------------------------------------------------------------
# Columns to drop before modeling
# ------------------------------------------------------------------
DROP_COLS = [
    "date", "season", "matchweek",
    "home_team", "away_team",
    "home_goals", "away_goals",
    "home_xg", "away_xg",
    "league", "result",
]
