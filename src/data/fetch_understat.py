#!/usr/bin/env python3
"""
Fetch historical match data from Understat via soccerdata.
Supports multiple leagues: ENG-Premier League, ESP-La Liga, ITA-Serie A, etc.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import soccerdata as sd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = ["2022-2023", "2023-2024", "2024-2025", "2025-2026"]
DEFAULT_LEAGUES = ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1"]


def fetch_season(league: str, season: str) -> pd.DataFrame:
    """Fetch match schedule + xG for a single league+season."""
    print(f"[fetch] {league} {season} ...")
    understat = sd.Understat(leagues=league, seasons=season)
    matches = understat.read_schedule()

    if isinstance(matches.index, pd.MultiIndex):
        matches = matches.reset_index(drop=True)

    completed = matches[matches["is_result"] == True].copy()
    print(f"  -> {len(completed)} / {len(matches)} matches completed")
    return completed


def main():
    parser = argparse.ArgumentParser(description="Fetch Understat data")
    parser.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES, help="Leagues to fetch")
    parser.add_argument("--seasons", nargs="+", default=SEASONS, help="Seasons to fetch")
    args = parser.parse_args()

    all_frames = []
    for league in args.leagues:
        league_frames = []
        for season in args.seasons:
            try:
                df = fetch_season(league, season)
                df["season"] = season
                df["league"] = league
                out_path = RAW_DIR / f"understat_{league.replace(' ', '_').replace('-', '_')}_{season.replace('-', '_')}.csv"
                df.to_csv(out_path, index=False)
                print(f"  -> saved to {out_path}")
                league_frames.append(df)
            except Exception as e:
                print(f"  -> ERROR fetching {league} {season}: {e}", file=sys.stderr)
                continue

        if league_frames:
            league_combined = pd.concat(league_frames, ignore_index=True)
            league_path = RAW_DIR / f"understat_{league.replace(' ', '_').replace('-', '_')}_all.csv"
            league_combined.to_csv(league_path, index=False)
            print(f"[done] {league} combined {len(league_combined)} matches -> {league_path}")
            all_frames.append(league_combined)

    if all_frames:
        total = pd.concat(all_frames, ignore_index=True)
        total_path = RAW_DIR / "understat_all_leagues.csv"
        total.to_csv(total_path, index=False)
        print(f"\n[done] ALL LEAGUES combined {len(total)} matches -> {total_path}")
    else:
        print("\n[warn] No data fetched.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
