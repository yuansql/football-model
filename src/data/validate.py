#!/usr/bin/env python3
"""
Validate raw Understat data: completeness, continuity, no duplicates.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

REQUIRED_COLS = [
    "date", "home_team", "away_team",
    "home_goals", "away_goals",
    "home_xg", "away_xg",
    "is_result",
]


def validate() -> dict:
    csv_path = RAW_DIR / "understat_epl_all.csv"
    if not csv_path.exists():
        print(f"[error] {csv_path} not found. Run fetch_understat.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path, parse_dates=["date"])
    report = {"file": str(csv_path), "total_rows": len(df)}

    # 1. Required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    report["missing_columns"] = missing
    report["columns_ok"] = len(missing) == 0

    # 2. Completion rate
    report["completion_rate"] = round(
        df[REQUIRED_COLS].notna().sum().sum()
        / (len(df) * len(REQUIRED_COLS)),
        4,
    )

    # 3. Duplicates (same teams, same date)
    dup_mask = df.duplicated(subset=["date", "home_team", "away_team"], keep=False)
    report["duplicate_matches"] = int(dup_mask.sum())

    # 4. Season continuity
    if "season" in df.columns:
        season_counts = df["season"].value_counts().to_dict()
        report["season_counts"] = season_counts
        # EPL = 380 matches per season
        for season, count in season_counts.items():
            if count < 370:
                report[f"season_{season}_warning"] = f"Only {count} matches (expect ~380)"

    # 5. Team name consistency
    home_teams = set(df["home_team"].dropna().unique())
    away_teams = set(df["away_team"].dropna().unique())
    report["unique_home_teams"] = len(home_teams)
    report["unique_away_teams"] = len(away_teams)

    return report


def main():
    report = validate()
    print("=" * 50)
    print("Data Validation Report")
    print("=" * 50)
    for k, v in report.items():
        print(f"  {k}: {v}")

    ok = (
        report.get("columns_ok", False)
        and report.get("completion_rate", 0) >= 0.95
        and report.get("duplicate_matches", 1) == 0
    )
    print("=" * 50)
    if ok:
        print("[PASS] Data validation passed.")
        sys.exit(0)
    else:
        print("[FAIL] Data validation failed. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
