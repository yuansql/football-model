#!/usr/bin/env python3
"""
Build rolling pre-match features from Understat raw data.
Supports multiple leagues — rolling/H2H/rank are computed WITHIN each league.
CRITICAL: No lookahead.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

MAPPING_PATH = ROOT / "config" / "team_name_mapping.json"
WINDOWS = [3, 5, 10]


def normalize_team_name(name: str, mapping: dict) -> str:
    return mapping.get("aliases", {}).get(name, name)


def build_features(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv, parse_dates=["date"])
    df = df.sort_values(["league", "date"]).reset_index(drop=True)

    # Normalize team names
    mapping = json.loads(MAPPING_PATH.read_text()) if MAPPING_PATH.exists() else {}
    df["home_team"] = df["home_team"].apply(lambda x: normalize_team_name(x, mapping))
    df["away_team"] = df["away_team"].apply(lambda x: normalize_team_name(x, mapping))

    # Result label
    df["result"] = np.where(
        df["home_goals"] > df["away_goals"], "H",
        np.where(df["home_goals"] < df["away_goals"], "A", "D")
    )

    df["season"] = df.get("season", "unknown")
    df["matchweek"] = df.groupby(["league", "season"]).cumcount() + 1

    # ---------- Per-league team history ----------
    team_records = []
    for _, row in df.iterrows():
        base = {
            "league": row["league"], "date": row["date"], "season": row["season"],
            "goals_for": None, "goals_against": None,
            "xg_for": None, "xg_against": None,
            "points": None, "win": None, "draw": None, "loss": None,
        }
        # Home
        r = base.copy()
        r["team"] = row["home_team"]
        r["is_home"] = 1
        r["goals_for"] = row["home_goals"]
        r["goals_against"] = row["away_goals"]
        r["xg_for"] = row["home_xg"]
        r["xg_against"] = row["away_xg"]
        r["points"] = 3 if row["result"] == "H" else (1 if row["result"] == "D" else 0)
        r["win"] = 1 if row["result"] == "H" else 0
        r["draw"] = 1 if row["result"] == "D" else 0
        r["loss"] = 1 if row["result"] == "A" else 0
        team_records.append(r)
        # Away
        r = base.copy()
        r["team"] = row["away_team"]
        r["is_home"] = 0
        r["goals_for"] = row["away_goals"]
        r["goals_against"] = row["home_goals"]
        r["xg_for"] = row["away_xg"]
        r["xg_against"] = row["home_xg"]
        r["points"] = 3 if row["result"] == "A" else (1 if row["result"] == "D" else 0)
        r["win"] = 1 if row["result"] == "A" else 0
        r["draw"] = 1 if row["result"] == "D" else 0
        r["loss"] = 1 if row["result"] == "H" else 0
        team_records.append(r)

    tr = pd.DataFrame(team_records).sort_values(["league", "team", "date"]).reset_index(drop=True)

    # ---------- Rolling features (within league+team) ----------
    roll_cols = []
    for w in WINDOWS:
        for col in ["goals_for", "goals_against", "xg_for", "xg_against", "points", "win", "draw", "loss"]:
            cname = f"{col}_roll{w}"
            tr[cname] = tr.groupby(["league", "team"])[col].shift(1).rolling(w, min_periods=1).mean().values
            roll_cols.append(cname)

        for col in ["goals_for", "goals_against", "xg_for", "xg_against", "points"]:
            tr[f"{col}_home_roll{w}"] = (
                tr[tr["is_home"] == 1].groupby(["league", "team"])[col]
                .shift(1).rolling(w, min_periods=1).mean()
                .reindex(tr.index).values
            )
            tr[f"{col}_away_roll{w}"] = (
                tr[tr["is_home"] == 0].groupby(["league", "team"])[col]
                .shift(1).rolling(w, min_periods=1).mean()
                .reindex(tr.index).values
            )
            roll_cols.append(f"{col}_home_roll{w}")
            roll_cols.append(f"{col}_away_roll{w}")

    # ---------- H2H (within league) ----------
    h2h_list = []
    for _, row in df.iterrows():
        mask = (
            (df["league"] == row["league"]) &
            (((df["home_team"] == row["home_team"]) & (df["away_team"] == row["away_team"])) |
             ((df["home_team"] == row["away_team"]) & (df["away_team"] == row["home_team"])))
            & (df["date"] < row["date"])
        )
        past = df[mask].tail(5)
        h2h_list.append({
            "league": row["league"], "date": row["date"],
            "home_team": row["home_team"], "away_team": row["away_team"],
            "h2h_home_win_rate": (past["result"] == "H").mean() if len(past) > 0 else np.nan,
            "h2h_draw_rate": (past["result"] == "D").mean() if len(past) > 0 else np.nan,
            "h2h_matches_count": len(past),
        })
    h2h_df = pd.DataFrame(h2h_list)

    # ---------- League table ranking (within league+season) ----------
    rank_list = []
    for (league, season), season_df in df.groupby(["league", "season"]):
        season_df = season_df.sort_values("date").reset_index(drop=True)
        teams = set(season_df["home_team"]) | set(season_df["away_team"])
        table = {t: {"pts": 0, "gd": 0, "played": 0} for t in teams}
        for _, row in season_df.iterrows():
            home_pts = 3 if row["result"] == "H" else (1 if row["result"] == "D" else 0)
            away_pts = 3 if row["result"] == "A" else (1 if row["result"] == "D" else 0)
            table[row["home_team"]]["pts"] += home_pts
            table[row["home_team"]]["gd"] += row["home_goals"] - row["away_goals"]
            table[row["home_team"]]["played"] += 1
            table[row["away_team"]]["pts"] += away_pts
            table[row["away_team"]]["gd"] += row["away_goals"] - row["home_goals"]
            table[row["away_team"]]["played"] += 1

            ranked = sorted(table.items(), key=lambda x: (x[1]["pts"], x[1]["gd"]), reverse=True)
            rank = {team: i + 1 for i, (team, _) in enumerate(ranked)}
            rank_list.append({
                "league": league, "date": row["date"],
                "home_team": row["home_team"], "away_team": row["away_team"],
                "home_rank": rank.get(row["home_team"], np.nan),
                "away_rank": rank.get(row["away_team"], np.nan),
                "rank_diff": rank.get(row["away_team"], np.nan) - rank.get(row["home_team"], np.nan),
            })
    rank_df = pd.DataFrame(rank_list)

    # ---------- Merge ----------
    base = df[["league", "date", "season", "matchweek", "home_team", "away_team",
               "home_goals", "away_goals", "home_xg", "away_xg", "result"]].copy()

    # Home team rolling
    home_tr = tr[tr["is_home"] == 1][["league", "date", "team"] + roll_cols].rename(columns={"team": "home_team"})
    home_tr = home_tr.rename(columns={c: f"home_{c}" for c in roll_cols})
    base = base.merge(home_tr, on=["league", "home_team", "date"], how="left")

    # Away team rolling
    away_tr = tr[tr["is_home"] == 0][["league", "date", "team"] + roll_cols].rename(columns={"team": "away_team"})
    away_tr = away_tr.rename(columns={c: f"away_{c}" for c in roll_cols})
    base = base.merge(away_tr, on=["league", "away_team", "date"], how="left")

    base = base.merge(h2h_df, on=["league", "date", "home_team", "away_team"], how="left")
    base = base.merge(rank_df, on=["league", "date", "home_team", "away_team"], how="left")

    # League encoding
    base["league_enc"] = base["league"].astype("category").cat.codes

    # Drop insufficient history
    base = base.dropna(subset=["home_goals_for_roll3", "away_goals_for_roll3"]).reset_index(drop=True)

    base.to_csv(output_csv, index=False)
    print(f"[done] Features -> {output_csv}")
    print(f"  rows: {len(base)} | cols: {len(base.columns)}")
    for lg, cnt in base["league"].value_counts().sort_index().items():
        print(f"    {lg}: {cnt} rows")
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=RAW_DIR / "understat_all_leagues.csv")
    parser.add_argument("--output", default=PROC_DIR / "features_multi_league_v1.csv")
    args = parser.parse_args()
    build_features(args.input, args.output)


if __name__ == "__main__":
    main()
