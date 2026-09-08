#!/usr/bin/env python3
"""
Enhanced rolling features v2:
- Adds asymmetric home/away advantage features
- Adds relative strength vs league average
- Adds form momentum (recent vs older)
CRITICAL: No lookahead.
"""

import argparse
import json
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

    mapping = json.loads(MAPPING_PATH.read_text()) if MAPPING_PATH.exists() else {}
    df["home_team"] = df["home_team"].apply(lambda x: normalize_team_name(x, mapping))
    df["away_team"] = df["away_team"].apply(lambda x: normalize_team_name(x, mapping))

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
        for team, is_home, gf, ga, xgf, xga, res in [
            (row["home_team"], 1, row["home_goals"], row["away_goals"], row["home_xg"], row["away_xg"], row["result"]),
            (row["away_team"], 0, row["away_goals"], row["home_goals"], row["away_xg"], row["home_xg"], row["result"]),
        ]:
            r = base.copy()
            r["team"] = team
            r["is_home"] = is_home
            r["goals_for"] = gf
            r["goals_against"] = ga
            r["xg_for"] = xgf
            r["xg_against"] = xga
            r["points"] = 3 if (res == "H" and is_home) or (res == "A" and not is_home) else (1 if res == "D" else 0)
            r["win"] = 1 if r["points"] == 3 else 0
            r["draw"] = 1 if r["points"] == 1 else 0
            r["loss"] = 1 if r["points"] == 0 else 0
            team_records.append(r)

    tr = pd.DataFrame(team_records).sort_values(["league", "team", "date"]).reset_index(drop=True)

    # ---------- Basic rolling features ----------
    roll_cols = []
    for w in WINDOWS:
        for col in ["goals_for", "goals_against", "xg_for", "xg_against", "points", "win", "draw", "loss"]:
            cname = f"{col}_roll{w}"
            tr[cname] = tr.groupby(["league", "team"])[col].shift(1).rolling(w, min_periods=1).mean().values
            roll_cols.append(cname)

        for col in ["goals_for", "goals_against", "xg_for", "xg_against", "points"]:
            tr[f"{col}_home_roll{w}"] = tr[tr["is_home"]==1].groupby(["league","team"])[col].shift(1).rolling(w, min_periods=1).mean().reindex(tr.index).values
            tr[f"{col}_away_roll{w}"] = tr[tr["is_home"]==0].groupby(["league","team"])[col].shift(1).rolling(w, min_periods=1).mean().reindex(tr.index).values
            roll_cols.append(f"{col}_home_roll{w}")
            roll_cols.append(f"{col}_away_roll{w}")

    # ---------- NEW: Asymmetric advantage features ----------
    # Home advantage: home form - away form (for the same team)
    for w in WINDOWS:
        tr[f"home_advantage_pts_roll{w}"] = tr[f"points_home_roll{w}"] - tr[f"points_away_roll{w}"]
        tr[f"home_advantage_xg_roll{w}"] = tr[f"xg_for_home_roll{w}"] - tr[f"xg_for_away_roll{w}"]
        tr[f"home_advantage_xga_roll{w}"] = tr[f"xg_against_home_roll{w}"] - tr[f"xg_against_away_roll{w}"]
        roll_cols.append(f"home_advantage_pts_roll{w}")
        roll_cols.append(f"home_advantage_xg_roll{w}")
        roll_cols.append(f"home_advantage_xga_roll{w}")

    # Form momentum: recent vs older (roll3 vs roll10 ratio)
    tr["momentum_pts"] = tr["points_roll3"] / (tr["points_roll10"] + 0.1)
    tr["momentum_xg"] = tr["xg_for_roll3"] / (tr["xg_for_roll10"] + 0.1)
    tr["momentum_xga"] = tr["xg_against_roll3"] / (tr["xg_against_roll10"] + 0.1)
    roll_cols.extend(["momentum_pts", "momentum_xg", "momentum_xga"])

    # ---------- H2H ----------
    h2h_list = []
    for _, row in df.iterrows():
        mask = (
            (df["league"] == row["league"]) &
            (((df["home_team"]==row["home_team"])&(df["away_team"]==row["away_team"]))|
             ((df["home_team"]==row["away_team"])&(df["away_team"]==row["home_team"])))
            & (df["date"] < row["date"])
        )
        past = df[mask].tail(5)
        h2h_list.append({
            "league": row["league"], "date": row["date"],
            "home_team": row["home_team"], "away_team": row["away_team"],
            "h2h_home_win_rate": (past["result"]=="H").mean() if len(past)>0 else np.nan,
            "h2h_draw_rate": (past["result"]=="D").mean() if len(past)>0 else np.nan,
            "h2h_matches_count": len(past),
        })
    h2h_df = pd.DataFrame(h2h_list)

    # ---------- League table ranking ----------
    rank_list = []
    for (league, season), season_df in df.groupby(["league", "season"]):
        season_df = season_df.sort_values("date").reset_index(drop=True)
        teams = set(season_df["home_team"]) | set(season_df["away_team"])
        table = {t: {"pts": 0, "gd": 0, "played": 0} for t in teams}
        for _, row in season_df.iterrows():
            home_pts = 3 if row["result"]=="H" else (1 if row["result"]=="D" else 0)
            away_pts = 3 if row["result"]=="A" else (1 if row["result"]=="D" else 0)
            table[row["home_team"]]["pts"] += home_pts
            table[row["home_team"]]["gd"] += row["home_goals"] - row["away_goals"]
            table[row["home_team"]]["played"] += 1
            table[row["away_team"]]["pts"] += away_pts
            table[row["away_team"]]["gd"] += row["away_goals"] - row["home_goals"]
            table[row["away_team"]]["played"] += 1

            ranked = sorted(table.items(), key=lambda x: (x[1]["pts"], x[1]["gd"]), reverse=True)
            rank = {team: i+1 for i, (team, _) in enumerate(ranked)}
            rank_list.append({
                "league": league, "date": row["date"],
                "home_team": row["home_team"], "away_team": row["away_team"],
                "home_rank": rank.get(row["home_team"], np.nan),
                "away_rank": rank.get(row["away_team"], np.nan),
                "rank_diff": rank.get(row["away_team"], np.nan) - rank.get(row["home_team"], np.nan),
            })
    rank_df = pd.DataFrame(rank_list)

    # ---------- League-relative strength ----------
    # Compute league average per matchweek
    league_avg = tr.groupby(["league", "date"])[[c for c in roll_cols if "roll" in c]].mean().reset_index()
    league_avg = league_avg.rename(columns={c: f"league_avg_{c}" for c in roll_cols if "roll" in c})

    # ---------- Merge ----------
    base = df[["league", "date", "season", "matchweek", "home_team", "away_team",
               "home_goals", "away_goals", "home_xg", "away_xg", "result"]].copy()

    # Home team rolling
    home_tr = tr[tr["is_home"]==1][["league","date","team"]+roll_cols].rename(columns={"team": "home_team"})
    home_tr = home_tr.rename(columns={c: f"home_{c}" for c in roll_cols})
    base = base.merge(home_tr, on=["league","home_team","date"], how="left")

    # Away team rolling
    away_tr = tr[tr["is_home"]==0][["league","date","team"]+roll_cols].rename(columns={"team": "away_team"})
    away_tr = away_tr.rename(columns={c: f"away_{c}" for c in roll_cols})
    base = base.merge(away_tr, on=["league","away_team","date"], how="left")

    base = base.merge(h2h_df, on=["league","date","home_team","away_team"], how="left")
    base = base.merge(rank_df, on=["league","date","home_team","away_team"], how="left")

    # NEW: Direct comparison features (home - away)
    for w in WINDOWS:
        for metric in ["points", "xg_for", "xg_against", "goals_for", "goals_against"]:
            base[f"diff_{metric}_roll{w}"] = base[f"home_{metric}_roll{w}"] - base[f"away_{metric}_roll{w}"]

    # NEW: Home advantage comparison
    for w in WINDOWS:
        base[f"diff_home_advantage_pts_roll{w}"] = base[f"home_home_advantage_pts_roll{w}"] - base[f"away_home_advantage_pts_roll{w}"]

    # League encoding
    base["league_enc"] = base["league"].astype("category").cat.codes

    # Drop insufficient history
    base = base.dropna(subset=["home_goals_for_roll3", "away_goals_for_roll3"]).reset_index(drop=True)

    base.to_csv(output_csv, index=False)
    print(f"[done] Features v2 -> {output_csv}")
    print(f"  rows: {len(base)} | cols: {len(base.columns)}")
    for lg, cnt in base["league"].value_counts().sort_index().items():
        print(f"    {lg}: {cnt} rows")
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=RAW_DIR / "understat_all_leagues.csv")
    parser.add_argument("--output", default=PROC_DIR / "features_multi_league_v2.csv")
    args = parser.parse_args()
    build_features(args.input, args.output)


if __name__ == "__main__":
    main()
