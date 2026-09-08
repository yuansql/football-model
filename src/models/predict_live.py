#!/usr/bin/env python3
"""
Live prediction for upcoming fixtures.
Two modes:
  --mode history    : Query historical match from dataset
  --mode simulate   : Simulate a future match using latest rolling stats

Simulate mode workflow:
  1. Fetch latest league data from Understat
  2. Compute rolling features for all teams
  3. Extract home/away team's latest form snapshot
  4. Predict P(H)/P(D)/P(A)
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import soccerdata as sd

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
MAPPING_PATH = ROOT / "config" / "team_name_mapping.json"
WINDOWS = [3, 5, 10]


def normalize_team_name(name: str, mapping: dict) -> str:
    return mapping.get("aliases", {}).get(name, name)


def load_model(model_stem: str):
    model_path = MODEL_DIR / f"{model_stem}.json"
    meta_path = MODEL_DIR / f"{model_stem}.pkl"
    model = xgb.Booster()
    model.load_model(str(model_path))
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return model, meta["label_encoder"], meta["feature_cols"]


def fetch_latest_league_data(league: str):
    """Fetch ALL matches for a league (all seasons Understat has)."""
    print(f"[fetch] Pulling latest data for {league} ...")
    understat = sd.Understat(leagues=league)
    matches = understat.read_schedule()
    if isinstance(matches.index, pd.MultiIndex):
        matches = matches.reset_index(drop=True)
    completed = matches[matches["is_result"] == True].copy()
    completed = completed.sort_values("date").reset_index(drop=True)
    print(f"  -> {len(completed)} completed matches")
    return completed


def build_latest_rolling(league_df: pd.DataFrame, home_team: str, away_team: str, mapping: dict):
    """
    Build a 'virtual match' feature vector for home_team vs away_team
    using their latest rolling stats as of the most recent match in league_df.
    """
    df = league_df.copy()
    df["home_team"] = df["home_team"].apply(lambda x: normalize_team_name(x, mapping))
    df["away_team"] = df["away_team"].apply(lambda x: normalize_team_name(x, mapping))

    # Result labels
    df["result"] = np.where(
        df["home_goals"] > df["away_goals"], "H",
        np.where(df["home_goals"] < df["away_goals"], "A", "D")
    )

    # Matchweek (sequential within the dataset)
    df["matchweek"] = range(1, len(df) + 1)
    latest_date = df["date"].max()

    # Team records
    team_records = []
    for _, row in df.iterrows():
        team_records.append({
            "date": row["date"], "team": row["home_team"], "is_home": 1,
            "goals_for": row["home_goals"], "goals_against": row["away_goals"],
            "xg_for": row["home_xg"], "xg_against": row["away_xg"],
            "points": 3 if row["result"] == "H" else (1 if row["result"] == "D" else 0),
            "win": 1 if row["result"] == "H" else 0,
            "draw": 1 if row["result"] == "D" else 0,
            "loss": 1 if row["result"] == "A" else 0,
        })
        team_records.append({
            "date": row["date"], "team": row["away_team"], "is_home": 0,
            "goals_for": row["away_goals"], "goals_against": row["home_goals"],
            "xg_for": row["away_xg"], "xg_against": row["home_xg"],
            "points": 3 if row["result"] == "A" else (1 if row["result"] == "D" else 0),
            "win": 1 if row["result"] == "A" else 0,
            "draw": 1 if row["result"] == "D" else 0,
            "loss": 1 if row["result"] == "H" else 0,
        })

    tr = pd.DataFrame(team_records).sort_values(["team", "date"]).reset_index(drop=True)

    # Compute rolling features
    roll_cols = []
    for w in WINDOWS:
        for col in ["goals_for", "goals_against", "xg_for", "xg_against", "points", "win", "draw", "loss"]:
            cname = f"{col}_roll{w}"
            tr[cname] = tr.groupby("team")[col].shift(1).rolling(w, min_periods=1).mean().values
            roll_cols.append(cname)

        for col in ["goals_for", "goals_against", "xg_for", "xg_against", "points"]:
            home_mask = tr["is_home"] == 1
            away_mask = tr["is_home"] == 0
            tr[f"{col}_home_roll{w}"] = tr[home_mask].groupby("team")[col].shift(1).rolling(w, min_periods=1).mean().reindex(tr.index).values
            tr[f"{col}_away_roll{w}"] = tr[away_mask].groupby("team")[col].shift(1).rolling(w, min_periods=1).mean().reindex(tr.index).values
            roll_cols.append(f"{col}_home_roll{w}")
            roll_cols.append(f"{col}_away_roll{w}")

    # Extract latest stats for home and away teams
    home_latest = tr[tr["team"] == home_team].iloc[-1] if len(tr[tr["team"] == home_team]) > 0 else None
    away_latest = tr[tr["team"] == away_team].iloc[-1] if len(tr[tr["team"] == away_team]) > 0 else None

    if home_latest is None:
        raise ValueError(f"Home team '{home_team}' not found in league data")
    if away_latest is None:
        raise ValueError(f"Away team '{away_team}' not found in league data")

    # H2H: last 5 meetings between these two
    h2h_mask = (
        ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
        ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    )
    h2h_past = df[h2h_mask].tail(5)
    h2h_home_win = (h2h_past["result"] == "H").mean() if len(h2h_past) > 0 else 0.5
    h2h_draw = (h2h_past["result"] == "D").mean() if len(h2h_past) > 0 else 0.25
    h2h_count = len(h2h_past)

    # League table: current standings
    table = {}
    for _, row in df.iterrows():
        for team, goals_for, goals_against, result, is_home in [
            (row["home_team"], row["home_goals"], row["away_goals"], row["result"], True),
            (row["away_team"], row["away_goals"], row["home_goals"], row["result"], False),
        ]:
            if team not in table:
                table[team] = {"pts": 0, "gd": 0, "played": 0}
            pts = 3 if (result == "H" and is_home) or (result == "A" and not is_home) else (1 if result == "D" else 0)
            table[team]["pts"] += pts
            table[team]["gd"] += goals_for - goals_against
            table[team]["played"] += 1

    ranked = sorted(table.items(), key=lambda x: (x[1]["pts"], x[1]["gd"]), reverse=True)
    rank = {team: i + 1 for i, (team, _) in enumerate(ranked)}
    home_rank = rank.get(home_team, len(rank) // 2)
    away_rank = rank.get(away_team, len(rank) // 2)
    rank_diff = away_rank - home_rank

    # Build virtual match row
    virtual = {}
    virtual["league_enc"] = 0  # Will be set by caller if multi-league

    for col in roll_cols:
        virtual[f"home_{col}"] = home_latest[col] if col in home_latest else np.nan
        virtual[f"away_{col}"] = away_latest[col] if col in away_latest else np.nan

    virtual["h2h_home_win_rate"] = h2h_home_win
    virtual["h2h_draw_rate"] = h2h_draw
    virtual["h2h_matches_count"] = h2h_count
    virtual["home_rank"] = home_rank
    virtual["away_rank"] = away_rank
    virtual["rank_diff"] = rank_diff

    return pd.Series(virtual), latest_date, table


def predict(model, le, feature_cols, row):
    # Ensure all required features present
    missing = [c for c in feature_cols if c not in row.index]
    if missing:
        for c in missing:
            row[c] = 0.0  # fill missing with neutral

    X = pd.DataFrame([row[feature_cols].astype(float).values], columns=feature_cols)
    probs = model.predict(xgb.DMatrix(X))[0]
    pred_idx = int(np.argmax(probs))
    return {
        "P_H": float(probs[le.transform(["H"])[0]]),
        "P_D": float(probs[le.transform(["D"])[0]]),
        "P_A": float(probs[le.transform(["A"])[0]]),
        "predicted": le.inverse_transform([pred_idx])[0],
        "confidence": float(probs[pred_idx]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgb_multi_league_v1")
    parser.add_argument("--league", required=True)
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--mode", default="simulate", choices=["history", "simulate"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model, le, feature_cols = load_model(args.model)
    mapping = json.loads(MAPPING_PATH.read_text()) if MAPPING_PATH.exists() else {}
    home_norm = normalize_team_name(args.home, mapping)
    away_norm = normalize_team_name(args.away, mapping)

    if args.mode == "history":
        # Query from processed features
        proc_path = ROOT / "data" / "processed" / "features_multi_league_v1.csv"
        df = pd.read_csv(proc_path, parse_dates=["date"])
        mask = (df["league"] == args.league) & (df["home_team"] == home_norm) & (df["away_team"] == away_norm)
        matches = df[mask]
        if len(matches) == 0:
            print(f"[error] No historical match found. Try --mode simulate", file=sys.stderr)
            sys.exit(1)
        row = matches.iloc[-1]
        pred = predict(model, le, feature_cols, row)
        actual = row.get("result")
        date = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "?"
    else:
        # Simulate future match
        league_df = fetch_latest_league_data(args.league)
        virtual_row, latest_date, table = build_latest_rolling(league_df, home_norm, away_norm, mapping)
        # Set league_enc
        proc_path = ROOT / "data" / "processed" / "features_multi_league_v1.csv"
        proc_df = pd.read_csv(proc_path)
        league_enc = proc_df[proc_df["league"] == args.league]["league_enc"].iloc[0] if len(proc_df[proc_df["league"] == args.league]) > 0 else 0
        virtual_row["league_enc"] = league_enc
        pred = predict(model, le, feature_cols, virtual_row)
        actual = None
        date = f"future (as of {latest_date.strftime('%Y-%m-%d')})"

    # Output
    conf_label = "high" if pred["confidence"] >= 0.6 else ("medium" if pred["confidence"] >= 0.5 else "low")

    if args.json:
        out = {
            "match": {"date": date, "league": args.league, "home": home_norm, "away": away_norm, "actual": actual},
            "probabilities": {"H": round(pred["P_H"], 4), "D": round(pred["P_D"], 4), "A": round(pred["P_A"], 4)},
            "predicted_direction": pred["predicted"],
            "confidence": round(pred["confidence"], 4),
            "confidence_label": conf_label,
        }
        if args.mode == "simulate":
            out["data_date"] = str(latest_date)
            out["standing"] = {
                "home_rank": int(virtual_row["home_rank"]),
                "away_rank": int(virtual_row["away_rank"]),
                "rank_diff": int(virtual_row["rank_diff"]),
            }
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print("=" * 60)
        print(f"football-model Live Prediction ({args.mode})")
        print("=" * 60)
        print(f"Match: {date}")
        print(f"       {args.league}")
        print(f"       {home_norm} vs {away_norm}")
        print()
        print(f"P(主胜) = {pred['P_H']:.4f}")
        print(f"P(平)   = {pred['P_D']:.4f}")
        print(f"P(客胜) = {pred['P_A']:.4f}")
        print(f"Sum     = {pred['P_H'] + pred['P_D'] + pred['P_A']:.4f}")
        print()
        print(f"预测方向: {pred['predicted']} | 置信度: {pred['confidence']:.4f} ({conf_label})")
        if actual:
            print(f"实际结果: {actual}")
        if args.mode == "simulate":
            print()
            print(f"【当前排名】")
            print(f"  {home_norm}: 第 {int(virtual_row['home_rank'])} 名")
            print(f"  {away_norm}: 第 {int(virtual_row['away_rank'])} 名")
            print(f"  排名差: {int(virtual_row['rank_diff'])}")
        print("=" * 60)


if __name__ == "__main__":
    main()
