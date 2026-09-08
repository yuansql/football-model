#!/usr/bin/env python3
"""
ROI backtest simulation.
NOTE: Without real historical odds data (football-data.co.uk 503),
this uses MODEL-IMPLIED fair odds as an upper-bound reference only.
Real-world ROI will be lower due to bookmaker margin.

Strategies tested:
  1. Naive: Bet 1 unit on predicted direction every match
  2. Confidence filter: Only bet when confidence >= threshold
  3. Kelly criterion: Bet fractionally based on edge vs model-implied odds
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DATES = {"train_end": "2024-08-01", "val_end": "2025-01-15"}
DROP_COLS = ["date", "season", "matchweek", "home_team", "away_team",
             "home_goals", "away_goals", "home_xg", "away_xg", "league", "result"]


def load_model(model_stem: str):
    model = xgb.Booster()
    model.load_model(str(MODEL_DIR / f"{model_stem}.json"))
    with open(MODEL_DIR / f"{model_stem}.pkl", "rb") as f:
        meta = pickle.load(f)
    return model, meta["label_encoder"], meta["feature_cols"]


def backtest(model, le, feature_cols, features_csv: Path):
    df = pd.read_csv(features_csv, parse_dates=["date"])
    test_mask = df["date"] >= SPLIT_DATES["val_end"]
    test_df = df[test_mask].copy()
    X_test = test_df[feature_cols].astype(float)
    y_true = le.transform(test_df["result"].values)

    probs = model.predict(xgb.DMatrix(X_test))
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    # Model-implied fair odds (no bookmaker margin) — theoretical upper bound
    fair_odds = 1.0 / probs[np.arange(len(probs)), preds]
    # Assume 10% bookmaker margin: market_odds = fair_odds * 0.9
    market_odds = fair_odds * 0.9

    results = []

    # Strategy 1: Naive (bet every predicted direction, 1 unit)
    returns = np.where(preds == y_true, market_odds - 1, -1.0)
    results.append({
        "strategy": "1_Naive_EveryMatch",
        "n_bets": len(returns),
        "hit_rate": (preds == y_true).mean(),
        "total_return": returns.sum(),
        "avg_return": returns.mean(),
        "roi_pct": returns.mean() * 100,
        "max_drawdown": np.minimum.accumulate(np.cumsum(returns)).min(),
        "sharpe": returns.mean() / (returns.std() + 1e-9) if returns.std() > 0 else 0,
    })

    # Strategy 2: Confidence thresholds
    for threshold in [0.50, 0.55, 0.60, 0.65]:
        mask = confidences >= threshold
        if mask.sum() == 0:
            continue
        r = returns[mask]
        results.append({
            "strategy": f"2_Conf{int(threshold*100):02d}",
            "n_bets": int(mask.sum()),
            "hit_rate": (preds[mask] == y_true[mask]).mean(),
            "total_return": r.sum(),
            "avg_return": r.mean(),
            "roi_pct": r.mean() * 100,
            "max_drawdown": np.minimum.accumulate(np.cumsum(r)).min() if len(r) > 0 else 0,
            "sharpe": r.mean() / (r.std() + 1e-9) if r.std() > 0 else 0,
        })

    # Strategy 3: Kelly criterion (fractional bet sizing)
    # Edge = p_model - 1/odds_market; Kelly fraction = Edge / (odds_market - 1)
    p_model = probs[np.arange(len(probs)), preds]
    edge = p_model - 1.0 / market_odds
    kelly_fraction = np.clip(edge / (market_odds - 1), 0, 1)
    kelly_fraction = np.where(edge > 0, kelly_fraction, 0)  # no bet if no edge

    kelly_returns = np.where(preds == y_true, kelly_fraction * (market_odds - 1), -kelly_fraction)
    mask_kelly = kelly_fraction > 0
    if mask_kelly.sum() > 0:
        rk = kelly_returns[mask_kelly]
        results.append({
            "strategy": "3_Kelly_Fractional",
            "n_bets": int(mask_kelly.sum()),
            "hit_rate": (preds[mask_kelly] == y_true[mask_kelly]).mean(),
            "total_return": rk.sum(),
            "avg_return": rk.mean(),
            "roi_pct": rk.sum() / kelly_fraction.sum() * 100 if kelly_fraction.sum() > 0 else 0,
            "max_drawdown": np.minimum.accumulate(np.cumsum(rk)).min(),
            "sharpe": rk.mean() / (rk.std() + 1e-9) if rk.std() > 0 else 0,
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgb_multi_league_v1")
    parser.add_argument("--features", default=PROC_DIR / "features_multi_league_v1.csv")
    args = parser.parse_args()

    model, le, feature_cols = load_model(args.model)
    df = backtest(model, le, feature_cols, args.features)

    print("=" * 75)
    print("ROI Backtest Report (Model-Implied Fair Odds · Theoretical Upper Bound)")
    print("=" * 75)
    print(f"{'Strategy':<22} {'Bets':>5} {'Hit%':>7} {'ROI%':>8} {'Total':>9} {'Sharpe':>7}")
    print("-" * 75)
    for _, row in df.iterrows():
        print(f"{row['strategy']:<22} {row['n_bets']:>5} {row['hit_rate']*100:>6.1f}% {row['roi_pct']:>7.2f}% {row['total_return']:>+8.2f} {row['sharpe']:>7.3f}")
    print("=" * 75)
    print("\n[WARNING] Odds are model-implied (fair, no margin). Real-world ROI will be LOWER.")
    print("          Use this for relative strategy comparison only, not absolute profit.")

    # Save
    out_path = OUT_DIR / "backtest_roi.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
