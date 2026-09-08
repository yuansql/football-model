#!/usr/bin/env python3
"""
v17 实战测试：批量预测近期比赛，生成完整报告。
Usage:
    .venv/bin/python3 src/models/batch_predict.py \
        --league "ENG-Premier League" --date-from "2026-05-01" --date-to "2026-05-24"
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
PROC_DIR = ROOT / "data" / "processed"

def load_model(model_stem: str):
    model = xgb.Booster()
    model.load_model(str(MODEL_DIR / f"{model_stem}.json"))
    with open(MODEL_DIR / f"{model_stem}.pkl", "rb") as f:
        meta = pickle.load(f)
    return model, meta["label_encoder"], meta["feature_cols"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgb_multi_league_v1")
    parser.add_argument("--features", default=PROC_DIR / "features_multi_league_v1.csv")
    parser.add_argument("--league", required=True)
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    args = parser.parse_args()

    model, le, feature_cols = load_model(args.model)
    df = pd.read_csv(args.features, parse_dates=["date"])

    mask = (df["league"] == args.league) & (df["date"] >= args.date_from) & (df["date"] <= args.date_to)
    matches = df[mask].copy()

    if len(matches) == 0:
        print(f"[error] No matches found in range", file=sys.stderr)
        sys.exit(1)

    X = matches[feature_cols].astype(float)
    probs = model.predict(xgb.DMatrix(X))
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    matches["pred"] = le.inverse_transform(preds)
    matches["conf"] = confidences
    matches["P_H"] = probs[:, le.transform(["H"])[0]]
    matches["P_D"] = probs[:, le.transform(["D"])[0]]
    matches["P_A"] = probs[:, le.transform(["A"])[0]]
    matches["hit"] = matches["pred"] == matches["result"]

    # Filter high confidence
    high_conf = matches[matches["conf"] >= args.min_confidence]

    print("=" * 70)
    print(f"v17 实战测试报告")
    print(f"联赛: {args.league} | 日期: {args.date_from} ~ {args.date_to}")
    print(f"模型: {args.model} | 置信度阈值: {args.min_confidence}")
    print("=" * 70)
    print(f"\n总场次: {len(matches)}")
    print(f"整体命中率: {matches['hit'].mean():.1%}")
    print(f"\n高置信场次 (≥{args.min_confidence}): {len(high_conf)}")
    print(f"高置信命中率: {high_conf['hit'].mean():.1%}" if len(high_conf) > 0 else "无")

    print(f"\n{'Date':<12} {'Home':<20} {'Away':<20} {'Pred':<5} {'Act':<5} {'Conf':<6} {'P(H)':<6} {'P(D)':<6} {'P(A)':<6}")
    print("-" * 90)
    for _, row in matches.iterrows():
        marker = "✓" if row["hit"] else "✗"
        print(f"{row['date'].strftime('%m-%d'):<12} {row['home_team']:<20} {row['away_team']:<20} "
              f"{row['pred']:<5} {row['result']:<5} {row['conf']:.3f}  "
              f"{row['P_H']:.3f}  {row['P_D']:.3f}  {row['P_A']:.3f}  {marker}")

    print(f"\n{'=' * 70}")
    print(f"按方向命中率:")
    for direction in ["H", "D", "A"]:
        sub = matches[matches["pred"] == direction]
        if len(sub) > 0:
            print(f"  预测{direction}: {sub['hit'].mean():.1%} ({sub['hit'].sum()}/{len(sub)})")

    print(f"\n按置信度区间:")
    for low, high in [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]:
        sub = matches[(matches["conf"] >= low) & (matches["conf"] < high)]
        if len(sub) > 0:
            print(f"  conf [{low:.1f}-{high:.1f}): {sub['hit'].mean():.1%} ({sub['hit'].sum()}/{len(sub)})")

    if args.output:
        matches[["date", "home_team", "away_team", "result", "pred", "conf", "P_H", "P_D", "P_A", "hit"]].to_csv(args.output, index=False)
        print(f"\n[save] {args.output}")

if __name__ == "__main__":
    main()
