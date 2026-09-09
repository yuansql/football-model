#!/usr/bin/env python3
"""
Batch binary prediction for v17 testing.
Compares binary model (主不败 vs 客不败) to actual results.
"""

import argparse
import pickle
import sys
from pathlib import Path

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
    return model, meta["feature_cols"]


def to_binary(result):
    return 0 if result in ("H", "D") else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgb_binary_tuned")
    parser.add_argument("--features", default=PROC_DIR / "features_multi_league_v2.csv")
    parser.add_argument("--league", required=True)
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold for 客不败")
    args = parser.parse_args()

    model, feature_cols = load_model(args.model)
    df = pd.read_csv(args.features, parse_dates=["date"])

    mask = (df["league"] == args.league) & (df["date"] >= args.date_from) & (df["date"] <= args.date_to)
    matches = df[mask].copy()
    if len(matches) == 0:
        print("[error] No matches found", file=sys.stderr)
        sys.exit(1)

    X = matches[feature_cols].astype(float)
    probs = model.predict(xgb.DMatrix(X))
    preds = (probs >= 0.5).astype(int)

    matches["pred_binary"] = preds
    matches["pred_label"] = matches["pred_binary"].map({0: "主不败", 1: "客不败"})
    matches["prob_away"] = probs  # prob of 客不败(A)
    matches["actual_binary"] = matches["result"].apply(to_binary)
    matches["hit"] = matches["pred_binary"] == matches["actual_binary"]

    # High confidence away predictions
    away_high = matches[matches["prob_away"] >= args.threshold]

    print("=" * 70)
    print(f"二分类实战测试 | {args.league} | {args.date_from} ~ {args.date_to}")
    print("=" * 70)
    print(f"\n总场次: {len(matches)}")
    print(f"整体命中率: {matches['hit'].mean():.1%}")
    print(f"\n客不败高置信 (prob≥{args.threshold}): {len(away_high)} 场")
    if len(away_high) > 0:
        print(f"客不败命中率: {away_high['hit'].mean():.1%}")
        print(f"客不败实际分布: {away_high['actual_binary'].value_counts().to_dict()}")

    print(f"\n{'Date':<12} {'Home':<20} {'Away':<20} {'Pred':<8} {'Act':<8} {'Prob_A':<8} {'P(H)':<6} {'P(D)':<6} {'P(A)':<6}")
    print("-" * 100)
    for _, row in matches.iterrows():
        act_label = "主不败" if row["actual_binary"] == 0 else "客不败"
        marker = "✓" if row["hit"] else "✗"
        print(f"{row['date'].strftime('%m-%d'):<12} {row['home_team']:<20} {row['away_team']:<20} "
              f"{row['pred_label']:<8} {act_label:<8} {row['prob_away']:.3f}   "
              f"{row.get('home_win', 0):.2f}  {row.get('draw', 0):.2f}  {row.get('away_win', 0):.2f}  {marker}")

    print(f"\n{'=' * 70}")
    print(f"按预测方向:")
    for pred in [0, 1]:
        sub = matches[matches["pred_binary"] == pred]
        label = "主不败" if pred == 0 else "客不败"
        print(f"  预测{label}: {sub['hit'].mean():.1%} ({sub['hit'].sum()}/{len(sub)})")

    print(f"\n按置信度区间 (客不败概率):")
    for low, high in [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]:
        sub = matches[(matches["prob_away"] >= low) & (matches["prob_away"] < high)]
        if len(sub) > 0:
            print(f"  prob [{low:.1f}-{high:.1f}): {sub['hit'].mean():.1%} ({sub['hit'].sum()}/{len(sub)})")


if __name__ == "__main__":
    main()
