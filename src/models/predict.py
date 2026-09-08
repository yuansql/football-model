#!/usr/bin/env python3
"""
Predict P(H)/P(D)/P(A) for a given fixture using the trained XGBoost model.
Supports multi-league features.
"""

import argparse
import json
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
    model_path = MODEL_DIR / f"{model_stem}.json"
    meta_path = MODEL_DIR / f"{model_stem}.pkl"
    if not model_path.exists():
        print(f"[error] Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    model = xgb.Booster()
    model.load_model(str(model_path))
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return model, meta["label_encoder"], meta["feature_cols"]


def predict_match(model, le, feature_cols, features_row: pd.Series) -> dict:
    X = pd.DataFrame([features_row[feature_cols].astype(float).values], columns=feature_cols)
    dmat = xgb.DMatrix(X)
    probs = model.predict(dmat)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = le.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])

    return {
        "P_H": round(float(probs[le.transform(["H"])[0]]), 4),
        "P_D": round(float(probs[le.transform(["D"])[0]]), 4),
        "P_A": round(float(probs[le.transform(["A"])[0]]), 4),
        "predicted_direction": pred_label,
        "confidence": confidence,
        "confidence_label": "high" if confidence >= 0.6 else ("medium" if confidence >= 0.5 else "low"),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict match outcome probabilities")
    parser.add_argument("--model", default="xgb_multi_league_v1", help="Model stem name")
    parser.add_argument("--features", default=PROC_DIR / "features_multi_league_v1.csv", help="Features CSV")
    parser.add_argument("--league", required=False, help="League name filter")
    parser.add_argument("--home", required=False, help="Home team name")
    parser.add_argument("--away", required=False, help="Away team name")
    parser.add_argument("--date", required=False, help="Match date (YYYY-MM-DD)")
    parser.add_argument("--output", default="json", choices=["json", "text"])
    args = parser.parse_args()

    model, le, feature_cols = load_model(args.model)
    df = pd.read_csv(args.features, parse_dates=["date"])

    # Filter
    mask = pd.Series([True] * len(df), index=df.index)
    if args.league:
        mask &= df["league"] == args.league
    if args.home:
        mask &= df["home_team"] == args.home
    if args.away:
        mask &= df["away_team"] == args.away
    if args.date:
        mask &= df["date"].dt.strftime("%Y-%m-%d") == args.date

    matches = df[mask]
    if len(matches) == 0:
        print(f"[error] No match found with given filters.", file=sys.stderr)
        sys.exit(1)

    results = []
    for _, row in matches.iterrows():
        pred = predict_match(model, le, feature_cols, row)
        pred["match"] = f"{row['home_team']} vs {row['away_team']}"
        pred["date"] = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else None
        pred["league"] = row.get("league", "unknown")
        pred["actual"] = row.get("result", None)
        results.append(pred)

    if args.output == "json":
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for r in results:
            print(f"\n{r['date']} | {r['league']} | {r['match']}")
            print(f"  P(主胜)={r['P_H']:.3f}  P(平)={r['P_D']:.3f}  P(客胜)={r['P_A']:.3f}")
            print(f"  预测方向: {r['predicted_direction']} | 置信度: {r['confidence']:.3f} ({r['confidence_label']})")
            if r.get("actual"):
                print(f"  实际结果: {r['actual']}")


if __name__ == "__main__":
    main()
