#!/usr/bin/env python3
"""
Bridge: Model P(H)/P(D)/P(A) → v17 p_model input.
Replaces the manual Poisson λ→p_model step in v17 framework.

Usage:
    .venv/bin/python src/models/predict_v17_bridge.py \
        --league "ENG-Premier League" \
        --home "Arsenal" --away "Chelsea" \
        --date "2025-03-16"

Output format (v17 hard-gate compatible):
    p_model_H=0.457  p_model_D=0.148  p_model_A=0.396
    predicted_direction=H  confidence=0.457 (low)
    → Use P_H as p_model when direction=主胜
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
    model = xgb.Booster()
    model.load_model(str(model_path))
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return model, meta["label_encoder"], meta["feature_cols"]


def find_match(features_csv, league, home, away, date=None):
    df = pd.read_csv(features_csv, parse_dates=["date"])
    mask = (df["league"] == league) & (df["home_team"] == home) & (df["away_team"] == away)
    if date:
        mask &= df["date"].dt.strftime("%Y-%m-%d") == date
    matches = df[mask]
    if len(matches) == 0:
        return None
    return matches.iloc[-1]  # most recent if multiple


def predict(model, le, feature_cols, row):
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


def v17_format(pred, match_info, actual=None):
    """Output in v17 hard-gate compatible format."""
    P_H, P_D, P_A = pred["P_H"], pred["P_D"], pred["P_A"]
    direction = pred["predicted"]
    conf = pred["confidence"]
    conf_label = "high" if conf >= 0.6 else ("medium" if conf >= 0.5 else "low")

    # p_model mapping for v17
    p_model_val = P_H if direction == "H" else (P_D if direction == "D" else P_A)

    lines = []
    lines.append("=" * 60)
    lines.append(f"v17 p_model Bridge Output")
    lines.append("=" * 60)
    lines.append(f"Match: {match_info.get('date', '?')} | {match_info.get('league', '?')}")
    lines.append(f"       {match_info['home_team']} vs {match_info['away_team']}")
    lines.append("")
    lines.append(f"【Model Probabilities】")
    lines.append(f"  P(H) = {P_H:.4f}  P(D) = {P_D:.4f}  P(A) = {P_A:.4f}")
    lines.append(f"  Sum  = {P_H+P_D+P_A:.4f}")
    lines.append("")
    lines.append(f"【v17 p_model Mapping】")
    lines.append(f"  方向=主胜  → p_model = P(H) = {P_H:.4f}")
    lines.append(f"  方向=平局  → p_model = P(D) = {P_D:.4f}")
    lines.append(f"  方向=客胜  → p_model = P(A) = {P_A:.4f}")
    lines.append(f"  方向=主不败 → p_model = P(H)+P(D) = {P_H+P_D:.4f} (for reference)")
    lines.append(f"  方向=客不败 → p_model = P(D)+P(A) = {P_D+P_A:.4f} (for reference)")
    lines.append("")
    lines.append(f"【Predicted Direction】")
    lines.append(f"  模型预测: {direction} | 置信度: {conf:.4f} ({conf_label})")
    lines.append(f"  若按模型方向({direction}) → p_model = {p_model_val:.4f}")
    lines.append("")
    lines.append(f"【Hard-Gate Fields (paste into v17)】")
    lines.append(f"  p_model_H={P_H:.4f}")
    lines.append(f"  p_model_D={P_D:.4f}")
    lines.append(f"  p_model_A={P_A:.4f}")
    lines.append(f"  model_direction={direction}")
    lines.append(f"  model_confidence={conf:.4f}")
    lines.append(f"  model_conf_label={conf_label}")
    lines.append("")
    if actual:
        lines.append(f"【Actual】 {actual}")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgb_multi_league_v1")
    parser.add_argument("--features", default=PROC_DIR / "features_multi_league_v1.csv")
    parser.add_argument("--league", required=True)
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--date", default=None)
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    model, le, feature_cols = load_model(args.model)
    row = find_match(args.features, args.league, args.home, args.away, args.date)
    if row is None:
        print(f"[error] No match found: {args.league} | {args.home} vs {args.away}", file=sys.stderr)
        sys.exit(1)

    pred = predict(model, le, feature_cols, row)
    actual = row.get("result") if "result" in row else None

    if args.json:
        out = {
            "match": {
                "date": row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else None,
                "league": args.league,
                "home": args.home,
                "away": args.away,
                "actual": actual,
            },
            "probabilities": {"H": round(pred["P_H"], 4), "D": round(pred["P_D"], 4), "A": round(pred["P_A"], 4)},
            "predicted_direction": pred["predicted"],
            "confidence": round(pred["confidence"], 4),
            "v17_p_model": {
                "主胜": round(pred["P_H"], 4),
                "平局": round(pred["P_D"], 4),
                "客胜": round(pred["P_A"], 4),
            },
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        match_info = {"date": row["date"], "league": args.league,
                      "home_team": args.home, "away_team": args.away}
        print(v17_format(pred, match_info, actual))


if __name__ == "__main__":
    main()
