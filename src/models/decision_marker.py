#!/usr/bin/env python3
"""
Decision Marker: Model outputs a CREDIBILITY TAG, not a direction.
Analyst uses the tag to decide how much weight to give model vs intelligence.

Tags:
  - CONFIDENT_HOME    : Model strongly favors home double chance, high confidence
  - LEAN_HOME         : Model tends home
  - TOSS_UP_HOME      : Model weakly home (ignore)
  - TOSS_UP_AWAY      : Model weakly away (ignore)
  - LEAN_AWAY         : Model tends away (weak signal historically)
  - CONFIDENT_AWAY    : Model strongly away (rare, verify carefully)
  - INSUFFICIENT_DATA : Team has < 5 games
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

# Allow imports from project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import MODEL_DIR, PROC_DIR, CONFIG_DIR
from src.core.features_live import normalize_team_name, compute_live_features


def load_model(model_stem: str):
    model = xgb.Booster()
    model.load_model(str(MODEL_DIR / f"{model_stem}.json"))
    with open(MODEL_DIR / f"{model_stem}.pkl", "rb") as f:
        meta = pickle.load(f)
    return model, meta["feature_cols"]


def predict_binary(model, feature_cols, row):
    missing = [c for c in feature_cols if c not in row.index]
    for c in missing:
        row[c] = 0.0
    X = pd.DataFrame([row[feature_cols].astype(float).values], columns=feature_cols)
    return float(model.predict(xgb.DMatrix(X))[0])


def get_tag(prob_away: float, form: dict) -> tuple[str, str]:
    """Returns (tag, advice)."""
    if prob_away <= 0.20:
        return "CONFIDENT_HOME", "模型主导，情报验证"
    elif prob_away <= 0.35:
        return "LEAN_HOME", "模型参考，情报平衡"
    elif prob_away <= 0.50:
        return "TOSS_UP_HOME", "情报主导，模型辅助"
    elif prob_away <= 0.65:
        return "TOSS_UP_AWAY", "情报主导，模型辅助"
    elif prob_away <= 0.80:
        return "LEAN_AWAY", "情报主导，模型弱信号"
    else:
        return "CONFIDENT_AWAY", "情报逐条验证，模型罕见强信号"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgb_binary_tuned")
    parser.add_argument("--league", required=True)
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    mapping_path = CONFIG_DIR / "team_name_mapping.json"
    mapping = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    home_norm = normalize_team_name(args.home, mapping)
    away_norm = normalize_team_name(args.away, mapping)

    model, feature_cols = load_model(args.model)

    # Load league_enc from processed data
    proc_df = pd.read_csv(PROC_DIR / "features_multi_league_v2.csv")
    league_enc = (
        proc_df[proc_df["league"] == args.league]["league_enc"].iloc[0]
        if len(proc_df[proc_df["league"] == args.league]) > 0 else 0
    )

    virtual_row, form_or_err, _ = compute_live_features(
        args.league, home_norm, away_norm, mapping, league_enc
    )

    if virtual_row is None:
        tag = "INSUFFICIENT_DATA"
        advice = f"数据不足: {form_or_err}"
        prob_away = None
        conf_home = None
        form = form_or_err
    else:
        prob_away = predict_binary(model, feature_cols, virtual_row)
        tag, advice = get_tag(prob_away, form_or_err)
        conf_home = 1 - prob_away
        form = form_or_err

    if args.json:
        out = {
            "match": {"league": args.league, "home": home_norm, "away": away_norm},
            "timestamp": datetime.now().isoformat(),
            "tag": tag,
            "model_probability": {
                "主不败": round(conf_home, 4) if conf_home is not None else None,
                "客不败": round(prob_away, 4) if prob_away is not None else None,
            },
            "form_snapshot": form,
            "advice": advice,
            "v17_action": {
                "CONFIDENT_HOME": "模型主导，情报验证",
                "LEAN_HOME": "模型参考，情报平衡",
                "TOSS_UP_HOME": "情报主导，模型辅助",
                "TOSS_UP_AWAY": "情报主导，模型辅助",
                "LEAN_AWAY": "情报主导，模型弱信号",
                "CONFIDENT_AWAY": "情报逐条验证，模型罕见强信号",
                "INSUFFICIENT_DATA": "跳过模型，纯情报分析",
            }.get(tag, ""),
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print("=" * 70)
        print("football-model Decision Marker")
        print("=" * 70)
        print(f"{args.league}")
        print(f"{home_norm} (主) vs {away_norm} (客)")
        print()
        print(f"【模型标记】 {tag}")
        if prob_away is not None:
            print(f"  主不败概率: {conf_home:.3f}")
            print(f"  客不败概率: {prob_away:.3f}")
        print()
        if isinstance(form, dict) and "home" in form:
            print(f"【形势快照】")
            print(f"  {home_norm}: 排名{form['home']['rank']} | 近5积分{form['home']['last5_pts']:.1f} | xG{form['home']['last5_xg']:.2f}")
            print(f"  {away_norm}: 排名{form['away']['rank']} | 近5积分{form['away']['last5_pts']:.1f} | xG{form['away']['last5_xg']:.2f}")
            print(f"  H2H: {form['h2h']['matches_count']}场 主胜率{form['h2h']['home_win_rate']:.0%}")
            print()
        print(f"【v17 建议】")
        print(f"  {advice}")
        print()
        print(f"【操作指南】")
        print({
            "CONFIDENT_HOME": "→ 模型主导。情报层只需确认无重大利空即可采用模型方向。",
            "LEAN_HOME": "→ 模型参考。正常情报分析，模型作为基本面锚点。",
            "TOSS_UP_HOME": "→ 情报主导。模型信号弱，盘口/伤停/战意权重提高。",
            "TOSS_UP_AWAY": "→ 情报主导。模型信号弱，盘口/伤停/战意权重提高。",
            "LEAN_AWAY": "→ 情报主导。客不败历史命中率低，须有明确反剧本证据才考虑。",
            "CONFIDENT_AWAY": "→ 罕见强信号！逐条验证: ①客队战意 ②主队核心伤停 ③赛程疲劳 ④H2H客场优势。",
            "INSUFFICIENT_DATA": "→ 跳过模型。纯情报分析，或等更多比赛数据。",
        }.get(tag, ""))
        print("=" * 70)


if __name__ == "__main__":
    main()
