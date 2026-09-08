#!/usr/bin/env python3
"""
Decision Marker: Model outputs a CREDIBILITY TAG, not a direction.
Analyst uses the tag to decide how much weight to give model vs intelligence.

Tags:
  - CONFIDENT_HOME    : Model strongly favors home double chance, high confidence
  - HESITANT_HOME     : Model slightly favors home, low confidence → intelligence matters more
  - HESITANT_AWAY     : Model slightly favors away, low confidence → intelligence matters more
  - CONFIDENT_AWAY    : Model strongly favors away double chance (rare but high value when hit)
  - TOSS_UP           : Model is split 50/50 → ignore model, rely fully on intelligence
  - INSUFFICIENT_DATA : Team has < 5 games this season
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
import soccerdata as sd

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
PROC_DIR = ROOT / "data" / "processed"
MAPPING_PATH = ROOT / "config" / "team_name_mapping.json"
WINDOWS = [3, 5, 10]


def normalize_team_name(name: str, mapping: dict) -> str:
    return mapping.get("aliases", {}).get(name, name)


def load_model(model_stem: str):
    model = xgb.Booster()
    model.load_model(str(MODEL_DIR / f"{model_stem}.json"))
    with open(MODEL_DIR / f"{model_stem}.pkl", "rb") as f:
        meta = pickle.load(f)
    return model, meta["feature_cols"]


def fetch_and_compute(league: str, home_team: str, away_team: str, mapping: dict):
    understat = sd.Understat(leagues=league)
    matches = understat.read_schedule()
    if isinstance(matches.index, pd.MultiIndex):
        matches = matches.reset_index(drop=True)
    df = matches[matches["is_result"] == True].copy()
    df = df.sort_values("date").reset_index(drop=True)

    df["home_team"] = df["home_team"].apply(lambda x: normalize_team_name(x, mapping))
    df["away_team"] = df["away_team"].apply(lambda x: normalize_team_name(x, mapping))
    df["result"] = np.where(df["home_goals"] > df["away_goals"], "H",
                            np.where(df["home_goals"] < df["away_goals"], "A", "D"))

    # Check if teams have enough data
    home_games = len(df[(df["home_team"] == home_team) | (df["away_team"] == home_team)])
    away_games = len(df[(df["home_team"] == away_team) | (df["away_team"] == away_team)])
    if home_games < 5 or away_games < 5:
        return None, {"home_games": home_games, "away_games": away_games}

    # Build rolling (simplified, reuse predict_live logic)
    records = []
    for _, row in df.iterrows():
        for team, is_home, gf, ga, xgf, xga, res in [
            (row["home_team"], 1, row["home_goals"], row["away_goals"], row["home_xg"], row["away_xg"], row["result"]),
            (row["away_team"], 0, row["away_goals"], row["home_goals"], row["away_xg"], row["home_xg"], row["result"]),
        ]:
            pts = 3 if (res == "H" and is_home) or (res == "A" and not is_home) else (1 if res == "D" else 0)
            records.append({
                "date": row["date"], "team": team, "is_home": is_home,
                "goals_for": gf, "goals_against": ga, "xg_for": xgf, "xg_against": xga,
                "points": pts, "win": 1 if pts == 3 else 0, "draw": 1 if pts == 1 else 0, "loss": 1 if pts == 0 else 0,
            })

    tr = pd.DataFrame(records).sort_values(["team", "date"]).reset_index(drop=True)

    roll_cols = []
    for w in WINDOWS:
        for col in ["goals_for", "goals_against", "xg_for", "xg_against", "points", "win", "draw", "loss"]:
            cname = f"{col}_roll{w}"
            tr[cname] = tr.groupby("team")[col].shift(1).rolling(w, min_periods=1).mean().values
            roll_cols.append(cname)
        for col in ["goals_for", "goals_against", "xg_for", "xg_against", "points"]:
            tr[f"{col}_home_roll{w}"] = tr[tr["is_home"]==1].groupby("team")[col].shift(1).rolling(w, min_periods=1).mean().reindex(tr.index).values
            tr[f"{col}_away_roll{w}"] = tr[tr["is_home"]==0].groupby("team")[col].shift(1).rolling(w, min_periods=1).mean().reindex(tr.index).values
            roll_cols.append(f"{col}_home_roll{w}")
            roll_cols.append(f"{col}_away_roll{w}")

        # Home advantage
        tr[f"home_advantage_pts_roll{w}"] = tr[f"points_home_roll{w}"] - tr[f"points_away_roll{w}"]
        tr[f"home_advantage_xg_roll{w}"] = tr[f"xg_for_home_roll{w}"] - tr[f"xg_for_away_roll{w}"]
        tr[f"home_advantage_xga_roll{w}"] = tr[f"xg_against_home_roll{w}"] - tr[f"xg_against_away_roll{w}"]
        roll_cols.extend([f"home_advantage_pts_roll{w}", f"home_advantage_xg_roll{w}", f"home_advantage_xga_roll{w}"])

    tr["momentum_pts"] = tr["points_roll3"] / (tr["points_roll10"] + 0.1)
    tr["momentum_xg"] = tr["xg_for_roll3"] / (tr["xg_for_roll10"] + 0.1)
    tr["momentum_xga"] = tr["xg_against_roll3"] / (tr["xg_against_roll10"] + 0.1)
    roll_cols.extend(["momentum_pts", "momentum_xg", "momentum_xga"])

    home_latest = tr[tr["team"] == home_team].iloc[-1]
    away_latest = tr[tr["team"] == away_team].iloc[-1]

    # H2H
    h2h = df[(((df["home_team"]==home_team)&(df["away_team"]==away_team))|((df["home_team"]==away_team)&(df["away_team"]==home_team)))].tail(5)
    h2h_home_win = (h2h["result"] == "H").mean() if len(h2h) > 0 else 0.5
    h2h_draw = (h2h["result"] == "D").mean() if len(h2h) > 0 else 0.25

    # Table
    table = {}
    for _, row in df.iterrows():
        for t, gf, ga, res, is_h in [(row["home_team"], row["home_goals"], row["away_goals"], row["result"], True),
                                       (row["away_team"], row["away_goals"], row["home_goals"], row["result"], False)]:
            if t not in table: table[t] = {"pts": 0, "gd": 0, "played": 0}
            pts = 3 if (res == "H" and is_h) or (res == "A" and not is_h) else (1 if res == "D" else 0)
            table[t]["pts"] += pts
            table[t]["gd"] += gf - ga
            table[t]["played"] += 1
    ranked = sorted(table.items(), key=lambda x: (x[1]["pts"], x[1]["gd"]), reverse=True)
    rank = {team: i+1 for i, (team, _) in enumerate(ranked)}
    home_rank = rank.get(home_team, len(rank)//2)
    away_rank = rank.get(away_team, len(rank)//2)

    virtual = {}
    proc_df = pd.read_csv(PROC_DIR / "features_multi_league_v2.csv")
    virtual["league_enc"] = proc_df[proc_df["league"]==league]["league_enc"].iloc[0] if len(proc_df[proc_df["league"]==league]) > 0 else 0
    for col in roll_cols:
        virtual[f"home_{col}"] = home_latest[col] if col in home_latest else 0
        virtual[f"away_{col}"] = away_latest[col] if col in away_latest else 0

    virtual["h2h_home_win_rate"] = h2h_home_win
    virtual["h2h_draw_rate"] = h2h_draw
    virtual["h2h_matches_count"] = len(h2h)
    virtual["home_rank"] = home_rank
    virtual["away_rank"] = away_rank
    virtual["rank_diff"] = away_rank - home_rank

    # diff features
    for w in WINDOWS:
        for metric in ["points", "xg_for", "xg_against", "goals_for", "goals_against"]:
            virtual[f"diff_{metric}_roll{w}"] = virtual[f"home_{metric}_roll{w}"] - virtual[f"away_{metric}_roll{w}"]
        virtual[f"diff_home_advantage_pts_roll{w}"] = virtual[f"home_home_advantage_pts_roll{w}"] - virtual[f"away_home_advantage_pts_roll{w}"]

    form = {
        "home": {"rank": home_rank, "last5_pts": home_latest.get("points_roll5", 0), "last5_xg": home_latest.get("xg_for_roll5", 0)},
        "away": {"rank": away_rank, "last5_pts": away_latest.get("points_roll5", 0), "last5_xg": away_latest.get("xg_for_roll5", 0)},
        "h2h": {"matches": len(h2h), "home_win_rate": h2h_home_win},
    }
    return pd.Series(virtual), form


def predict_binary(model, feature_cols, row):
    missing = [c for c in feature_cols if c not in row.index]
    for c in missing:
        row[c] = 0.0
    X = pd.DataFrame([row[feature_cols].astype(float).values], columns=feature_cols)
    prob_away = float(model.predict(xgb.DMatrix(X))[0])
    return prob_away


def get_tag(prob_away: float, form: dict) -> tuple[str, str]:
    """
    Returns (tag, advice) based on model probability and form context.
    """
    rank_diff = form["away"]["rank"] - form["home"]["rank"]

    if prob_away <= 0.20:
        return "CONFIDENT_HOME", f"模型强信号: 主不败极高信度。情报层只需验证无重大利空即可。"
    elif prob_away <= 0.35:
        return "LEAN_HOME", f"模型倾向主不败。情报层正常权重。注意客队是否有{form['away']['last5_xg']:.2f} xG级别的爆冷可能。"
    elif prob_away <= 0.50:
        return "TOSS_UP_HOME", f"模型微倾向主不败但信心不足(conf={1-prob_away:.2f})。情报层权重提高，模型仅作参考。"
    elif prob_away <= 0.65:
        return "TOSS_UP_AWAY", f"模型微倾向客不败但信心不足(conf={prob_away:.2f})。情报层权重提高，模型仅作参考。客队近5={form['away']['last5_pts']:.1f}分。"
    elif prob_away <= 0.80:
        return "LEAN_AWAY", f"模型倾向客不败。注意: 历史客不败预测命中率低，须强情报支持(伤停/战意/主场疲态)才可考虑。"
    else:
        return "CONFIDENT_AWAY", f"模型强信号: 客不败极高信度。罕见标记！须逐条验证反剧本(客队是否有必须取胜动力/主队核心缺阵)。"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgb_binary_v1")
    parser.add_argument("--league", required=True)
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    mapping = json.loads(MAPPING_PATH.read_text()) if MAPPING_PATH.exists() else {}
    home_norm = normalize_team_name(args.home, mapping)
    away_norm = normalize_team_name(args.away, mapping)

    model, feature_cols = load_model(args.model)
    virtual_row, form = fetch_and_compute(args.league, home_norm, away_norm, mapping)

    if virtual_row is None:
        tag = "INSUFFICIENT_DATA"
        advice = f"数据不足: 主队/客队赛季场次不够(各需≥5场)。"
        prob_away = None
    else:
        prob_away = predict_binary(model, feature_cols, virtual_row)
        tag, advice = get_tag(prob_away, form)

    conf_home = 1 - prob_away if prob_away is not None else None
    conf_away = prob_away

    if args.json:
        out = {
            "match": {"league": args.league, "home": home_norm, "away": away_norm},
            "timestamp": datetime.now().isoformat(),
            "tag": tag,
            "model_probability": {"主不败": round(conf_home, 4) if conf_home else None, "客不败": round(conf_away, 4) if conf_away else None},
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
            print(f"  客不败概率: {conf_away:.3f}")
        print()
        print(f"【形势快照】")
        print(f"  {home_norm}: 排名{form['home']['rank']} | 近5积分{form['home']['last5_pts']:.1f} | xG{form['home']['last5_xg']:.2f}")
        print(f"  {away_norm}: 排名{form['away']['rank']} | 近5积分{form['away']['last5_pts']:.1f} | xG{form['away']['last5_xg']:.2f}")
        print(f"  H2H: {form['h2h']['matches']}场 主胜率{form['h2h']['home_win_rate']:.0%}")
        print()
        print(f"【v17 建议】")
        print(f"  {advice}")
        print()
        print(f"【操作指南】")
        print({
            "CONFIDENT_HOME": "→ 模型主导。情报层只需确认无重大利空(核心伤停/战意缺失)即可采用模型方向。",
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
