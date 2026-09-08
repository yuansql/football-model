#!/usr/bin/env python3
"""
Generate a complete v17-compatible prediction report.
Combines model probabilities with v17 hard-gate structure.

Usage:
    .venv/bin/python3 src/models/v17_full_report.py \
        --league "ENG-Premier League" --home "Arsenal" --away "Chelsea"

Output: Full v17 report with model p_model pre-filled.
Analyst fills in: 情报叙事, 取证清单, SP/赔率, 反剧本收据.
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
    return model, meta["label_encoder"], meta["feature_cols"]


def fetch_and_compute(league: str, home_team: str, away_team: str, mapping: dict):
    """Fetch latest data and build feature snapshot."""
    understat = sd.Understat(leagues=league)
    matches = understat.read_schedule()
    if isinstance(matches.index, pd.MultiIndex):
        matches = matches.reset_index(drop=True)
    df = matches[matches["is_result"] == True].copy()
    df = df.sort_values("date").reset_index(drop=True)

    df["home_team"] = df["home_team"].apply(lambda x: normalize_team_name(x, mapping))
    df["away_team"] = df["away_team"].apply(lambda x: normalize_team_name(x, mapping))

    df["result"] = np.where(
        df["home_goals"] > df["away_goals"], "H",
        np.where(df["home_goals"] < df["away_goals"], "A", "D")
    )

    # Build rolling stats
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

    home_latest = tr[tr["team"] == home_team].iloc[-1] if len(tr[tr["team"] == home_team]) > 0 else None
    away_latest = tr[tr["team"] == away_team].iloc[-1] if len(tr[tr["team"] == away_team]) > 0 else None
    if home_latest is None or away_latest is None:
        raise ValueError(f"Team not found in {league}")

    # H2H
    h2h = df[(((df["home_team"]==home_team)&(df["away_team"]==away_team))|((df["home_team"]==away_team)&(df["away_team"]==home_team)))].tail(5)
    h2h_win = (h2h["result"] == "H").mean() if len(h2h) > 0 else 0.5
    h2h_draw = (h2h["result"] == "D").mean() if len(h2h) > 0 else 0.25

    # Table
    table = {}
    for _, row in df.iterrows():
        for t, gf, ga, res, is_h in [(row["home_team"], row["home_goals"], row["away_goals"], row["result"], True),
                                       (row["away_team"], row["away_goals"], row["home_goals"], row["result"], False)]:
            if t not in table: table[t] = {"pts": 0, "gd": 0, "played": 0}
            pts = 3 if (res=="H" and is_h) or (res=="A" and not is_h) else (1 if res=="D" else 0)
            table[t]["pts"] += pts
            table[t]["gd"] += gf - ga
            table[t]["played"] += 1
    ranked = sorted(table.items(), key=lambda x: (x[1]["pts"], x[1]["gd"]), reverse=True)
    rank = {t: i+1 for i, (t, _) in enumerate(ranked)}
    home_rank, away_rank = rank.get(home_team, 10), rank.get(away_team, 10)

    # Build virtual row
    virtual = {}
    proc_df = pd.read_csv(PROC_DIR / "features_multi_league_v1.csv")
    virtual["league_enc"] = proc_df[proc_df["league"]==league]["league_enc"].iloc[0] if len(proc_df[proc_df["league"]==league]) > 0 else 0
    for col in roll_cols:
        virtual[f"home_{col}"] = home_latest[col] if col in home_latest else 0
        virtual[f"away_{col}"] = away_latest[col] if col in away_latest else 0
    virtual["h2h_home_win_rate"] = h2h_win
    virtual["h2h_draw_rate"] = h2h_draw
    virtual["h2h_matches_count"] = len(h2h)
    virtual["home_rank"] = home_rank
    virtual["away_rank"] = away_rank
    virtual["rank_diff"] = away_rank - home_rank

    # Form summary for report
    form = {
        "home": {
            "last5_pts": home_latest.get("points_roll5", 0),
            "last5_xg": home_latest.get("xg_for_roll5", 0),
            "last5_xga": home_latest.get("xg_against_roll5", 0),
            "last10_pts": home_latest.get("points_roll10", 0),
        },
        "away": {
            "last5_pts": away_latest.get("points_roll5", 0),
            "last5_xg": away_latest.get("xg_for_roll5", 0),
            "last5_xga": away_latest.get("xg_against_roll5", 0),
            "last10_pts": away_latest.get("points_roll10", 0),
        },
        "h2h": {"matches": len(h2h), "home_win_rate": h2h_win, "draw_rate": h2h_draw},
        "table": {"home_rank": home_rank, "away_rank": away_rank, "diff": away_rank - home_rank},
    }
    return pd.Series(virtual), form


def predict(model, le, feature_cols, row):
    missing = [c for c in feature_cols if c not in row.index]
    for c in missing:
        row[c] = 0.0
    X = pd.DataFrame([row[feature_cols].astype(float).values], columns=feature_cols)
    probs = model.predict(xgb.DMatrix(X))[0]
    pred_idx = int(np.argmax(probs))
    return {
        "P_H": probs[le.transform(["H"])[0]],
        "P_D": probs[le.transform(["D"])[0]],
        "P_A": probs[le.transform(["A"])[0]],
        "predicted": le.inverse_transform([pred_idx])[0],
        "confidence": probs[pred_idx],
    }


def generate_report(league, home, away, pred, form, today):
    conf_label = "high" if pred["confidence"] >= 0.6 else ("medium" if pred["confidence"] >= 0.5 else "low")
    direction = pred["predicted"]
    p_model_val = pred["P_H"] if direction == "H" else (pred["P_D"] if direction == "D" else pred["P_A"])

    direction_text = {"H": "主胜", "D": "平局", "A": "客胜"}[direction]

    lines = []
    lines.append("=" * 70)
    lines.append("【2足球框架 V17.4.9 · 模型辅助报告】")
    lines.append(f"生成时间: {today} | 数据来源: football-model v0.1.0")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"联赛: {league}")
    lines.append(f"对阵: {home} (主) vs {away} (客)")
    lines.append("")
    lines.append("-" * 70)
    lines.append("【取证清单】（模型预填部分 · 分析师补充 SOURCE_URL / 摘要）")
    lines.append("-" * 70)
    lines.append(f"| 槽位         | 状态    | 来源/备注 |")
    lines.append(f"| 积分榜       | HIT     | {home} 第{form['table']['home_rank']}名 vs {away} 第{form['table']['away_rank']}名 (差{form['table']['diff']}) |")
    lines.append(f"| 近5场积分    | HIT     | 主队:{form['home']['last5_pts']:.1f}分 客队:{form['away']['last5_pts']:.1f}分 |")
    lines.append(f"| 近5 xG       | HIT     | 主队:{form['home']['last5_xg']:.2f} 客队:{form['away']['last5_xg']:.2f} |")
    lines.append(f"| H2H近5       | HIT     | 共{form['h2h']['matches']}场 主胜率{form['h2h']['home_win_rate']:.1%} |")
    lines.append(f"| 伤停         | UNKNOWN | [待补充] |")
    lines.append(f"| SP/赔率      | UNKNOWN | [待粘贴] |")
    lines.append("")
    lines.append("-" * 70)
    lines.append("【模型概率输出】（替代手算泊松 p_model）")
    lines.append("-" * 70)
    lines.append(f"  P(主胜) = {pred['P_H']:.4f}")
    lines.append(f"  P(平)   = {pred['P_D']:.4f}")
    lines.append(f"  P(客胜) = {pred['P_A']:.4f}")
    lines.append(f"  模型方向: {direction}({direction_text}) | 置信度: {pred['confidence']:.4f} ({conf_label})")
    lines.append("")
    lines.append("-" * 70)
    lines.append("【硬闸自检】")
    lines.append("-" * 70)
    lines.append(f"联赛层级     = Tier1 ({league})")
    lines.append(f"口径         = MODEL (XGBoost, 7081场训练)")
    lines.append(f"p_model_H    = {pred['P_H']:.4f}")
    lines.append(f"p_model_D    = {pred['P_D']:.4f}")
    lines.append(f"p_model_A    = {pred['P_A']:.4f}")
    lines.append(f"方向         = {direction_text}")
    lines.append(f"p_model      = {p_model_val:.4f} (按方向取)")
    lines.append(f"model_conf   = {pred['confidence']:.4f} ({conf_label})")
    lines.append(f"JC_SP        = [待粘贴]")
    lines.append(f"p_fair       = [待计算]")
    lines.append(f"Edge_raw     = [待计算: p_model - p_fair]")
    lines.append(f"出票通道     = [待判断]")
    lines.append("")
    lines.append("-" * 70)
    lines.append("【情报叙事 / 球队画像 / 比赛剧本】")
    lines.append("-" * 70)
    lines.append("[此处由分析师填写：伤停、战意、赛程、风格克制、反剧本收据等]")
    lines.append("")
    lines.append("-" * 70)
    lines.append("【研究推荐】")
    lines.append("-" * 70)
    lines.append(f"星级         = ★★★☆☆ (模型置信度 {conf_label})")
    lines.append(f"胜平负       = {direction_text} (模型方向)")
    lines.append(f"单子倾向     = {'主胜' if direction=='H' else ('平局' if direction=='D' else '客胜')} (模型推荐)")
    lines.append(f"进球数       = [待补充]")
    lines.append(f"比分         = [待补充]")
    lines.append("")
    lines.append("-" * 70)
    lines.append("【出票】")
    lines.append("-" * 70)
    lines.append("状态: [待分析师填: 可介入 / 观望 / 不荐]")
    lines.append("原因: [Edge / 取证不足 / 反剧本 / 结构闸等]")
    lines.append("")
    lines.append("=" * 70)
    lines.append("【使用说明】")
    lines.append("1. 上方【模型概率输出】可直接替代 p_model手算.txt 的泊松步骤")
    lines.append("2. 分析师需补充: 伤停、SP、情报叙事、反剧本收据")
    lines.append("3. Edge计算: p_model(按方向) - p_fair → 按v17闸门判断")
    lines.append("4. 模型conf=low时，情报层权重应提高，不硬冲TOP1")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgb_multi_league_v1")
    parser.add_argument("--league", required=True)
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--output", default=None, help="Output file path (default: stdout)")
    args = parser.parse_args()

    mapping = json.loads(MAPPING_PATH.read_text()) if MAPPING_PATH.exists() else {}
    home_norm = normalize_team_name(args.home, mapping)
    away_norm = normalize_team_name(args.away, mapping)

    model, le, feature_cols = load_model(args.model)
    virtual_row, form = fetch_and_compute(args.league, home_norm, away_norm, mapping)
    pred = predict(model, le, feature_cols, virtual_row)

    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = generate_report(args.league, home_norm, away_norm, pred, form, today)

    if args.output:
        Path(args.output).write_text(report)
        print(f"[save] Report -> {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
