#!/usr/bin/env python3
"""
Batch Decision Marker for upcoming fixtures.
Outputs a structured Markdown report for v17 analysts.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
import soccerdata as sd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import MODEL_DIR, PROC_DIR, REPORTS_DIR, LEAGUES
from src.core.features_live import (
    normalize_team_name, compute_live_features,
)


def load_model(model_stem: str):
    model = xgb.Booster()
    model.load_model(str(MODEL_DIR / f"{model_stem}.json"))
    with open(MODEL_DIR / f"{model_stem}.pkl", "rb") as f:
        meta = pickle.load(f)
    return model, meta["feature_cols"]


def fetch_fixtures(league: str, days_ahead: int = 7):
    """Return unplayed fixtures in the next N days."""
    understat = sd.Understat(leagues=league)
    sched = understat.read_schedule()
    if isinstance(sched.index, pd.MultiIndex):
        sched = sched.reset_index(drop=True)
    sched = sched[~sched["is_result"]].copy()
    if "date" not in sched.columns:
        return pd.DataFrame()
    sched["date"] = pd.to_datetime(sched["date"])
    cutoff = datetime.now() + timedelta(days=days_ahead)
    sched = sched[sched["date"] <= cutoff]
    return sched.sort_values("date")


def predict_binary(model, feature_cols, row):
    missing = [c for c in feature_cols if c not in row.index]
    for c in missing:
        row[c] = 0.0
    X = pd.DataFrame([row[feature_cols].astype(float).values], columns=feature_cols)
    return float(model.predict(xgb.DMatrix(X))[0])


def get_tag(prob_away: float) -> tuple[str, str]:
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


def tag_emoji(tag: str) -> str:
    return {
        "CONFIDENT_HOME": "🟢",
        "LEAN_HOME": "🟡",
        "TOSS_UP_HOME": "⚪",
        "TOSS_UP_AWAY": "⚪",
        "LEAN_AWAY": "🟠",
        "CONFIDENT_AWAY": "🔴",
        "INSUFFICIENT_DATA": "⚫",
    }.get(tag, "❓")


def process_match(model, feature_cols, league, home, away, mapping, league_enc, date_str):
    """Process a single match and return result dict."""
    virtual_row, form_or_err, _ = compute_live_features(league, home, away, mapping, league_enc)

    if virtual_row is None:
        return {
            "date": date_str, "league": league, "home": home, "away": away,
            "tag": "INSUFFICIENT_DATA", "emoji": tag_emoji("INSUFFICIENT_DATA"),
            "prob_home": None, "prob_away": None,
            "home_rank": "-", "away_rank": "-",
            "home_form": 0.0, "away_form": 0.0,
            "v17_action": "跳过模型，纯情报分析",
        }

    prob_away = predict_binary(model, feature_cols, virtual_row)
    tag, action = get_tag(prob_away)
    conf_home = 1 - prob_away
    form = form_or_err

    return {
        "date": date_str, "league": league, "home": home, "away": away,
        "tag": tag, "emoji": tag_emoji(tag),
        "prob_home": round(conf_home, 3), "prob_away": round(prob_away, 3),
        "home_rank": form["home"]["rank"], "away_rank": form["away"]["rank"],
        "home_form": round(form["home"]["last5_pts"], 1),
        "away_form": round(form["away"]["last5_pts"], 1),
        "v17_action": action,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgb_binary_tuned")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--leagues", nargs="+", default=LEAGUES)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--simulate-count", type=int, default=10)
    parser.add_argument("--fixtures-csv", type=str, default="")
    args = parser.parse_args()

    mapping_path = Path(__file__).resolve().parents[2] / "config" / "team_name_mapping.json"
    mapping = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    model, feature_cols = load_model(args.model)

    # League encodings
    proc_df = pd.read_csv(PROC_DIR / "features_multi_league_v2.csv")
    league_enc_map = {lg: int(proc_df[proc_df["league"] == lg]["league_enc"].iloc[0])
                      for lg in proc_df["league"].unique()}

    results = []

    # Mode 1: Manual CSV
    if args.fixtures_csv:
        fixtures_csv = pd.read_csv(args.fixtures_csv)
        required = {"league", "date", "home", "away"}
        if not required.issubset(fixtures_csv.columns):
            print(f"CSV must contain columns: {required}", file=sys.stderr)
            sys.exit(1)
        for _, row in fixtures_csv.iterrows():
            league = row["league"]
            home = normalize_team_name(row["home"], mapping)
            away = normalize_team_name(row["away"], mapping)
            date = str(row["date"])
            enc = league_enc_map.get(league, 0)
            try:
                results.append(process_match(model, feature_cols, league, home, away, mapping, enc, date))
            except Exception as e:
                print(f"  [skip] {league}: {home} vs {away} — {e}", file=sys.stderr)
                results.append({
                    "date": date, "league": league, "home": home, "away": away,
                    "tag": "ERROR", "emoji": "❌",
                    "prob_home": None, "prob_away": None,
                    "home_rank": "-", "away_rank": "-",
                    "home_form": 0.0, "away_form": 0.0,
                    "v17_action": f"处理失败: {e}",
                })

    else:
        # Mode 2: Auto-fetch fixtures
        for league in args.leagues:
            print(f"Fetching fixtures for {league} ...", file=sys.stderr)
            try:
                fixtures = fetch_fixtures(league, args.days)
            except Exception as e:
                print(f"  [skip] Failed to fetch fixtures for {league}: {e}", file=sys.stderr)
                continue
            enc = league_enc_map.get(league, 0)

            if fixtures.empty:
                print(f"  No upcoming fixtures found.", file=sys.stderr)
                continue

            for _, row in fixtures.iterrows():
                home = normalize_team_name(row["home_team"], mapping)
                away = normalize_team_name(row["away_team"], mapping)
                date = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
                try:
                    results.append(process_match(model, feature_cols, league, home, away, mapping, enc, date))
                except Exception as e:
                    print(f"  [skip] {league}: {home} vs {away} — {e}", file=sys.stderr)
                    results.append({
                        "date": date, "league": league, "home": home, "away": away,
                        "tag": "ERROR", "emoji": "❌",
                        "prob_home": None, "prob_away": None,
                        "home_rank": "-", "away_rank": "-",
                        "home_form": 0.0, "away_form": 0.0,
                        "v17_action": f"处理失败: {e}",
                    })

    # Fallback: simulate mode when no fixtures found
    if not results and args.simulate:
        print("Simulating historical fixtures as upcoming matches...", file=sys.stderr)
        for league in args.leagues:
            enc = league_enc_map.get(league, 0)
            try:
                from src.core.features_live import fetch_league_matches
                df = fetch_league_matches(league, mapping)
            except Exception as e:
                print(f"  [skip] Failed to fetch history for {league}: {e}", file=sys.stderr)
                continue
            if df.empty:
                continue
            fake = df.tail(args.simulate_count).copy()
            for _, row in fake.iterrows():
                home = normalize_team_name(row["home_team"], mapping)
                away = normalize_team_name(row["away_team"], mapping)
                date = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
                try:
                    results.append(process_match(model, feature_cols, league, home, away, mapping, enc, date))
                except Exception as e:
                    print(f"  [skip] {league}: {home} vs {away} — {e}", file=sys.stderr)
                    results.append({
                        "date": date, "league": league, "home": home, "away": away,
                        "tag": "ERROR", "emoji": "❌",
                        "prob_home": None, "prob_away": None,
                        "home_rank": "-", "away_rank": "-",
                        "home_form": 0.0, "away_form": 0.0,
                        "v17_action": f"处理失败: {e}",
                    })

    if not results:
        print("No fixtures found.")
        return

    df_out = pd.DataFrame(results)

    if args.json:
        print(df_out.to_json(orient="records", force_ascii=False, indent=2))
        return

    # Markdown report
    lines = [
        "# football-model 周报：未来 {} 天 Decision Marker".format(args.days),
        "",
        "生成时间: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M")),
        "",
        "## 速查表",
        "",
        "| 日期 | 联赛 | 主队 | 客队 | 标记 | 主不败 | 客不败 | 排名 | 近5积分 | v17 建议 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df_out.iterrows():
        lg_short = r["league"].replace("ENG-Premier League", "EPL").replace("GER-Bundesliga", "Bundesliga")
        lines.append("| {} | {} | {} | {} | {} {} | {} | {} | {}v{} | {}v{} | {} |".format(
            r["date"], lg_short, r["home"], r["away"], r["emoji"], r["tag"],
            r["prob_home"] if r["prob_home"] is not None else "-",
            r["prob_away"] if r["prob_away"] is not None else "-",
            r["home_rank"], r["away_rank"],
            r["home_form"], r["away_form"],
            r["v17_action"],
        ))

    lines.extend([
        "",
        "## 标记说明",
        "",
        "| 标记 | 含义 | v17 操作 |",
        "|---|---|---|",
        "| 🟢 CONFIDENT_HOME | 模型强信号主不败 | 模型主导，情报验证利空 |",
        "| 🟡 LEAN_HOME | 模型倾向主不败 | 模型参考，情报平衡 |",
        "| ⚪ TOSS_UP | 模型无明确方向 | 情报主导，忽略模型 |",
        "| 🟠 LEAN_AWAY | 模型倾向客不败 | **情报主导**，模型弱信号（客不败历史命中低） |",
        "| 🔴 CONFIDENT_AWAY | 模型强信号客不败 | 罕见！逐条验证反剧本 |",
        "| ⚫ INSUFFICIENT_DATA | 数据不足 | 跳过模型 |",
        "",
        "## 详细分场",
        "",
    ])
    for _, r in df_out.iterrows():
        lines.append("### {} {} vs {}".format(r["emoji"], r["home"], r["away"]))
        lines.append("- **日期**: {} | **联赛**: {}".format(r["date"], r["league"]))
        lines.append("- **标记**: `{}`".format(r["tag"]))
        if r["prob_home"] is not None:
            lines.append("- **模型概率**: 主不败 {} | 客不败 {}".format(r["prob_home"], r["prob_away"]))
        lines.append("- **排名**: {} 排{} vs {} 排{}".format(r["home"], r["home_rank"], r["away"], r["away_rank"]))
        lines.append("- **近5积分**: {} {}分 vs {} {}分".format(r["home"], r["home_form"], r["away"], r["away_form"]))
        lines.append("- **v17 建议**: {}".format(r["v17_action"]))
        lines.append("")

    md = "\n".join(lines)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(md, encoding="utf-8")
        print(f"Report written to {out_path}")
    else:
        print(md)


if __name__ == "__main__":
    main()
