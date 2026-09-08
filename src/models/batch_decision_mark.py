#!/usr/bin/env python3
"""
Batch Decision Marker for upcoming fixtures.

1. Fetches upcoming (unplayed) fixtures from Understat for all supported leagues.
2. Runs decision_marker logic on each match.
3. Outputs a structured Markdown report for v17 analysts.
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
MODEL_DIR = ROOT / "models"
PROC_DIR = ROOT / "data" / "processed"
MAPPING_PATH = ROOT / "config" / "team_name_mapping.json"
REPORTS_DIR = ROOT / "reports"
WINDOWS = [3, 5, 10]

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]


def normalize_team_name(name: str, mapping: dict) -> str:
    return mapping.get("aliases", {}).get(name, name)


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
    sched = sched[sched["is_result"] == False].copy()
    if "date" not in sched.columns:
        return pd.DataFrame()
    sched["date"] = pd.to_datetime(sched["date"])
    cutoff = datetime.now() + timedelta(days=days_ahead)
    sched = sched[sched["date"] <= cutoff]
    return sched.sort_values("date")


def build_team_records(league: str, mapping: dict):
    """Build historical team records up to now for feature computation."""
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

        tr[f"home_advantage_pts_roll{w}"] = tr[f"points_home_roll{w}"] - tr[f"points_away_roll{w}"]
        tr[f"home_advantage_xg_roll{w}"] = tr[f"xg_for_home_roll{w}"] - tr[f"xg_for_away_roll{w}"]
        tr[f"home_advantage_xga_roll{w}"] = tr[f"xg_against_home_roll{w}"] - tr[f"xg_against_away_roll{w}"]
        roll_cols.extend([f"home_advantage_pts_roll{w}", f"home_advantage_xg_roll{w}", f"home_advantage_xga_roll{w}"])

    tr["momentum_pts"] = tr["points_roll3"] / (tr["points_roll10"] + 0.1)
    tr["momentum_xg"] = tr["xg_for_roll3"] / (tr["xg_for_roll10"] + 0.1)
    tr["momentum_xga"] = tr["xg_against_roll3"] / (tr["xg_against_roll10"] + 0.1)
    roll_cols.extend(["momentum_pts", "momentum_xg", "momentum_xga"])

    return df, tr, roll_cols


def compute_virtual_row(df, tr, roll_cols, home_team, away_team, league):
    home_games = len(df[(df["home_team"] == home_team) | (df["away_team"] == home_team)])
    away_games = len(df[(df["home_team"] == away_team) | (df["away_team"] == away_team)])
    if home_games < 5 or away_games < 5:
        return None, {"home_games": home_games, "away_games": away_games}

    home_latest = tr[tr["team"] == home_team].iloc[-1]
    away_latest = tr[tr["team"] == away_team].iloc[-1]

    h2h = df[(((df["home_team"]==home_team)&(df["away_team"]==away_team))|((df["home_team"]==away_team)&(df["away_team"]==home_team)))].tail(5)
    h2h_home_win = (h2h["result"] == "H").mean() if len(h2h) > 0 else 0.5
    h2h_draw = (h2h["result"] == "D").mean() if len(h2h) > 0 else 0.25

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
    return float(model.predict(xgb.DMatrix(X))[0])


def get_tag(prob_away: float, form: dict) -> tuple:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgb_binary_v1")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--leagues", nargs="+", default=LEAGUES)
    parser.add_argument("--simulate", action="store_true", help="Use historical matches as fake fixtures for testing")
    parser.add_argument("--simulate-count", type=int, default=10, help="Number of simulated fixtures per league")
    parser.add_argument("--fixtures-csv", type=str, default="", help="Path to CSV with columns: league,date,home,away")
    args = parser.parse_args()

    mapping = json.loads(MAPPING_PATH.read_text()) if MAPPING_PATH.exists() else {}
    model, feature_cols = load_model(args.model)

    results = []

    # Mode 1: Manual CSV fixtures
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
            df, tr, roll_cols = build_team_records(league, mapping)
            if df.empty:
                print(f"Warning: no historical data for {league}, skipping.", file=sys.stderr)
                continue
            virtual_row, form = compute_virtual_row(df, tr, roll_cols, home, away, league)
            if virtual_row is None:
                tag = "INSUFFICIENT_DATA"
                prob_away = None
                conf_home = None
                action = "跳过模型，纯情报分析"
            else:
                prob_away = predict_binary(model, feature_cols, virtual_row)
                tag, action = get_tag(prob_away, form)
                conf_home = 1 - prob_away
            results.append({
                "date": date,
                "league": league,
                "home": home,
                "away": away,
                "tag": tag,
                "emoji": tag_emoji(tag),
                "prob_home": round(conf_home, 3) if conf_home is not None else None,
                "prob_away": round(prob_away, 3) if prob_away is not None else None,
                "home_rank": form.get("home", {}).get("rank", "-"),
                "away_rank": form.get("away", {}).get("rank", "-"),
                "home_form": round(form.get("home", {}).get("last5_pts", 0), 1),
                "away_form": round(form.get("away", {}).get("last5_pts", 0), 1),
                "v17_action": action,
            })
    else:
        # Mode 2: Auto-fetch from Understat
        for league in args.leagues:
            print(f"Fetching fixtures for {league} ...", file=sys.stderr)
            fixtures = fetch_fixtures(league, args.days)
            if fixtures.empty:
                print(f"  No upcoming fixtures found.", file=sys.stderr)
                continue

            df, tr, roll_cols = build_team_records(league, mapping)

            for _, row in fixtures.iterrows():
                home = normalize_team_name(row["home_team"], mapping)
                away = normalize_team_name(row["away_team"], mapping)
                date = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])

                virtual_row, form = compute_virtual_row(df, tr, roll_cols, home, away, league)
                if virtual_row is None:
                    tag = "INSUFFICIENT_DATA"
                    prob_away = None
                    conf_home = None
                    action = "跳过模型，纯情报分析"
                else:
                    prob_away = predict_binary(model, feature_cols, virtual_row)
                    tag, action = get_tag(prob_away, form)
                    conf_home = 1 - prob_away

                results.append({
                    "date": date,
                    "league": league,
                    "home": home,
                    "away": away,
                    "tag": tag,
                    "emoji": tag_emoji(tag),
                    "prob_home": round(conf_home, 3) if conf_home is not None else None,
                    "prob_away": round(prob_away, 3) if prob_away is not None else None,
                    "home_rank": form.get("home", {}).get("rank", "-"),
                    "away_rank": form.get("away", {}).get("rank", "-"),
                    "home_form": round(form.get("home", {}).get("last5_pts", 0), 1),
                    "away_form": round(form.get("away", {}).get("last5_pts", 0), 1),
                    "v17_action": action,
                })

    if not results and args.simulate:
        print("Simulating historical fixtures as upcoming matches...", file=sys.stderr)
        for league in args.leagues:
            df, tr, roll_cols = build_team_records(league, mapping)
            if df.empty:
                continue
            fake_fixtures = df.tail(args.simulate_count).copy()
            for _, row in fake_fixtures.iterrows():
                home = normalize_team_name(row["home_team"], mapping)
                away = normalize_team_name(row["away_team"], mapping)
                date = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
                virtual_row, form = compute_virtual_row(df, tr, roll_cols, home, away, league)
                if virtual_row is None:
                    tag = "INSUFFICIENT_DATA"
                    prob_away = None
                    conf_home = None
                    action = "跳过模型，纯情报分析"
                else:
                    prob_away = predict_binary(model, feature_cols, virtual_row)
                    tag, action = get_tag(prob_away, form)
                    conf_home = 1 - prob_away
                results.append({
                    "date": date,
                    "league": league,
                    "home": home,
                    "away": away,
                    "tag": tag,
                    "emoji": tag_emoji(tag),
                    "prob_home": round(conf_home, 3) if conf_home is not None else None,
                    "prob_away": round(prob_away, 3) if prob_away is not None else None,
                    "home_rank": form.get("home", {}).get("rank", "-"),
                    "away_rank": form.get("away", {}).get("rank", "-"),
                    "home_form": round(form.get("home", {}).get("last5_pts", 0), 1),
                    "away_form": round(form.get("away", {}).get("last5_pts", 0), 1),
                    "v17_action": action,
                })

    if not results:
        print("No fixtures found in the next {} days.".format(args.days))
        return

    df_out = pd.DataFrame(results)

    if args.json:
        print(df_out.to_json(orient="records", force_ascii=False, indent=2))
        return

    # Markdown report
    lines = []
    lines.append("# football-model 周报：未来 {} 天 Decision Marker".format(args.days))
    lines.append("")
    lines.append("生成时间: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M")))
    lines.append("")
    lines.append("## 速查表")
    lines.append("")
    lines.append("| 日期 | 联赛 | 主队 | 客队 | 标记 | 主不败 | 客不败 | 排名 | 近5积分 | v17 建议 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df_out.iterrows():
        lines.append("| {} | {} | {} | {} | {} {} | {} | {} | {}v{} | {}v{} | {} |".format(
            r["date"], r["league"].replace("ENG-Premier League", "EPL").replace("GER-Bundesliga", "Bundesliga"),
            r["home"], r["away"], r["emoji"], r["tag"],
            r["prob_home"] if r["prob_home"] is not None else "-",
            r["prob_away"] if r["prob_away"] is not None else "-",
            r["home_rank"], r["away_rank"],
            r["home_form"], r["away_form"],
            r["v17_action"],
        ))
    lines.append("")
    lines.append("## 标记说明")
    lines.append("")
    lines.append("| 标记 | 含义 | v17 操作 |")
    lines.append("|---|---|---|")
    lines.append("| 🟢 CONFIDENT_HOME | 模型强信号主不败 | 模型主导，情报验证利空 |")
    lines.append("| 🟡 LEAN_HOME | 模型倾向主不败 | 模型参考，情报平衡 |")
    lines.append("| ⚪ TOSS_UP | 模型无明确方向 | 情报主导，忽略模型 |")
    lines.append("| 🟠 LEAN_AWAY | 模型倾向客不败 | **情报主导**，模型弱信号（客不败历史命中低） |")
    lines.append("| 🔴 CONFIDENT_AWAY | 模型强信号客不败 | 罕见！逐条验证反剧本 |")
    lines.append("| ⚫ INSUFFICIENT_DATA | 数据不足 | 跳过模型 |")
    lines.append("")
    lines.append("## 详细分场")
    lines.append("")
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
