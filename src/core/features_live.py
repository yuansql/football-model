"""
Live feature computation for pre-match prediction.
Shared by decision_marker.py, predict_live.py, batch_decision_mark.py.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soccerdata as sd

# Allow absolute imports from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config import WINDOWS


def normalize_team_name(name: str, mapping: dict) -> str:
    return mapping.get("aliases", {}).get(name, name)


def fetch_league_matches(league: str, mapping: dict) -> pd.DataFrame:
    """Fetch completed matches from Understat for a league.
    Raises ValueError on empty data so callers can handle gracefully.
    """
    try:
        understat = sd.Understat(leagues=league)
        matches = understat.read_schedule()
    except Exception as e:
        raise ValueError(f"Failed to fetch schedule for {league}: {e}") from e

    if isinstance(matches.index, pd.MultiIndex):
        matches = matches.reset_index(drop=True)

    df = matches[matches["is_result"] == True].copy()
    if df.empty:
        raise ValueError(f"No completed matches found for {league}")

    df = df.sort_values("date").reset_index(drop=True)
    df["home_team"] = df["home_team"].apply(lambda x: normalize_team_name(x, mapping))
    df["away_team"] = df["away_team"].apply(lambda x: normalize_team_name(x, mapping))
    df["result"] = np.where(
        df["home_goals"] > df["away_goals"], "H",
        np.where(df["home_goals"] < df["away_goals"], "A", "D")
    )
    return df


def build_team_records(df: pd.DataFrame) -> pd.DataFrame:
    """Convert match-level DataFrame to team-level records."""
    records = []
    for _, row in df.iterrows():
        for team, is_home, gf, ga, xgf, xga, res in [
            (row["home_team"], 1, row["home_goals"], row["away_goals"],
             row["home_xg"], row["away_xg"], row["result"]),
            (row["away_team"], 0, row["away_goals"], row["home_goals"],
             row["away_xg"], row["home_xg"], row["result"]),
        ]:
            pts = 3 if (res == "H" and is_home) or (res == "A" and not is_home) else (1 if res == "D" else 0)
            records.append({
                "date": row["date"], "team": team, "is_home": is_home,
                "goals_for": gf, "goals_against": ga,
                "xg_for": xgf, "xg_against": xga,
                "points": pts,
                "win": 1 if pts == 3 else 0,
                "draw": 1 if pts == 1 else 0,
                "loss": 1 if pts == 0 else 0,
            })
    return pd.DataFrame(records).sort_values(["team", "date"]).reset_index(drop=True)


def compute_rolling_features(tr: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Add rolling windows to team records. Returns (tr, roll_cols)."""
    roll_cols = []
    for w in WINDOWS:
        for col in ["goals_for", "goals_against", "xg_for", "xg_against",
                    "points", "win", "draw", "loss"]:
            cname = f"{col}_roll{w}"
            tr[cname] = tr.groupby("team")[col].shift(1).rolling(w, min_periods=1).mean().values
            roll_cols.append(cname)

        for col in ["goals_for", "goals_against", "xg_for", "xg_against", "points"]:
            tr[f"{col}_home_roll{w}"] = (
                tr[tr["is_home"] == 1].groupby("team")[col].shift(1)
                .rolling(w, min_periods=1).mean().reindex(tr.index).values
            )
            tr[f"{col}_away_roll{w}"] = (
                tr[tr["is_home"] == 0].groupby("team")[col].shift(1)
                .rolling(w, min_periods=1).mean().reindex(tr.index).values
            )
            roll_cols.append(f"{col}_home_roll{w}")
            roll_cols.append(f"{col}_away_roll{w}")

        # Asymmetric home advantage
        tr[f"home_advantage_pts_roll{w}"] = (
            tr[f"points_home_roll{w}"] - tr[f"points_away_roll{w}"]
        )
        tr[f"home_advantage_xg_roll{w}"] = (
            tr[f"xg_for_home_roll{w}"] - tr[f"xg_for_away_roll{w}"]
        )
        tr[f"home_advantage_xga_roll{w}"] = (
            tr[f"xg_against_home_roll{w}"] - tr[f"xg_against_away_roll{w}"]
        )
        roll_cols.extend([
            f"home_advantage_pts_roll{w}",
            f"home_advantage_xg_roll{w}",
            f"home_advantage_xga_roll{w}",
        ])

    # Momentum
    tr["momentum_pts"] = tr["points_roll3"] / (tr["points_roll10"] + 0.1)
    tr["momentum_xg"] = tr["xg_for_roll3"] / (tr["xg_for_roll10"] + 0.1)
    tr["momentum_xga"] = tr["xg_against_roll3"] / (tr["xg_against_roll10"] + 0.1)
    roll_cols.extend(["momentum_pts", "momentum_xg", "momentum_xga"])

    return tr, roll_cols


def compute_h2h(df: pd.DataFrame, home_team: str, away_team: str) -> dict:
    """Compute head-to-head stats."""
    h2h = df[
        (((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
         ((df["home_team"] == away_team) & (df["away_team"] == home_team)))
    ].tail(5)
    return {
        "home_win_rate": (h2h["result"] == "H").mean() if len(h2h) > 0 else 0.5,
        "draw_rate": (h2h["result"] == "D").mean() if len(h2h) > 0 else 0.25,
        "matches_count": len(h2h),
    }


def compute_league_table(df: pd.DataFrame) -> tuple[dict, dict]:
    """Compute league table. Returns (table, rank dict)."""
    table = {}
    for _, row in df.iterrows():
        for team, gf, ga, res, is_h in [
            (row["home_team"], row["home_goals"], row["away_goals"], row["result"], True),
            (row["away_team"], row["away_goals"], row["home_goals"], row["result"], False),
        ]:
            if team not in table:
                table[team] = {"pts": 0, "gd": 0, "played": 0}
            pts = 3 if (res == "H" and is_h) or (res == "A" and not is_h) else (1 if res == "D" else 0)
            table[team]["pts"] += pts
            table[team]["gd"] += gf - ga
            table[team]["played"] += 1

    ranked = sorted(table.items(), key=lambda x: (x[1]["pts"], x[1]["gd"]), reverse=True)
    rank = {team: i + 1 for i, (team, _) in enumerate(ranked)}
    return table, rank


def build_virtual_row(
    df: pd.DataFrame,
    tr: pd.DataFrame,
    roll_cols: list,
    home_team: str,
    away_team: str,
    league: str,
    league_enc: int = 0,
) -> tuple[pd.Series, dict, dict]:
    """
    Build a virtual match feature row for home_team vs away_team.
    Returns (virtual_row, form_snapshot, h2h_stats).
    """
    home_latest = tr[tr["team"] == home_team].iloc[-1]
    away_latest = tr[tr["team"] == away_team].iloc[-1]

    h2h_stats = compute_h2h(df, home_team, away_team)
    table, rank = compute_league_table(df)
    home_rank = rank.get(home_team, len(rank) // 2)
    away_rank = rank.get(away_team, len(rank) // 2)

    virtual = {"league_enc": league_enc}
    for col in roll_cols:
        virtual[f"home_{col}"] = home_latest[col] if col in home_latest else 0.0
        virtual[f"away_{col}"] = away_latest[col] if col in away_latest else 0.0

    virtual["h2h_home_win_rate"] = h2h_stats["home_win_rate"]
    virtual["h2h_draw_rate"] = h2h_stats["draw_rate"]
    virtual["h2h_matches_count"] = h2h_stats["matches_count"]
    virtual["home_rank"] = home_rank
    virtual["away_rank"] = away_rank
    virtual["rank_diff"] = away_rank - home_rank

    # Diff features
    for w in WINDOWS:
        for metric in ["points", "xg_for", "xg_against", "goals_for", "goals_against"]:
            virtual[f"diff_{metric}_roll{w}"] = (
                virtual[f"home_{metric}_roll{w}"] - virtual[f"away_{metric}_roll{w}"]
            )
        virtual[f"diff_home_advantage_pts_roll{w}"] = (
            virtual[f"home_home_advantage_pts_roll{w}"] -
            virtual[f"away_home_advantage_pts_roll{w}"]
        )

    form = {
        "home": {
            "rank": home_rank,
            "last5_pts": home_latest.get("points_roll5", 0),
            "last5_xg": home_latest.get("xg_for_roll5", 0),
        },
        "away": {
            "rank": away_rank,
            "last5_pts": away_latest.get("points_roll5", 0),
            "last5_xg": away_latest.get("xg_for_roll5", 0),
        },
        "h2h": h2h_stats,
    }
    return pd.Series(virtual), form, h2h_stats


def compute_live_features(
    league: str,
    home_team: str,
    away_team: str,
    mapping: dict,
    league_enc: int = 0,
) -> tuple[pd.Series | None, dict | None, pd.DataFrame | None]:
    """
    High-level helper: fetch data → build records → rolling → virtual row.
    Returns (virtual_row, form, df) or (None, error_info, None) on failure.
    Never raises; all errors are returned as dict in the second element.
    """
    try:
        df = fetch_league_matches(league, mapping)
    except Exception as e:
        return None, {"error": str(e)}, None

    # Check team existence and game count
    home_games = len(df[(df["home_team"] == home_team) | (df["away_team"] == home_team)])
    away_games = len(df[(df["home_team"] == away_team) | (df["away_team"] == away_team)])
    if home_games < 5:
        return None, {"error": f"insufficient data: {home_team} has only {home_games} games (need ≥5)"}, None
    if away_games < 5:
        return None, {"error": f"insufficient data: {away_team} has only {away_games} games (need ≥5)"}, None

    try:
        tr = build_team_records(df)
        tr, roll_cols = compute_rolling_features(tr)
        virtual, form, _ = build_virtual_row(df, tr, roll_cols, home_team, away_team, league, league_enc)
    except Exception as e:
        return None, {"error": f"feature computation failed: {e}"}, None

    return virtual, form, df
