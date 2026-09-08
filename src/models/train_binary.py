#!/usr/bin/env python3
"""
Binary classifier: Home Double Chance (H/D) vs Away Double Chance (A).
Maps to v17's atomic directions: 主不败 vs 客不败.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DATES = {
    "train_end": "2024-08-01",
    "val_end": "2025-01-15",
}

DROP_COLS = ["date", "season", "matchweek", "home_team", "away_team",
             "home_goals", "away_goals", "home_xg", "away_xg", "league",
             "result"]


def load_features(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    print(f"[load] {len(df)} rows, {len(df.columns)} cols")
    return df


def time_split(df: pd.DataFrame):
    train = df[df["date"] < SPLIT_DATES["train_end"]]
    val = df[(df["date"] >= SPLIT_DATES["train_end"]) & (df["date"] < SPLIT_DATES["val_end"])]
    test = df[df["date"] >= SPLIT_DATES["val_end"]]
    print(f"[split] train={len(train)} | val={len(val)} | test={len(test)}")
    return train, val, test


def to_binary_label(result: str) -> int:
    """H/D -> 0 (主不败), A -> 1 (客不败)"""
    return 0 if result in ("H", "D") else 1


def prepare_xy(df: pd.DataFrame, feature_cols: list):
    X = df[feature_cols].astype(float)
    y = df["binary_label"].values
    return X, y


def train_model(train_df, val_df, feature_cols):
    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_val, y_val = prepare_xy(val_df, feature_cols)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Class balance
    classes, counts = np.unique(y_train, return_counts=True)
    weights = {int(c): float(counts.sum() / (len(classes) * counts[i])) for i, c in enumerate(classes)}
    print(f"[weights] {dict(zip(['主不败', '客不败'], [weights.get(0, 1.0), weights.get(1, 1.0)]))}")
    sample_weight = np.array([weights.get(int(y), 1.0) for y in y_train])
    dtrain.set_weight(sample_weight)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
        "random_state": 42,
    }

    model = xgb.train(
        params, dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    print(f"[train] best iteration: {model.best_iteration}, best score: {model.best_score:.4f}")
    return model, X_train.columns.tolist()


def evaluate(model, df, feature_cols, split_name="test"):
    X, y = prepare_xy(df, feature_cols)
    dtest = xgb.DMatrix(X)
    probs = model.predict(dtest)
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y, preds)
    ll = log_loss(y, probs)
    print(f"\n[{split_name}] Accuracy: {acc:.4f} | LogLoss: {ll:.4f}")
    print(f"[{split_name}] Classification report:")
    print(classification_report(y, preds, target_names=["主不败(H/D)", "客不败(A)"], digits=4))

    # Per-league
    if "league" in df.columns:
        print(f"[{split_name}] Per-league accuracy:")
        for lg in sorted(df["league"].unique()):
            lg_mask = df["league"] == lg
            if lg_mask.sum() > 0:
                lg_acc = accuracy_score(y[lg_mask], preds[lg_mask])
                print(f"    {lg}: {lg_acc:.4f} (n={lg_mask.sum()})")

    return {"accuracy": acc, "logloss": ll, "probs": probs, "preds": preds, "true": y}


def save_artifacts(model, feature_cols, output_stem: Path):
    model.save_model(str(output_stem.with_suffix(".json")))
    with open(output_stem.with_suffix(".pkl"), "wb") as f:
        pickle.dump({"feature_cols": feature_cols, "classes": ["主不败", "客不败"]}, f)
    print(f"[save] Model -> {output_stem}.json")
    print(f"[save] Meta  -> {output_stem}.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=PROC_DIR / "features_multi_league_v2.csv")
    parser.add_argument("--output", default=MODEL_DIR / "xgb_binary_v1")
    args = parser.parse_args()

    df = load_features(args.input)
    df["binary_label"] = df["result"].apply(to_binary_label)

    train, val, test = time_split(df)

    feature_cols = [c for c in df.columns if c not in DROP_COLS + ["binary_label"]]
    print(f"[features] Using {len(feature_cols)} features")

    # Label balance
    for name, split in [("train", train), ("val", val), ("test", test)]:
        labels = split["binary_label"].value_counts(normalize=True).sort_index()
        print(f"  {name} label balance: 主不败={labels.get(0, 0):.1%} 客不败={labels.get(1, 0):.1%}")

    model, final_features = train_model(train, val, feature_cols)
    save_artifacts(model, final_features, Path(args.output))

    print("\n" + "=" * 50)
    evaluate(model, train, final_features, "train")
    evaluate(model, val, final_features, "val")
    evaluate(model, test, final_features, "test")
    print("=" * 50)

    importance = model.get_score(importance_type="gain")
    top10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n[Top 10 features by gain]")
    for feat, score in top10:
        print(f"  {feat}: {score:.1f}")


if __name__ == "__main__":
    main()
