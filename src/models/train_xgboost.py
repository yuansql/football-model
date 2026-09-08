#!/usr/bin/env python3
"""
Train XGBoost 3-class classifier for H/D/A prediction.
Time-based split: no random shuffle to prevent leakage.
Supports multi-league data.
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

# Time-based split boundaries (4 seasons: 22/23, 23/24, 24/25, 25/26)
SPLIT_DATES = {
    "train_end": "2024-08-01",   # 22/23 + 23/24 + 24/25 first half
    "val_end": "2025-01-15",     # 24/25 second half
    # test = 25/26 season
}

TARGET = "result"
DROP_COLS = ["date", "season", "matchweek", "home_team", "away_team",
             "home_goals", "away_goals", "home_xg", "away_xg", "league",
             "result"]


def load_features(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    print(f"[load] {len(df)} rows, {len(df.columns)} cols from {csv_path}")
    for lg, cnt in df["league"].value_counts().sort_index().items():
        print(f"       {lg}: {cnt}")
    return df


def time_split(df: pd.DataFrame):
    train = df[df["date"] < SPLIT_DATES["train_end"]]
    val = df[(df["date"] >= SPLIT_DATES["train_end"]) & (df["date"] < SPLIT_DATES["val_end"])]
    test = df[df["date"] >= SPLIT_DATES["val_end"]]
    print(f"[split] train={len(train)} | val={len(val)} | test={len(test)}")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name}:")
        for lg, cnt in split["league"].value_counts().sort_index().items():
            print(f"    {lg}: {cnt}")
    return train, val, test


def prepare_xy(df: pd.DataFrame, feature_cols: list, le: LabelEncoder = None):
    X = df[feature_cols].astype(float)
    y_raw = df[TARGET].values
    if le is None:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
    else:
        y = le.transform(y_raw)
    return X, y, le


def train_model(train_df, val_df, feature_cols, class_weights=None):
    X_train, y_train, le = prepare_xy(train_df, feature_cols)
    X_val, y_val, _ = prepare_xy(val_df, feature_cols, le)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    if class_weights == "balanced":
        classes, counts = np.unique(y_train, return_counts=True)
        weights = {int(c): float(counts.sum() / (len(classes) * counts[i])) for i, c in enumerate(classes)}
        print(f"[weights] {dict(zip(le.classes_, [weights.get(i, 1.0) for i in range(len(le.classes_))]))}")
        sample_weight = np.array([weights.get(int(y), 1.0) for y in y_train])
        dtrain.set_weight(sample_weight)

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
        "random_state": 42,
    }

    evals = [(dtrain, "train"), (dval, "val")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    print(f"[train] best iteration: {model.best_iteration}, best score: {model.best_score:.4f}")
    return model, le, X_train.columns.tolist()


def evaluate(model, df, feature_cols, le, split_name="test"):
    X, y, _ = prepare_xy(df, feature_cols, le)
    dtest = xgb.DMatrix(X)
    probs = model.predict(dtest)
    preds = np.argmax(probs, axis=1)

    acc = accuracy_score(y, preds)
    ll = log_loss(y, probs)
    print(f"\n[{split_name}] Accuracy: {acc:.4f} | LogLoss: {ll:.4f}")
    print(f"[{split_name}] Classification report:")
    print(classification_report(y, preds, target_names=le.classes_, digits=4))

    # Calibration
    print(f"[{split_name}] Calibration (top-1 predicted class):")
    confidences = np.max(probs, axis=1)
    correct = (preds == y)
    bins = [0.0, 0.4, 0.5, 0.6, 0.8, 1.0]
    for i in range(len(bins) - 1):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        actual = correct[mask].mean()
        mean_conf = confidences[mask].mean()
        print(f"  bin [{bins[i]:.1f}-{bins[i+1]:.1f}): n={mask.sum():3d} | avg_conf={mean_conf:.3f} | actual_acc={actual:.3f} | gap={mean_conf-actual:+.3f}")

    # Per-league breakdown
    if "league" in df.columns:
        print(f"[{split_name}] Per-league accuracy:")
        for lg in sorted(df["league"].unique()):
            lg_mask = df["league"] == lg
            if lg_mask.sum() > 0:
                lg_acc = accuracy_score(y[lg_mask], preds[lg_mask])
                print(f"    {lg}: {lg_acc:.4f} (n={lg_mask.sum()})")

    return {"accuracy": acc, "logloss": ll, "probs": probs, "preds": preds, "true": y}


def save_artifacts(model, le, feature_cols, output_stem: Path):
    model.save_model(str(output_stem.with_suffix(".json")))
    with open(output_stem.with_suffix(".pkl"), "wb") as f:
        pickle.dump({"label_encoder": le, "feature_cols": feature_cols}, f)
    print(f"[save] Model -> {output_stem}.json")
    print(f"[save] Meta  -> {output_stem}.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=PROC_DIR / "features_multi_league_v1.csv")
    parser.add_argument("--output", default=MODEL_DIR / "xgb_multi_league_v1")
    parser.add_argument("--class-weights", default="balanced", choices=["none", "balanced"])
    args = parser.parse_args()

    df = load_features(args.input)
    train, val, test = time_split(df)

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    print(f"[features] Using {len(feature_cols)} features")

    model, le, final_features = train_model(train, val, feature_cols, class_weights=args.class_weights)
    save_artifacts(model, le, final_features, Path(args.output))

    print("\n" + "=" * 50)
    evaluate(model, train, final_features, le, "train")
    evaluate(model, val, final_features, le, "val")
    evaluate(model, test, final_features, le, "test")
    print("=" * 50)

    importance = model.get_score(importance_type="gain")
    top10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n[Top 10 features by gain]")
    for feat, score in top10:
        print(f"  {feat}: {score:.1f}")


if __name__ == "__main__":
    main()
