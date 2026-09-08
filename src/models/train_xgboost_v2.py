#!/usr/bin/env python3
"""
Train XGBoost 3-class classifier v2.
Anti-bias class weights: away > draw > home (to counteract home-win shortcut).
Uses enhanced v2 features with asymmetric home advantage.
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

TARGET = "result"
DROP_COLS = ["date", "season", "matchweek", "home_team", "away_team",
             "home_goals", "away_goals", "home_xg", "away_xg", "league",
             "result"]

# Anti-bias weights: counteract home-win shortcut
# H is over-predicted -> lower weight; A is under-predicted -> higher weight
ANTI_BIAS_WEIGHTS = {0: 0.7, 1: 1.5, 2: 2.0}  # H, D, A (will be mapped by label encoder)


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


def train_model(train_df, val_df, feature_cols, class_weights="anti_bias"):
    X_train, y_train, le = prepare_xy(train_df, feature_cols)
    X_val, y_val, _ = prepare_xy(val_df, feature_cols, le)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Apply class weights
    if class_weights == "anti_bias":
        # Map class index to anti-bias weight
        weight_map = {le.transform(["H"])[0]: 0.7,
                      le.transform(["D"])[0]: 1.5,
                      le.transform(["A"])[0]: 2.0}
        print(f"[weights] Anti-bias: {dict(zip(le.classes_, [weight_map.get(le.transform([c])[0], 1.0) for c in le.classes_]))}")
        sample_weight = np.array([weight_map.get(int(y), 1.0) for y in y_train])
        dtrain.set_weight(sample_weight)
    elif class_weights == "balanced":
        classes, counts = np.unique(y_train, return_counts=True)
        weights = {int(c): float(counts.sum() / (len(classes) * counts[i])) for i, c in enumerate(classes)}
        print(f"[weights] Balanced: {dict(zip(le.classes_, [weights.get(le.transform([c])[0], 1.0) for c in le.classes_]))}")
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

    # Per-class accuracy
    for i, cls in enumerate(le.classes_):
        mask = y == i
        if mask.sum() > 0:
            cls_acc = (preds[mask] == y[mask]).mean()
            cls_pred_count = (preds == i).sum()
            print(f"  {cls}: recall={cls_acc:.3f} | predicted={cls_pred_count} times")

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
    parser.add_argument("--input", default=PROC_DIR / "features_multi_league_v2.csv")
    parser.add_argument("--output", default=MODEL_DIR / "xgb_multi_league_v2")
    parser.add_argument("--class-weights", default="anti_bias", choices=["none", "balanced", "anti_bias"])
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
