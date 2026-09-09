#!/usr/bin/env python3
"""
Regression approach: predict xG difference (home_xg - away_xg).
Positive = home stronger → likely 主不败
Negative = away stronger → likely 客不败

This avoids the class imbalance problem by predicting a continuous target.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, roc_auc_score, f1_score

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


def prepare_xy(df: pd.DataFrame, feature_cols: list, target_col: str):
    X = df[feature_cols].astype(float)
    y = df[target_col].values
    return X, y


def train_model(X_train, y_train, X_val, y_val):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "reg_alpha": 0.5,
        "random_state": 42,
    }

    model = xgb.train(
        params, dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    print(f"[train] best_iteration={model.best_iteration}, best_score={model.best_score:.4f}")
    return model


def evaluate_regression(y_true, y_pred, name="test"):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"[{name}] RMSE={np.sqrt(mse):.4f} MAE={mae:.4f} R²={r2:.4f}")
    return {"rmse": np.sqrt(mse), "mae": mae, "r2": r2}


def evaluate_binary_from_regression(y_true_xgdiff, y_pred_xgdiff, y_true_binary, name="test", threshold=0.0):
    """
    Convert xG difference prediction to binary classification.
    pred_xgdiff > threshold → 主不败 (0)
    pred_xgdiff <= threshold → 客不败 (1)
    """
    preds_binary = np.where(y_pred_xgdiff > threshold, 0, 1)
    acc = accuracy_score(y_true_binary, preds_binary)
    auc = roc_auc_score(y_true_binary, y_pred_xgdiff)  # Use continuous score for AUC
    f1 = f1_score(y_true_binary, preds_binary)
    print(f"[{name}→binary] Acc={acc:.4f} AUC={auc:.4f} F1={f1:.4f} (threshold={threshold})")

    # Distribution
    print(f"  Pred distribution: 主不败={(preds_binary==0).mean():.1%} 客不败={(preds_binary==1).mean():.1%}")
    print(f"  True distribution: 主不败={(y_true_binary==0).mean():.1%} 客不败={(y_true_binary==1).mean():.1%}")

    return {"accuracy": acc, "auc": auc, "f1": f1}


def save_artifacts(model, feature_cols, output_stem: Path):
    model.save_model(str(output_stem.with_suffix(".json")))
    with open(output_stem.with_suffix(".pkl"), "wb") as f:
        pickle.dump({"feature_cols": feature_cols, "target": "xg_diff"}, f)
    print(f"\n[save] {output_stem}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=PROC_DIR / "features_multi_league_v2.csv")
    parser.add_argument("--output", default=MODEL_DIR / "xgb_regression_xgdiff")
    args = parser.parse_args()

    df = load_features(args.input)

    # Target: xG difference (home - away)
    df["xg_diff"] = df["home_xg"] - df["away_xg"]
    # Binary for comparison
    df["binary_label"] = np.where(df["result"].isin(["H", "D"]), 0, 1)

    train, val, test = time_split(df)

    feature_cols = [c for c in df.columns if c not in DROP_COLS + ["xg_diff", "binary_label"]]
    print(f"[features] Using {len(feature_cols)} features")

    X_train, y_train = prepare_xy(train, feature_cols, "xg_diff")
    X_val, y_val = prepare_xy(val, feature_cols, "xg_diff")
    X_test, y_test = prepare_xy(test, feature_cols, "xg_diff")

    model = train_model(X_train, y_train, X_val, y_val)
    save_artifacts(model, feature_cols, Path(args.output))

    print("\n" + "=" * 60)
    print("REGRESSION METRICS (xG diff)")
    print("=" * 60)

    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        preds = model.predict(xgb.DMatrix(X))
        evaluate_regression(y, preds, name)

    print("\n" + "=" * 60)
    print("BINARY CONVERSION (xG diff → 主不败/客不败)")
    print("=" * 60)

    for name, X, y_true_bin in [("train", X_train, train["binary_label"].values),
                                 ("val", X_val, val["binary_label"].values),
                                 ("test", X_test, test["binary_label"].values)]:
        preds = model.predict(xgb.DMatrix(X))
        evaluate_binary_from_regression(None, preds, y_true_bin, name)

    print("=" * 60)

    # Try different thresholds
    print("\n[Threshold sweep on validation set]")
    val_preds = model.predict(xgb.DMatrix(X_val))
    for thr in [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]:
        preds_bin = np.where(val_preds > thr, 0, 1)
        acc = accuracy_score(val["binary_label"].values, preds_bin)
        f1 = f1_score(val["binary_label"].values, preds_bin)
        print(f"  threshold={thr:+.1f}: Acc={acc:.4f} F1={f1:.4f}")

    # Top features
    importance = model.get_score(importance_type="gain")
    top10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n[Top 10 features by gain]")
    for feat, score in top10:
        print(f"  {feat}: {score:.1f}")


if __name__ == "__main__":
    main()
