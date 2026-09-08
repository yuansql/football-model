#!/usr/bin/env python3
"""
Hyperparameter tuning with Optuna for XGBoost multi-class model.
Maximizes validation accuracy while keeping model complexity in check.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
import optuna
from optuna.samplers import TPESampler

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DATES = {"train_end": "2024-08-01", "val_end": "2025-01-15"}
DROP_COLS = ["date", "season", "matchweek", "home_team", "away_team",
             "home_goals", "away_goals", "home_xg", "away_xg", "league", "result"]


def load_data(features_csv: Path):
    df = pd.read_csv(features_csv, parse_dates=["date"])
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    train = df[df["date"] < SPLIT_DATES["train_end"]]
    val = df[(df["date"] >= SPLIT_DATES["train_end"]) & (df["date"] < SPLIT_DATES["val_end"])]
    test = df[df["date"] >= SPLIT_DATES["val_end"]]

    le = LabelEncoder()
    y_train = le.fit_transform(train["result"].values)
    y_val = le.transform(val["result"].values)
    y_test = le.transform(test["result"].values)

    return (train[feature_cols].astype(float), y_train,
            val[feature_cols].astype(float), y_val,
            test[feature_cols].astype(float), y_test,
            feature_cols, le)


def objective(trial, X_train, y_train, X_val, y_val, classes, counts):
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "random_state": 42,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Class weights
    weights = {int(c): float(counts.sum() / (len(classes) * counts[i])) for i, c in enumerate(classes)}
    sample_weight = np.array([weights.get(int(y), 1.0) for y in y_train])
    dtrain.set_weight(sample_weight)

    model = xgb.train(
        params, dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    preds = np.argmax(model.predict(dval), axis=1)
    acc = accuracy_score(y_val, preds)
    # Also penalize large gap between train and val (overfitting)
    train_preds = np.argmax(model.predict(dtrain), axis=1)
    train_acc = accuracy_score(y_train, train_preds)
    overfit_penalty = max(0, (train_acc - acc - 0.15)) * 0.5  # penalize if gap > 15%

    return acc - overfit_penalty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=PROC_DIR / "features_multi_league_v1.csv")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--output", default=MODEL_DIR / "xgb_tuned")
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, le = load_data(args.features)
    classes, counts = np.unique(y_train, return_counts=True)

    print(f"[data] Train={len(X_train)} Val={len(X_val)} Test={len(X_test)} Features={len(feature_cols)}")
    print(f"[optuna] Starting {args.trials} trials...")

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, classes, counts),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    print(f"\n[best] Trial #{study.best_trial.number}")
    print(f"  Score: {study.best_value:.4f}")
    print(f"  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Train final model with best params
    best_params = {
        "objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
        "random_state": 42, **study.best_params,
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    weights = {int(c): float(counts.sum() / (len(classes) * counts[i])) for i, c in enumerate(classes)}
    sample_weight = np.array([weights.get(int(y), 1.0) for y in y_train])
    dtrain.set_weight(sample_weight)

    final_model = xgb.train(
        best_params, dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # Evaluate
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        probs = final_model.predict(xgb.DMatrix(X))
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y, preds)
        ll = log_loss(y, probs)
        print(f"[{name}] Accuracy: {acc:.4f} | LogLoss: {ll:.4f}")

    # Save
    final_model.save_model(str(args.output.with_suffix(".json")))
    with open(args.output.with_suffix(".pkl"), "wb") as f:
        pickle.dump({"label_encoder": le, "feature_cols": feature_cols}, f)
    print(f"\n[save] Tuned model -> {args.output}.json")

    # Save study
    study_path = OUT_DIR / "optuna_study.json"
    study_path.write_text(json.dumps({
        "best_trial": study.best_trial.number,
        "best_score": study.best_value,
        "best_params": study.best_params,
        "trials": args.trials,
    }, indent=2))
    print(f"[save] Study results -> {study_path}")


if __name__ == "__main__":
    main()
