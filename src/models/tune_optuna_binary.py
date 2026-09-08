#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for binary XGBoost (Home Double Chance vs Away).
Maximizes validation AUC + penalizes overfitting.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score
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
    df["binary_label"] = np.where(df["result"].isin(["H", "D"]), 0, 1)
    feature_cols = [c for c in df.columns if c not in DROP_COLS + ["binary_label"]]

    train = df[df["date"] < SPLIT_DATES["train_end"]]
    val = df[(df["date"] >= SPLIT_DATES["train_end"]) & (df["date"] < SPLIT_DATES["val_end"])]
    test = df[df["date"] >= SPLIT_DATES["val_end"]]

    return (train[feature_cols].astype(float), train["binary_label"].values,
            val[feature_cols].astype(float), val["binary_label"].values,
            test[feature_cols].astype(float), test["binary_label"].values,
            feature_cols)


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 3.0),
        "random_state": 42,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params, dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # Predictions
    val_probs = model.predict(dval)
    val_preds = (val_probs >= 0.5).astype(int)
    train_probs = model.predict(dtrain)
    train_preds = (train_probs >= 0.5).astype(int)

    # Metrics
    val_auc = roc_auc_score(y_val, val_probs)
    val_f1 = f1_score(y_val, val_preds)
    val_acc = accuracy_score(y_val, val_preds)
    train_acc = accuracy_score(y_train, train_preds)

    # Composite objective: AUC + F1 (minority class boost) - overfit penalty
    overfit_penalty = max(0, (train_acc - val_acc - 0.10)) * 0.3
    composite = val_auc + val_f1 * 0.5 - overfit_penalty

    return composite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=PROC_DIR / "features_multi_league_v2.csv")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--output", default=MODEL_DIR / "xgb_binary_tuned")
    parser.add_argument("--timeout", type=int, default=3600, help="Max seconds per trial")
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data(args.features)
    print(f"[data] Train={len(X_train)} Val={len(X_val)} Test={len(X_test)} Features={len(feature_cols)}")
    print(f"[label] 主不败={np.mean(y_train==0):.1%} 客不败={np.mean(y_train==1):.1%}")
    print(f"[optuna] {args.trials} trials, timeout={args.timeout}s")

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=args.trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    print(f"\n[best] Trial #{study.best_trial.number}")
    print(f"  Score: {study.best_value:.4f}")
    print(f"  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Train final model
    best_params = {
        "objective": "binary:logistic", "eval_metric": "logloss",
        "random_state": 42, **study.best_params,
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    final_model = xgb.train(
        best_params, dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # Evaluate
    for name, X, y, d in [("train", X_train, y_train, xgb.DMatrix(X_train)),
                           ("val", X_val, y_val, xgb.DMatrix(X_val)),
                           ("test", X_test, y_test, xgb.DMatrix(X_test))]:
        probs = final_model.predict(d)
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y, preds)
        ll = log_loss(y, probs)
        auc = roc_auc_score(y, probs)
        f1 = f1_score(y, preds)
        print(f"[{name}] Acc={acc:.4f} AUC={auc:.4f} F1={f1:.4f} LogLoss={ll:.4f}")

    # Save
    final_model.save_model(str(args.output.with_suffix(".json")))
    with open(args.output.with_suffix(".pkl"), "wb") as f:
        pickle.dump({"feature_cols": feature_cols, "classes": ["主不败", "客不败"]}, f)
    print(f"\n[save] {args.output}.json")

    study_path = OUT_DIR / "optuna_binary_study.json"
    study_path.write_text(json.dumps({
        "best_trial": study.best_trial.number,
        "best_score": study.best_value,
        "best_params": study.best_params,
        "trials": len(study.trials),
    }, indent=2))
    print(f"[save] {study_path}")


if __name__ == "__main__":
    main()
