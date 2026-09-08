#!/usr/bin/env python3
"""
Calibrate model predicted probabilities using Platt Scaling and Isotonic Regression.
Fits on validation set, evaluates on test set.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
PROC_DIR = ROOT / "data" / "processed"

# Time split boundaries (must match train_xgboost.py)
SPLIT_DATES = {
    "train_end": "2024-01-15",
    "val_end": "2024-08-01",
}

TARGET = "result"
DROP_COLS = ["date", "season", "matchweek", "home_team", "away_team",
             "home_goals", "away_goals", "home_xg", "away_xg", "league",
             "result"]


def load_data(features_csv: Path):
    df = pd.read_csv(features_csv, parse_dates=["date"])
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].astype(float)
    y = df[TARGET].values

    train_mask = df["date"] < SPLIT_DATES["train_end"]
    val_mask = (df["date"] >= SPLIT_DATES["train_end"]) & (df["date"] < SPLIT_DATES["val_end"])
    test_mask = df["date"] >= SPLIT_DATES["val_end"]

    return (
        X[train_mask], y[train_mask],
        X[val_mask], y[val_mask],
        X[test_mask], y[test_mask],
        df[test_mask].copy(),
        feature_cols,
    )


def load_model(model_stem: str):
    model_path = MODEL_DIR / f"{model_stem}.json"
    meta_path = MODEL_DIR / f"{model_stem}.pkl"
    model = xgb.Booster()
    model.load_model(str(model_path))
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return model, meta["label_encoder"], meta["feature_cols"]


def get_probs(model, X):
    dmat = xgb.DMatrix(X)
    return model.predict(dmat)


def evaluate_calibration(y_true, probs, le, name="uncalibrated"):
    preds = np.argmax(probs, axis=1)
    acc = (preds == y_true).mean()
    ll = log_loss(y_true, probs)

    # Brier score per class (multi-class)
    brier = 0.0
    for i, cls in enumerate(le.classes_):
        brier += brier_score_loss((y_true == i).astype(int), probs[:, i])
    brier /= len(le.classes_)

    print(f"\n[{name}] Accuracy: {acc:.4f} | LogLoss: {ll:.4f} | Brier: {brier:.4f}")

    # Top-1 calibration bins
    confidences = np.max(probs, axis=1)
    correct = (preds == y_true)
    bins = [0.0, 0.4, 0.5, 0.6, 0.8, 1.0]
    max_gap = 0.0
    print(f"[{name}] Top-1 calibration bins:")
    for i in range(len(bins) - 1):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        actual = correct[mask].mean()
        mean_conf = confidences[mask].mean()
        gap = abs(mean_conf - actual)
        max_gap = max(max_gap, gap)
        print(f"  bin [{bins[i]:.1f}-{bins[i+1]:.1f}): n={mask.sum():3d} | conf={mean_conf:.3f} | acc={actual:.3f} | gap={gap:.3f}")
    print(f"[{name}] Max calibration gap: {max_gap:.3f}")
    return {"accuracy": acc, "logloss": ll, "brier": brier, "max_gap": max_gap}


def platt_scaling(val_probs, val_y, test_probs, le):
    """Platt scaling: logistic regression on raw classifier outputs."""
    calibrated = np.zeros_like(test_probs)
    for i in range(len(le.classes_)):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(val_probs[:, i].reshape(-1, 1), (val_y == i).astype(int))
        calibrated[:, i] = lr.predict_proba(test_probs[:, i].reshape(-1, 1))[:, 1]

    # Normalize to sum to 1
    calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
    return calibrated


def isotonic_scaling(val_probs, val_y, test_probs, le):
    """Isotonic regression per class."""
    calibrated = np.zeros_like(test_probs)
    for i in range(len(le.classes_)):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_probs[:, i], (val_y == i).astype(int))
        calibrated[:, i] = iso.predict(test_probs[:, i])

    calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
    return calibrated


def temperature_scaling(val_probs, val_y, test_probs, le, lr=0.01, max_iter=1000):
    """Simple temperature scaling: optimize a single temperature parameter."""
    # Binary search for temperature T that minimizes log loss on val set
    best_T = 1.0
    best_ll = float('inf')
    for T in np.linspace(0.5, 3.0, 50):
        scaled = np.exp(val_probs / T)
        scaled = scaled / scaled.sum(axis=1, keepdims=True)
        ll = log_loss(val_y, scaled)
        if ll < best_ll:
            best_ll = ll
            best_T = T

    scaled = np.exp(test_probs / best_T)
    scaled = scaled / scaled.sum(axis=1, keepdims=True)
    print(f"[temperature] Best T={best_T:.3f} (val ll={best_ll:.4f})")
    return scaled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgb_multi_league_v1")
    parser.add_argument("--features", default=PROC_DIR / "features_multi_league_v1.csv")
    args = parser.parse_args()

    print("=" * 60)
    print("Probability Calibration Report")
    print("=" * 60)

    model, le, _ = load_model(args.model)
    X_train, y_train, X_val, y_val, X_test, y_test, test_df, feat_cols = load_data(args.features)

    # Raw predictions
    train_probs = get_probs(model, X_train)
    val_probs = get_probs(model, X_val)
    test_probs = get_probs(model, X_test)

    # Encode y
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    # Baseline (uncalibrated)
    print("\n--- Uncalibrated ---")
    base = evaluate_calibration(y_test_enc, test_probs, le, "uncalibrated")

    # Platt Scaling
    print("\n--- Platt Scaling ---")
    platt_test = platt_scaling(val_probs, y_val_enc, test_probs, le)
    platt = evaluate_calibration(y_test_enc, platt_test, le, "platt")

    # Isotonic Regression
    print("\n--- Isotonic Regression ---")
    iso_test = isotonic_scaling(val_probs, y_val_enc, test_probs, le)
    iso = evaluate_calibration(y_test_enc, iso_test, le, "isotonic")

    # Temperature Scaling
    print("\n--- Temperature Scaling ---")
    temp_test = temperature_scaling(val_probs, y_val_enc, test_probs, le)
    temp = evaluate_calibration(y_test_enc, temp_test, le, "temperature")

    # Summary
    print("\n" + "=" * 60)
    print("Summary (test set)")
    print("=" * 60)
    print(f"{'Method':<20} {'Acc':>8} {'LogLoss':>10} {'Brier':>10} {'MaxGap':>10}")
    print("-" * 60)
    for name, res in [("uncalibrated", base), ("platt", platt), ("isotonic", iso), ("temperature", temp)]:
        print(f"{name:<20} {res['accuracy']:>8.4f} {res['logloss']:>10.4f} {res['brier']:>10.4f} {res['max_gap']:>10.4f}")

    # Pick best by log loss on test
    best_name = min([("uncalibrated", base), ("platt", platt), ("isotonic", iso), ("temperature", temp)],
                    key=lambda x: x[1]["logloss"])[0]
    print(f"\n[best] {best_name} (by LogLoss)")

    # Save calibrated probabilities for downstream use
    out_dir = PROC_DIR
    if best_name == "platt":
        np.save(out_dir / "calibrated_test_probs.npy", platt_test)
    elif best_name == "isotonic":
        np.save(out_dir / "calibrated_test_probs.npy", iso_test)
    elif best_name == "temperature":
        np.save(out_dir / "calibrated_test_probs.npy", temp_test)
    else:
        np.save(out_dir / "calibrated_test_probs.npy", test_probs)
    print(f"[save] Calibrated probs -> {out_dir}/calibrated_test_probs.npy")


if __name__ == "__main__":
    main()
