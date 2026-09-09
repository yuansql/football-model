#!/usr/bin/env python3
"""
SHAP explainability for binary XGBoost model (Home Double Chance vs Away).
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DATES = {"train_end": "2024-08-01", "val_end": "2025-01-15"}
DROP_COLS = ["date", "season", "matchweek", "home_team", "away_team",
             "home_goals", "away_goals", "home_xg", "away_xg", "league",
             "result", "binary_label"]


def load_model(model_stem: str):
    model = xgb.Booster()
    model.load_model(str(MODEL_DIR / f"{model_stem}.json"))
    with open(MODEL_DIR / f"{model_stem}.pkl", "rb") as f:
        meta = pickle.load(f)
    return model, meta["feature_cols"]


def load_data(features_csv: Path):
    df = pd.read_csv(features_csv, parse_dates=["date"])
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    test_df = df[df["date"] >= SPLIT_DATES["val_end"]].copy()
    X_test = test_df[feature_cols].astype(float)
    return X_test, test_df, feature_cols


def shap_global(model, X_test, feature_cols, out_dir: Path, top_n: int = 20):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.values)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    json_path = out_dir / "shap_binary_top_features.json"
    json_path.write_text(importance.head(top_n).to_json(orient="records", indent=2))
    print(f"[save] Top {top_n} features -> {json_path}")

    txt_path = out_dir / "shap_binary_top_features.txt"
    with open(txt_path, "w") as f:
        f.write("SHAP Global Feature Importance (Binary: 主不败 vs 客不败)\n")
        f.write("=" * 55 + "\n")
        for _, row in importance.iterrows():
            f.write(f"  {row['feature']:40s} {row['mean_abs_shap']:.6f}\n")
    print(f"[save] Full ranking -> {txt_path}")

    plt.figure(figsize=(10, 8))
    top = importance.head(top_n).iloc[::-1]
    plt.barh(range(len(top)), top["mean_abs_shap"].values, color="steelblue")
    plt.yticks(range(len(top)), top["feature"].values, fontsize=8)
    plt.xlabel("mean |SHAP value|", fontsize=10)
    plt.title(f"Top {top_n} Features by SHAP Importance (Binary)", fontsize=12)
    plt.tight_layout()
    plot_path = out_dir / "shap_binary_global_importance.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[save] Bar plot -> {plot_path}")

    plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_values, X_test.values, feature_names=feature_cols,
                      show=False, max_display=top_n, plot_size=(10, 10))
    plot_path2 = out_dir / "shap_binary_beeswarm.png"
    plt.tight_layout()
    plt.savefig(plot_path2, dpi=150)
    plt.close()
    print(f"[save] Beeswarm -> {plot_path2}")

    return importance


def shap_local(model, X_test, test_df, feature_cols, out_dir: Path, n_samples: int = 5):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.values)
    probs = model.predict(xgb.DMatrix(X_test))
    preds = (probs >= 0.5).astype(int)

    rng = np.random.RandomState(42)
    samples = rng.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)

    out_lines = ["Local SHAP Explanations (per-match)\n", "=" * 60 + "\n"]

    for idx in samples:
        row = test_df.iloc[idx]
        pred_label = "客不败" if preds[idx] == 1 else "主不败"
        prob_away = probs[idx]
        actual = row.get("result", "?")

        out_lines.append(f"\nMatch: {row.get('date', '?')} | {row.get('home_team','?')} vs {row.get('away_team','?')}\n")
        out_lines.append(f"  Predicted: {pred_label} (客不败_prob={prob_away:.3f}) | Actual: {actual}\n")
        out_lines.append(f"  Top 3 driving features:\n")

        sv = shap_values[idx]
        top_idx = np.argsort(np.abs(sv))[::-1][:3]
        for i in top_idx:
            direction = " pushes →" if sv[i] > 0 else " pushes ←"
            out_lines.append(f"    {feature_cols[i]:35s} SHAP={sv[i]:+.4f}{direction}\n")

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[idx],
            base_values=float(explainer.expected_value),
            data=X_test.iloc[idx].values,
            feature_names=feature_cols,
        ), show=False, max_display=10)
        plt.tight_layout()
        safe_name = f"{row.get('home_team','H')}_vs_{row.get('away_team','A')}_{idx}".replace(" ", "_").replace("/", "_")
        plot_path = out_dir / f"shap_binary_waterfall_{safe_name}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        out_lines.append(f"  [plot] {plot_path}\n")

    local_path = out_dir / "shap_binary_local_explanations.txt"
    local_path.write_text("".join(out_lines))
    print(f"[save] Local explanations -> {local_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgb_binary_tuned")
    parser.add_argument("--features", default=PROC_DIR / "features_multi_league_v2.csv")
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--local-samples", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("SHAP Binary Explainability Analysis")
    print("=" * 60)

    model, feature_cols = load_model(args.model)
    X_test, test_df, _ = load_data(args.features)
    print(f"[data] Test set: {len(X_test)} matches, {len(feature_cols)} features")

    print("\n--- Global Feature Importance ---")
    importance = shap_global(model, X_test, feature_cols, Path(args.out), args.top_n)

    print(f"\nTop 10 features by SHAP:")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']:40s} {row['mean_abs_shap']:.6f}")

    print("\n--- Local Explanations ---")
    shap_local(model, X_test, test_df, feature_cols, Path(args.out), args.local_samples)

    print("\n[done] All SHAP outputs saved to", args.out)


if __name__ == "__main__":
    main()
