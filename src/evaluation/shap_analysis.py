#!/usr/bin/env python3
"""
SHAP explainability analysis for the trained XGBoost model.
Outputs:
  - Global feature importance plot
  - Top N features list (text + JSON)
  - Per-sample explanations for a few test matches
"""

import argparse
import json
import pickle
import sys
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

SPLIT_DATES = {
    "train_end": "2024-08-01",
    "val_end": "2025-01-15",
}
DROP_COLS = ["date", "season", "matchweek", "home_team", "away_team",
             "home_goals", "away_goals", "home_xg", "away_xg", "league",
             "result"]
CLASS_NAMES = ["H (Home Win)", "D (Draw)", "A (Away Win)"]


def load_model(model_stem: str):
    model_path = MODEL_DIR / f"{model_stem}.json"
    meta_path = MODEL_DIR / f"{model_stem}.pkl"
    model = xgb.Booster()
    model.load_model(str(model_path))
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return model, meta["label_encoder"], meta["feature_cols"]


def load_data(features_csv: Path):
    df = pd.read_csv(features_csv, parse_dates=["date"])
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    test_mask = df["date"] >= SPLIT_DATES["val_end"]
    test_df = df[test_mask].copy()
    X_test = test_df[feature_cols].astype(float)
    return X_test, test_df, feature_cols


def shap_global(model, X_test, feature_cols, out_dir: Path, top_n: int = 20):
    """Global SHAP summary: which features drive predictions overall."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.values)

    # shap_values shape: (n_samples, n_features, n_classes=3)
    # Average absolute SHAP across all classes
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))  # avg over samples & classes

    importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    # Save top features JSON
    top_features = importance.head(top_n)
    json_path = out_dir / "shap_top_features.json"
    json_path.write_text(top_features.to_json(orient="records", indent=2))
    print(f"[save] Top {top_n} features -> {json_path}")

    # Text report
    txt_path = out_dir / "shap_top_features.txt"
    with open(txt_path, "w") as f:
        f.write("SHAP Global Feature Importance (mean |SHAP| across H/D/A)\n")
        f.write("=" * 55 + "\n")
        for _, row in importance.iterrows():
            f.write(f"  {row['feature']:40s} {row['mean_abs_shap']:.6f}\n")
    print(f"[save] Full ranking -> {txt_path}")

    # Plot: bar chart of top N
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_features["mean_abs_shap"].values[::-1], color="steelblue")
    plt.yticks(y_pos, top_features["feature"].values[::-1], fontsize=8)
    plt.xlabel("mean |SHAP value|", fontsize=10)
    plt.title(f"Top {top_n} Features by SHAP Importance", fontsize=12)
    plt.tight_layout()
    plot_path = out_dir / "shap_global_importance.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[save] Bar plot -> {plot_path}")

    # Plot: beeswarm for Home Win (class 0)
    plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_values[:, :, 0], X_test.values, feature_names=feature_cols,
                      show=False, max_display=top_n, plot_size=(10, 10))
    plot_path2 = out_dir / "shap_beeswarm_H.png"
    plt.tight_layout()
    plt.savefig(plot_path2, dpi=150)
    plt.close()
    print(f"[save] Beeswarm (Home Win) -> {plot_path2}")

    # Plot: beeswarm for Draw (class 1)
    plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_values[:, :, 1], X_test.values, feature_names=feature_cols,
                      show=False, max_display=top_n, plot_size=(10, 10))
    plot_path3 = out_dir / "shap_beeswarm_D.png"
    plt.tight_layout()
    plt.savefig(plot_path3, dpi=150)
    plt.close()
    print(f"[save] Beeswarm (Draw) -> {plot_path3}")

    return importance


def shap_local(model, X_test, test_df, feature_cols, out_dir: Path, n_samples: int = 5):
    """Per-sample explanations: why did the model predict X for this match?"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.values)
    probs = model.predict(xgb.DMatrix(X_test))
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    # Sample: high-conf of each class + random
    samples = []
    for target_class in [0, 1, 2]:
        mask = preds == target_class
        if mask.sum() > 0:
            idx = np.where(mask)[0][np.argmax(confidences[mask])]
            samples.append(idx)
    rng = np.random.RandomState(42)
    extra = rng.choice([i for i in range(len(X_test)) if i not in samples],
                       size=min(2, len(X_test)-len(samples)), replace=False)
    samples.extend(extra)

    out_lines = []
    out_lines.append("Local SHAP Explanations (per-match)\n")
    out_lines.append("=" * 60 + "\n")

    for idx in samples[:n_samples]:
        row = test_df.iloc[idx]
        pred_label = ["H", "D", "A"][preds[idx]]
        conf = confidences[idx]
        actual = row.get("result", "?")

        out_lines.append(f"\nMatch: {row.get('date', '?')} | {row.get('home_team','?')} vs {row.get('away_team','?')}\n")
        out_lines.append(f"  Predicted: {pred_label} (conf={conf:.3f}) | Actual: {actual}\n")
        out_lines.append(f"  P_H={probs[idx][0]:.3f} P_D={probs[idx][1]:.3f} P_A={probs[idx][2]:.3f}\n")
        out_lines.append(f"  Top 3 driving features:\n")

        sv = shap_values[idx, :, preds[idx]]
        top_idx = np.argsort(np.abs(sv))[::-1][:3]
        for i in top_idx:
            direction = " pushes →" if sv[i] > 0 else " pushes ←"
            out_lines.append(f"    {feature_cols[i]:35s} SHAP={sv[i]:+.4f}{direction}\n")

        # Waterfall plot
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[idx, :, preds[idx]],
            base_values=float(explainer.expected_value[preds[idx]]),
            data=X_test.iloc[idx].values,
            feature_names=feature_cols,
        ), show=False, max_display=10)
        plt.tight_layout()
        safe_name = f"{row.get('home_team','H')}_vs_{row.get('away_team','A')}_{idx}".replace(" ", "_").replace("/", "_")
        plot_path = out_dir / f"shap_waterfall_{safe_name}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        out_lines.append(f"  [plot] {plot_path}\n")

    txt_path = out_dir / "shap_local_explanations.txt"
    with open(txt_path, "w") as f:
        f.writelines(out_lines)
    print(f"[save] Local explanations -> {txt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgb_multi_league_v1")
    parser.add_argument("--features", default=PROC_DIR / "features_multi_league_v1.csv")
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--local-samples", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("SHAP Explainability Analysis")
    print("=" * 60)

    model, le, _ = load_model(args.model)
    X_test, test_df, feature_cols = load_data(args.features)
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
