#!/usr/bin/env python3
"""
Monthly retraining pipeline.
Runs full data + feature + training cycle, archives old model.

Usage:
    cd /Users/wumm/学习/AQQ/football-model
    .venv/bin/python3 src/models/retrain.py
"""

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
LOG_PATH = ROOT / "logs" / "retrain.log"

def run(cmd, cwd=ROOT):
    print(f"[run] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return r.returncode, r.stdout, r.stderr

def archive_old_model():
    today = datetime.now().strftime("%Y%m%d")
    for stem in ["xgb_multi_league_v1"]:
        for ext in [".json", ".pkl"]:
            src = MODEL_DIR / f"{stem}{ext}"
            if src.exists():
                dst = MODEL_DIR / f"{stem}_{today}{ext}"
                shutil.copy2(src, dst)
                print(f"[archive] {src} -> {dst}")

def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = [f"=== Retrain {datetime.now().isoformat()} ===\n"]

    # 1. Update data
    log.append("\n[1/4] Updating data...\n")
    rc, out, err = run([".venv/bin/python3", "src/data/update_weekly.py"])
    log.extend([out, err or ""])
    if rc != 0:
        log.append("[ERROR] Data update failed\n")
        LOG_PATH.write_text("".join(log)); sys.exit(1)

    # 2. Archive
    log.append("\n[2/4] Archiving old model...\n")
    archive_old_model()

    # 3. Train
    log.append("\n[3/4] Training new model...\n")
    rc, out, err = run([".venv/bin/python3", "src/models/train_xgboost.py"])
    log.extend([out, err or ""])
    if rc != 0:
        log.append("[ERROR] Training failed\n")
        LOG_PATH.write_text("".join(log)); sys.exit(1)

    # 4. Evaluate
    log.append("\n[4/4] Running backtest...\n")
    rc, out, err = run([".venv/bin/python3", "src/evaluation/backtest_roi.py"])
    log.extend([out, err or ""])

    log.append("\n[done] Retrain completed.\n")
    LOG_PATH.write_text("".join(log))
    print("[done] Retrain completed. Log:", LOG_PATH)

if __name__ == "__main__":
    main()
