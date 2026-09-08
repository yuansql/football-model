#!/usr/bin/env python3
"""
Weekly data update pipeline.
Fetches latest Understat data for all leagues and rebuilds features.

Usage (via cron or manual):
    cd /Users/wumm/学习/AQQ/football-model
    .venv/bin/python3 src/data/update_weekly.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "logs" / "update_weekly.log"


def run(cmd: list[str], cwd: Path = ROOT) -> tuple[int, str, str]:
    print(f"[run] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return result.returncode, result.stdout, result.stderr


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_lines = [f"=== Weekly Update {datetime.now().isoformat()} ===\n"]

    # Step 1: Fetch raw data
    log_lines.append("\n[1/3] Fetching latest Understat data...\n")
    rc, out, err = run([".venv/bin/python3", "src/data/fetch_understat.py"])
    log_lines.append(out)
    if err:
        log_lines.append(f"[stderr] {err}\n")
    if rc != 0:
        log_lines.append("[ERROR] Data fetch failed!\n")
        LOG_PATH.write_text("".join(log_lines))
        sys.exit(1)

    # Step 2: Validate
    log_lines.append("\n[2/3] Validating data...\n")
    rc, out, err = run([".venv/bin/python3", "src/data/validate.py"])
    log_lines.append(out)
    if err:
        log_lines.append(f"[stderr] {err}\n")
    if rc != 0:
        log_lines.append("[ERROR] Data validation failed!\n")
        LOG_PATH.write_text("".join(log_lines))
        sys.exit(1)

    # Step 3: Rebuild features
    log_lines.append("\n[3/3] Rebuilding features...\n")
    rc, out, err = run([".venv/bin/python3", "src/features/build_rolling_features.py"])
    log_lines.append(out)
    if err:
        log_lines.append(f"[stderr] {err}\n")
    if rc != 0:
        log_lines.append("[ERROR] Feature build failed!\n")
        LOG_PATH.write_text("".join(log_lines))
        sys.exit(1)

    log_lines.append("\n[done] Weekly update completed successfully.\n")
    LOG_PATH.write_text("".join(log_lines))
    print("[done] Weekly update completed. Log:", LOG_PATH)


if __name__ == "__main__":
    main()
