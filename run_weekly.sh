#!/bin/bash
# 周末联赛预测一键脚本
# Usage: ./run_weekly.sh

set -e

cd "$(dirname "$0")"
source .venv/bin/activate

DATE=$(date +%Y-%m-%d)
REPORT_DIR="reports/${DATE}_weekly"
mkdir -p "$REPORT_DIR"

echo "===== 足球模型周末预测 ====="
echo "日期: $DATE"
echo ""

# 1. 更新数据
echo "[1/3] 更新历史数据..."
python3 src/data/update_weekly.py || echo "数据更新部分失败，继续..."

# 2. 生成预测报告
echo "[2/3] 生成未来7天预测报告..."
python3 src/models/batch_decision_mark.py \
  --days 7 \
  --out "$REPORT_DIR/decision_report" \
  --json

# 3. 汇总
echo "[3/3] 完成！"
echo "报告已保存到: $REPORT_DIR/"
ls -la "$REPORT_DIR/"
