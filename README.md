# football-model

五大联赛（英超/西甲/意甲/德甲/法甲）足球比赛胜平负概率预测模型。
基于 XGBoost，从 Understat xG 数据学习，输出 P(主胜)/P(平)/P(客胜) 供 v17 规则框架消费。

> ⚠️ **定位**：概率输入层，替代手算泊松 `p_model`，**不替代**情报分析/反剧本/结构闸。

---

## 快速开始

```bash
cd /Users/wumm/学习/AQQ/football-model

# 1. 环境（已配好）
source .venv/bin/activate

# 2. 预测一场（实时模式：自动拉最新数据）
.venv/bin/python3 src/models/predict_live.py \
  --league "ENG-Premier League" \
  --home "Arsenal" --away "Chelsea" \
  --mode simulate

# 3. 生成完整 v17 报告
.venv/bin/python3 src/models/v17_full_report.py \
  --league "ENG-Premier League" \
  --home "Arsenal" --away "Chelsea"
```

---

## 功能一览

| 脚本 | 功能 |
|---|---|
| `src/data/fetch_understat.py` | 拉取 Understat 历史数据（五大联赛） |
| `src/data/validate.py` | 数据完整性验证 |
| `src/data/update_weekly.py` | **周度自动更新** |
| `src/features/build_rolling_features.py` | 赛前快照特征工程（无泄漏） |
| `src/models/train_xgboost.py` | XGBoost 三分类训练 |
| `src/models/retrain.py` | **月度自动重训练** |
| `src/models/predict.py` | 历史比赛查询预测 |
| `src/models/predict_live.py` | **实时/未来比赛预测** |
| `src/models/predict_v17_bridge.py` | v17 p_model 对接 |
| `src/models/v17_full_report.py` | **一键生成 v17 完整报告** |
| `src/models/decision_marker.py` | **可信度标记**（替代硬预测，输出情报权重建议） |
| `src/models/batch_decision_mark.py` | **批量标记**（未来一轮赛程，输出 Markdown 报告；支持 `--simulate` 模拟测试） |
| `src/evaluation/shap_analysis.py` | SHAP 可解释性分析 |
| `src/evaluation/backtest_roi.py` | ROI 回测模拟 |
| `src/evaluation/calibrate.py` | 概率校准（Platt/Isotonic） |

---

## 模型性能

**数据**：7081 场（22/23 ~ 25/26 赛季，五大联赛）

| 指标 | 值 |
|---|---|
| 测试集准确率 | **55.8%**（阈值 45%） |
| LogLoss | 0.930 |
| 平局 Precision | 0.31（弱项） |
| 主胜 Precision | 0.66 |
| 客胜 Precision | 0.58 |

**Top 5 特征**：`rank_diff` >> `home_rank` > `away_rank` > `home_points_roll10` > `away_xg_for_roll5`

---

## 目录结构

```text
football-model/
├── data/
│   ├── raw/                 # Understat 原始数据
│   └── processed/           # 赛前特征
├── models/                  # 模型权重 + 归档
├── reports/                 # SHAP 图 + 回测结果
├── logs/                    # 更新/训练日志
├── src/
│   ├── data/                # 数据采集
│   ├── features/            # 特征工程
│   ├── models/              # 训练/预测
│   └── evaluation/          # 评估/解释
├── config/
│   └── team_name_mapping.json
├── docs/
│   └── v17-integration.md   # v17 深度集成文档
├── requirements.txt
└── PRD.md                   # 产品需求文档
```

---

## 支持联赛

| 联赛 | Understat 代码 | 每季场次 |
|---|---|---|
| 英超 | `ENG-Premier League` | 380 |
| 西甲 | `ESP-La Liga` | 380 |
| 意甲 | `ITA-Serie A` | 380 |
| 德甲 | `GER-Bundesliga` | 306 |
| 法甲 | `FRA-Ligue 1` | 306~380 |

---

## 自动化

```bash
# 周度更新（赛后拉新数据）
.venv/bin/python3 src/data/update_weekly.py

# 月度重训练（归档旧模型 + 训练新模型 + 回测）
.venv/bin/python3 src/models/retrain.py
```

可配 cron：
```cron
# 每周一 6:00 更新数据
0 6 * * 1 cd /Users/wumm/学习/AQQ/football-model && .venv/bin/python3 src/data/update_weekly.py

# 每月 1 日 7:00 重训练
0 7 1 * * cd /Users/wumm/学习/AQQ/football-model && .venv/bin/python3 src/models/retrain.py
```

---

## 与 v17 集成

模型输出直接替代 `p_model手算.txt` 的泊松步骤：

```text
# 旧流程
λ_H, λ_A → 泊松表 → P_H/P_D/P_A → p_model

# 新流程
模型 → P_H/P_D/P_A → p_model
```

详细集成说明见 [`docs/v17-integration.md`](docs/v17-integration.md)。

### 每周分析师周报（批量标记）

```bash
# 自动拉取未来 7 天赛程并生成报告
.venv/bin/python3 src/models/batch_decision_mark.py --days 7 --out reports/weekly_$(date +%Y%m%d).md

# 手动输入赛程（休赛期/数据源不可用）
echo 'league,date,home,away
ENG-Premier League,2026-09-15,Arsenal,Chelsea
GER-Bundesliga,2026-09-16,Bayern Munich,Borussia Dortmund' > /tmp/fixtures.csv
.venv/bin/python3 src/models/batch_decision_mark.py --fixtures-csv /tmp/fixtures.csv --out reports/manual.md

# 模拟模式（用历史比赛测试报告格式）
.venv/bin/python3 src/models/batch_decision_mark.py --simulate --simulate-count 5 --leagues "ENG-Premier League"
```

---

## 已知限制

1. **无赔率特征**：football-data.co.uk 503，当前模型不含市场赔率
2. **实时预测需数据**：`simulate` 模式需要球队已有本赛季比赛记录（滚动窗口）
3. **平局弱**：precision 仅 0.31，平局预测可靠性低
4. **ROI 为负**：模型独立盈利不可行，必须叠加 v17 情报层

---

## 依赖

```text
soccerdata>=1.9.0
xgboost>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
shap>=0.42.0
matplotlib>=3.7.0
```

完整列表见 `requirements.txt`。

---

## License

仅供研究与学习；不构成投注建议。
