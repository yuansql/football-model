# football-model → v17 集成指南

> 版本: v0.1.0 | 日期: 2026-09-07
> 模型: `xgb_multi_league_v1` (五大联赛 · 5330场)

## 1. 集成定位

`football-model` 的 XGBoost 输出 **替代/增强** v17 的泊松手算 `p_model`，**不替代**情报层、反剧本收据、结构闸、RMA 复盘。

```text
输入: 对阵 + 联赛
  ↓
football-model (P_H, P_D, P_A)
  ↓
v17 情报层叠加 (伤停/战意/反剧本/结构闸)
  ↓
Edge 计算 → 出票判断
```

## 2. 快速使用

```bash
cd /Users/wumm/学习/AQQ/football-model

# 输出 v17 格式的 p_model
.venv/bin/python src/models/predict_v17_bridge.py \
  --league "ENG-Premier League" \
  --home "Arsenal" \
  --away "Chelsea" \
  --date "2025-03-16"

# JSON 格式（供脚本消费）
.venv/bin/python src/models/predict_v17_bridge.py \
  --league "ENG-Premier League" \
  --home "Arsenal" \
  --away "Chelsea" \
  --json
```

## 3. 输出字段映射

| v17 字段 | football-model 来源 | 说明 |
|---|---|---|
| `p_model_H` | `P(H)` | 主胜概率 |
| `p_model_D` | `P(D)` | 平局概率 |
| `p_model_A` | `P(A)` | 客胜概率 |
| `model_direction` | `argmax(P_H, P_D, P_A)` | 模型预测方向 |
| `model_confidence` | `max(P_H, P_D, P_A)` | 置信度 |
| `model_conf_label` | high/medium/low | 分级标签 |

### p_model 取值规则（与 v17 方向对齐）

```text
方向=主胜   → p_model = p_model_H
方向=平局   → p_model = p_model_D
方向=客胜   → p_model = p_model_A
方向=主不败 → 不用于单选 Edge（v17 规则）
方向=客不败 → 不用于单选 Edge（v17 规则）
```

## 4. 与 v17 `p_model手算.txt` 的替换说明

### 原流程（泊松）
```text
λ_H, λ_A → 泊松表 0~5 → P_H/P_D/P_A → p_model
```

### 新流程（模型）
```text
模型直接输出 P_H/P_D/P_A → p_model
（跳过 λ 和泊松表，保留 Edge 计算和后续闸门）
```

### 在 v17 硬闸中的位置

在 `【硬闸自检】` 一节中，原：
```text
λ_H=· λ_A=· → λ'_H=· λ'_A=·（RAW 必填；npxG 可写 λ'=λ）
```

替换为（当使用模型时）：
```text
model_p_H=· model_p_D=· model_p_A=·  model_direction=·  model_conf=·
p_model = 按方向取对应项（见上表）
```

或简写：
```text
p_model来源=football-model_v0.1 | p_model=· | model_conf=·
```

## 5. 置信度使用建议

| 置信度 | 标签 | v17 建议 |
|---|---|---|
| ≥ 0.60 | high | 模型信号强，研究星级保底 ★★★☆☆ |
| 0.50 ~ 0.59 | medium | 正常参考，情报层权重提高 |
| < 0.50 | low | 模型犹豫，优先听情报层/反剧本 |

**禁止**：模型 confidence < 0.45 却硬冲 TOP1；此时应降权或观望。

## 6. 数据更新

```bash
# 每周赛后更新数据
.venv/bin/python src/data/fetch_understat.py
.venv/bin/python src/features/build_rolling_features.py

# 月度重训练
.venv/bin/python src/models/train_xgboost.py
```

## 7. 限制与已知问题

1. **无赔率特征**：football-data.co.uk 503，当前模型不含赔率隐含概率
2. **无实时数据**：只能预测历史库中已有的比赛（赛前快照需要滚动窗口）
3. **平局弱**：precision 仅 0.31，平局预测可靠性低
4. **德甲/法甲样本少**：德甲 306场/季，法甲 306~380场/季，模型对这两联赛的把握弱于英西意

## 8. 文件清单

```
football-model/
├── src/models/predict_v17_bridge.py   # ← 本指南对应的脚本
├── src/models/predict.py              # 通用预测脚本
├── src/models/train_xgboost.py        # 训练脚本
├── models/xgb_multi_league_v1.json    # 模型权重
├── models/xgb_multi_league_v1.pkl     # 元数据
└── docs/v17-integration.md            # ← 本文件
```
