# football-model × v17 实战运行手册

> 版本: 2026-09-08 | 模型: `xgb_binary_tuned` | 适用联赛: 五大联赛

---

## 1. 每周工作流程（赛季中）

```
周一 06:00  ┌─ 自动：数据更新 ─┐
            │  update_weekly.py │
            └────────┬──────────┘
                     ▼
            ┌─ 自动：批量标记 ─┐
            │ batch_decision_mark.py --days 7 │
            └────────┬──────────┘
                     ▼
            ┌─ 分析师：读报告 ─┐
            │ reports/weekly_YYYYMMDD.md      │
            └────────┬──────────┘
                     ▼
            ┌─ 分析师：情报补充 ─┐
            │ 伤停 / 战意 / 盘口变化         │
            └────────┬──────────┘
                     ▼
            ┌─ 分析师：输出 v17 报告 ─┐
            │ v17_full_report.py      │
            └─────────────────────────┘
```

### 1.1 一键生成周报

```bash
cd /Users/wumm/学习/AQQ/football-model

# 自动拉取未来 7 天赛程
.venv/bin/python3 src/models/batch_decision_mark.py \
  --days 7 \
  --out reports/weekly_$(date +%Y%m%d).md

# 或：手动输入赛程（休赛期/数据源异常时）
cat > /tmp/fixtures.csv << 'EOF'
league,date,home,away
ENG-Premier League,2026-09-15,Arsenal,Chelsea
GER-Bundesliga,2026-09-16,Bayern Munich,Borussia Dortmund
EOF

.venv/bin/python3 src/models/batch_decision_mark.py \
  --fixtures-csv /tmp/fixtures.csv \
  --out reports/manual_$(date +%Y%m%d).md
```

### 1.2 单场快速查询

```bash
.venv/bin/python3 src/models/decision_marker.py \
  --league "ENG-Premier League" \
  --home "Arsenal" --away "Chelsea"
```

---

## 2. 标记速查表

| 标记 | 概率范围 | v17 操作建议 | 历史参考 |
|---|---|---|---|
| 🟢 **CONFIDENT_HOME** | 主不败 ≥ 80% | **模型主导**，情报层只验证利空 | 准确率 ~85% |
| 🟡 **LEAN_HOME** | 主不败 65~80% | **模型参考**，情报正常权重 | 准确率 ~78% |
| ⚪ **TOSS_UP** | 双方 40~60% | **情报主导**，模型仅辅助 | 准确率 ~55% |
| 🟠 **LEAN_AWAY** | 客不败 65~80% | **情报主导**，模型弱信号 | 客不败准确率 ~22% ⚠️ |
| 🔴 **CONFIDENT_AWAY** | 客不败 ≥ 80% | **罕见！逐条验证反剧本** | 样本极少，高波动 |
| ⚫ **INSUFFICIENT_DATA** | — | **跳过模型**，纯情报分析 | — |

### 关键纪律

- **🟠 LEAN_AWAY / 🔴 CONFIDENT_AWAY**：历史客不败预测命中率极低，必须有 **强情报支持**（核心伤停/战意/主场疲态）才可考虑
- **⚪ TOSS_UP**：模型无方向，此时情报层权重 100%
- 绝不单独凭模型下注，必须叠加 v17 情报层

---

## 3. 模型输出 → v17 报告对接

### 3.1 获取概率值

```bash
# JSON 输出，直接解析 P_主不败 / P_客不败
.venv/bin/python3 src/models/decision_marker.py \
  --league "ENG-Premier League" \
  --home "Arsenal" --away "Chelsea" \
  --json
```

输出示例：
```json
{
  "match": {"league": "ENG-Premier League", "home": "Arsenal", "away": "Chelsea"},
  "tag": "LEAN_HOME",
  "model_probability": {"主不败": 0.790, "客不败": 0.210},
  "v17_action": "模型参考，情报平衡"
}
```

### 3.2 填入 v17 报告

在 `v17_full_report.py` 生成的报告中，模型已预填以下字段：

| v17 字段 | 来源 |
|---|---|
| `p_model` | 模型输出的主不败/客不败概率 |
| `基本面` | 排名、近5积分、xG、H2H |
| `模型标记` | CONFIDENT_HOME / LEAN_HOME / ... |

分析师只需补充：
- 伤停情报
- 盘口解读
- 战意/赛程因素
- 最终方向与信心度

---

## 4. 数据更新与模型维护

### 4.1 周度更新（ cron 已配置）

```cron
# 每周一 6:00
0 6 * * 1 cd /Users/wumm/学习/AQQ/football-model && .venv/bin/python3 src/data/update_weekly.py
```

更新内容：
- 拉取上周末赛果
- 验证数据完整性
- 重建滚动特征

### 4.2 月度重训练（ cron 已配置）

```cron
# 每月 1 日 7:00
0 7 1 * * cd /Users/wumm/学习/AQQ/football-model && .venv/bin/python3 src/models/retrain.py
```

重训练流程：
1. 归档旧模型到 `models/archive/`
2. 更新全部历史数据
3. 重新训练 `xgb_binary_tuned`
4. 输出评估报告

### 4.3 手动触发重训练

```bash
# 紧急重训练（如重大数据修复后）
.venv/bin/python3 src/models/train_binary.py \
  --input data/processed/features_multi_league_v2.csv \
  --output models/xgb_binary_tuned
```

---

## 5. 已知限制与风险

| 限制 | 影响 | 应对 |
|---|---|---|
| 无赔率特征 | 缺少市场信号，模型独立盈利不可行 | 必须叠加 v17 情报层 |
| 平局弱 | 三分类模型平局 precision 仅 0.31 | 已转二分类（主不败 vs 客不败） |
| 客不败弱 | 二分类客不败 F1 仅 0.60 | 客不败标记须强情报支持 |
| 休赛期无赛程 | Understat 不发布未来比赛 | 手动 CSV 输入或等赛季开打 |
| 扩联赛受阻 | 葡超/荷甲无免费高质量 xG 数据 | 暂限五大联赛 |
| 模型过拟合 | Train AUC 0.94 vs Test AUC 0.78 | 控制训练窗口，月度重训练 |

---

## 6. 决策流程图

```
              ┌─────────────┐
              │ 收到比赛列表 │
              └──────┬──────┘
                     ▼
         ┌─────────────────────┐
         │ batch_decision_mark  │
         │ 生成标记 + 概率       │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
    ┌────┤    看标记颜色        ├────┐
    │    └─────────────────────┘    │
    ▼                                 ▼
🟢 CONFIDENT_HOME              🟠/🔴 客不败方向
├─ 模型主导                     ├─ 情报主导 100%
├─ 确认无重大利空               ├─ 逐条验证：
└─ 采用模型方向                 │   ① 客队战意
                                │   ② 主队核心伤停
⚪ TOSS_UP                      │   ③ 赛程疲劳
├─ 情报主导                     │   ④ H2H 客场优势
└─ 模型仅供参考                 └─ 有强证据才考虑

    情报层补充：伤停 / 盘口 / 战意 / 天气
                    ▼
            ┌─────────────┐
            │  输出 v17   │
            │  完整报告   │
            └─────────────┘
```

---

## 7. 紧急联系

| 场景 | 操作 |
|---|---|
| 脚本报错 | 检查 `.venv` 激活、依赖安装、`soccerdata` 缓存 |
| Understat 拉取失败 | 检查网络、可能是休赛期无数据 |
| 模型标记全部一样 | 检查 `models/xgb_binary_tuned.json` 是否存在 |
| 需要扩联赛 | 暂不支持，等数据源 |

---

*本手册随模型迭代更新。当前模型版本: `xgb_binary_tuned` (Optuna 100 trials, Test AUC=0.7846)*
