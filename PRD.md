# Plan: football-model 数据驱动预测模型建设

- 日期: 2026-09-07
- 来源需求: "在 football-model 目录下训练一个足球模型"

## 1. 一句话目标

在 `/Users/wumm/学习/AQQ/football-model/` 下建设一条**可复现、可评估**的数据驱动预测 pipeline：先从历史赛果+赔率数据构建结构化数据集，再用轻量 ML 模型（XGBoost/LightGBM）预测胜平负方向概率；模型输出作为 `v17` 规则框架的**概率输入层**（替代/增强手算泊松 p_model），而非取代整个情报分析体系。MVP 目标：单联赛（如英超）方向预测准确率显著优于随机基线（≥45%），且模型置信度与真实胜率呈正相关。

## 2. 用户与场景

- **谁**：用户（AQQ）已有成熟的规则驱动预测框架 `football-predict-v17`，具备情报分析、反剧本收据、结构闸、RMA 复盘等复杂规则体系，用于体彩/竞彩推荐。
- **为什么现在要**：v17 框架的 p_model 目前依赖手算泊松（λ→比分表→P_H/D/A），这是一个强假设的生成式模型。用户希望引入数据驱动的判别式 ML 模型，用历史数据学习特征-结果映射，可能提升以下环节：
  1. 替代手算泊松，提供更准确的基准胜率；
  2. 为情报层无法覆盖的场次提供" fallback "方向概率；
  3. 用模型置信度辅助 v17 的【研究星级】评定；
  4. 长期来看，用模型输出反哺规则更新（如三桶补偿、结构闸阈值校准）。
- **场景**：每日竞彩场次分析前，模型先跑一轮给出 P(主胜)/P(平)/P(客胜)，v17 情报层在此基础上叠加修正，最终由 Edge/出票闸决定是否介入。

## 3. 范围内 / 明确不做

**范围内（MVP 阶段）：**
1. 单联赛 MVP：先以英超（EPL）为试验田，跑通数据→特征→模型→评估全链路。
2. 预测目标：胜平负（1X2）三分类概率 —— 与 v17 的 p_model 直接对齐。
3. 数据源：Understat（xG/npxG，五大联赛）、FBref（历史赛果+统计）、OddsPortal/体彩 SP（赔率）。
4. 特征工程：球队近 N 场表现、xG 差、主客场形态、H2H、赔率隐含概率、联赛排名差。
5. 模型：XGBoost 或 LightGBM（表格数据 SOTA，可解释，轻量）。
6. 评估：准确率、LogLoss、ROC-AUC（OvR）、Calibration、ROI 回测。
7. 与 v17 集成：输出 JSON/CSV 格式的 P_H/P_D/P_A，可被 v17 的 `p_model` 字段消费。

**明确不做（至少前 3 个月）：**
1. **不做比分预测模型** —— v17 的比分篮由情景路径+三桶补偿生成，ML 比分预测需要 36+ 分类且样本极稀疏，ROI 性价比极低。
2. **不做进球数/大小球模型** —— 降维通道的让球/进球 Edge 仍用泊松手算，ML 替换优先级低于 1X2。
3. **不做深度学习/神经网络** —— 数据量（单联赛每季 380 场）不足以支撑 NN；XGBoost 是更务实的起点。
4. **不做实时数据流/自动投注** —— 仅做赛前静态预测，不碰任何自动下单或资金管理。
5. **不替代 v17 情报层** —— 模型只提供"数据面"概率，情报/伤停/战意/反剧本收据仍由 v17 规则层处理。
6. **不碰多联赛统一模型（至少 MVP 不做）** —— 不同联赛风格差异大，Tier1/Tier2/Tier3 需分别建模，MVP 只锁定英超。

## 4. 主路径

用户（或自动化脚本）的完整使用流程：

1. **数据采集**（可手动或定时脚本）
   - 从 Understat 拉取英超近 3 个赛季逐场 xG、npxG、赛果。
   - 从 FBref/SofaScore 拉取逐场统计（射门、射正、控球、PPDA 等）。
   - 从体彩官网/OddsPortal 收集赛前 SP（胜平负、让球、大小球）。
2. **数据清洗与结构化**
   - 统一球队名映射（不同源队名不一致）。
   - 生成每场记录的 "赛前快照"：两队近 5/10 场滚动统计、H2H、排名差、xG 差。
   - 标注结果：H/D/A（主胜/平/客胜）。
3. **特征工程**
   - 基础特征：主客场近 5 场 xG 均值、xGA 均值、实际进球均值、积分排名差。
   - 进阶特征：xG 差趋势（近 5 vs 近 10）、主客场火力比、H2H 近 5 场主胜率、赔率隐含概率。
   - 时间特征：赛季轮次、周中/周末、是否为德比。
4. **模型训练**
   - 按时间切分：训练集 = 前 2.5 季，验证集 = 中间 0.5 季，测试集 = 最后 0.5 季（严禁随机切分，防止数据泄漏）。
   - 三分类 XGBoost（`objective=multi:softprob`）。
   - 超参搜索：Optuna 或网格搜索（learning_rate, max_depth, subsample, colsample_bytree）。
5. **评估与校准**
   - 测试集上计算 Accuracy、LogLoss、Top-1 Calibration（分 bin 看预测概率 vs 实际频率）。
   - 若校准差 → 用 Platt Scaling 或 Isotonic Regression 校准。
   - ROI 回测：用模型最高置信度场次的模拟投注，按 v17 的 Edge 逻辑计算假设 ROI。
6. **与 v17 集成**
   - 模型输出 JSON：`{"match": "A vs B", "P_H": 0.42, "P_D": 0.28, "P_A": 0.30, "confidence": "medium", "model_version": "v0.1.0"}`
   - v17 的 `p_model手算.txt` 增加分支："若有 model 输出 → 优先采用 model P_H/D/A 作为 p_model；无则 fallback 泊松手算"。
   - 研究星级校准：模型置信度 ≥0.55 的方向 → 研究星级保底 ★★★☆☆。
7. **持续迭代**
   - 每周赛后补充数据，月度重训练。
   - 用 v17 的 RMA 复盘结果（方向 hit/miss）作为模型反馈，标记错误样本分析特征缺口。

## 5. 验收标准

1. **数据层**：英超近 3 个赛季（约 1140 场）结构化数据文件存在，字段完整率 ≥95%，球队名统一映射表存在。
2. **特征工程**：每场记录 ≥20 个有效特征，无 NaN 泄漏（赛前快照不得包含赛后数据）。
3. **模型性能**：测试集（按时间切分的最后 1/6 数据）上，三分类准确率 ≥45%（随机基线 33%），LogLoss ≤ 1.10。
4. **校准质量**：预测概率分 5 个 bin（0.2–0.4, 0.4–0.5, 0.5–0.6, 0.6–0.8, 0.8+），每个 bin 内实际频率与预测概率的绝对偏差 ≤0.08。
5. **集成可用**：模型输出 JSON 能被 v17 的 `p_model` 字段消费，至少完成 1 场端到端演示（模型预测 → v17 情报叠加 → 出票判断）。

```json
[
  {"cmd": "python -m py_compile football-model/src/features.py", "expect": "exit 0", "onMissing": "BLOCKED"},
  {"cmd": "python football-model/src/evaluate.py --league EPL --seasons 3", "expect": "Accuracy >= 0.45", "onMissing": "BLOCKED"},
  {"cmd": "python football-model/src/evaluate.py --check-calibration", "expect": "max_bin_error <= 0.08", "onMissing": "BLOCKED"},
  {"cmd": "python football-model/src/predict.py --match 'Arsenal vs Chelsea' --output json", "expect": "P_H + P_D + P_A ≈ 1.0", "onMissing": "BLOCKED"},
  {"cmd": "ls football-model/data/processed/epl_rolling_features.csv", "expect": "file exists", "onMissing": "BLOCKED"}
]
```

## 6. 分叉

**分叉 A：模型目标（三分类 vs 二分类）**
- **选项 1（默认 · 推荐）**：三分类（主胜/平/客胜）。与 v17 p_model 完全对齐，但平局类天然不平衡（约 25%），模型容易欠拟合平局。
- **选项 2（备选）**：二分类（主不败 vs 客不败）。避开平局难题，与 v17 的"方向原子"（主不败/客不败）更一致，但丢失精确胜平负概率，Edge 计算会损失精度。
- **倾向**：选选项 1（三分类），因为 v17 的 p_model 需要 P_H/P_D/P_A 三项来算 Edge；平局可通过 class_weight 或 focal loss 缓解不平衡。

**分叉 B：特征核心（xG 主导 vs 赔率主导）**
- **选项 1（默认 · 推荐）**：xG + 球队统计主导，赔率仅作辅助特征。与 v17 "情报优先"哲学一致，模型提供"基本面"概率，Edge 由赔率层计算。
- **选项 2（备选）**：赔率隐含概率 + 市场信号主导。直接用 P_fair 作强特征，模型学的是"市场偏差"，类似传统价值投注。但依赖稳定赔率数据源，且与 v17 情报层因果冲突。
- **倾向**：选选项 1，模型定位是"基本面概率引擎"，不是"市场套利引擎"。

**分叉 C：数据获取方式（爬虫自建 vs 第三方库）**
- **选项 1（默认 · 推荐）**：用现有 Python 库 `understat` / `fbref-api` / `soccerdata` 拉取，本地存 CSV/Parquet。开发快、维护成本可控。
- **选项 2（备选）**：自建爬虫抓 Understat/FBref。灵活但维护成本高，且 Understat robots.txt Disallow:/，有合规风险。
- **倾向**：选选项 1，优先用现成库；若库不可用再评估自建爬虫。

**分叉 D：模型类型（XGBoost vs 泊松回归 vs 其他）**
- **选项 1（默认 · 推荐）**：XGBoost/LightGBM。表格数据 SOTA，训练快，可解释（SHAP），不需要大量数据。
- **选项 2（备选）**：Dixon-Coles 泊松回归（统计足球建模经典）。与 v17 现有泊松体系兼容好，但假设强（进球独立、泊松分布），对情报信号无法融合。
- **选项 3（备选）**：逻辑回归 + 手动特征交叉。可解释性最强，但表达能力弱，大概率打不过 XGBoost。
- **倾向**：选选项 1，XGBoost 是平衡性能与可解释性的最优解。

## 7. 执行步骤

### 阶段 1：数据基建（预计 2–3 天）
1. **创建目录结构**
   ```
   football-model/
   ├── data/
   │   ├── raw/              # 原始爬取/下载数据
   │   ├── processed/        # 清洗后的结构化数据
   │   └── external/         # 手动补充（如体彩 SP 截图）
   ├── src/
   │   ├── data/             # 数据采集脚本
   │   ├── features/         # 特征工程
   │   ├── models/           # 训练与预测
   │   └── evaluation/       # 评估与回测
   ├── notebooks/            # 探索性分析（EDA）
   ├── config/               # 联赛配置、球队映射
   ├── tests/                # 单元测试
   └── requirements.txt
   ```
2. **安装依赖**：`pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `requests`, `beautifulsoup4` / `soccerdata`，写入 `requirements.txt`。
3. **数据采集脚本**：
   - `src/data/fetch_understat.py`：拉取英超逐场 xG、赛果（近 3 季）。
   - `src/data/fetch_fbref.py`：拉取逐场统计（射门、控球、PPDA 等）。
   - `src/data/fetch_odds.py`：拉取赛前 SP（备用源：football-data.co.uk 免费历史赔率）。
4. **球队名统一映射表**：`config/team_name_mapping.json`（处理 "Manchester City" / "Man City" 等别名）。
5. **数据验证**：跑通 `python src/data/validate.py`，检查完整率、时间连续性、无重复场。

### 阶段 2：特征工程（预计 2–3 天）
6. **赛前快照生成器**：`src/features/build_rolling_features.py`
   - 输入：逐场原始数据（按时间排序）。
   - 输出：每场记录的赛前特征向量（不得泄漏赛后信息）。
   - 滚动窗口：近 3/5/10 场（主场/客场/不分主客分别计算）。
7. **特征列表（初版）**：
   - `home_xg_roll5_mean`, `away_xg_roll5_mean`
   - `home_xga_roll5_mean`, `away_xga_roll5_mean`
   - `home_goals_roll5_mean`, `away_goals_roll5_mean`
   - `home_points_roll5`, `away_points_roll5`
   - `table_rank_diff`（主队排名 − 客队排名，赛前）
   - `h2h_home_win_rate_last5`
   - `odds_implied_prob_h`, `odds_implied_prob_d`, `odds_implied_prob_a`
   - `matchweek`（赛季轮次）
8. **特征存储**：`data/processed/epl_features_v1.csv`（每场 1 行，含标签 H/D/A）。

### 阶段 3：模型训练（预计 2–3 天）
9. **训练脚本**：`src/models/train_xgboost.py`
   - 按时间切分：2022/23 + 2023/24 前半 → train；2023/24 后半 → val；2024/25 → test。
   - 超参搜索：Optuna，目标最小化 val LogLoss。
   - 类别不平衡：用 `scale_pos_weight` 或 `sample_weight`。
10. **模型保存**：`models/xgb_epl_v1.json`（XGBoost 原生格式）。
11. **可解释性**：输出 SHAP summary plot，识别 Top10 重要特征。

### 阶段 4：评估与校准（预计 1–2 天）
12. **评估脚本**：`src/evaluation/evaluate.py`
    - 输出 Accuracy、LogLoss、Confusion Matrix、Calibration plot。
    - 若校准差 → `src/evaluation/calibrate.py`（Platt/Isotonic）。
13. **ROI 回测脚本**：`src/evaluation/backtest_roi.py`
    - 模拟策略：只投注模型置信度最高的方向且 P ≥ 0.50 的场次。
    - 按 v17 Edge 逻辑：若模型 P − p_fair ≥ 1% → 模拟投注 1 unit。
    - 输出假设 ROI、胜率、最大回撤。

### 阶段 5：v17 集成（预计 1–2 天）
14. **预测服务脚本**：`src/models/predict.py`
    - 输入：主队名、客队名、日期。
    - 输出：JSON 格式的 P_H/P_D/P_A + 置信度标签。
15. **v17 集成文档**：在 v17 的 `p_model手算.txt` 中增加"模型输入"分支说明。
16. **端到端演示**：选 1 场近期英超，跑通 model → v17 情报叠加 → 出票判断。

### 阶段 6：持续迭代（长期）
17. **周度数据更新脚本**：`src/data/update_weekly.py`
18. **月度重训练脚本**：`src/models/retrain.py`
19. **错误分析流水线**：用 v17 RMA 复盘结果，自动标记模型 miss 场次，生成特征缺口报告。

## 8. 风险与依赖

**红线（执行必须 handoff 或用户确认）：**
1. **数据合规风险**：Understat robots.txt 为 `Disallow: /`，批量爬取存在法律/道德风险。**缓解**：优先用 `soccerdata` 等封装库（内部可能已处理合规边界）；若必须自建爬虫，限制请求频率（≥5 秒/次），只拉历史数据不拉实时。
2. **无历史数据**：当前 v17 的 `archive/data/prediction_log.jsonl` **仅有 2 条记录**，`edge_history.csv` 仅 384 字节，完全不足以训练任何模型。**缓解**：必须从外部源（Understat/FBref/football-data.co.uk）重新拉取历史数据；这是前置条件，无法跳过。

**BLOCKED 条件：**
1. **若 soccerdata/understat 库无法获取英超历史数据** → BLOCKED，需用户确认是否接受手动下载 CSV（如 football-data.co.uk 免费数据）。
2. **若拉取后发现近 3 季有效样本 < 500 场**（如大量缺失 xG 或赔率） → BLOCKED，需降级为更简单的特征集或换联赛。
3. **若测试集准确率 < 40%（连随机基线都打不过）** → BLOCKED，需重新审视特征工程或放弃该项目。
4. **若用户期望的是"用 AI 替代 v17 整个框架做全自动预测"** → BLOCKED，本 Plan 的范围仅限概率输入层，不替代情报层。

**其他风险：**
- **过拟合**：表格数据+树模型在小样本下易过拟合。**缓解**：严格时间切分、早停、正则化、交叉验证用 TimeSeriesSplit。
- **冷启动**：赛季初/新升级球队滚动特征窗口不足。**缓解**：用上赛季末数据填充、增加"样本不足"标记字段。
- **概念漂移**：足球风格、规则、球队实力随时间变化。**缓解**：月度重训练、只保留近 3 季数据、用验证集监控漂移。
- **小联赛扩展难**：英超有 Understat xG，但挪超/芬超等 Tier3 没有。**缓解**：MVP 只做英超；其他联赛需单独评估数据源可用性。
- **赔率数据缺失**：历史 SP 不易获取。**缓解**：用 football-data.co.uk 的 Pinnacle 终盘作为代理（更接近市场真实概率）。

## 9. 下一步

**等人点头后再执行。**

点头前需要用户确认以下**关键假设**（任一会改变 Plan 走向）：

1. **模型定位确认**：您是否接受"模型只做 1X2 概率输入层，不替代 v17 情报层"？还是您期望的是端到端全自动预测（输入对阵→输出完整 v17 格式报告）？
2. **联赛范围确认**：MVP 先做英超（数据最完整），还是您有偏好的其他联赛？
3. **数据获取方式**：您是否愿意使用第三方数据服务（如 football-data.co.uk 免费 CSV、或付费 API）？还是坚持只使用 v17 现有数据（目前仅 2 条记录，不足以训练）？

**点头后的第一句建议**：
按步骤 1 开工——先创建目录结构 + 安装依赖 + 跑通 `soccerdata` 拉取英超近 3 季数据，验证数据完整率后进入特征工程。做完用 `/test-engineer` 对照第 5 节验收标准跑第一轮数据检查。

---

## 附录：岗位反思（产品经理自检）

**未核验的假设：**
- 假设用户说的"足球模型"是指"数据驱动的 ML 判别模型"，而非 v17 已有的泊松生成模型。若用户其实只是想改进 p_model 手算流程，则本 Plan 过度工程化。
- 假设 Understat/sportsdata 库在 2026-09-07 仍可正常获取英超历史数据（未实际调用验证）。
- 假设用户对 ML 有基本认知（知道模型不是"稳赢"，只是概率）。若用户期望的是"训练出高准确率模型后靠它赚钱"，则需要重新设定预期。

**一条能推翻本 Plan 的反例：**
如果用户现有 v17 框架的预测准确率已经在 55% 以上（方向命中率），那么一个基于 1000+ 样本的 XGBoost 模型大概率打不过这个高度工程化的规则系统（因为情报层的伤停/战意等信号无法结构化进特征）。此时"训练模型"的投入产出比为负，建议改为"用模型辅助校准 v17 的泊松 p_model"而非独立预测。

**哪条缺失会让执行猜错：**
- 缺少"用户是否有现有历史投注/预测记录"——如果用户有私藏的大量 v17 预测历史（含方向命中结果），这些是最宝贵的标签数据，可替代外部数据源直接做模型。但当前只发现了 2 条记录，可能用户有未纳入仓库的数据。
