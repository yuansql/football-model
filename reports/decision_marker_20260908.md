# football-model 周报：未来 7 天 Decision Marker

生成时间: 2026-09-08 10:46

## 速查表

| 日期 | 联赛 | 主队 | 客队 | 标记 | 主不败 | 客不败 | 排名 | 近5积分 | v17 建议 |
|---|---|---|---|---|---|---|---|---|---|
| 2026-05-24 | EPL | Crystal Palace | Arsenal | 🟠 LEAN_AWAY | 0.223 | 0.777 | 11v2 | 0.4v2.4 | 情报主导，模型弱信号 |
| 2026-05-24 | EPL | Burnley | Wolverhampton Wanderers | 🟠 LEAN_AWAY | 0.241 | 0.759 | 20v14 | 0.2v0.4 | 情报主导，模型弱信号 |
| 2026-05-24 | EPL | Brighton | Manchester United | ⚪ TOSS_UP_AWAY | 0.413 | 0.587 | 9v4 | 1.4v2.6 | 情报主导，模型辅助 |
| 2026-05-24 | EPL | Fulham | Newcastle United | 🔴 CONFIDENT_AWAY | 0.144 | 0.856 | 15v7 | 1.0v1.4 | 情报逐条验证，模型罕见强信号 |
| 2026-05-24 | EPL | West Ham | Leeds | 🟢 CONFIDENT_HOME | 0.928 | 0.072 | 10v19 | 0.8v2.2 | 模型主导，情报验证 |
| 2026-05-16 | Bundesliga | Borussia M.Gladbach | Hoffenheim | ⚪ TOSS_UP_HOME | 0.524 | 0.476 | 12v9 | 1.0v2.2 | 情报主导，模型辅助 |
| 2026-05-16 | Bundesliga | Bayern Munich | FC Cologne | 🟢 CONFIDENT_HOME | 0.955 | 0.045 | 1v15 | 2.6v1.0 | 模型主导，情报验证 |
| 2026-05-16 | Bundesliga | Bayer Leverkusen | Hamburger SV | 🟢 CONFIDENT_HOME | 0.962 | 0.038 | 3v22 | 1.8v1.2 | 模型主导，情报验证 |
| 2026-05-16 | Bundesliga | Eintracht Frankfurt | VfB Stuttgart | 🟢 CONFIDENT_HOME | 0.864 | 0.136 | 5v7 | 0.8v1.6 | 模型主导，情报验证 |
| 2026-05-16 | Bundesliga | Werder Bremen | Borussia Dortmund | 🟠 LEAN_AWAY | 0.244 | 0.756 | 14v2 | 0.8v1.2 | 情报主导，模型弱信号 |
| 2026-05-23 | ESP-La Liga | Espanyol | Real Sociedad | 🔴 CONFIDENT_AWAY | 0.071 | 0.929 | 17v6 | 1.4v0.6 | 情报逐条验证，模型罕见强信号 |
| 2026-05-23 | ESP-La Liga | Celta Vigo | Sevilla | 🟡 LEAN_HOME | 0.676 | 0.324 | 9v8 | 1.4v1.8 | 模型参考，情报平衡 |
| 2026-05-23 | ESP-La Liga | Alaves | Rayo Vallecano | 🟠 LEAN_AWAY | 0.232 | 0.768 | 16v13 | 2.0v1.8 | 情报主导，模型弱信号 |
| 2026-05-23 | ESP-La Liga | Getafe | Osasuna | 🟠 LEAN_AWAY | 0.23 | 0.77 | 12v10 | 0.8v0.6 | 情报主导，模型弱信号 |
| 2026-05-24 | ESP-La Liga | Villarreal | Atletico Madrid | 🟡 LEAN_HOME | 0.743 | 0.257 | 4v3 | 1.4v2.4 | 模型参考，情报平衡 |

## 标记说明

| 标记 | 含义 | v17 操作 |
|---|---|---|
| 🟢 CONFIDENT_HOME | 模型强信号主不败 | 模型主导，情报验证利空 |
| 🟡 LEAN_HOME | 模型倾向主不败 | 模型参考，情报平衡 |
| ⚪ TOSS_UP | 模型无明确方向 | 情报主导，忽略模型 |
| 🟠 LEAN_AWAY | 模型倾向客不败 | **情报主导**，模型弱信号（客不败历史命中低） |
| 🔴 CONFIDENT_AWAY | 模型强信号客不败 | 罕见！逐条验证反剧本 |
| ⚫ INSUFFICIENT_DATA | 数据不足 | 跳过模型 |

## 详细分场

### 🟠 Crystal Palace vs Arsenal
- **日期**: 2026-05-24 | **联赛**: ENG-Premier League
- **标记**: `LEAN_AWAY`
- **模型概率**: 主不败 0.223 | 客不败 0.777
- **排名**: Crystal Palace 排11 vs Arsenal 排2
- **近5积分**: Crystal Palace 0.4分 vs Arsenal 2.4分
- **v17 建议**: 情报主导，模型弱信号

### 🟠 Burnley vs Wolverhampton Wanderers
- **日期**: 2026-05-24 | **联赛**: ENG-Premier League
- **标记**: `LEAN_AWAY`
- **模型概率**: 主不败 0.241 | 客不败 0.759
- **排名**: Burnley 排20 vs Wolverhampton Wanderers 排14
- **近5积分**: Burnley 0.2分 vs Wolverhampton Wanderers 0.4分
- **v17 建议**: 情报主导，模型弱信号

### ⚪ Brighton vs Manchester United
- **日期**: 2026-05-24 | **联赛**: ENG-Premier League
- **标记**: `TOSS_UP_AWAY`
- **模型概率**: 主不败 0.413 | 客不败 0.587
- **排名**: Brighton 排9 vs Manchester United 排4
- **近5积分**: Brighton 1.4分 vs Manchester United 2.6分
- **v17 建议**: 情报主导，模型辅助

### 🔴 Fulham vs Newcastle United
- **日期**: 2026-05-24 | **联赛**: ENG-Premier League
- **标记**: `CONFIDENT_AWAY`
- **模型概率**: 主不败 0.144 | 客不败 0.856
- **排名**: Fulham 排15 vs Newcastle United 排7
- **近5积分**: Fulham 1.0分 vs Newcastle United 1.4分
- **v17 建议**: 情报逐条验证，模型罕见强信号

### 🟢 West Ham vs Leeds
- **日期**: 2026-05-24 | **联赛**: ENG-Premier League
- **标记**: `CONFIDENT_HOME`
- **模型概率**: 主不败 0.928 | 客不败 0.072
- **排名**: West Ham 排10 vs Leeds 排19
- **近5积分**: West Ham 0.8分 vs Leeds 2.2分
- **v17 建议**: 模型主导，情报验证

### ⚪ Borussia M.Gladbach vs Hoffenheim
- **日期**: 2026-05-16 | **联赛**: GER-Bundesliga
- **标记**: `TOSS_UP_HOME`
- **模型概率**: 主不败 0.524 | 客不败 0.476
- **排名**: Borussia M.Gladbach 排12 vs Hoffenheim 排9
- **近5积分**: Borussia M.Gladbach 1.0分 vs Hoffenheim 2.2分
- **v17 建议**: 情报主导，模型辅助

### 🟢 Bayern Munich vs FC Cologne
- **日期**: 2026-05-16 | **联赛**: GER-Bundesliga
- **标记**: `CONFIDENT_HOME`
- **模型概率**: 主不败 0.955 | 客不败 0.045
- **排名**: Bayern Munich 排1 vs FC Cologne 排15
- **近5积分**: Bayern Munich 2.6分 vs FC Cologne 1.0分
- **v17 建议**: 模型主导，情报验证

### 🟢 Bayer Leverkusen vs Hamburger SV
- **日期**: 2026-05-16 | **联赛**: GER-Bundesliga
- **标记**: `CONFIDENT_HOME`
- **模型概率**: 主不败 0.962 | 客不败 0.038
- **排名**: Bayer Leverkusen 排3 vs Hamburger SV 排22
- **近5积分**: Bayer Leverkusen 1.8分 vs Hamburger SV 1.2分
- **v17 建议**: 模型主导，情报验证

### 🟢 Eintracht Frankfurt vs VfB Stuttgart
- **日期**: 2026-05-16 | **联赛**: GER-Bundesliga
- **标记**: `CONFIDENT_HOME`
- **模型概率**: 主不败 0.864 | 客不败 0.136
- **排名**: Eintracht Frankfurt 排5 vs VfB Stuttgart 排7
- **近5积分**: Eintracht Frankfurt 0.8分 vs VfB Stuttgart 1.6分
- **v17 建议**: 模型主导，情报验证

### 🟠 Werder Bremen vs Borussia Dortmund
- **日期**: 2026-05-16 | **联赛**: GER-Bundesliga
- **标记**: `LEAN_AWAY`
- **模型概率**: 主不败 0.244 | 客不败 0.756
- **排名**: Werder Bremen 排14 vs Borussia Dortmund 排2
- **近5积分**: Werder Bremen 0.8分 vs Borussia Dortmund 1.2分
- **v17 建议**: 情报主导，模型弱信号

### 🔴 Espanyol vs Real Sociedad
- **日期**: 2026-05-23 | **联赛**: ESP-La Liga
- **标记**: `CONFIDENT_AWAY`
- **模型概率**: 主不败 0.071 | 客不败 0.929
- **排名**: Espanyol 排17 vs Real Sociedad 排6
- **近5积分**: Espanyol 1.4分 vs Real Sociedad 0.6分
- **v17 建议**: 情报逐条验证，模型罕见强信号

### 🟡 Celta Vigo vs Sevilla
- **日期**: 2026-05-23 | **联赛**: ESP-La Liga
- **标记**: `LEAN_HOME`
- **模型概率**: 主不败 0.676 | 客不败 0.324
- **排名**: Celta Vigo 排9 vs Sevilla 排8
- **近5积分**: Celta Vigo 1.4分 vs Sevilla 1.8分
- **v17 建议**: 模型参考，情报平衡

### 🟠 Alaves vs Rayo Vallecano
- **日期**: 2026-05-23 | **联赛**: ESP-La Liga
- **标记**: `LEAN_AWAY`
- **模型概率**: 主不败 0.232 | 客不败 0.768
- **排名**: Alaves 排16 vs Rayo Vallecano 排13
- **近5积分**: Alaves 2.0分 vs Rayo Vallecano 1.8分
- **v17 建议**: 情报主导，模型弱信号

### 🟠 Getafe vs Osasuna
- **日期**: 2026-05-23 | **联赛**: ESP-La Liga
- **标记**: `LEAN_AWAY`
- **模型概率**: 主不败 0.23 | 客不败 0.77
- **排名**: Getafe 排12 vs Osasuna 排10
- **近5积分**: Getafe 0.8分 vs Osasuna 0.6分
- **v17 建议**: 情报主导，模型弱信号

### 🟡 Villarreal vs Atletico Madrid
- **日期**: 2026-05-24 | **联赛**: ESP-La Liga
- **标记**: `LEAN_HOME`
- **模型概率**: 主不败 0.743 | 客不败 0.257
- **排名**: Villarreal 排4 vs Atletico Madrid 排3
- **近5积分**: Villarreal 1.4分 vs Atletico Madrid 2.4分
- **v17 建议**: 模型参考，情报平衡
