# 大规模回测验证计划

## 背景

新实现的功能需要验证：
1. **min_profit_loss_ratio = 3.0** (中长线最低盈亏比)
2. **T2 独立触发** (跌破 MA10 减仓 50%)
3. **结构止损** (entry_prev_low, lowest_3d_low)
4. **RSI 动态阈值** (短周期更敏感)

---

## 一、参数敏感性分析

### 1.1 核心参数网格扫描

| 参数 | 扫描范围 | 步长 | 默认值 |
|------|---------|------|--------|
| min_profit_loss_ratio | [1.5, 2.0, 2.5, 3.0, 3.5, 4.0] | - | 3.0 |
| atr_stop_multiplier | [1.5, 2.0, 2.5, 3.0] | - | 2.0 |
| atr_trailing_multiplier | [2.0, 2.5, 3.0, 3.5, 4.0] | - | 3.0 |
| rsi_oversold | [25, 30, 35, 40] | - | 35 |

### 1.2 结构止损专项测试

| 测试场景 | entry_prev_low | lowest_3d_low | 预期 |
|---------|----------------|---------------|------|
| 结构止损1触发 | 启用 | 禁用 | 仅 structure_stop_1 触发 |
| 结构止损2触发 | 禁用 | 启用 | 仅 structure_stop_2 触发 |
| 双重结构止损 | 启用 | 启用 | 按优先级触发 |
| ATR兜底 | 禁用 | 禁用 | 仅 ATR 止损 |

---

## 二、市场环境覆盖

### 2.1 时间段划分

| 时期 | 名称 | 特征 | 覆盖年份 |
|------|------|------|---------|
| 牛市 | Bull | 指数单边上涨 | 2019-2020 |
| 熊市 | Bear | 指数单边下跌 | 2022 |
| 震荡 | Sideways | 区间波动 | 2021, 2023 |
| 反弹 | Recovery | 下跌后反弹 | 2024 |

### 2.2 板块覆盖

重点测试板块：
- 半导体概念
- 人工智能
- 新能源
- 消费
- 金融

---

## 三、绩效指标体系

### 3.1 收益指标
- 总收益率
- 年化收益率
- 超额收益（相对于基准）

### 3.2 风险指标
- 夏普比率 (Sharpe Ratio)
- 索提诺比率 (Sortino Ratio)
- 卡玛比率 (Calmar Ratio)
- 最大回撤 (Max Drawdown)
- 最大回撤持续时间

### 3.3 交易指标
- 胜率 (Win Rate)
- 盈亏比 (Profit Factor)
- 交易次数 (Total Trades)
- 年均交易次数
- 平均持仓周期

---

## 四、测试用例设计

### 4.1 回归测试 (vs. 修复前)

```python
def test_compare_before_after():
    """对比修复前后的绩效差异"""
    # 运行修复前参数
    # 运行修复后参数
    # 对比关键指标
```

### 4.2 边界条件测试

```python
# 最小盈亏比边界
test_min_profit_loss_ratio_edge_cases():
    # ratio = 1.0 极端宽松
    # ratio = 5.0 极端严格
    # 验证过滤效果
```

### 4.3 结构止损触发频率

```python
def test_structure_stop_frequency():
    """统计结构止损触发占比"""
    # 统计 structure_stop_1 vs structure_stop_2 vs ATR_stop
```

---

## 五、实现步骤

### Phase 1: 基础设施 (1天)
- [ ] 扩展 `run_parameter_scan.py` 支持网格扫描
- [ ] 实现 `BacktestGridRunner` 并行回测类
- [ ] 配置 SQLite 数据源验证

### Phase 2: 单参数扫描 (2天)
- [ ] min_profit_loss_ratio 敏感性测试
- [ ] ATR 止损/追踪参数扫描
- [ ] RSI 阈值参数扫描

### Phase 3: 多参数组合 (3天)
- [ ] 结构止损效果对比
- [ ] 参数组合优化
- [ ] 生成敏感性分析报告

### Phase 4: 全市场验证 (5天)
- [ ] 全 A 股回测 (T+1 过滤)
- [ ] 多板块共振测试
- [ ] 跨时间段稳健性验证

### Phase 5: 报告生成 (1天)
- [ ] 生成绩效对比报告
- [ ] 参数推荐清单
- [ ] 风险提示清单

---

## 六、预期交付物

1. **参数敏感性报告** (`reports/parameter_sensitivity_*.html`)
2. **全市场回测报告** (`reports/full_backtest_*.html`)
3. **推荐参数清单** (`config/recommended_params.json`)
4. **风险事件统计** (`reports/risk_events_*.csv`)

---

## 七、风险控制

1. **数据质量**: 排除停牌、涨跌停等异常数据
2. **过拟合警告**: 跨时间段验证参数稳健性
3. **交易成本**: 佣金+印花税+滑点全量计入
4. **流动性**: 最小交易金额过滤
