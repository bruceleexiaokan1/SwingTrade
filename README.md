# SwingTrade

个人波段量化交易系统

## 项目概述

专注于A股波段交易的量化投资系统，支持数据采集、策略回测和实盘交易。

**核心定位**：基于行为理解的智能波段交易系统 — 通过Pattern知识库、行为分析、归因反馈实现持续进化的交易能力。

---

## 核心架构

```
数据采集 → Pattern验证 → 信号生成 → 回测优化 → 归因反馈 → 策略迭代
    ↓           ↓           ↓          ↓          ↓         ↓
  akshare   41 Pattern   多周期共振   向量化回测   教训知识库   因子优化
  tushare   90%+置信度   四维筛选    蒙特卡洛     自动归因    拥挤度检测
```

---

## 核心功能

### 1. 数据系统
- **数据源**: tushare、akshare、baostock 多源并行
- **存储**: SQLite (索引) + Parquet (行情数据)
- **覆盖**: 5,488只A股，2021-2026约5年数据
- **加载器**: `StockDataLoader` - 支持热/温/冷分层访问

### 2. 技术指标 (`src/data/indicators/`)
| 模块 | 文件 | 功能 |
|------|------|------|
| MA | `ma.py` | 均线计算、金叉死叉检测 |
| MACD | `macd.py` | MACD指标 |
| RSI | `rsi.py` | RSI超买超卖、背离检测 |
| 布林带 | `bollinger.py` | 布林带压缩扩张 |
| ATR | `atr.py` | 真实波幅、止损计算 |
| ADX | `adx.py` | 趋势强度 |
| 成交量 | `volume.py` | 放量缩量、连续涨停 |
| 波浪 | `wave.py` | Elliott Wave |
| 缠论 | `chan_theory.py` | 笔/段/中枢/买卖点 |
| 共振 | `resonance.py` | 板块共振检测 |
| 板块RS | `sector_rs.py` | 相对强度排名 |

### 3. 信号检测 (`SwingSignals`)
- **三屏系统**：方向(MA/MACD) → 时机(RSI/布林) → 确认(成交量)
- **入场信号**：金叉回踩、突破布林带、RSI超卖反弹、游资操盘(连续涨停)
- **出场信号**：ATR止损、追踪止损、MA死叉、RSI超买、分批止盈(T1/T2)

### 4. 回测引擎 (`src/backtest/`)
| 模块 | 功能 |
|------|------|
| `engine.py` | 波段回测核心，支持T+1、滑点、涨跌停 |
| `matching.py` | 订单撮合引擎 |
| `models.py` | 数据模型：Trade、Position、BacktestResult |
| `multi_cycle.py` | 月/周/日三周期共振检测 |
| `market_state.py` | 市场状态识别 |
| `resonance.py` | 板块共振检测 |
| `position_sizer.py` | Kelly公式仓位管理 |
| `vectorized/engine.py` | 向量化回测引擎(预计算所有指标) |
| `performance.py` | 绩效分析 |

### 5. 因子库 (`src/factors/`)
- **因子注册表**: `FactorRegistry` - 线程安全单例
- **价格因子**: 动量、波动率
- **资金流因子**: 主力资金、北向资金
- **估值因子**: 市盈率、市净率
- **质量因子**: 盈利质量
- **评估框架**: IC/IR分析

---

## 技术栈

- **数据**: pandas、numpy
- **存储**: SQLite + Parquet
- **分析**: Jupyter、matplotlib、seaborn

---

## 项目结构

```
SwingTrade/
├── src/
│   ├── backtest/              # 回测框架 (~1800行)
│   │   ├── engine.py         # 核心回测引擎 (825行)
│   │   ├── matching.py       # 撮合引擎
│   │   ├── models.py         # 数据模型
│   │   ├── resonance.py      # 板块共振
│   │   ├── multi_cycle.py    # 多周期共振 (513行)
│   │   ├── market_state.py   # 市场状态
│   │   ├── position_sizer.py # Kelly仓位
│   │   ├── strategy_params.py # 策略参数 (219行)
│   │   └── vectorized/       # 向量化回测
│   │       └── engine.py    # 向量化引擎 (719行)
│   ├── data/
│   │   ├── indicators/       # 技术指标 (~17,870行)
│   │   │   ├── ma.py        # 均线
│   │   │   ├── macd.py      # MACD
│   │   │   ├── rsi.py       # RSI
│   │   │   ├── bollinger.py # 布林带
│   │   │   ├── atr.py       # ATR
│   │   │   ├── volume.py    # 成交量 (506行)
│   │   │   ├── chan_theory.py # 缠论 (697行)
│   │   │   ├── signals.py   # 综合信号 (673行)
│   │   │   └── ...
│   │   ├── vectorized/       # 向量化计算
│   │   ├── fetcher/         # 数据采集
│   │   └── loader.py        # 数据加载器
│   ├── factors/              # 因子库
│   │   ├── registry.py      # 因子注册表
│   │   ├── factor_base.py   # 因子基类
│   │   └── price_volume/    # 价格/动量因子
│   └── portfolio/            # 组合管理
├── scripts/                  # 分析脚本 (119个)
│   ├── pattern_loader.py    # Pattern定义加载器
│   ├── behavior_analysis_system.py # 行为分析系统
│   ├── comprehensive_pattern_validation.py # Pattern验证
│   ├── causal_attribution_engine.py # 因果归因
│   └── ...
├── config/                   # 配置文件
├── tests/                    # 测试套件 (942 tests)
│   ├── test_backtest/       # 回测测试
│   ├── test_indicators/     # 指标测试
│   └── ...
└── docs/                     # 设计文档 (30+)
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
export STOCKDATA_ROOT="/Users/bruce/workspace/trade/StockData"
export TUSHARE_TOKEN="your_token_here"
```

### 3. 运行回测

```python
from src.backtest.engine import SwingBacktester
from src.backtest.strategy_params import StrategyParams

params = StrategyParams.current_market()
backtester = SwingBacktester(
    initial_capital=1_000_000,
    strategy_params=params
)

result = backtester.run(
    stock_codes=["600519", "000858"],
    start_date="2025-01-01",
    end_date="2026-03-28"
)

print(result.summary())
```

### 4. 向量化回测

```python
from src.backtest.vectorized.engine import VectorizedBacktester, BacktestConfig

config = BacktestConfig(initial_capital=1_000_000)
backtester = VectorizedBacktester(config=config)

result = backtester.run(
    stock_codes=["600519", "000858"],
    start_date="2025-01-01",
    end_date="2026-03-28"
)
```

---

## 策略参数 (`StrategyParams`)

```python
# 指标参数
ma_short: 20        # 短期均线
ma_long: 60         # 长期均线
rsi_period: 14      # RSI周期
rsi_oversold: 55    # RSI超卖(强势市场)
rsi_overbought: 80  # RSI超买

# 风控参数
atr_stop_multiplier: 2.0      # ATR止损倍数
atr_trailing_multiplier: 3.0  # 追踪止损倍数
max_single_loss_pct: 0.02     # 单笔最大亏损2%
max_open_positions: 5          # 最大持仓数
```

预设配置：
- `StrategyParams.default()` - 默认参数
- `StrategyParams.current_market()` - 强势市场
- `StrategyParams.aggressive()` - 激进
- `StrategyParams.conservative()` - 保守

---

## 多周期共振系统

```python
from src.backtest.multi_cycle import MultiCycleResonance, MultiCycleLevel

resonance = MultiCycleResonance(stockdata_root="/path/to/StockData")
result = resonance.check_resonance("600519", "2026-03-28")

print(f"共振等级: {result.level_label}")  # 三周期共振/强信号/中信号/禁止操作
print(f"仓位上限: {result.position_limit}")  # 0.0~0.8
```

共振等级：
| 等级 | 标签 | 仓位上限 | 条件 |
|------|------|---------|------|
| 5 | 三周期共振 | 80% | 月周周日全部向上 |
| 4 | 强信号 | 60% | 月周共振，日线待确认 |
| 3 | 中信号 | 20% | 只有日线信号 |
| 0 | 禁止操作 | 0% | 三层逆势 |

---

## Pattern知识库

```python
from scripts.pattern_loader import PatternLoader

loader = PatternLoader()
p = loader.get_pattern("MM1")

# 信号前条件检查（预测）
result = loader.check_pre_conditions("MM1", stock_data)

# 信号后确认（用于确认）
result = loader.check_post_confirmation("MM1", post_data)
```

---

## 测试覆盖

```bash
# 运行所有测试
python -m pytest tests/ -v

# 特定模块
python -m pytest tests/test_backtest/ -v
python -m pytest tests/test_multi_cycle.py -v
```

| 测试模块 | 说明 |
|----------|------|
| test_backtest | 回测引擎测试 |
| test_multi_cycle | 多周期共振测试 |
| test_indicators | 技术指标测试 |
| test_bayesian | 贝叶斯分析 |
| test_chan_theory | 缠论测试 |
| test_wave | 波浪理论测试 |

---

## 文档体系

| 文档 | 说明 |
|------|------|
| `docs/PATTERN_SYSTEM_README.md` | Pattern系统说明 |
| `docs/pattern_knowledge_base.md` | Pattern知识库 |
| `docs/OPTIMAL_ENTRY_EXIT_SYSTEM.md` | 最优出入场系统 |
| `docs/comprehensive_system_design.md` | 完整系统设计 |
| `docs/FAILURE_WARNING_SYSTEM.md` | 失败预警系统 |
| `docs/STRATEGY_v1.4_FINAL.md` | 最终策略文档 |

---

## 核心结论

### 真正的起涨点

| 方法 | 样本 | 胜率 | 20日期望值 |
|------|------|------|-----------|
| Phase 1-5: 放量突破RSI>50 | 137 | 31.8% | -3.73% |
| Phase 6: 缩量震仓+高恢复 | 450 | 52.0% | +2.92% |
| 等待3日确认 | 837 | **61.4%** | **+4.52%** |
| 等待5日确认 | 839 | **66.2%** | **+5.63%** |

### 归因分解

- **技能贡献**: ~60%（入场时机+条件判断）
- **市场贡献**: ~30%（大盘涨跌）
- **运气贡献**: ~10%（随机波动）

### 核心原则

1. **等待确认**: 信号发生时的指标无法预测走势，唯一的"提前确认"是等待后续走势
2. **量能是核心**: 缩量=主力控盘（真洗盘），巨量=散户恐慌（真跌）
3. **止损纪律**: 单笔亏损≤2%，总暴露≤15%

---

## 数据覆盖（2026-03-31修正）

| 板块 | 覆盖数量 | 覆盖率 |
|------|---------|--------|
| 沪市主板(600/601/603xxx) | 1,701只 | 94.5% |
| 深市主板(000/001xxx) | 527只 | 263% |
| 中小板(002xxx) | 921只 | 92.1% |
| 创业板(300xxx) | 938只 | 72.2% |
| 科创板(688xxx) | 602只 | 120% |
| 北交所(920xxx) | 301只 | 100% |
| **总计** | **5,488只** | **~5年数据** |
