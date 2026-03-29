# SwingTrade

个人波段量化交易系统

## 项目概述

专注于A股波段交易的量化投资系统，支持数据采集、策略回测和实盘交易。

## 核心功能

- **数据采集**: tushare、akshare、baostock 多数据源
- **数据存储**: SQLite (索引) + Parquet (行情数据)
- **技术指标**: MA、MACD、RSI、布林带、ATR、ADX、波浪理论、缠论、HMM市场状态
- **策略框架**: 波段交易策略信号生成
- **回测引擎**: T+1 成交、滑点、涨跌停、持仓管理
- **共振系统**: 板块共振 + 多周期共振 (月/周/日)
- **仓位管理**: Kelly公式 + 波动率自适应
- **绩效分析**: 夏普比率、最大回撤、胜率、盈亏比
- **事件驱动**: 财报预期差、解禁风险、指数调样、股东增持
- **期权分析**: BS定价、隐含波动率、希腊字母、波动率套利
- **因子库**: 30个核心因子，IC验证体系，多因子合成

## 技术栈

- **数据**: pandas、numpy
- **存储**: SQLite + Parquet
- **分析**: Jupyter、matplotlib、seaborn

## 项目结构

```
SwingTrade/
├── src/
│   ├── backtest/                 # 回测框架
│   │   ├── engine.py            # 核心回测引擎
│   │   ├── matching.py          # 撮合引擎（T+1、滑点、涨跌停）
│   │   ├── models.py            # 数据模型（Trade、Position、BacktestResult）
│   │   ├── performance.py       # 绩效分析
│   │   ├── reporter.py          # 报告生成
│   │   ├── resonance.py         # 板块共振检测
│   │   ├── resonance_backtester.py  # 共振回测器
│   │   ├── resonance_position.py    # 共振仓位管理
│   │   ├── multi_cycle.py       # 多周期共振（月/周/日）
│   │   ├── market_state.py      # 市场状态识别
│   │   ├── position_sizer.py    # Kelly公式仓位管理
│   │   ├── expectancy.py        # 正期望计算
│   │   ├── walk_forward.py      # Walk-Forward分析
│   │   └── ...
│   └── data/
│       ├── indicators/          # 技术指标
│       │   ├── ma.py           # 移动平均线
│       │   ├── macd.py         # MACD
│       │   ├── rsi.py          # RSI
│       │   ├── bollinger.py     # 布林带
│       │   ├── atr.py          # ATR
│       │   ├── adx.py          # ADX（平均方向指数）
│       │   ├── volume.py        # 成交量指标
│       │   ├── wave.py         # 波浪理论
│       │   ├── chan_theory.py   # 缠论
│       │   ├── resonance.py     # 共振检测
│       │   ├── sector_signals.py # 板块信号
│       │   ├── sector_rs.py     # 板块相对强度
│       │   ├── signals.py       # 综合信号
│       │   ├── crowding.py      # 因子拥挤度
│       │   ├── hmm_model.py    # HMM市场状态识别
│       │   ├── event_driven.py  # 事件驱动策略
│       │   ├── options_volatility.py  # 期权波动率
│       │   ├── fama_french.py  # Fama-French因子
│       │   ├── fundamental.py  # 基本面量化
│       │   └── microstructure.py # 市场微观结构
│       ├── fetcher/             # 数据采集
│       │   ├── sector_fetcher.py   # 板块数据获取
│       │   └── ...
│       └── loader.py            # 数据加载器
├── config/                      # 配置文件
├── tests/                       # 测试（417 tests）
├── docs/                        # 设计文档
└── scripts/                     # 脚本

StockData/                       # 数据仓库（独立仓库）
├── raw/daily/                   # 日线原始数据 (Parquet)
├── sqlite/                      # SQLite 索引库
├── cache/                       # 热数据缓存
├── status/                      # 状态文件
└── logs/                        # 日志
```

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
from src.backtest.resonance_backtester import ResonanceBacktester

backtester = ResonanceBacktester(
    sector_config_path="config/sectors/sector_portfolio.json",
    initial_capital=1_000_000
)

result = backtester.run(
    sector_names=["半导体概念", "人工智能"],
    start_date="2025-01-01",
    end_date="2026-03-28"
)

print(result.summary())
```

## 核心模块

### 回测引擎 (SwingBacktester)

支持：
- T+1 开盘价成交
- 滑点控制
- 涨跌停检测
- ATR 追踪止损
- 分批止盈 (T1/T2)
- Kelly 仓位管理
- 市场状态感知

### 共振系统

**板块共振** (ResonanceBacktester):
- 8 条件评分 (S/A/B/C 四级)
- 每日动态检测
- 进度持久化

**多周期共振** (MultiCycleResonance):
- 月线定方向 (MA5/MA10)
- 周线定趋势
- 日线定入场

### 技术指标

| 指标 | 文件 | 说明 |
|------|------|------|
| ADX | `adx.py` | 趋势强度判断 |
| 波浪 | `wave.py` | Elliott Wave 定位 |
| 缠论 | `chan_theory.py` | 笔/段/中枢/买卖点 |
| 板块RS | `sector_rs.py` | 相对强度排名 |

### 因子库

**验证通过的因子**:

| 因子 | IC | p-value | 年化(毛) | 权重 |
|------|-----|---------|---------|------|
| ret_6m | 0.050 | 0.0006 | 62.2% | 30% |
| fund_flow | 0.065 | 0.000007 | 54.1% | 70% |

**最优策略**: 30% ret_6m + 70% fund_flow, 持仓10日

| 参数 | 数值 |
|------|------|
| 预期年化 | 80.4% |
| 扣除成本后 | ~76% |
| IC | 0.043 |
| p-value | 0.003 |
| 置信度 | 99.5% |

**因子模块**:
- `factors/price_volume/` - 动量因子、波动率因子
- `factors/flow/` - 资金流因子、北向资金因子
- `factors/valuation/` - 估值因子 (需财务数据)
- `factors/quality/` - 质量因子 (需财务数据)
- `factors/evaluation/` - IC/IR评估框架

## 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 特定模块测试
python -m pytest tests/test_wave.py -v
python -m pytest tests/test_multi_cycle.py -v
```

**测试覆盖**: 838 tests

## 开发进度

| 模块 | 状态 |
|------|------|
| 数据存储架构 | ✅ 完成 |
| 数据采集 | ✅ 完成 |
| 回测引擎 | ✅ 完成 |
| 共振系统 | ✅ 完成 |
| 技术指标 | ✅ 完成 |
| 绩效分析 | ✅ 完成 |
| 因子库验证 | ✅ 完成 |
| 文档 | ✅ 完成 |

## 文档

- [设计文档](docs/design.md) - 完整设计方案
- [数据系统设计](docs/SwingTrade_Data_System_Design.md) - 数据存储架构
- [因子库规划](docs/factor_library_planning.md) - 因子库建设路线图
- [回测发现](docs/backtest_findings.md) - 回测结果分析
