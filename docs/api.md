# SwingTrade API 文档

**版本**: v1.0
**状态**: 已完成
**测试**: 417 tests passed

---

## 一、模块概览

```
SwingTrade/src/
├── backtest/                      # 回测框架
│   ├── engine.py                # 核心回测引擎
│   ├── models.py                # 数据模型
│   ├── resonance.py             # 板块共振检测
│   ├── resonance_backtester.py  # 共振回测器
│   ├── multi_cycle.py          # 多周期共振
│   ├── market_state.py          # 市场状态识别
│   ├── position_sizer.py        # 仓位管理
│   ├── expectancy.py            # 正期望计算
│   ├── walk_forward.py          # Walk-Forward分析
│   └── ...
│
└── data/indicators/            # 技术指标
    ├── ma.py, macd.py, rsi.py  # 基础指标
    ├── adx.py                  # ADX指标
    ├── wave.py                # 波浪理论
    ├── chan_theory.py         # 缠论
    ├── sector_rs.py           # 板块相对强度
    └── signals.py             # 综合信号
```

---

## 二、核心数据模型

### 2.1 Position (持仓)

```python
from src.backtest.models import Position

@dataclass
class Position:
    position_id: str              # 唯一ID
    code: str                      # 股票代码
    direction: str = "long"        # "long" / "short"

    # 入场信息
    entry_date: str               # 入场日期
    entry_price: float            # 入场价格
    shares: int                   # 持股数量
    original_shares: int          # 原始持股数量（分批止盈用）

    # ATR止损
    atr: float                    # 入场时ATR
    stop_loss: float              # 止损价
    trailing_stop: float          # 追踪止损价
    highest_price: float          # 持仓期间最高价

    # 分批止盈状态
    t1_triggered: bool            # T1止盈是否已触发
    t2_triggered: bool            # T2止盈是否已触发

    # 状态
    status: str                   # "open" / "closed"
    exit_date: Optional[str]       # 出场日期
    exit_reason: Optional[str]    # 出场原因

    # 当前市场价格（回测引擎每日更新）
    current_price: float

    # 计算属性
    market_value: float           # current_price * shares
    unrealized_pnl: float         # (current_price - entry_price) * shares
    unrealized_pnl_pct: float     # (current_price - entry_price) / entry_price
```

**示例**:
```python
position = Position(
    code="600519",
    entry_price=100.0,
    shares=1000,
    atr=2.0,
    current_price=110.0
)

print(position.market_value)       # 110000.0
print(position.unrealized_pnl)    # 10000.0
print(position.unrealized_pnl_pct) # 0.10
```

### 2.2 Trade (成交)

```python
from src.backtest.models import Trade

@dataclass
class Trade:
    trade_id: str
    date: str                     # 成交日期 (T+1)
    code: str
    direction: str = "long"
    entry_price: float
    exit_price: float             # 出场价格
    shares: int
    turnover: float               # 成交额
    commission: float             # 手续费
    signal_type: str              # "golden", "breakout"
    signal_reason: str
    slippage: float              # 滑点成本
    limit_hit: bool               # 是否触及涨跌停

    # 计算属性
    pnl: float                    # (exit_price - entry_price) * shares
    pnl_pct: float               # (exit_price - entry_price) / entry_price
```

### 2.3 BacktestResult (回测结果)

```python
from src.backtest.models import BacktestResult

result = BacktestResult(
    start_date="2025-01-01",
    end_date="2026-03-28",
    initial_capital=1_000_000,
    final_capital=1_200_000,
    total_trades=50,
    winning_trades=35,
    win_rate=0.70,
    total_return=0.20,
    sharpe_ratio=1.5,
    max_drawdown=0.10,
    # ... 更多指标
)

print(result.summary())
```

---

## 三、回测引擎

### 3.1 SwingBacktester

```python
from src.backtest.engine import SwingBacktester

backtester = SwingBacktester(
    initial_capital=1_000_000,     # 初始资金
    commission_rate=0.0003,        # 佣金率 0.03%
    stamp_tax=0.0001,              # 印花税 0.01%
    max_open_positions=5,         # 最大持仓数
    trial_position_pct=0.10,       # 试探仓比例 10%
    max_single_loss_pct=0.02,     # 单笔最大亏损 2%
    atr_stop_multiplier=2.0,      # ATR止损倍数
    strategy_params={...}          # 策略参数
)

result = backtester.run(
    stock_codes=["600519", "000001"],
    start_date="2025-01-01",
    end_date="2026-03-28",
)
```

### 3.2 关键方法

| 方法 | 说明 |
|------|------|
| `run()` | 执行回测 |
| `_detect_entries()` | 检测入场信号 |
| `_execute_orders()` | 执行订单 |
| `_update_and_record()` | 更新持仓权益 |
| `_close_positions()` | 检查止损/止盈 |

---

## 四、共振系统

### 4.1 板块共振 (ResonanceBacktester)

```python
from src.backtest.resonance_backtester import ResonanceBacktester

backtester = ResonanceBacktester(
    sector_config_path="config/sectors/sector_portfolio.json",
    initial_capital=1_000_000,
    max_positions=5,
)

result = backtester.run(
    sector_names=["半导体概念", "人工智能"],
    start_date="2025-01-01",
    end_date="2026-03-28",
    min_resonance_level=ResonanceLevel.C
)
```

### 4.2 共振等级 (ResonanceLevel)

```python
from src.backtest.resonance import ResonanceLevel

class ResonanceLevel(IntEnum):
    INVALID = 0   # 无效/禁止操作
    C = 1        # 观察 (2-3/8 条件满足)
    B = 2        # 弱共振 (4-5/8 条件满足)
    A = 3        # 强共振 (6-7/8 条件满足)
    S = 4        # 完美共振 (8/8 条件满足)
```

### 4.3 多周期共振 (MultiCycleResonance)

```python
from src.backtest.multi_cycle import MultiCycleResonance, MultiCycleLevel

resonance = MultiCycleResonance(stockdata_root="/path/to/StockData")

result = resonance.check_resonance("600519", "2026-03-28")

print(f"共振等级: {result.level_label}")
print(f"仓位上限: {result.position_limit}")
print(f"月线趋势: {result.monthly_trend}")
print(f"周线趋势: {result.weekly_trend}")
print(f"日线趋势: {result.daily_trend}")
```

| MultiCycleLevel | 条件 | 仓位上限 |
|-----------------|------|---------|
| THREE_CYCLE (5) | 月周周日全部向上 | 80% |
| MONTHLY_WEEKLY (4) | 月周共振，日线待确认 | 60% |
| DAILY_ONLY (3) | 只有日线信号 | 20% |
| FORBIDDEN (0) | 三层逆势 | 0% |

---

## 五、技术指标

### 5.1 ADX (平均方向指数)

```python
from src.data.indicators.adx import calculate_adx

df = calculate_adx(df, period=14)

# 辅助函数
is_strong = adx_strong_trend(adx_value)       # ADX > 25
is_weak = adx_weak_trend(adx_value)           # ADX < 20
is_rising = adx_rising(adx_series)           # ADX 在上升
is_bullish = adx_bullish_signal(adx, plus_di, minus_di)
is_bearish = adx_bearish_signal(adx, plus_di, minus_di)
```

### 5.2 波浪理论 (WaveIndicators)

```python
from src.data.indicators.wave import WaveIndicators

wave = WaveIndicators()
result = wave.analyze(df)

print(f"波浪评分: {result.wave_score}")  # 0.0 ~ 1.0
print(f"趋势方向: {result.trend_direction}")  # "up" / "down" / "sideways"
print(f"局部高点: {result.wave_points}")  # [WavePoint, ...]
```

### 5.3 缠论 (ChanTheory)

```python
from src.data.indicators.chan_theory import ChanTheory, detect_chan_signals

# 计算缠论
result = calculate_chan(df)

# 检测买卖点
signals = detect_chan_signals(df)
for signal in signals:
    print(f"{signal.date}: {signal.signal_type} @ {signal.price}")
```

### 5.4 板块相对强度 (SectorRelativeStrength)

```python
from src.data.indicators.sector_rs import SectorRelativeStrength

rs = SectorRelativeStrength(stockdata_root="/path/to/StockData")

# 计算 RS 值
rs_values = rs.calculate_rs("半导体概念", stock_codes, date, lookback=20)

# 计算 RPS 排名 (0.0 ~ 1.0)
rps_values = rs.calculate_rps("半导体概念", stock_codes, date)

# 获取最强板块
top_sectors = rs.get_top_sectors(sector_names, date, top_n=5)
```

---

## 六、市场状态识别

```python
from src.backtest.market_state import MarketState, detect_market_state

state = detect_market_state(df)

if state == MarketState.TREND:
    # 趋势市：ADX > 25，只做顺势波段，放宽止损
    entry_size *= 1.0
elif state == MarketState.VOLATILE:
    # 震荡市：ADX < 20，空仓或切换期权/网格
    entry_size *= 0.5
elif state == MarketState.TRANSITION:
    # 转折市：波动率急剧放大，谨慎操作
    entry_size *= 0.8
```

---

## 七、仓位管理

### 7.1 Kelly公式

```python
from src.backtest.position_sizer import KellyPositionSizer

sizer = KellyPositionSizer(
    max_risk_pct=0.02,        # 单笔最大风险 2%
    max_position_pct=0.15    # 最大持仓比例 15%
)

# 计算仓位
position_value = sizer.calculate_position(
    account_value=100_000,
    win_rate=0.6,
    avg_win=5000,
    avg_loss=2000,
    atr_pct=0.03
)
```

### 7.2 正期望计算

```python
from src.backtest.expectancy import calculate_expectancy, filter_by_expectancy

# 计算期望收益
result = calculate_expectancy(trades)

print(f"期望收益: {result.expectancy}")
print(f"胜率: {result.win_rate}")
print(f"盈亏比: {result.profit_ratio}")
print(f"是否可行: {result.is_viable}")

# 过滤可行交易
viable_trades = filter_by_expectancy(trades, min_expectancy=0.01)
```

---

## 八、Walk-Forward 分析

```python
from src.backtest.walk_forward import WalkForwardAnalyzer

analyzer = WalkForwardAnalyzer(
    backtester=backtester,
    train_window=756,      # 训练窗口 (~3年)
    test_window=126,      # 测试窗口 (~6个月)
    step=63               # 滚动步长 (~1个月)
)

results = analyzer.run(
    stock_codes=["600519"],
    start_date="2020-01-01",
    end_date="2026-03-28"
)

# WFR = OOS_Sharpe / IS_Sharpe
# 稳健策略: WFR >= 0.6
print(f"WFR: {results.robustness_ratio}")
```

---

## 九、回测结果示例

```python
result = backtester.run(
    sector_names=["半导体概念"],
    start_date="2025-01-01",
    end_date="2026-03-28"
)

print(result.summary())
```

**输出示例**:
```
=== Backtest Result ===
Period: 2025-01-01 ~ 2026-03-28
Initial: 1,000,000 → Final: 1,185,000
Return: 18.50% | Annual: 14.20%
Sharpe: 1.45 | MaxDD: 8.30% | Calmar: 1.71
Trades: 47 | Win Rate: 68.09% | Profit Factor: 2.31
Avg Holding: 12.3 days | Trades/Year: 28.5
```

---

## 十、测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定模块测试
python -m pytest tests/test_wave.py -v
python -m pytest tests/test_multi_cycle.py -v
python -m pytest tests/test_resonance.py -v

# 生成覆盖率报告
python -m pytest tests/ --cov=src --cov-report=html
```

**测试统计**: 417 passed, 1 skipped
