"""回测框架模块

提供波段交易策略回测功能：
- 撮合引擎（T+1开盘价成交、滑点、涨跌停）
- 回测引擎（信号检测、订单执行、持仓管理）
- 绩效分析（夏普比率、最大回撤、胜率、盈亏比）
- 报告生成（HTML权益曲线、回撤图）
"""

from .models import (
    Trade,
    Position,
    EquityRecord,
    PerformanceMetrics,
    BacktestResult,
    EntrySignal,
    ExitSignal,
    MatchResult,
    generate_id,
)

from .matching import OrderMatcher, calculate_slippage, check_limit_hit

from .engine import SwingBacktester

from .performance import PerformanceAnalyzer

from .reporter import BacktestReporter

from .position_sizer import KellyPositionSizer
from .market_state import MarketState, detect_market_state, MarketStateResult
from .multi_cycle import MultiCycleResonance, MultiCycleResult, MultiCycleLevel
from .walk_forward import WalkForwardAnalyzer, WalkForwardResult
from .expectancy import calculate_expectancy, filter_by_expectancy, ExpectancyResult

from .execution import (
    # 执行算法
    twap_execution,
    vwap_execution,
    iceberg_order,
    adaptive_execution,
    estimate_market_impact,
    monitor_execution_quality,
    order_slicer,
    execute_order,
    # 数据类
    ExecutionSlice,
    ExecutionResult,
    ExecutionStatus,
    MarketImpact,
    ExecutionQuality,
)

__all__ = [
    # 数据模型
    "Trade",
    "Position",
    "EquityRecord",
    "PerformanceMetrics",
    "BacktestResult",
    "EntrySignal",
    "ExitSignal",
    "MatchResult",
    "generate_id",
    # 撮合引擎
    "OrderMatcher",
    "calculate_slippage",
    "check_limit_hit",
    # 回测引擎
    "SwingBacktester",
    # 绩效分析
    "PerformanceAnalyzer",
    # 报告生成
    "BacktestReporter",
    # 仓位管理
    "KellyPositionSizer",
    # 市场状态识别
    "MarketState",
    "detect_market_state",
    "MarketStateResult",
    # 多周期共振
    "MultiCycleResonance",
    "MultiCycleResult",
    "MultiCycleLevel",
    # Walk-Forward 分析
    "WalkForwardAnalyzer",
    "WalkForwardResult",
    # 正期望
    "calculate_expectancy",
    "filter_by_expectancy",
    "ExpectancyResult",
    # 执行算法
    "twap_execution",
    "vwap_execution",
    "iceberg_order",
    "adaptive_execution",
    "estimate_market_impact",
    "monitor_execution_quality",
    "order_slicer",
    "execute_order",
    # 执行数据类
    "ExecutionSlice",
    "ExecutionResult",
    "ExecutionStatus",
    "MarketImpact",
    "ExecutionQuality",
]
