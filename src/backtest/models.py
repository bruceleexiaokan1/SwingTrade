"""回测数据模型

定义回测引擎所需的核心数据类：
- Trade: 单笔成交记录
- Position: 持仓记录
- BacktestResult: 回测结果
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
import uuid
import pandas as pd


def generate_id() -> str:
    """生成唯一ID"""
    return str(uuid.uuid4())[:8]


@dataclass
class Trade:
    """单笔成交记录"""
    trade_id: str = field(default_factory=generate_id)
    date: str = ""                       # 成交日期 (T+1)
    code: str = ""                        # 股票代码

    # 方向
    direction: str = "long"              # "long" / "short"

    # 成交价格
    entry_price: float = 0.0             # 入场价格
    exit_price: float = 0.0              # 出场价格 (仅平仓时)

    # 数量
    shares: int = 0                      # 成交股数

    # 金额
    turnover: float = 0.0                # 成交额
    commission: float = 0.0               # 手续费

    # 信号
    signal_type: str = ""                 # "golden", "breakout"
    signal_reason: str = ""              # 信号原因

    # 执行信息
    slippage: float = 0.0                # 滑点成本
    limit_hit: bool = False               # 是否触及涨跌停

    # 持仓关联
    position_id: str = ""                 # 关联仓位ID

    @property
    def pnl(self) -> float:
        """盈亏金额"""
        if self.exit_price == 0:
            return 0.0
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def pnl_pct(self) -> float:
        """盈亏比例"""
        if self.entry_price == 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price


@dataclass
class Position:
    """持仓记录"""
    position_id: str = field(default_factory=generate_id)
    code: str = ""                        # 股票代码
    direction: str = "long"               # "long" / "short"

    # 入场信息
    entry_date: str = ""                  # 入场日期
    entry_price: float = 0.0             # 入场价格
    shares: int = 0                       # 持股数量
    original_shares: int = 0              # 原始持股数量（用于分批止盈计算）

    # ATR止损
    atr: float = 0.0                     # 入场时ATR
    stop_loss: float = 0.0               # 止损价
    trailing_stop: float = 0.0           # 追踪止损价
    highest_price: float = 0.0           # 持仓期间最高价

    # 关键结构破坏止损（知识库定义）
    entry_prev_low: float = 0.0         # 入场后前一根K线最低点
    lowest_3d_low: float = 0.0           # 前3日最低点

    # 分批止盈状态
    t1_triggered: bool = False            # T1止盈是否已触发
    t2_triggered: bool = False            # T2止盈是否已触发

    # 状态
    status: str = "open"                 # "open" / "closed"
    exit_date: Optional[str] = None       # 出场日期
    exit_reason: Optional[str] = None   # 出场原因

    # 当前市场价格（由回测引擎在每个交易日更新）
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        """市值（使用当前市场价格）"""
        return self.current_price * self.shares

    @property
    def unrealized_pnl(self) -> float:
        """浮动盈亏"""
        if self.current_price == 0 or self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) * self.shares

    @property
    def unrealized_pnl_pct(self) -> float:
        """浮动盈亏比例"""
        if self.current_price == 0 or self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    def reduce_shares(self, ratio: float) -> int:
        """
        减仓并返回减仓数量

        Args:
            ratio: 减仓比例（0.0~1.0）

        Returns:
            减仓的股数
        """
        shares_to_reduce = int(self.shares * ratio)
        self.shares -= shares_to_reduce
        return shares_to_reduce


@dataclass
class EquityRecord:
    """权益记录（每日）"""
    date: str
    equity: float                         # 总权益
    cash: float                          # 现金
    market_value: float                  # 市值
    daily_return: float = 0.0           # 当日收益


@dataclass
class PerformanceMetrics:
    """绩效指标"""
    # 基础
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 0.0
    final_capital: float = 0.0

    # 交易统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # 收益指标
    total_return: float = 0.0
    annualized_return: float = 0.0
    profit_factor: float = 0.0          # 盈亏比
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # 风险指标
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0      # 最大回撤持续天数
    calmar_ratio: float = 0.0

    # 持仓指标
    avg_holding_days: float = 0.0
    total_trading_days: int = 0
    trades_per_year: float = 0.0


@dataclass
class BacktestResult:
    """完整回测结果"""
    # 基础信息
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 0.0
    final_capital: float = 0.0

    # 交易统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # 收益指标
    total_return: float = 0.0
    annualized_return: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # 风险指标
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    calmar_ratio: float = 0.0

    # 持仓指标
    avg_holding_days: float = 0.0
    total_trading_days: int = 0
    trades_per_year: float = 0.0

    # 序列数据
    equity_curve: pd.DataFrame | None = None  # type: ignore
    trades: List[Trade] = field(default_factory=list)
    positions: List[Position] = field(default_factory=list)

    def to_metrics(self) -> PerformanceMetrics:
        """转换为 PerformanceMetrics"""
        return PerformanceMetrics(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            final_capital=self.final_capital,
            total_trades=self.total_trades,
            winning_trades=self.winning_trades,
            losing_trades=self.losing_trades,
            win_rate=self.win_rate,
            total_return=self.total_return,
            annualized_return=self.annualized_return,
            profit_factor=self.profit_factor,
            avg_win=self.avg_win,
            avg_loss=self.avg_loss,
            sharpe_ratio=self.sharpe_ratio,
            sortino_ratio=self.sortino_ratio,
            max_drawdown=self.max_drawdown,
            max_drawdown_duration=self.max_drawdown_duration,
            calmar_ratio=self.calmar_ratio,
            avg_holding_days=self.avg_holding_days,
            total_trading_days=self.total_trading_days,
            trades_per_year=self.trades_per_year,
        )

    def summary(self) -> str:
        """回测结果摘要"""
        return f"""=== Backtest Result ===
Period: {self.start_date} ~ {self.end_date}
Initial: {self.initial_capital:,.0f} → Final: {self.final_capital:,.0f}
Return: {self.total_return:.2%} | Annual: {self.annualized_return:.2%}
Sharpe: {self.sharpe_ratio:.2f} | MaxDD: {self.max_drawdown:.2%} | Calmar: {self.calmar_ratio:.2f}
Trades: {self.total_trades} | Win Rate: {self.win_rate:.2%} | Profit Factor: {self.profit_factor:.2f}
Avg Holding: {self.avg_holding_days:.1f} days | Trades/Year: {self.trades_per_year:.1f}
"""


@dataclass
class EntrySignal:
    """入场信号"""
    code: str
    signal_type: str                      # "golden", "breakout"
    confidence: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    atr: float = 0.0
    reason: str = ""
    market_state: Optional[str] = None    # 市场状态: "trend"/"volatile"/"transition"


@dataclass
class ExitSignal:
    """出场信号"""
    position_id: str
    code: str
    exit_signal: str                       # "stop_loss", "trailing_stop", "ma_cross", "rsi_overbought", "take_profit_1", "take_profit_2"
    exit_price: float = 0.0
    reason: str = ""
    reduce_only: bool = False              # 是否只是减仓（不清仓）
    reduce_ratio: float = 1.0              # 减仓比例（1.0=全部平仓）


@dataclass
class MatchResult:
    """撮合结果"""
    success: bool = False
    match_date: str = ""
    match_price: float = 0.0
    reason: str = ""

    # 滑点信息
    slippage: float = 0.0
    slippage_reason: str = ""

    # 限制信息
    limit_hit: bool = False
    limit_type: Optional[str] = None     # "limit_up", "limit_down"

    # 数量信息
    filled_shares: int = 0
    turnover: float = 0.0
    commission: float = 0.0
