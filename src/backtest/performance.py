"""绩效分析

计算回测绩效指标：
- 收益类：总收益、年化收益、盈亏比
- 风险类：夏普比率、最大回撤、卡玛比率
- 交易类：胜率、持仓周期、年均交易次数
"""

import logging
from typing import List, Tuple
import pandas as pd
import numpy as np
from math import sqrt

from .models import Trade, Position, PerformanceMetrics, EquityRecord

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    绩效分析器

    计算指标：
    - 收益类：总收益、年化收益、盈亏比
    - 风险类：夏普比率、最大回撤、卡玛比率
    - 交易类：胜率、持仓周期、年均交易次数
    """

    def __init__(self, risk_free_rate: float = 0.03, trading_days: int = 252):
        """
        初始化绩效分析器

        Args:
            risk_free_rate: 无风险利率，默认 3%
            trading_days: 年交易天数，默认 252
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def analyze(
        self,
        equity_curve: pd.DataFrame,
        trades: List[Trade],
        positions: List[Position],
        initial_capital: float
    ) -> PerformanceMetrics:
        """
        完整绩效分析

        Args:
            equity_curve: 权益曲线 DataFrame
            trades: 成交记录列表
            positions: 持仓记录列表
            initial_capital: 初始资金

        Returns:
            PerformanceMetrics: 绩效指标
        """
        metrics = PerformanceMetrics()

        if equity_curve.empty:
            return metrics

        # 基础信息
        metrics.initial_capital = initial_capital
        metrics.final_capital = equity_curve["equity"].iloc[-1]
        metrics.start_date = equity_curve["date"].iloc[0]
        metrics.end_date = equity_curve["date"].iloc[-1]

        # 收益指标
        metrics.total_return = self._calc_total_return(equity_curve, initial_capital)
        metrics.annualized_return = self._calc_annualized_return(
            metrics.total_return, len(equity_curve)
        )

        # 交易统计
        metrics.total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        metrics.winning_trades = len(winning_trades)
        metrics.losing_trades = len(losing_trades)

        # 盈亏比
        avg_win, avg_loss = self._calc_avg_win_loss(trades)
        metrics.avg_win = avg_win
        metrics.avg_loss = avg_loss
        metrics.profit_factor = self._calc_profit_factor(trades)

        # 胜率
        metrics.win_rate = self._calc_win_rate(trades)

        # 风险指标
        metrics.sharpe_ratio = self._calc_sharpe_ratio(equity_curve)
        metrics.sortino_ratio = self._calc_sortino_ratio(equity_curve)
        metrics.max_drawdown, metrics.max_drawdown_duration = self._calc_max_drawdown(equity_curve)

        # 卡玛比率
        if metrics.max_drawdown != 0:
            metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown)

        # 持仓指标
        metrics.avg_holding_days = self._calc_avg_holding_days(positions)
        metrics.total_trading_days = len(equity_curve)
        metrics.trades_per_year = self._calc_trades_per_year(trades, len(equity_curve))

        return metrics

    def _calc_total_return(
        self,
        equity_curve: pd.DataFrame,
        initial_capital: float
    ) -> float:
        """计算总收益率"""
        if initial_capital <= 0:
            return 0.0
        final_equity = equity_curve["equity"].iloc[-1]
        return (final_equity - initial_capital) / initial_capital

    def _calc_annualized_return(
        self,
        total_return: float,
        trading_days: int
    ) -> float:
        """计算年化收益率"""
        if trading_days <= 0:
            return 0.0
        years = trading_days / self.trading_days
        if years <= 0:
            return 0.0
        return (1 + total_return) ** (1 / years) - 1

    def _calc_sharpe_ratio(self, equity_curve: pd.DataFrame) -> float:
        """
        夏普比率 = (年化收益 - 无风险利率) / 年化波动率

        标准：
        - > 1.5: 优秀
        - 1.0~1.5: 良好
        - < 1.0: 一般
        """
        if len(equity_curve) < 2:
            return 0.0

        returns = equity_curve["daily_return"].dropna()

        if len(returns) < 2:
            return 0.0

        # 年化
        annual_return = returns.mean() * self.trading_days
        annual_volatility = returns.std() * sqrt(self.trading_days)

        if annual_volatility == 0:
            return 0.0

        sharpe = (annual_return - self.risk_free_rate) / annual_volatility
        return round(sharpe, 2)

    def _calc_sortino_ratio(self, equity_curve: pd.DataFrame) -> float:
        """
        索提诺比率 = (年化收益 - 无风险利率) / 下行波动率

        只考虑下行波动率，对收益分布不对称策略更公平
        """
        if len(equity_curve) < 2:
            return 0.0

        returns = equity_curve["daily_return"].dropna()

        if len(returns) < 2:
            return 0.0

        # 下行波动率（只考虑负收益）
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return 0.0

        downside_std = negative_returns.std() * sqrt(self.trading_days)

        if downside_std == 0:
            return 0.0

        annual_return = returns.mean() * self.trading_days
        sortino = (annual_return - self.risk_free_rate) / downside_std

        return round(sortino, 2)

    def _calc_max_drawdown(
        self,
        equity_curve: pd.DataFrame
    ) -> Tuple[float, int]:
        """
        最大回撤 = max(peak - valley) / peak

        返回：(回撤比例, 持续天数)
        """
        if equity_curve.empty:
            return 0.0, 0

        equity = equity_curve["equity"].values
        peak = equity[0]
        max_dd = 0.0
        max_dd_duration = 0

        current_duration = 0
        peak_idx = 0

        for i, value in enumerate(equity):
            if value > peak:
                peak = value
                peak_idx = i
                current_duration = 0
            else:
                current_duration = i - peak_idx

            dd = (peak - value) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_duration = current_duration

        return round(max_dd, 4), max_dd_duration

    def _calc_profit_factor(self, trades: List[Trade]) -> float:
        """
        盈亏比 = 总盈利 / 总亏损

        标准：
        - > 3.0: 理想
        - > 1.5: 核心标准
        """
        if not trades:
            return 0.0

        total_profit = sum(t.pnl for t in trades if t.pnl > 0)
        total_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

        if total_loss == 0:
            return float("inf") if total_profit > 0 else 0.0

        return round(total_profit / total_loss, 2)

    def _calc_win_rate(self, trades: List[Trade]) -> float:
        """
        胜率 = 盈利交易数 / 总交易数

        标准：40%~60%
        """
        if not trades:
            return 0.0

        winning = sum(1 for t in trades if t.pnl > 0)
        return round(winning / len(trades), 4)

    def _calc_avg_win_loss(self, trades: List[Trade]) -> Tuple[float, float]:
        """计算平均盈利和平均亏损"""
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0.0

        return round(avg_win, 2), round(avg_loss, 2)

    def _calc_avg_holding_days(self, positions: List[Position]) -> float:
        """
        平均持仓周期

        标准：> 20天 (波段交易特征)
        """
        closed_positions = [
            p for p in positions
            if p.status == "closed"
            and p.exit_date is not None
            and p.entry_date
        ]

        if not closed_positions:
            return 0.0

        total_days = 0
        for p in closed_positions:
            try:
                entry_dt = pd.to_datetime(p.entry_date)
                exit_dt = pd.to_datetime(p.exit_date)
                total_days += (exit_dt - entry_dt).days
            except Exception as e:
                logger.debug(f"Failed to parse dates for position: {e}")

        return round(total_days / len(closed_positions), 1) if closed_positions else 0.0

    def _calc_trades_per_year(
        self,
        trades: List[Trade],
        trading_days: int
    ) -> float:
        """计算年均交易次数"""
        if not trades or trading_days <= 0:
            return 0.0

        years = trading_days / self.trading_days
        if years <= 0:
            return 0.0

        return round(len(trades) / years, 1)
