"""回测框架模块测试"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtest.models import (
    Trade, Position, BacktestResult, EquityRecord,
    EntrySignal, ExitSignal, MatchResult, generate_id
)
from src.backtest.matching import OrderMatcher, calculate_slippage, check_limit_hit
from src.backtest.performance import PerformanceAnalyzer
from src.backtest.engine import SwingBacktester


class TestModels:
    """数据模型测试"""

    def test_generate_id(self):
        """ID生成测试"""
        id1 = generate_id()
        id2 = generate_id()
        assert len(id1) == 8
        assert len(id2) == 8
        assert id1 != id2

    def test_trade_pnl(self):
        """Trade盈亏计算"""
        trade = Trade(
            trade_id="test001",
            date="2024-01-10",
            code="600519",
            entry_price=100.0,
            exit_price=110.0,
            shares=1000
        )
        assert trade.pnl == 10000.0  # (110-100)*1000
        assert trade.pnl_pct == 0.10  # 10%

    def test_position(self):
        """Position测试"""
        position = Position(
            position_id="pos001",
            code="600519",
            entry_date="2024-01-01",
            entry_price=100.0,
            shares=1000,
            atr=2.0,
            stop_loss=96.0,
            trailing_stop=96.0,
            highest_price=100.0,
            current_price=100.0,  # 当前价格等于入场价格
        )
        assert position.position_id == "pos001"
        assert position.market_value == 100000.0  # current_price * shares
        assert position.unrealized_pnl == 0.0      # 未盈利（current_price == entry_price）
        assert position.unrealized_pnl_pct == 0.0  # 盈亏比例 0%
        assert position.status == "open"

    def test_position_profit(self):
        """持仓盈利测试"""
        position = Position(
            position_id="pos002",
            code="600519",
            entry_date="2024-01-01",
            entry_price=100.0,
            shares=1000,
            atr=2.0,
            current_price=110.0,  # 上涨到110
        )
        assert position.market_value == 110000.0  # 110 * 1000
        assert position.unrealized_pnl == 10000.0  # (110-100) * 1000
        assert abs(position.unrealized_pnl_pct - 0.10) < 0.001  # 10% 盈利

    def test_position_loss(self):
        """持仓亏损测试"""
        position = Position(
            position_id="pos003",
            code="600519",
            entry_date="2024-01-01",
            entry_price=100.0,
            shares=1000,
            atr=2.0,
            current_price=90.0,  # 下跌到90
        )
        assert position.market_value == 90000.0   # 90 * 1000
        assert position.unrealized_pnl == -10000.0  # (90-100) * 1000
        assert abs(position.unrealized_pnl_pct - (-0.10)) < 0.001  # -10% 亏损


class TestMatching:
    """撮合引擎测试"""

    def setup_method(self):
        """测试初始化"""
        self.matcher = OrderMatcher()

    def test_check_limit_up(self):
        """涨停检测"""
        # 涨幅10%，触及涨停
        assert check_limit_hit(11.0, 10.0, 0.10, -0.10) == (True, "limit_up")

    def test_check_limit_down(self):
        """跌停检测"""
        # 跌幅10%，触及跌停
        assert check_limit_hit(9.0, 10.0, 0.10, -0.10) == (True, "limit_down")

    def test_check_no_limit(self):
        """正常价格"""
        # 涨幅5%，未触及涨跌停
        assert check_limit_hit(10.5, 10.0, 0.10, -0.10) == (False, None)

    def test_calculate_slippage_base(self):
        """基准滑点计算"""
        slippage, reason = calculate_slippage(
            target_price=10.0,
            open_price=10.1,
            turnover=10000.0,
            avg_daily_turnover=1000000.0  # 成交额占比 1%
        )
        assert slippage == 0.001
        assert reason == "base"

    def test_calculate_slippage_liquidity(self):
        """流动性折价滑点"""
        slippage, reason = calculate_slippage(
            target_price=10.0,
            open_price=10.1,
            turnover=50000.0,  # 成交额占比 5%
            avg_daily_turnover=1000000.0
        )
        assert slippage > 0.001
        assert "liquidity" in reason

    def test_match_buy_normal(self):
        """正常买入撮合"""
        df_next = pd.DataFrame({
            'date': ['2024-01-02'],
            'open': [100.0],
            'close': [101.0],
            'high': [102.0],
            'low': [99.0],
            'volume': [1000000]
        })

        result = self.matcher.match_buy(
            signal_date="2024-01-01",
            code="600519",
            target_price=100.0,
            df_next=df_next,
            shares=1000,
            avg_daily_turnover=1000000.0
        )

        assert result.success == True
        assert result.match_price > 0
        assert result.commission > 0

    def test_match_buy_limit_up(self):
        """涨停无法买入"""
        # open=11.0 涨停（前一交易日close=10.0）
        # match_buy 内部用 df_next["close"] 作为 prev_close，所以这里 close=10.0
        df_next = pd.DataFrame({
            'date': ['2024-01-02'],
            'open': [11.0],  # 开盘涨停价
            'close': [10.0],  # 前收盘价（作为 prev_close）
            'high': [11.0],
            'low': [11.0],
            'volume': [100000]
        })

        result = self.matcher.match_buy(
            signal_date="2024-01-01",
            code="600519",
            target_price=10.0,
            df_next=df_next,
            shares=1000,
            avg_daily_turnover=1000000.0
        )

        assert result.success == False
        assert result.limit_hit == True
        assert result.limit_type == "limit_up"

    def test_match_sell_normal(self):
        """正常卖出撮合"""
        df_next = pd.DataFrame({
            'date': ['2024-01-02'],
            'open': [100.0],
            'close': [99.0],
            'high': [101.0],
            'low': [98.0],
            'volume': [1000000]
        })

        result = self.matcher.match_sell(
            signal_date="2024-01-01",
            code="600519",
            target_price=100.0,
            df_next=df_next,
            shares=1000,
            avg_daily_turnover=1000000.0
        )

        assert result.success == True
        assert result.match_price > 0


class TestPerformance:
    """绩效分析测试"""

    def setup_method(self):
        """测试初始化"""
        self.analyzer = PerformanceAnalyzer()

    def test_win_rate(self):
        """胜率计算"""
        trades = [
            Trade(entry_price=100, exit_price=110, shares=100),  # 盈利
            Trade(entry_price=100, exit_price=90, shares=100),   # 亏损
            Trade(entry_price=100, exit_price=105, shares=100),  # 盈利
        ]

        win_rate = self.analyzer._calc_win_rate(trades)
        assert win_rate == pytest.approx(2/3, rel=0.01)

    def test_profit_factor(self):
        """盈亏比计算"""
        trades = [
            Trade(entry_price=100, exit_price=110, shares=100),   # 盈利 +1000
            Trade(entry_price=100, exit_price=90, shares=100),    # 亏损 -1000
            Trade(entry_price=100, exit_price=120, shares=100),   # 盈利 +2000
        ]

        profit_factor = self.analyzer._calc_profit_factor(trades)
        assert profit_factor == 3.0  # (1000+2000)/1000 = 3

    def test_avg_holding_days(self):
        """平均持仓天数计算"""
        positions = [
            Position(
                entry_date="2024-01-01",
                exit_date="2024-01-21",
                status="closed"
            ),
            Position(
                entry_date="2024-02-01",
                exit_date="2024-02-11",
                status="closed"
            ),
        ]

        avg_days = self.analyzer._calc_avg_holding_days(positions)
        assert avg_days == 15.0  # (20+10)/2 = 15

    def test_max_drawdown(self):
        """最大回撤计算"""
        equity_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
            'equity': [100000, 110000, 105000, 115000, 120000],
            'cash': [100000, 110000, 105000, 115000, 120000],
            'market_value': [0, 0, 0, 0, 0],
            'daily_return': [0, 0.1, -0.045, 0.095, 0.043]
        })

        max_dd, duration = self.analyzer._calc_max_drawdown(equity_df)
        # 从 110000 跌到 105000，回撤约 4.5%
        assert max_dd > 0
        assert max_dd < 0.1  # < 10%

    def test_sharpe_ratio(self):
        """夏普比率计算"""
        equity_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
            'equity': [100000, 101000, 102000, 103000, 104000],
            'cash': [100000, 101000, 102000, 103000, 104000],
            'market_value': [0, 0, 0, 0, 0],
            'daily_return': [0, 0.01, 0.01, 0.01, 0.01]
        })

        sharpe = self.analyzer._calc_sharpe_ratio(equity_df)
        assert sharpe > 0  # 正收益

    def test_full_performance_analysis(self):
        """完整绩效分析"""
        equity_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
            'equity': [100000, 105000, 110000, 105000, 115000],
            'cash': [100000, 105000, 110000, 105000, 115000],
            'market_value': [0, 0, 0, 0, 0],
            'daily_return': [0, 0.05, 0.0476, -0.0455, 0.0952]
        })

        trades = [
            Trade(entry_price=100, exit_price=105, shares=1000),
        ]
        positions = [
            Position(entry_date="2024-01-01", exit_date="2024-01-03", status="closed"),
        ]

        metrics = self.analyzer.analyze(
            equity_df,
            trades,
            positions,
            initial_capital=100000
        )

        assert metrics.total_return > 0
        assert metrics.final_capital == 115000
        assert metrics.sharpe_ratio >= 0


class TestSwingBacktester:
    """回测引擎测试"""

    def setup_method(self):
        """测试初始化"""
        self.backtester = SwingBacktester(
            initial_capital=1_000_000,
            atr_stop_multiplier=2.0,
            atr_trailing_multiplier=3.0,
        )

    def test_initialization(self):
        """回测器初始化"""
        assert self.backtester.initial_capital == 1_000_000
        assert self.backtester.cash == 1_000_000
        assert self.backtester.atr_stop_multiplier == 2.0

    def test_empty_result(self):
        """空回测结果"""
        result = self.backtester._generate_empty_result("2024-01-01", "2024-12-31")
        assert result.final_capital == 1_000_000
        assert result.total_trades == 0


class TestBacktestResult:
    """回测结果测试"""

    def test_summary(self):
        """结果摘要"""
        result = BacktestResult(
            start_date="2024-01-01",
            end_date="2024-12-31",
            initial_capital=1_000_000,
            final_capital=1_150_000,
            total_trades=10,
            win_rate=0.6,
            profit_factor=2.0,
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            calmar_ratio=2.0,
            avg_holding_days=20,
            trades_per_year=5,
        )

        summary = result.summary()
        assert "2024-01-01" in summary
        assert "1,150,000" in summary
        assert "1.50" in summary  # sharpe


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
