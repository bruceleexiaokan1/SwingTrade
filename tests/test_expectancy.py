"""Expectancy Tests"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.backtest.expectancy import (
    calculate_expectancy,
    filter_by_expectancy,
    calculate_expectancy_from_stats,
    is_viable_strategy,
    ExpectancyResult,
)
from src.backtest.models import Trade, EntrySignal


class TestExpectancyCalculation:
    """正期望值计算测试"""

    def test_positive_expectancy(self):
        """正期望计算 - 盈利场景"""
        trades = [
            Trade(entry_price=100, exit_price=110, shares=1000),   # +10000
            Trade(entry_price=100, exit_price=105, shares=1000),    # +5000
            Trade(entry_price=100, exit_price=90, shares=1000),     # -10000
        ]

        result = calculate_expectancy(trades)

        assert result.total_trades == 3
        assert result.winning_trades == 2
        assert result.losing_trades == 1
        assert result.win_rate == pytest.approx(2/3, rel=0.01)
        assert result.avg_win == 7500.0  # (10000+5000)/2
        assert result.avg_loss == 10000.0
        assert result.profit_loss_ratio == 0.75  # 7500/10000 = 0.75:1

    def test_negative_expectancy(self):
        """负期望计算 - 亏损场景"""
        trades = [
            Trade(entry_price=100, exit_price=90, shares=1000),     # -10000
            Trade(entry_price=100, exit_price=80, shares=1000),    # -20000
            Trade(entry_price=100, exit_price=110, shares=1000),   # +10000
        ]

        result = calculate_expectancy(trades)

        assert result.total_trades == 3
        assert result.winning_trades == 1
        assert result.losing_trades == 2
        assert result.win_rate == pytest.approx(1/3, rel=0.01)
        assert result.avg_win == 10000.0
        assert result.avg_loss == 15000.0  # (10000+20000)/2
        assert result.profit_loss_ratio == pytest.approx(0.67, rel=0.01)  # 10000/15000

    def test_empty_trades(self):
        """空交易列表"""
        result = calculate_expectancy([])

        assert result.expectancy == 0.0
        assert result.win_rate == 0.0
        assert result.total_trades == 0

    def test_all_winning_trades(self):
        """全盈利交易"""
        trades = [
            Trade(entry_price=100, exit_price=110, shares=1000),
            Trade(entry_price=100, exit_price=120, shares=1000),
        ]

        result = calculate_expectancy(trades)

        assert result.win_rate == 1.0
        assert result.losing_trades == 0
        assert result.avg_loss == 0.0
        assert result.profit_loss_ratio == 0.0  # 无法计算（无亏损）

    def test_all_losing_trades(self):
        """全亏损交易"""
        trades = [
            Trade(entry_price=100, exit_price=90, shares=1000),
            Trade(entry_price=100, exit_price=80, shares=1000),
        ]

        result = calculate_expectancy(trades)

        assert result.win_rate == 0.0
        assert result.winning_trades == 0
        assert result.avg_win == 0.0
        assert result.profit_loss_ratio == 0.0  # 无法计算（无盈利）

    def test_commission_cost(self):
        """包含佣金成本的计算"""
        # 佣金率 0.03%，印花税 0.01%
        trades = [
            Trade(entry_price=100, exit_price=110, shares=1000),
        ]

        result = calculate_expectancy(trades, commission_rate=0.0003, stamp_tax=0.0001)

        # 买入佣金: 100 * 1000 * 0.0003 = 30
        # 卖出佣金: 110 * 1000 * 0.0003 = 33
        # 印花税: 110 * 1000 * 0.0001 = 11
        # 总成本: 30 + 33 + 11 = 74
        assert result.total_cost == pytest.approx(74.0, rel=1e-9)


class TestExpectancyResult:
    """ExpectancyResult 测试"""

    def test_is_positive(self):
        """正期望判断"""
        positive = ExpectancyResult(
            expectancy=1000.0,
            win_rate=0.4,
            avg_win=5000.0,
            avg_loss=3000.0,
            profit_loss_ratio=1.67,
            total_trades=10,
            winning_trades=4,
            losing_trades=6,
            total_cost=500.0,
        )
        assert positive.is_positive == True

        negative = ExpectancyResult(
            expectancy=-500.0,
            win_rate=0.4,
            avg_win=3000.0,
            avg_loss=5000.0,
            profit_loss_ratio=0.6,
            total_trades=10,
            winning_trades=4,
            losing_trades=6,
            total_cost=500.0,
        )
        assert negative.is_positive == False

    def test_passes_filter(self):
        """盈亏比过滤器判断"""
        # 盈亏比 3:1 - 通过
        good = ExpectancyResult(
            expectancy=1000.0,
            win_rate=0.35,
            avg_win=6000.0,
            avg_loss=2000.0,
            profit_loss_ratio=3.0,
            total_trades=10,
            winning_trades=3,
            losing_trades=7,
            total_cost=500.0,
        )
        assert good.passes_filter == True

        # 盈亏比 2:1 - 不通过
        bad = ExpectancyResult(
            expectancy=-100.0,
            win_rate=0.4,
            avg_win=4000.0,
            avg_loss=2000.0,
            profit_loss_ratio=2.0,
            total_trades=10,
            winning_trades=4,
            losing_trades=6,
            total_cost=500.0,
        )
        assert bad.passes_filter == False

    def test_summary(self):
        """摘要生成测试"""
        result = ExpectancyResult(
            expectancy=1500.0,
            win_rate=0.35,
            avg_win=6000.0,
            avg_loss=2000.0,
            profit_loss_ratio=3.0,
            total_trades=20,
            winning_trades=7,
            losing_trades=13,
            total_cost=800.0,
        )

        summary = result.summary()
        assert "20" in summary
        assert "7" in summary
        assert "3.0" in summary


class TestFilterByExpectancy:
    """正期望过滤测试"""

    def test_passes_filter_with_good_history(self):
        """历史数据良好 - 不过滤"""
        signals = [
            EntrySignal(code="600519", signal_type="golden", confidence=0.8),
            EntrySignal(code="000858", signal_type="breakout", confidence=0.7),
        ]
        trades = [
            Trade(entry_price=100, exit_price=110, shares=1000),  # 盈利
            Trade(entry_price=100, exit_price=105, shares=1000),  # 盈利
            Trade(entry_price=100, exit_price=90, shares=1000),   # 亏损
            Trade(entry_price=100, exit_price=115, shares=1000),  # 盈利
            Trade(entry_price=100, exit_price=90, shares=1000),   # 亏损
            Trade(entry_price=100, exit_price=120, shares=1000),  # 盈利
            Trade(entry_price=100, exit_price=85, shares=1000),   # 亏损
            Trade(entry_price=100, exit_price=110, shares=1000),  # 盈利
            Trade(entry_price=100, exit_price=90, shares=1000),   # 亏损
            Trade(entry_price=100, exit_price=105, shares=1000),  # 盈利
        ]

        result = filter_by_expectancy(signals, trades, min_ratio=3.0)

        # 盈亏比约 7000/3500 = 2:1 < 3:1，应该被过滤
        assert len(result) == 0

    def test_insufficient_history(self):
        """历史数据不足 - 保守返回空"""
        signals = [
            EntrySignal(code="600519", signal_type="golden", confidence=0.8),
        ]
        trades = [
            Trade(entry_price=100, exit_price=110, shares=1000),
        ]

        result = filter_by_expectancy(signals, trades)
        assert len(result) == 0


class TestCalculateExpectancyFromStats:
    """从统计数据计算正期望测试"""

    def test_from_stats(self):
        """从统计数据计算"""
        result = calculate_expectancy_from_stats(
            win_rate=0.35,
            avg_win=6000.0,
            avg_loss=2000.0,
            total_trades=100,
        )

        assert result.win_rate == 0.35
        assert result.avg_win == 6000.0
        assert result.avg_loss == 2000.0
        assert result.profit_loss_ratio == 3.0
        assert result.total_trades == 100
        assert result.winning_trades == 35
        assert result.losing_trades == 65

    def test_expectancy_formula(self):
        """期望公式验证"""
        # E = P_win × Avg_Win - P_loss × Avg_Loss - Cost
        result = calculate_expectancy_from_stats(
            win_rate=0.4,
            avg_win=5000.0,
            avg_loss=2500.0,
            total_trades=100,
            avg_commission_pct=0.0,  # 忽略成本
        )

        # E = 0.4 × 5000 - 0.6 × 2500 = 2000 - 1500 = 500
        assert result.expectancy == 500.0


class TestIsViableStrategy:
    """策略可行性判断测试"""

    def test_viable_strategy(self):
        """可行策略"""
        result = ExpectancyResult(
            expectancy=1000.0,
            win_rate=0.35,
            avg_win=6000.0,
            avg_loss=2000.0,
            profit_loss_ratio=3.0,
            total_trades=100,
            winning_trades=35,
            losing_trades=65,
            total_cost=500.0,
        )

        assert is_viable_strategy(result) == True

    def test_low_win_rate(self):
        """低胜率策略不可行"""
        result = ExpectancyResult(
            expectancy=500.0,
            win_rate=0.20,  # < 25%
            avg_win=6000.0,
            avg_loss=2000.0,
            profit_loss_ratio=3.0,
            total_trades=100,
            winning_trades=20,
            losing_trades=80,
            total_cost=500.0,
        )

        assert is_viable_strategy(result) == False

    def test_low_ratio(self):
        """低盈亏比策略不可行"""
        result = ExpectancyResult(
            expectancy=100.0,
            win_rate=0.40,
            avg_win=4000.0,  # 盈亏比 2:1
            avg_loss=2000.0,
            profit_loss_ratio=2.0,  # < 3:1
            total_trades=100,
            winning_trades=40,
            losing_trades=60,
            total_cost=500.0,
        )

        assert is_viable_strategy(result) == False

    def test_negative_expectancy(self):
        """负期望不可行"""
        result = ExpectancyResult(
            expectancy=-500.0,
            win_rate=0.35,
            avg_win=5000.0,
            avg_loss=3000.0,
            profit_loss_ratio=1.67,
            total_trades=100,
            winning_trades=35,
            losing_trades=65,
            total_cost=500.0,
        )

        assert is_viable_strategy(result) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
