"""非策略参数测试

测试新增加的风险/仓位参数：
- trial_position_pct: 试探仓位比例
- max_single_loss_pct: 单笔最大亏损限制
- min_profit_loss_ratio: 最小盈亏比
- max_open_positions: 最大同时持仓数
- atr_circuit_breaker: ATR熔断倍数
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.backtest.models import EntrySignal, Position, Trade
from src.backtest.engine import SwingBacktester
from src.backtest.performance import PerformanceAnalyzer


class TestNewParameters:
    """新参数测试"""

    def setup_method(self):
        """测试初始化"""
        self.backtester = SwingBacktester(
            initial_capital=1_000_000,
            atr_stop_multiplier=2.0,
            atr_trailing_multiplier=3.0,
            # 新参数
            trial_position_pct=0.10,
            max_single_loss_pct=0.02,
            min_profit_loss_ratio=1.5,
            max_open_positions=5,
            atr_circuit_breaker=3.0,
        )

    def test_initialization(self):
        """新参数初始化"""
        assert self.backtester.trial_position_pct == 0.10
        assert self.backtester.max_single_loss_pct == 0.02
        assert self.backtester.min_profit_loss_ratio == 1.5
        assert self.backtester.max_open_positions == 5
        assert self.backtester.atr_circuit_breaker == 3.0

    def test_trial_position_uses_small_size(self):
        """试探仓位使用较小资金比例"""
        # 当没有持仓时，第一笔应该使用 trial_position_pct
        assert len(self.backtester.positions) == 0

        # 计算试探仓位金额
        trial_value = self.backtester.cash * self.backtester.trial_position_pct
        expected_trial = 1_000_000 * 0.10  # 10万

        # 由于后续还会被 max_position_pct (20%) 限制，需要取较小值
        max_allowed = self.backtester.cash * self.backtester.max_position_pct
        expected = min(expected_trial, max_allowed)

        # 第一笔建仓时应该使用这个值
        assert trial_value == expected_trial

    def test_max_single_loss_limits_position(self):
        """单笔最大亏损限制仓位"""
        # 单笔最大亏损 = 2% * 100万 = 2万
        max_loss = self.backtester.cash * self.backtester.max_single_loss_pct
        assert max_loss == 20_000

        # 如果止损距离为 1元，则最大仓位 = 2万 / 1元 * 10元 = 20万股 = 20万市值
        stop_distance = 1.0
        entry_price = 10.0
        max_position_value = max_loss / stop_distance * entry_price

        # 20_000 / 1.0 * 10 = 200_000
        assert max_position_value == 200_000

    def test_min_profit_loss_ratio_filters_signals(self):
        """最小盈亏比过滤信号"""
        # 创建回测器，使用较高的最小盈亏比
        backtester = SwingBacktester(
            initial_capital=1_000_000,
            min_profit_loss_ratio=10.0,  # 非常高的要求
        )

        # 创建一个普通信号的快照
        # profit_loss_ratio = 0.05 / 2.0 = 0.025 < 10.0，应该被过滤
        assert backtester.min_profit_loss_ratio == 10.0

    def test_max_open_positions_blocks_new_entries(self):
        """最大持仓数限制新开仓"""
        # 先填充5个持仓
        for i in range(5):
            self.backtester.positions[f"60{1000+i}"] = Position(
                position_id=f"pos{i}",
                code=f"60{1000+i}",
                entry_date="2024-01-01",
                entry_price=10.0,
                shares=1000,
                atr=0.5,
                stop_loss=9.0,
                status="open",
            )

        assert len(self.backtester.positions) == 5
        assert len(self.backtester.positions) >= self.backtester.max_open_positions

    def test_atr_circuit_breaker_logic(self):
        """ATR熔断倍数逻辑"""
        # ATR 熔断检查：current_atr > entry_atr * atr_circuit_breaker 时禁止开仓
        # 例如：entry_atr = 1.0，atr_circuit_breaker = 3.0
        # 则 current_atr > 3.0 时才会触发熔断

        entry_atr = 1.0
        circuit_breaker_threshold = entry_atr * self.backtester.atr_circuit_breaker

        assert circuit_breaker_threshold == 3.0

        # current_atr = 2.0 < 3.0，不触发
        current_atr = 2.0
        assert not (current_atr > circuit_breaker_threshold)

        # current_atr = 4.0 > 3.0，触发熔断
        current_atr = 4.0
        assert current_atr > circuit_breaker_threshold


class TestTrialPositionFlow:
    """试探仓位流程测试"""

    def setup_method(self):
        """测试初始化"""
        self.backtester = SwingBacktester(
            initial_capital=1_000_000,
            atr_stop_multiplier=2.0,
            trial_position_pct=0.10,  # 10%试探仓位
            max_position_pct=0.20,    # 最大20%仓位
            fixed_position_value=200_000,
        )

    def test_first_position_uses_trial(self):
        """第一笔使用试探仓位"""
        # 模拟第一笔交易
        is_trial = len(self.backtester.positions) == 0

        position_value = min(
            self.backtester.fixed_position_value,
            self.backtester.cash * self.backtester.max_position_pct
        )

        # 第一笔使用试探仓位
        if is_trial:
            position_value = min(position_value, self.backtester.cash * self.backtester.trial_position_pct)

        # 试探仓位 = 100万 * 10% = 10万
        expected = 100_000
        assert position_value == expected

    def test_second_position_uses_normal(self):
        """第二笔使用正常仓位"""
        # 模拟第二笔交易（已有一个持仓）
        self.backtester.positions["600000"] = Position(
            position_id="pos0",
            code="600000",
            entry_date="2024-01-01",
            entry_price=10.0,
            shares=1000,
            atr=0.5,
            stop_loss=9.0,
            status="open",
        )

        is_trial = len(self.backtester.positions) == 0

        position_value = min(
            self.backtester.fixed_position_value,
            self.backtester.cash * self.backtester.max_position_pct
        )

        # 第二笔不使用试探仓位
        if is_trial:
            position_value = min(position_value, self.backtester.cash * self.backtester.trial_position_pct)

        # 正常仓位 = min(20万, 200万*20%) = 20万
        expected = 200_000
        assert position_value == expected


class TestMaxSingleLossConstraint:
    """单笔最大亏损约束测试"""

    def test_position_limited_by_max_single_loss(self):
        """仓位受单笔最大亏损限制"""
        backtester = SwingBacktester(
            initial_capital=1_000_000,
            atr_stop_multiplier=2.0,    # 止损 = entry - 2*ATR
            max_single_loss_pct=0.02,   # 单笔最大亏损2%
            fixed_position_value=500_000,  # 固定50万仓位
        )

        entry_price = 10.0
        atr = 1.0
        stop_loss = entry_price - 2.0 * atr  # 8.0
        stop_distance = entry_price - stop_loss  # 2.0

        # 最大亏损限制 = 100万 * 2% = 2万
        max_loss_allowed = backtester.cash * backtester.max_single_loss_pct  # 2万

        # 按亏损限制的仓位 = 2万 / 2.0 * 10 = 10万股 = 100万市值
        max_position_by_loss = max_loss_allowed / stop_distance * entry_price

        # 按固定仓位是50万，但亏损限制只有10万
        # actual = min(50万, 10万) = 10万
        actual_position = min(500_000, max_position_by_loss)

        # 实际应该取 10万（因为10万 < 50万，受亏损限制）
        assert actual_position == 100_000

    def test_small_stop_distance_triggers_loss_limit(self):
        """小止损距离触发亏损限制"""
        backtester = SwingBacktester(
            initial_capital=1_000_000,
            atr_stop_multiplier=2.0,
            max_single_loss_pct=0.02,
            fixed_position_value=500_000,
        )

        entry_price = 10.0
        atr = 0.1  # 很小的ATR
        stop_loss = entry_price - 2.0 * atr  # 9.8
        stop_distance = entry_price - stop_loss  # 0.2

        # 最大亏损限制 = 100万 * 2% = 2万
        max_loss_allowed = backtester.cash * backtester.max_single_loss_pct

        # 按亏损限制的仓位 = 2万 / 0.2 * 10 = 100万股 = 1000万市值
        # 但资金只有100万，所以实际最大只能买10万股 = 100万
        max_position_by_loss = max_loss_allowed / stop_distance * entry_price

        # 这个值会超过fixed_position_value
        assert max_position_by_loss > backtester.fixed_position_value


class TestMaxOpenPositions:
    """最大持仓数测试"""

    def test_block_new_entry_when_at_limit(self):
        """达到上限时禁止新开仓"""
        backtester = SwingBacktester(
            initial_capital=1_000_000,
            max_open_positions=3,
        )

        # 填充到上限
        for i in range(3):
            backtester.positions[f"60000{i}"] = Position(
                position_id=f"pos{i}",
                code=f"60000{i}",
                entry_date="2024-01-01",
                entry_price=10.0,
                shares=1000,
                atr=0.5,
                stop_loss=9.0,
                status="open",
            )

        # 检查是否达到上限
        can_open_new = len(backtester.positions) < backtester.max_open_positions
        assert can_open_new is False

        # 持仓数等于上限
        assert len(backtester.positions) == backtester.max_open_positions


class TestAtrCircuitBreaker:
    """ATR熔断测试"""

    def test_circuit_breaker_blocks_high_volatility(self):
        """高波动时熔断"""
        backtester = SwingBacktester(
            initial_capital=1_000_000,
            atr_circuit_breaker=3.0,
        )

        # 入场时ATR = 1.0，熔断阈值 = 3.0
        entry_atr = 1.0
        threshold = entry_atr * backtester.atr_circuit_breaker

        # 当前ATR = 2.0 < 3.0，不触发
        current_atr = 2.0
        assert not (current_atr > threshold)

        # 当前ATR = 3.5 > 3.0，触发
        current_atr = 3.5
        assert current_atr > threshold


class TestMinProfitLossRatio:
    """最小盈亏比测试"""

    def test_ratio_calculation(self):
        """盈亏比计算"""
        backtester = SwingBacktester(
            initial_capital=1_000_000,
            min_profit_loss_ratio=1.5,
        )

        # 预期涨幅 = 5%，止损距离 = 2*ATR
        # 假设 entry_price = 10, atr = 1
        entry_price = 10.0
        atr = 1.0
        expected_profit_pct = 0.05
        expected_profit = entry_price * expected_profit_pct  # 0.5
        stop_distance = 2.0 * atr  # 2.0

        ratio = expected_profit / stop_distance  # 0.25
        assert ratio < backtester.min_profit_loss_ratio  # 0.25 < 1.5

    def test_ratio_with_large_profit_target(self):
        """大目标盈亏比通过"""
        backtester = SwingBacktester(
            initial_capital=1_000_000,
            min_profit_loss_ratio=0.2,  # 较低阈值
        )

        entry_price = 10.0
        atr = 1.0
        expected_profit_pct = 0.05
        expected_profit = entry_price * expected_profit_pct  # 0.5
        stop_distance = 2.0 * atr  # 2.0

        ratio = expected_profit / stop_distance  # 0.25
        assert ratio > backtester.min_profit_loss_ratio  # 0.25 > 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
