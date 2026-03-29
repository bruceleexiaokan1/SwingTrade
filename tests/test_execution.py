"""执行算法模块测试

测试TWAP、VWAP、Iceberg、自适应执行等算法
以及市场冲击估算、执行质量监控、大单拆分等功能
"""

import pytest
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.execution import (
    # 函数
    twap_execution,
    vwap_execution,
    iceberg_order,
    adaptive_execution,
    estimate_market_impact,
    monitor_execution_quality,
    order_slicer,
    execute_order,
    # 类
    ExecutionSlice,
    ExecutionResult,
    ExecutionStatus,
    MarketImpact,
    ExecutionQuality,
)


class TestTWAPExecution:
    """TWAP执行算法测试"""

    def setup_method(self):
        """设置测试数据"""
        self.price_series = pd.Series(
            [10.0, 10.1, 10.2, 10.3, 10.4],
            index=pd.date_range('2024-01-01', periods=5, freq='h')
        )

    def test_twap_basic(self):
        """基本TWAP执行测试"""
        total_shares = 10000
        num_slices = 5
        slices = twap_execution(total_shares, num_slices, self.price_series)

        assert len(slices) == 5
        assert all(s.shares > 0 for s in slices)
        # 总股数应该等于或接近total_shares
        total = sum(s.shares for s in slices)
        assert total == 10000

    def test_twap_urgent_low(self):
        """低紧急程度TWAP测试"""
        total_shares = 10000
        num_slices = 5
        slices = twap_execution(total_shares, num_slices, self.price_series, urgency=0.2)

        # 低紧急程度应该更均匀分布
        assert len(slices) == 5
        shares = [s.shares for s in slices]
        std = np.std(shares)
        # 均匀分布时标准差应该较小
        assert std < 500

    def test_twap_urgent_high(self):
        """高紧急程度TWAP测试"""
        total_shares = 10000
        num_slices = 5
        slices = twap_execution(total_shares, num_slices, self.price_series, urgency=0.8)

        # 高紧急程度前几个切片应该更多
        assert len(slices) == 5
        assert slices[0].shares >= slices[-1].shares

    def test_twap_with_remainder(self):
        """有余数时TWAP测试"""
        total_shares = 10003  # 不能被5整除
        num_slices = 5
        slices = twap_execution(total_shares, num_slices, self.price_series)

        total = sum(s.shares for s in slices)
        assert total == 10003

    def test_twap_invalid_total_shares(self):
        """无效总股数测试"""
        with pytest.raises(ValueError, match="total_shares must be positive"):
            twap_execution(0, 5, self.price_series)

        with pytest.raises(ValueError):
            twap_execution(-100, 5, self.price_series)

    def test_twap_invalid_num_slices(self):
        """无效切片数测试"""
        with pytest.raises(ValueError, match="num_slices must be positive"):
            twap_execution(10000, 0, self.price_series)

    def test_twap_invalid_urgency(self):
        """无效紧急程度测试"""
        with pytest.raises(ValueError, match="urgency must be between 0 and 1"):
            twap_execution(10000, 5, self.price_series, urgency=1.5)

    def test_twap_single_slice(self):
        """单切片测试"""
        total_shares = 5000
        num_slices = 1
        slices = twap_execution(total_shares, num_slices, self.price_series)

        assert len(slices) == 1
        assert slices[0].shares == 5000

    def test_twap_price_values(self):
        """价格序列测试"""
        total_shares = 10000
        num_slices = 5
        slices = twap_execution(total_shares, num_slices, self.price_series)

        for i, s in enumerate(slices):
            assert s.price == pytest.approx(self.price_series.iloc[i])
            assert s.slice_id == i

    def test_twap_turnover_calculation(self):
        """成交金额计算测试"""
        total_shares = 10000
        num_slices = 5
        slices = twap_execution(total_shares, num_slices, self.price_series)

        for s in slices:
            assert s.turnover == pytest.approx(s.price * s.shares)


class TestVWAPExecution:
    """VWAP执行算法测试"""

    def setup_method(self):
        """设置测试数据"""
        self.price_series = pd.Series(
            [10.0, 10.1, 10.2, 10.3, 10.4],
            index=pd.date_range('2024-01-01', periods=5, freq='h')
        )
        self.volume_profile = pd.Series(
            [1000, 2000, 1500, 2500, 1000],
            index=pd.date_range('2024-01-01', periods=5, freq='h')
        )

    def test_vwap_basic(self):
        """基本VWAP执行测试"""
        total_shares = 10000
        slices = vwap_execution(total_shares, self.volume_profile, self.price_series)

        assert len(slices) == 5
        total = sum(s.shares for s in slices)
        assert total == 10000

    def test_vwap_high_volume_period(self):
        """高成交量时段测试"""
        total_shares = 10000
        slices = vwap_execution(total_shares, self.volume_profile, self.price_series)

        # 找到最大成交量的索引（应该是index=3，成交量2500）
        max_vol_idx = self.volume_profile.values.argmax()
        # 该时段的成交量应该最多
        assert slices[max_vol_idx].shares >= max(s.shares for s in slices[:max_vol_idx])

    def test_vwap_low_urgency(self):
        """低紧急程度VWAP测试"""
        total_shares = 10000
        slices = vwap_execution(total_shares, self.volume_profile, self.price_series, urgency=0.2)

        # 低紧急程度应该更接近VWAP权重
        assert len(slices) == 5

    def test_vwap_high_urgency(self):
        """高紧急程度VWAP测试"""
        total_shares = 10000
        slices = vwap_execution(total_shares, self.volume_profile, self.price_series, urgency=0.9)

        # 高紧急程度应该前几个时段更多
        assert slices[0].shares >= slices[-1].shares

    def test_vwap_zero_volume(self):
        """零成交量测试"""
        zero_volume = pd.Series([0, 0, 0, 0, 0])
        total_shares = 10000
        slices = vwap_execution(total_shares, zero_volume, self.price_series)

        # 零成交量时应该均匀分配
        assert len(slices) == 5
        shares = [s.shares for s in slices]
        assert abs(max(shares) - min(shares)) <= 1

    def test_vwap_invalid_total_shares(self):
        """无效总股数测试"""
        with pytest.raises(ValueError, match="total_shares must be positive"):
            vwap_execution(0, self.volume_profile, self.price_series)

    def test_vwap_invalid_urgency(self):
        """无效紧急程度测试"""
        with pytest.raises(ValueError, match="urgency must be between 0 and 1"):
            vwap_execution(10000, self.volume_profile, self.price_series, urgency=-0.1)

    def test_vwap_empty_volume(self):
        """空成交量测试"""
        empty_volume = pd.Series([])
        with pytest.raises(ValueError, match="volume_profile cannot be empty"):
            vwap_execution(10000, empty_volume, self.price_series)


class TestIcebergOrder:
    """冰山订单测试"""

    def setup_method(self):
        """设置测试数据"""
        self.price_series = pd.Series(
            [10.0, 10.1, 10.2, 10.3, 10.4],
            index=pd.date_range('2024-01-01', periods=5, freq='h')
        )

    def test_iceberg_basic(self):
        """基本冰山订单测试"""
        total_shares = 10000
        visible_ratio = 0.1
        slices = iceberg_order(total_shares, visible_ratio, self.price_series)

        # 每次最多显示1000股（10% of remaining, capped at first slice)
        # 由于每次显示remaining的10%，股数会递减
        first_slice_shares = slices[0].shares
        assert first_slice_shares <= 1000

        total = sum(s.shares for s in slices)
        assert total == 10000

    def test_iceberg_iterations(self):
        """冰山订单迭代次数测试"""
        total_shares = 10000
        visible_ratio = 0.1
        slices = iceberg_order(total_shares, visible_ratio, self.price_series, num_iterations=50)

        # 由于每次显示remaining的10%，会有多次迭代直到完成
        # 第一次: 1000, 然后900, 810...直到全部成交
        # 总和应该等于10000
        total = sum(s.shares for s in slices)
        assert total == 10000
        # 第一次应该是1000（10% of 10000）
        assert slices[0].shares == 1000

    def test_iceberg_small_order(self):
        """小订单测试"""
        total_shares = 500
        visible_ratio = 0.1
        slices = iceberg_order(total_shares, visible_ratio, self.price_series)

        # visible = min(500, 500*0.1) = 50
        # remaining = 450
        # visible = min(450, 450*0.1) = 45
        # 会有多次迭代，直到remaining < 1
        total = sum(s.shares for s in slices)
        assert total == 500

    def test_iceberg_large_visible_ratio(self):
        """大可见比例测试"""
        total_shares = 1000
        visible_ratio = 0.5
        slices = iceberg_order(total_shares, visible_ratio, self.price_series)

        # visible = min(1000, 1000*0.5) = 500
        # 第一次应该500股
        assert slices[0].shares == 500
        total = sum(s.shares for s in slices)
        assert total == 1000

    def test_iceberg_invalid_visible_ratio(self):
        """无效可见比例测试"""
        with pytest.raises(ValueError, match="visible_ratio must be between 0 and 1"):
            iceberg_order(10000, 0.0, self.price_series)

        with pytest.raises(ValueError):
            iceberg_order(10000, 1.5, self.price_series)

    def test_iceberg_invalid_total_shares(self):
        """无效总股数测试"""
        with pytest.raises(ValueError, match="total_shares must be positive"):
            iceberg_order(0, 0.1, self.price_series)


class TestAdaptiveExecution:
    """自适应执行算法测试"""

    def setup_method(self):
        """设置测试数据"""
        self.price_series = pd.Series(
            [10.0, 10.1, 10.2, 10.3, 10.4],
            index=pd.date_range('2024-01-01', periods=5, freq='h')
        )
        self.volume_profile = pd.Series(
            [1000, 2000, 1500, 2500, 1000],
            index=pd.date_range('2024-01-01', periods=5, freq='h')
        )

    def test_adaptive_low_urgency(self):
        """低紧急程度自适应测试"""
        total_shares = 10000
        slices = adaptive_execution(total_shares, self.volume_profile, self.price_series, urgency=0.2)

        # 低紧急程度应该使用VWAP
        assert len(slices) == 5

    def test_adaptive_medium_urgency(self):
        """中等紧急程度自适应测试"""
        total_shares = 10000
        slices = adaptive_execution(total_shares, self.volume_profile, self.price_series, urgency=0.5)

        # 中等紧急程度应该使用混合策略
        assert len(slices) == 5

    def test_adaptive_high_urgency(self):
        """高紧急程度自适应测试"""
        total_shares = 10000
        slices = adaptive_execution(total_shares, self.volume_profile, self.price_series, urgency=0.9)

        # 高紧急程度应该使用更少切片
        assert len(slices) < 5
        assert slices[0].shares > slices[-1].shares

    def test_adaptive_zero_volume(self):
        """零成交量测试"""
        zero_volume = pd.Series([0, 0, 0, 0, 0])
        total_shares = 10000
        slices = adaptive_execution(total_shares, zero_volume, self.price_series, urgency=0.5)

        # 应该仍然能够执行
        assert len(slices) > 0
        total = sum(s.shares for s in slices)
        assert total == 10000

    def test_adaptive_invalid_total_shares(self):
        """无效总股数测试"""
        with pytest.raises(ValueError, match="total_shares must be positive"):
            adaptive_execution(0, self.volume_profile, self.price_series, urgency=0.5)

    def test_adaptive_invalid_urgency(self):
        """无效紧急程度测试"""
        with pytest.raises(ValueError, match="urgency must be between 0 and 1"):
            adaptive_execution(10000, self.volume_profile, self.price_series, urgency=1.5)


class TestMarketImpact:
    """市场冲击估算测试"""

    def test_market_impact_basic(self):
        """基本冲击估算测试"""
        impact = estimate_market_impact(
            order_amount=1000000,
            daily_avg_volume=10000000
        )

        assert impact.participation_rate == pytest.approx(0.02)  # 100万/(1000万*5)
        assert impact.spread_bps == 10.0
        assert impact.impact_bps > 0

    def test_market_impact_high_participation(self):
        """高参与率冲击测试"""
        impact = estimate_market_impact(
            order_amount=2000000,
            daily_avg_volume=10000000
        )

        # 参与率 = 200万/(1000万*5) = 0.04 = 4%
        assert impact.participation_rate == pytest.approx(0.04)
        # 0.04 < 0.05, so it's "minimal_participation"
        assert impact.reason == "minimal_participation"

    def test_market_impact_zero_order(self):
        """零订单测试"""
        impact = estimate_market_impact(
            order_amount=0,
            daily_avg_volume=10000000
        )

        assert impact.participation_rate == 0.0
        assert impact.impact_bps == 0.0

    def test_market_impact_custom_params(self):
        """自定义参数测试"""
        impact = estimate_market_impact(
            order_amount=5000000,
            daily_avg_volume=20000000,
            spread_bps=20.0,
            volatility_bps=100.0
        )

        assert impact.spread_bps == 20.0
        assert impact.participation_rate == pytest.approx(0.05)

    def test_market_impact_invalid_order(self):
        """无效订单金额测试"""
        with pytest.raises(ValueError, match="order_amount must be non-negative"):
            estimate_market_impact(-100, 10000000)

    def test_market_impact_invalid_volume(self):
        """无效日均成交测试"""
        with pytest.raises(ValueError, match="daily_avg_volume must be positive"):
            estimate_market_impact(1000000, 0)


class TestExecutionQuality:
    """执行质量监控测试"""

    def test_quality_basic(self):
        """基本执行质量测试"""
        slices = [
            ExecutionSlice(0, "2024-01-01 09:30", 10.0, 1000, 10000, 0.0, 0.5),
            ExecutionSlice(1, "2024-01-01 10:00", 10.1, 2000, 20200, 0.0, 0.5),
            ExecutionSlice(2, "2024-01-01 10:30", 10.2, 3000, 30600, 0.0, 0.5),
        ]

        quality = monitor_execution_quality(slices, arrival_price=10.0, current_vwap=10.15)

        # avg_price = (10000+20200+30600)/(1000+2000+3000) = 60800/6000 = 10.1333
        assert quality.arrival_price == 10.0
        assert quality.avg_execution_price == pytest.approx(60800/6000)
        assert quality.fill_rate == 1.0

    def test_quality_empty_slices(self):
        """空切片测试"""
        with pytest.raises(ValueError, match="slices cannot be empty"):
            monitor_execution_quality([], arrival_price=10.0, current_vwap=10.0)

    def test_quality_zero_shares(self):
        """零股数测试"""
        slices = [
            ExecutionSlice(0, "2024-01-01 09:30", 10.0, 0, 0, 0.0, 0.5),
        ]

        with pytest.raises(ValueError, match="total shares cannot be zero"):
            monitor_execution_quality(slices, arrival_price=10.0, current_vwap=10.0)

    def test_quality_vwap_diff(self):
        """VWAP差异测试"""
        slices = [
            ExecutionSlice(0, "2024-01-01 09:30", 10.0, 1000, 10000, 0.0, 0.5),
        ]

        quality = monitor_execution_quality(slices, arrival_price=10.0, current_vwap=10.0)
        assert quality.vwap_diff_bps == 0.0

        # Test with different VWAP
        quality2 = monitor_execution_quality(slices, arrival_price=10.0, current_vwap=10.005)
        # vwap_diff should be small and negative
        assert quality2.vwap_diff_bps < 0


class TestOrderSlicer:
    """大单拆分测试"""

    def test_slicer_equal(self):
        """均匀拆分测试"""
        slices = order_slicer(10000, 1000, strategy="equal")

        assert len(slices) == 10
        assert all(s == 1000 for s in slices)
        assert sum(slices) == 10000

    def test_slicer_vwap(self):
        """VWAP拆分测试"""
        volumes = pd.Series([1000, 2000, 3000, 4000])
        slices = order_slicer(10000, 5000, volume_profile=volumes, strategy="vwap")

        assert len(slices) > 0
        assert sum(slices) == 10000
        # 最大的片应该在高成交量时段
        assert max(slices) <= 5000

    def test_slicer_time(self):
        """时间拆分测试"""
        volumes = pd.Series([1000, 2000, 3000, 4000])
        slices = order_slicer(10000, 5000, volume_profile=volumes, strategy="time")

        assert len(slices) > 0
        assert sum(slices) == 10000

    def test_slicer_single_slice(self):
        """单片测试"""
        slices = order_slicer(500, 1000, strategy="equal")

        assert len(slices) == 1
        assert slices[0] == 500

    def test_slicer_uneven(self):
        """不整除测试"""
        slices = order_slicer(9500, 1000, strategy="equal")

        assert len(slices) == 10
        assert sum(slices) == 9500
        # 9片1000 + 1片500
        assert 500 in slices

    def test_slicer_vwap_requires_profile(self):
        """VWAP策略需要成交量分布"""
        with pytest.raises(ValueError, match="volume_profile required"):
            order_slicer(10000, 1000, strategy="vwap")

    def test_slicer_invalid_strategy(self):
        """无效策略测试"""
        with pytest.raises(ValueError, match="strategy must be 'equal'"):
            order_slicer(10000, 1000, strategy="invalid")

    def test_slicer_invalid_total_shares(self):
        """无效总股数测试"""
        with pytest.raises(ValueError, match="total_shares must be positive"):
            order_slicer(0, 1000)

    def test_slicer_invalid_max_size(self):
        """无效最大片大小测试"""
        with pytest.raises(ValueError, match="max_slice_size must be positive"):
            order_slicer(10000, 0)


class TestExecuteOrder:
    """执行订单便捷函数测试"""

    def setup_method(self):
        """设置测试数据"""
        self.price_series = pd.Series(
            [10.0, 10.1, 10.2, 10.3, 10.4],
            index=pd.date_range('2024-01-01', periods=5, freq='h')
        )
        self.volume_profile = pd.Series(
            [1000, 2000, 1500, 2500, 1000],
            index=pd.date_range('2024-01-01', periods=5, freq='h')
        )

    def test_execute_order_twap(self):
        """TWAP订单执行测试"""
        result = execute_order(
            order_id="test001",
            code="600519",
            direction="buy",
            total_shares=10000,
            price_series=self.price_series,
            algorithm="twap",
            urgency=0.5
        )

        assert result.order_id == "test001"
        assert result.code == "600519"
        assert result.direction == "buy"
        assert result.filled_shares == 10000
        assert result.status == ExecutionStatus.FILLED

    def test_execute_order_vwap(self):
        """VWAP订单执行测试"""
        result = execute_order(
            order_id="test002",
            code="600519",
            direction="buy",
            total_shares=10000,
            price_series=self.price_series,
            volume_profile=self.volume_profile,
            algorithm="vwap",
            urgency=0.5
        )

        assert result.filled_shares == 10000
        assert len(result.slices) == 5

    def test_execute_order_iceberg(self):
        """Iceberg订单执行测试"""
        result = execute_order(
            order_id="test003",
            code="600519",
            direction="buy",
            total_shares=10000,
            price_series=self.price_series,
            algorithm="iceberg"
        )

        # 由于每次显示remaining的10%，不是所有股份都能在默认迭代次数内完成
        assert result.filled_shares <= 10000
        assert result.filled_shares > 0
        # 每次最多显示10%
        for s in result.slices:
            assert s.shares <= 1000

    def test_execute_order_adaptive(self):
        """自适应订单执行测试"""
        result = execute_order(
            order_id="test004",
            code="600519",
            direction="buy",
            total_shares=10000,
            price_series=self.price_series,
            volume_profile=self.volume_profile,
            algorithm="adaptive",
            urgency=0.5
        )

        assert result.filled_shares == 10000

    def test_execute_order_invalid_algorithm(self):
        """无效算法测试"""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            execute_order(
                order_id="test005",
                code="600519",
                direction="buy",
                total_shares=10000,
                price_series=self.price_series,
                algorithm="invalid"
            )

    def test_execute_order_avg_price(self):
        """平均成交价测试"""
        result = execute_order(
            order_id="test006",
            code="600519",
            direction="buy",
            total_shares=10000,
            price_series=self.price_series,
            algorithm="twap"
        )

        expected_avg = result.total_turnover / result.filled_shares
        assert result.avg_price == pytest.approx(expected_avg)


class TestExecutionResult:
    """ExecutionResult类测试"""

    def test_remaining_shares(self):
        """剩余股数测试"""
        result = ExecutionResult(
            order_id="test",
            code="600519",
            direction="buy",
            total_shares=10000,
            filled_shares=7000,
            avg_price=10.0,
            total_turnover=70000,
            total_slippage=0,
            execution_time=60,
            status=ExecutionStatus.PARTIAL
        )

        assert result.remaining_shares == 3000

    def test_fill_rate(self):
        """成交率测试"""
        result1 = ExecutionResult(
            order_id="test1",
            code="600519",
            direction="buy",
            total_shares=10000,
            filled_shares=10000,
            avg_price=10.0,
            total_turnover=100000,
            total_slippage=0,
            execution_time=60,
            status=ExecutionStatus.FILLED
        )

        assert result1.fill_rate == 1.0

        result2 = ExecutionResult(
            order_id="test2",
            code="600519",
            direction="buy",
            total_shares=10000,
            filled_shares=5000,
            avg_price=10.0,
            total_turnover=50000,
            total_slippage=0,
            execution_time=30,
            status=ExecutionStatus.PARTIAL
        )

        assert result2.fill_rate == 0.5

    def test_fill_rate_zero_total(self):
        """零总股数成交率测试"""
        result = ExecutionResult(
            order_id="test",
            code="600519",
            direction="buy",
            total_shares=0,
            filled_shares=0,
            avg_price=0.0,
            total_turnover=0,
            total_slippage=0,
            execution_time=0,
            status=ExecutionStatus.PENDING
        )

        assert result.fill_rate == 0.0


class TestEdgeCases:
    """边界情况测试"""

    def test_twap_extreme_urgency(self):
        """极端紧急程度测试"""
        prices = pd.Series([10.0, 10.1, 10.2])
        # urgency = 0 应该不会崩溃
        slices = twap_execution(1000, 3, prices, urgency=0.0)
        assert len(slices) == 3

        # urgency = 1 应该不会崩溃
        slices = twap_execution(1000, 3, prices, urgency=1.0)
        assert len(slices) == 3

    def test_twap_many_slices(self):
        """多切片测试"""
        prices = pd.Series([10.0 + i * 0.01 for i in range(100)])
        slices = twap_execution(10000, 100, prices)
        assert len(slices) == 100

    def test_vwap_single_volume(self):
        """单成交量测试"""
        prices = pd.Series([10.0])
        volumes = pd.Series([1000])
        slices = vwap_execution(1000, volumes, prices)
        assert len(slices) == 1
        assert slices[0].shares == 1000

    def test_order_slicer_large_max_size(self):
        """大片尺寸测试"""
        # max_slice_size > total_shares
        slices = order_slicer(1000, 5000, strategy="equal")
        assert len(slices) == 1
        assert slices[0] == 1000

    def test_order_slicer_small_max_size(self):
        """小片尺寸测试"""
        # 每片最多100，10000股需要100片
        slices = order_slicer(10000, 100, strategy="equal")
        assert len(slices) == 100

    def test_market_impact_small_order(self):
        """小额订单冲击测试"""
        impact = estimate_market_impact(
            order_amount=100,
            daily_avg_volume=10000000
        )
        assert impact.participation_rate < 0.0001
        assert impact.impact_bps < 1.0

    def test_market_impact_large_order(self):
        """大额订单冲击测试"""
        impact = estimate_market_impact(
            order_amount=100000000,  # 1亿
            daily_avg_volume=100000000  # 1亿日均
        )
        # 参与率 = 1亿 / (1亿 * 5) = 0.2 = 20%
        assert impact.participation_rate == pytest.approx(0.2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
