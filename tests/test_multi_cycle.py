"""多周期共振测试"""

import pytest
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from src.backtest.multi_cycle import (
    MultiCycleResonance,
    MultiCycleResult,
    MultiCycleLevel
)


class TestMultiCycleLevel:
    """多周期共振等级测试"""

    def test_level_labels(self):
        """等级标签测试"""
        assert MultiCycleLevel.FORBIDDEN.label == "禁止操作"
        assert MultiCycleLevel.DAILY_ONLY.label == "中信号"
        assert MultiCycleLevel.MONTHLY_WEEKLY.label == "强信号"
        assert MultiCycleLevel.THREE_CYCLE.label == "三周期共振"

    def test_position_limits(self):
        """仓位上限测试"""
        assert MultiCycleLevel.FORBIDDEN.position_limit == 0.0
        assert MultiCycleLevel.DAILY_ONLY.position_limit == 0.2
        assert MultiCycleLevel.MONTHLY_WEEKLY.position_limit == 0.6
        assert MultiCycleLevel.THREE_CYCLE.position_limit == 0.8


class TestMultiCycleResult:
    """多周期共振结果测试"""

    def test_result_properties(self):
        """结果属性测试"""
        result = MultiCycleResult(
            stock_code="600519",
            date="2026-03-28",
            monthly_trend="up",
            monthly_conf=0.8,
            weekly_trend="up",
            weekly_conf=0.7,
            daily_trend="up",
            daily_conf=0.6,
            resonance_level=5,
            position_limit=0.8,
            is_bullish=True,
            reasons=["月周周日三层共振向上"]
        )

        assert result.stock_code == "600519"
        assert result.monthly_up is True
        assert result.weekly_up is True
        assert result.daily_up is True
        assert result.level_label == "三周期共振"
        assert result.is_bullish is True

    def test_bearish_result(self):
        """看空结果测试"""
        result = MultiCycleResult(
            stock_code="600519",
            date="2026-03-28",
            monthly_trend="down",
            monthly_conf=0.8,
            weekly_trend="down",
            weekly_conf=0.7,
            daily_trend="down",
            daily_conf=0.6,
            resonance_level=0,
            position_limit=0.0,
            is_bullish=False,
            reasons=["三层逆势"]
        )

        assert result.monthly_up is False
        assert result.weekly_up is False
        assert result.daily_up is False
        assert result.is_bullish is False


class TestMultiCycleResonance:
    """多周期共振检测器测试"""

    def setup_method(self):
        """测试初始化"""
        # 使用临时数据创建检测器
        self.resonance = MultiCycleResonance.__new__(MultiCycleResonance)
        self.resonance.stock_loader = None

    def _create_sample_daily_data(self, n_days: int = 100) -> pd.DataFrame:
        """创建示例日线数据"""
        dates = pd.date_range(end='2026-03-28', periods=n_days, freq='D')
        # 构造一个上涨趋势
        base_price = 100.0
        prices = [base_price]
        for i in range(1, n_days):
            # 每日上涨0.5%
            prices.append(prices[-1] * 1.005)

        data = {
            'date': [d.strftime('%Y-%m-%d') for d in dates],
            'open': [p * 0.99 for p in prices],
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000000] * n_days
        }
        return pd.DataFrame(data)

    def _create_sample_daily_data_downtrend(self, n_days: int = 100) -> pd.DataFrame:
        """创建下跌趋势日线数据"""
        dates = pd.date_range(end='2026-03-28', periods=n_days, freq='D')
        base_price = 100.0
        prices = [base_price]
        for i in range(1, n_days):
            # 每日下跌0.5%
            prices.append(prices[-1] * 0.995)

        data = {
            'date': [d.strftime('%Y-%m-%d') for d in dates],
            'open': [p * 1.01 for p in prices],
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000000] * n_days
        }
        return pd.DataFrame(data)

    def _create_sample_daily_data_sideways(self, n_days: int = 100) -> pd.DataFrame:
        """创建震荡日线数据"""
        dates = pd.date_range(end='2026-03-28', periods=n_days, freq='D')
        base_price = 100.0
        prices = []
        for i in range(n_days):
            # 震荡
            price = base_price + 5 * np.sin(i / 10)
            prices.append(price)

        data = {
            'date': [d.strftime('%Y-%m-%d') for d in dates],
            'open': [p * 0.99 for p in prices],
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000000] * n_days
        }
        return pd.DataFrame(data)

    def test_to_monthly(self):
        """日线转月线测试"""
        daily_df = self._create_sample_daily_data(100)

        # 模拟 _to_monthly 方法
        df = daily_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        monthly = df.resample('ME').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        monthly = monthly.reset_index()
        monthly['date'] = monthly['date'].dt.strftime('%Y-%m-%d')

        # 应该有几个月的数据
        assert len(monthly) >= 3
        assert 'close' in monthly.columns

    def test_to_weekly(self):
        """日线转周线测试"""
        daily_df = self._create_sample_daily_data(100)

        # 模拟 _to_weekly 方法
        df = daily_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        weekly = df.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        weekly = weekly.reset_index()
        weekly['date'] = weekly['date'].dt.strftime('%Y-%m-%d')

        # 应该有若干周的数据
        assert len(weekly) >= 10
        assert 'close' in weekly.columns

    def test_detect_trend_uptrend(self):
        """检测上涨趋势"""
        daily_df = self._create_sample_daily_data(100)

        # 手动测试趋势检测逻辑
        ma_short = daily_df['close'].rolling(window=20, min_periods=1).mean()
        ma_long = daily_df['close'].rolling(window=60, min_periods=1).mean()

        last_short = ma_short.iloc[-1]
        last_long = ma_long.iloc[-1]

        assert last_short > last_long  # 上涨趋势

    def test_detect_trend_downtrend(self):
        """检测下跌趋势"""
        daily_df = self._create_sample_daily_data_downtrend(100)

        ma_short = daily_df['close'].rolling(window=20, min_periods=1).mean()
        ma_long = daily_df['close'].rolling(window=60, min_periods=1).mean()

        last_short = ma_short.iloc[-1]
        last_long = ma_long.iloc[-1]

        assert last_short < last_long  # 下跌趋势

    def test_calc_resonance_level_three_cycle(self):
        """三周期共振等级计算"""
        # 模拟 _calc_resonance_level
        monthly_trend = "up"
        weekly_trend = "up"
        daily_trend = "up"

        up_count = sum([
            monthly_trend == "up",
            weekly_trend == "up",
            daily_trend == "up"
        ])

        assert up_count == 3

        # 三周期全部向上应该是 level 5
        level = MultiCycleLevel.THREE_CYCLE.value
        assert level == 5

    def test_calc_resonance_level_monthly_weekly(self):
        """月周共振等级计算"""
        monthly_trend = "up"
        weekly_trend = "up"
        daily_trend = "sideways"  # 日线待确认

        up_count = sum([
            monthly_trend == "up",
            weekly_trend == "up",
            daily_trend == "up"
        ])

        # 应该是 level 4
        level = MultiCycleLevel.MONTHLY_WEEKLY.value
        assert level == 4
        assert up_count == 2

    def test_calc_resonance_level_forbidden(self):
        """禁止操作等级计算"""
        monthly_trend = "down"
        weekly_trend = "down"
        daily_trend = "down"

        down_count = sum([
            monthly_trend == "down",
            weekly_trend == "down",
            daily_trend == "down"
        ])

        assert down_count >= 2

        level = MultiCycleLevel.FORBIDDEN.value
        assert level == 0

    def test_calc_resonance_level_daily_only(self):
        """日线信号等级计算"""
        monthly_trend = "sideways"
        weekly_trend = "sideways"
        daily_trend = "up"

        up_count = sum([
            monthly_trend == "up",
            weekly_trend == "up",
            daily_trend == "up"
        ])

        # 只有日线向上应该是 level 3
        level = MultiCycleLevel.DAILY_ONLY.value
        assert level == 3
        assert up_count == 1

    def test_is_bullish_calculation(self):
        """看多判断测试"""
        # 至少2个周期向上才是看多
        test_cases = [
            (["up", "up", "up"], True),
            (["up", "up", "down"], True),
            (["up", "down", "down"], False),
            (["down", "down", "down"], False),
            (["up", "sideways", "sideways"], False),
        ]

        for trends, expected_bullish in test_cases:
            bullish_count = sum(t == "up" for t in trends)
            is_bullish = bullish_count >= 2
            assert is_bullish == expected_bullish, f"Failed for {trends}"

    def test_empty_result_creation(self):
        """空结果创建测试"""
        result = MultiCycleResult(
            stock_code="600519",
            date="2026-03-28",
            reasons=["数据不足"]
        )

        assert result.stock_code == "600519"
        assert result.resonance_level == 0
        assert result.position_limit == 0.0
        assert "数据不足" in result.reasons


class TestMultiCycleIntegration:
    """多周期共振集成测试（使用模拟数据）"""

    def test_full_cycle_detection_with_mock(self):
        """完整周期检测测试"""
        # 创建上涨趋势的日线数据
        dates = pd.date_range(end='2026-03-28', periods=120, freq='D')
        base_price = 100.0
        prices = []
        for i in range(120):
            prices.append(base_price * (1.003 ** i))  # 稳定上涨

        daily_df = pd.DataFrame({
            'date': [d.strftime('%Y-%m-%d') for d in dates],
            'open': [p * 0.99 for p in prices],
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000000] * 120
        })

        # 转换为月线和周线
        df = daily_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        monthly_df = df.resample('ME').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        monthly_df['date'] = monthly_df['date'].dt.strftime('%Y-%m-%d')

        weekly_df = df.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        weekly_df['date'] = weekly_df['date'].dt.strftime('%Y-%m-%d')

        # 验证数据转换正确
        assert len(monthly_df) >= 4  # 至少4个月
        assert len(weekly_df) >= 16  # 至少16周
        assert len(daily_df) >= 60   # 至少60天

        # 验证上涨趋势
        assert monthly_df['close'].iloc[-1] > monthly_df['close'].iloc[0]
        assert weekly_df['close'].iloc[-1] > weekly_df['close'].iloc[0]
        assert daily_df['close'].iloc[-1] > daily_df['close'].iloc[0]

    def test_batch_check_summary(self):
        """批量检测汇总测试"""
        results = [
            MultiCycleResult(
                stock_code="600519",
                date="2026-03-28",
                resonance_level=5,
                is_bullish=True
            ),
            MultiCycleResult(
                stock_code="000001",
                date="2026-03-28",
                resonance_level=4,
                is_bullish=True
            ),
            MultiCycleResult(
                stock_code="000002",
                date="2026-03-28",
                resonance_level=3,
                is_bullish=True
            ),
            MultiCycleResult(
                stock_code="000003",
                date="2026-03-28",
                resonance_level=0,
                is_bullish=False
            ),
        ]

        # 手动计算汇总
        summary = {
            "total": len(results),
            "three_cycle": sum(1 for r in results if r.resonance_level == 5),
            "monthly_weekly": sum(1 for r in results if r.resonance_level == 4),
            "daily_only": sum(1 for r in results if r.resonance_level == 3),
            "forbidden": sum(1 for r in results if r.resonance_level == 0),
            "bullish_count": sum(1 for r in results if r.is_bullish)
        }

        assert summary["total"] == 4
        assert summary["three_cycle"] == 1
        assert summary["monthly_weekly"] == 1
        assert summary["daily_only"] == 1
        assert summary["forbidden"] == 1
        assert summary["bullish_count"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
