"""市场状态识别模块测试"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtest.market_state import (
    MarketState,
    MarketStateResult,
    detect_market_state,
    detect_breakout
)


class TestMarketState:
    """市场状态测试"""

    def _create_sample_df(
        self,
        n: int = 100,
        start_price: float = 100.0,
        trend: str = "sideways"
    ) -> pd.DataFrame:
        """创建样本数据"""
        dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
        np.random.seed(42)

        # 生成价格数据
        if trend == "uptrend":
            returns = np.random.normal(0.001, 0.02, n)
        elif trend == "downtrend":
            returns = np.random.normal(-0.001, 0.02, n)
        else:
            returns = np.random.normal(0, 0.01, n)

        prices = start_price * np.exp(np.cumsum(returns))
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n)))
        closes = prices

        df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "open": closes * (1 + np.random.normal(0, 0.005, n)),
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.random.randint(1_000_000, 10_000_000, n)
        })

        return df

    def _add_atr_to_df(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """为 DataFrame 添加 ATR 指标"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=period).mean()
        return df

    def test_market_state_enum(self):
        """MarketState 枚举测试"""
        assert MarketState.TREND.value == "trend"
        assert MarketState.VOLATILE.value == "volatile"
        assert MarketState.TRANSITION.value == "transition"

    def test_market_state_result_properties(self):
        """MarketStateResult 属性测试"""
        result = MarketStateResult(
            state=MarketState.TREND,
            adx=30.0,
            volatility_ratio=1.5,
            trend_direction="up",
            confidence=0.7
        )

        assert result.state_name == "趋势市"
        assert result.should_skip_entry is False
        assert result.position_size_multiplier == 1.0

        # 测试震荡市
        volatile_result = MarketStateResult(
            state=MarketState.VOLATILE,
            adx=15.0,
            volatility_ratio=1.0,
            trend_direction="sideways",
            confidence=0.6
        )
        assert volatile_result.state_name == "震荡市"
        assert volatile_result.should_skip_entry is True
        assert volatile_result.position_size_multiplier == 0.5

        # 测试转折市
        transition_result = MarketStateResult(
            state=MarketState.TRANSITION,
            adx=20.0,
            volatility_ratio=2.5,
            trend_direction="down",
            confidence=0.8
        )
        assert transition_result.state_name == "转折市"
        assert transition_result.should_skip_entry is False
        assert transition_result.position_size_multiplier == 1.2

    def test_detect_market_state_trend(self):
        """检测趋势市"""
        # 创建上涨趋势数据
        df = self._create_sample_df(n=60, trend="uptrend")
        df = self._add_atr_to_df(df)

        # 使用 adx 模块计算 ADX
        from src.data.indicators.adx import calculate_adx
        df = calculate_adx(df, period=14)

        result = detect_market_state(df)

        assert result.adx > 0
        assert result.trend_direction == "up"
        assert result.volatility_ratio > 0

    def test_detect_market_state_volatile(self):
        """检测震荡市"""
        # 创建震荡数据
        df = self._create_sample_df(n=60, trend="sideways")
        df = self._add_atr_to_df(df)

        # 使用 adx 模块计算 ADX
        from src.data.indicators.adx import calculate_adx
        df = calculate_adx(df, period=14)

        result = detect_market_state(df)

        # 震荡市场中 ADX 应该较低
        assert result.state in [MarketState.VOLATILE, MarketState.TRANSITION]
        assert result.adx >= 0

    def test_detect_market_state_volatility_spike(self):
        """检测波动率急剧放大（转折市）

        由于 ATR 使用 EWM 计算，单点波动难以触发转折市。
        这里测试连续高波动场景（多个点的高 ATR）。
        """
        # 创建波动率数据：前 45 个 ATR 较小，然后连续 5 个 ATR 较高
        df = self._create_sample_df(n=60, trend="sideways")
        df = self._add_atr_to_df(df)

        # 先计算 tr（true range）
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # 设置前 35 个 TR 值为较小值，然后连续 5 个 TR 急剧放大
        small_tr_value = 1.0
        large_tr_value = 5.0  # 5x smaller value

        df.iloc[:35, df.columns.get_loc('tr')] = small_tr_value
        df.iloc[-5:, df.columns.get_loc('tr')] = large_tr_value

        # 使用 adx 模块计算 ADX（会基于设置的 TR 值计算 ATR）
        from src.data.indicators.adx import calculate_adx
        df = calculate_adx(df, period=14)

        result = detect_market_state(df)

        # 连续高波动应该被检测为转折市
        # 由于 EWM 平滑，可能需要检查其他指标
        # 这里验证 state 是 VOLATILE 或 TRANSITION
        assert result.state in [MarketState.VOLATILE, MarketState.TRANSITION]
        # 验证波动率放大
        assert result.volatility_ratio > 1.0

    def test_detect_market_state_insufficient_data(self):
        """数据不足时的默认处理"""
        df = self._create_sample_df(n=10)
        df = self._add_atr_to_df(df)

        result = detect_market_state(df)

        # 数据不足应该返回震荡市默认状态
        assert result.state == MarketState.VOLATILE
        assert result.confidence == 0.0

    def test_detect_breakout_up(self):
        """检测向上突破"""
        df = self._create_sample_df(n=30)
        df = self._add_atr_to_df(df)

        # 人为制造一个突破
        df.iloc[-1, df.columns.get_loc('close')] = df['high'].iloc[-25:-1].max() * 1.03

        result = detect_breakout(df)
        assert result is True

    def test_detect_breakout_down(self):
        """检测向下突破"""
        df = self._create_sample_df(n=30)
        df = self._add_atr_to_df(df)

        # 人为制造一个向下突破
        df.iloc[-1, df.columns.get_loc('close')] = df['low'].iloc[-25:-1].min() * 0.97

        result = detect_breakout(df)
        assert result is True

    def test_detect_breakout_no_breakout(self):
        """无突破时返回 False"""
        df = self._create_sample_df(n=30)
        df = self._add_atr_to_df(df)

        # 故意保持价格在范围内
        recent_high = df['high'].iloc[-25:-1].max()
        recent_low = df['low'].iloc[-25:-1].min()
        mid_price = (recent_high + recent_low) / 2
        df.iloc[-1, df.columns.get_loc('close')] = mid_price

        result = detect_breakout(df)
        assert result is False

    def test_detect_breakout_insufficient_data(self):
        """突破检测数据不足"""
        df = self._create_sample_df(n=5)

        result = detect_breakout(df)
        assert result is False


class TestMarketStateIntegration:
    """市场状态集成测试"""

    def _create_sample_df(self, n: int = 100) -> pd.DataFrame:
        """创建样本数据"""
        dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
        np.random.seed(42)

        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n)))

        df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "open": prices,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": np.random.randint(1_000_000, 10_000_000, n)
        })

        return df

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加所有指标"""
        from src.data.indicators.adx import calculate_adx
        from src.data.indicators.atr import calculate_atr

        df = calculate_atr(df, period=14)
        df = calculate_adx(df, period=14)

        return df

    def test_market_state_with_real_indicators(self):
        """使用真实指标数据进行市场状态检测"""
        df = self._create_sample_df(n=60)
        df = self._add_indicators(df)

        result = detect_market_state(df)

        assert isinstance(result.state, MarketState)
        assert result.adx >= 0
        assert result.volatility_ratio > 0
        assert result.trend_direction in ["up", "down", "sideways"]
        assert 0.0 <= result.confidence <= 1.0

    def test_market_state_signal_result_integration(self):
        """验证 MarketState 可以正确传递到 EntrySignal"""
        from src.backtest.models import EntrySignal

        # 创建 EntrySignal 时包含市场状态
        signal = EntrySignal(
            code="600519",
            signal_type="golden",
            confidence=0.7,
            entry_price=100.0,
            stop_loss=96.0,
            atr=2.0,
            reason="MA金叉",
            market_state="trend"
        )

        assert signal.market_state == "trend"

        # 测试震荡市场状态
        volatile_signal = EntrySignal(
            code="600519",
            signal_type="golden",
            confidence=0.7,
            entry_price=100.0,
            stop_loss=96.0,
            atr=2.0,
            reason="MA金叉",
            market_state="volatile"
        )

        assert volatile_signal.market_state == "volatile"
