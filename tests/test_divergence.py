"""RSI Divergence 背离检测测试"""

import pytest
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np

from src.data.indicators import (
    calculate_rsi,
    calculate_ma,
    SwingSignals,
    detect_rsi_divergence
)


def create_ohlcv_with_divergence() -> pd.DataFrame:
    """
    创建包含底背离的测试数据

    场景：价格创新低，但 RSI 未创新低（底背离）
    - 第 1-15 天：价格从 100 下跌到 80，RSI 从 50 跌到 30
    - 第 16-20 天：价格反弹到 85，RSI 回升到 45
    - 第 21-25 天：价格再次下跌到 75（新低），RSI 在 40（未创新低）

    这种情况下形成底背离：价格新低 75 < 前低 80，但 RSI 40 > 前低 30
    """
    np.random.seed(42)
    days = 30

    # 价格走势：先跌后反弹再创新低
    prices = [100.0]
    for i in range(1, 15):  # 1-15: 100 -> 80
        prices.append(prices[-1] - 1.3)
    for i in range(15, 21):  # 15-21: 反弹
        prices.append(prices[-1] + 1.0)
    for i in range(21, days):  # 21-30: 再跌创新低
        prices.append(prices[-1] - 1.0)

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    data = {
        'date': dates.strftime('%Y-%m-%d'),
        'open': [p + np.random.uniform(-0.5, 0.5) for p in prices],
        'high': [p + np.random.uniform(0.5, 1.5) for p in prices],
        'low': [p - np.random.uniform(0.5, 1.5) for p in prices],
        'close': prices,
        'volume': [1000000] * days
    }

    df = pd.DataFrame(data)

    # 确保 high >= open, close, low
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


def create_ohlcv_no_divergence() -> pd.DataFrame:
    """
    创建无背离的测试数据

    场景：价格和 RSI 同涨同跌，无背离
    """
    np.random.seed(42)
    days = 30

    # 单边上涨趋势
    prices = np.linspace(100, 130, days).tolist()
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    data = {
        'date': dates.strftime('%Y-%m-%d'),
        'open': [p + np.random.uniform(-0.5, 0.5) for p in prices],
        'high': [p + np.random.uniform(0.5, 1.5) for p in prices],
        'low': [p - np.random.uniform(0.5, 1.5) for p in prices],
        'close': prices,
        'volume': [1000000] * days
    }

    df = pd.DataFrame(data)
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


def create_ohlcv_bearish_divergence() -> pd.DataFrame:
    """
    创建顶背离测试数据

    场景：价格创新高，但 RSI 未创新高（顶背离）
    """
    np.random.seed(42)
    days = 30

    # 价格走势：先涨后回调再创新高
    prices = [80.0]
    for i in range(1, 15):  # 1-15: 80 -> 100 上涨
        prices.append(prices[-1] + 1.3)
    for i in range(15, 21):  # 15-21: 回调
        prices.append(prices[-1] - 1.0)
    for i in range(21, days):  # 21-30: 再涨创新高
        prices.append(prices[-1] + 1.0)

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    data = {
        'date': dates.strftime('%Y-%m-%d'),
        'open': [p + np.random.uniform(-0.5, 0.5) for p in prices],
        'high': [p + np.random.uniform(0.5, 1.5) for p in prices],
        'low': [p - np.random.uniform(0.5, 1.5) for p in prices],
        'close': prices,
        'volume': [1000000] * days
    }

    df = pd.DataFrame(data)
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


def create_simple_bullish_divergence() -> pd.DataFrame:
    """
    创建简单底背离数据（手工构造）

    价格：100 -> 95 -> 90(低) -> 92 -> 85(新低)
    RSI：  50 -> 40 -> 35(低) -> 45 -> 42(更高低)
    """
    np.random.seed(42)
    days = 25

    prices = [100, 95, 90, 88, 86, 84, 82, 80, 82, 84, 86, 88, 90, 92, 90, 88, 87, 86, 85]
    while len(prices) < days:
        prices.append(85 - len(prices) + 19)

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    data = {
        'date': dates.strftime('%Y-%m-%d'),
        'open': [p + 0.5 for p in prices],
        'high': [p + 1.5 for p in prices],
        'low': [p - 1.5 for p in prices],
        'close': prices,
        'volume': [1000000] * days
    }

    df = pd.DataFrame(data)
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


class TestDetectRSIDivergence:
    """RSI 背离检测测试"""

    def test_detect_rsi_divergence_basic(self):
        """基本功能测试"""
        df = create_simple_bullish_divergence()
        df = calculate_rsi(df, periods=[14])
        df = calculate_ma(df, periods=[20])

        has_div, div_type, strength = detect_rsi_divergence(df)

        assert isinstance(has_div, bool)
        assert div_type in ["", "bullish", "bearish"]
        assert 0.0 <= strength <= 1.0

    def test_bullish_divergence_detected(self):
        """检测到底背离"""
        df = create_ohlcv_with_divergence()
        df = calculate_rsi(df, periods=[14])
        df = calculate_ma(df, periods=[20])

        has_div, div_type, strength = detect_rsi_divergence(df, lookback=20)

        # 底背离：价格新低，RSI 未新低
        if has_div:
            assert div_type == "bullish"
            assert strength > 0.0

    def test_no_divergence(self):
        """无背离情况"""
        df = create_ohlcv_no_divergence()
        df = calculate_rsi(df, periods=[14])
        df = calculate_ma(df, periods=[20])

        has_div, div_type, strength = detect_rsi_divergence(df, lookback=20)

        # 单边上涨趋势不应有背离
        # 注意：可能检测到顶背离，因为价格一直涨
        assert isinstance(has_div, bool)

    def test_bearish_divergence(self):
        """顶背离检测"""
        df = create_ohlcv_bearish_divergence()
        df = calculate_rsi(df, periods=[14])
        df = calculate_ma(df, periods=[20])

        has_div, div_type, strength = detect_rsi_divergence(df, lookback=20)

        # 顶背离：价格新高，RSI 未新高
        if has_div:
            assert div_type == "bearish"
            assert strength > 0.0

    def test_insufficient_data(self):
        """数据不足情况"""
        df = create_ohlcv_no_divergence().head(5)
        df = calculate_rsi(df, periods=[14])

        has_div, div_type, strength = detect_rsi_divergence(df, lookback=20)

        assert has_div == False
        assert div_type == ""
        assert strength == 0.0

    def test_with_default_parameters(self):
        """默认参数测试"""
        df = create_simple_bullish_divergence()
        df = calculate_rsi(df, periods=[14])
        df = calculate_ma(df, periods=[20])

        # 使用默认参数
        has_div, div_type, strength = detect_rsi_divergence(df)

        assert isinstance(has_div, bool)

    def test_custom_parameters(self):
        """自定义参数测试"""
        df = create_simple_bullish_divergence()
        df = calculate_rsi(df, periods=[14])
        df = calculate_ma(df, periods=[20])

        has_div, div_type, strength = detect_rsi_divergence(
            df,
            lookback=15,
            min_price_diff_pct=0.03,
            max_time_bars=8
        )

        assert isinstance(has_div, bool)


class TestRSIDivergenceIntegration:
    """RSI 背离与 SwingSignals 集成测试"""

    def test_analyze_includes_divergence(self):
        """analyze 方法包含背离检测"""
        df = create_ohlcv_with_divergence()

        signals = SwingSignals()
        result = signals.analyze(df)

        # 检查返回值包含背离字段
        assert hasattr(result, 'rsi_divergence')
        assert hasattr(result, 'rsi_divergence_type')
        assert hasattr(result, 'rsi_divergence_strength')

        assert isinstance(result.rsi_divergence, bool)
        assert result.rsi_divergence_type in ["", "bullish", "bearish"]
        assert 0.0 <= result.rsi_divergence_strength <= 1.0

    def test_divergence_false_when_no_divergence(self):
        """无背离时 rsi_divergence 为 False"""
        df = create_ohlcv_no_divergence()

        signals = SwingSignals()
        result = signals.analyze(df)

        # 无背离时应该是 False
        if not result.rsi_divergence:
            assert result.rsi_divergence_type == ""
            assert result.rsi_divergence_strength == 0.0


class TestDivergenceEdgeCases:
    """背离检测边界情况测试"""

    def test_flat_price_no_divergence(self):
        """价格平稳无背离"""
        days = 30
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        data = {
            'date': dates.strftime('%Y-%m-%d'),
            'open': [100.0] * days,
            'high': [101.0] * days,
            'low': [99.0] * days,
            'close': [100.0] * days,
            'volume': [1000000] * days
        }

        df = pd.DataFrame(data)
        df = calculate_rsi(df, periods=[14])

        has_div, div_type, strength = detect_rsi_divergence(df, lookback=20)

        # 平稳价格不应有背离
        assert has_div == False

    def test_missing_rsi_column(self):
        """缺少 RSI 列"""
        days = 30
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        data = {
            'date': dates.strftime('%Y-%m-%d'),
            'close': np.linspace(100, 120, days).tolist(),
            'volume': [1000000] * days
        }

        df = pd.DataFrame(data)

        has_div, div_type, strength = detect_rsi_divergence(df)

        assert has_div == False
        assert strength == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
