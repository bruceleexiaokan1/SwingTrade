"""波浪理论指标测试"""

import pytest
import pandas as pd
import numpy as np

from src.data.indicators.wave import (
    WaveIndicators, WaveType, WaveDirection,
    WavePoint, WaveResult, calculate_wave_levels
)


def make_ohlcv(n=100, seed=42) -> pd.DataFrame:
    """生成测试用 OHLCV 数据"""
    np.random.seed(seed)
    dates = pd.date_range("2025-01-01", periods=n).strftime("%Y-%m-%d").tolist()
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_prices = low + np.random.rand(n) * (high - low)
    volume = np.random.randint(1_000_000, 10_000_000, n)

    return pd.DataFrame({
        "date": dates,
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestWaveIndicators:
    """WaveIndicators 单元测试"""

    def test_calculate_all_basic(self):
        """基本计算测试"""
        df = make_ohlcv(100)
        wi = WaveIndicators()
        result = wi.calculate_all(df)

        assert "retracement" in result.columns
        assert "wave_momentum" in result.columns
        assert "is_local_max" in result.columns
        assert "is_local_min" in result.columns
        assert len(result) == len(df)

    def test_calculate_all_preserves_data(self):
        """验证不丢失原始列"""
        df = make_ohlcv(50)
        wi = WaveIndicators()
        result = wi.calculate_all(df)

        for col in ["date", "open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_analyze_insufficient_data(self):
        """数据不足时返回默认值"""
        df = make_ohlcv(10)  # 少于 min_periods=20
        wi = WaveIndicators()
        result = wi.analyze(df)

        assert result.direction == WaveDirection.SIDEWAYS
        assert result.current_wave == WaveType.UNKNOWN
        assert result.wave_count == 0
        assert result.wave_count_confidence == 0.0

    def test_analyze_sufficient_data(self):
        """数据充足时能正常分析"""
        df = make_ohlcv(100)
        wi = WaveIndicators()
        result = wi.analyze(df)

        assert result.date == df["date"].iloc[-1]
        assert result.direction in (WaveDirection.UP, WaveDirection.DOWN, WaveDirection.SIDEWAYS)
        assert isinstance(result.momentum, float)
        assert 0.0 <= result.momentum <= 1.0

    def test_get_wave_score_uptrend(self):
        """上涨趋势评分"""
        # 构造一个明确上涨的趋势
        df = make_ohlcv(100)
        # 让价格持续上涨
        df.loc[:, "close"] = 50 + np.linspace(0, 30, 100)
        df.loc[:, "high"] = df["close"] + 2
        df.loc[:, "low"] = df["close"] - 2
        df.loc[:, "open"] = df["low"] + (df["high"] - df["low"]) * 0.5

        wi = WaveIndicators()
        score = wi.get_wave_score(df, wave_type="uptrend")
        assert 0.0 <= score <= 1.0

    def test_get_wave_score_downtrend(self):
        """下跌趋势评分"""
        df = make_ohlcv(100)
        # 构造一个明确下跌的趋势
        df.loc[:, "close"] = 80 - np.linspace(0, 30, 100)
        df.loc[:, "high"] = df["close"] + 2
        df.loc[:, "low"] = df["close"] - 2
        df.loc[:, "open"] = df["low"] + (df["high"] - df["low"]) * 0.5

        wi = WaveIndicators()
        score = wi.get_wave_score(df, wave_type="downtrend")
        assert 0.0 <= score <= 1.0

    def test_find_wave_points(self):
        """波浪转折点查找"""
        df = make_ohlcv(100)
        wi = WaveIndicators()
        df = wi.calculate_all(df)
        points = wi.find_wave_points(df)

        assert isinstance(points, list)
        # 转折点应该有类型标记
        for wp in points:
            assert isinstance(wp, WavePoint)
            assert wp.wave_type != WaveType.UNKNOWN or wp.strength > 0

    def test_find_wave_points_insufficient(self):
        """数据不足时返回空"""
        df = make_ohlcv(10)
        wi = WaveIndicators()
        points = wi.find_wave_points(df)
        assert points == []


class TestCalculateWaveLevels:
    """calculate_wave_levels 工具函数测试"""

    def test_insufficient_prices(self):
        """价格数量不足"""
        result = calculate_wave_levels([100, 110])
        assert result == {}

    def test_valid_prices(self):
        """正常价格序列"""
        prices = [100, 110, 105, 115, 112, 120]
        result = calculate_wave_levels(prices)

        assert "wave2_fib_0.382" in result
        assert "wave2_fib_0.5" in result
        assert "wave2_fib_0.618" in result
        assert "wave3_1618" in result
        assert "wave3_1272" in result

    def test_wave_levels_sensible(self):
        """波浪水平数值合理性"""
        prices = [100, 110, 105, 115, 112, 120]
        result = calculate_wave_levels(prices)

        # wave2 回撤应该低于 wave1 终点
        assert result["wave2_fib_0.618"] < 110
        # wave3 扩展应该高于 wave1 终点
        assert result["wave3_1618"] > 110


class TestWaveEnums:
    """波浪类型枚举测试"""

    def test_wave_type_values(self):
        """验证波浪类型枚举值"""
        assert WaveType.IMPULSE_1.value == "impulse_1"
        assert WaveType.CORRECTIVE_A.value == "corrective_a"

    def test_wave_direction_values(self):
        """验证波浪方向枚举值"""
        assert WaveDirection.UP.value == "up"
        assert WaveDirection.DOWN.value == "down"
        assert WaveDirection.SIDEWAYS.value == "sideways"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
