"""技术指标模块测试"""

import pytest
import sys
import os
import tempfile
import shutil
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np

from src.data.indicators import (
    calculate_ma, golden_cross, death_cross,
    calculate_macd, macd_bullish, macd_bearish,
    calculate_rsi, rsi_oversold, rsi_overbought,
    calculate_bollinger, bollinger_squeeze,
    calculate_atr, atr_stop_loss, atr_trailing_stop,
    calculate_volume_ma, volume_surge, volume_shrink,
    SwingSignals
)


def create_sample_ohlcv(days: int = 100, start_price: float = 100.0) -> pd.DataFrame:
    """创建样本 OHLCV 数据"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    data = {
        'date': dates.strftime('%Y-%m-%d'),
        'open': start_price + np.random.randn(days).cumsum(),
        'high': start_price + np.random.randn(days).cumsum() + abs(np.random.randn(days)),
        'low': start_price + np.random.randn(days).cumsum() - abs(np.random.randn(days)),
        'close': start_price + np.random.randn(days).cumsum(),
        'volume': (1000000 + np.random.randn(days) * 100000).astype(int)
    }

    df = pd.DataFrame(data)

    # 确保 high >= open, close, low
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


class TestMA:
    """MA 指标测试"""

    def test_calculate_ma(self):
        """MA 计算测试"""
        df = create_sample_ohlcv(100)
        result = calculate_ma(df, periods=[5, 10, 20, 60])

        assert 'ma5' in result.columns
        assert 'ma10' in result.columns
        assert 'ma20' in result.columns
        assert 'ma60' in result.columns

        # 检查非空值
        assert result['ma60'].notna().sum() > 0

    def test_ma_values(self):
        """MA 值正确性测试"""
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
            'close': [100.0, 102.0, 104.0, 106.0, 108.0],
            'volume': [1000, 1000, 1000, 1000, 1000]
        })

        result = calculate_ma(df, periods=[5])
        # 5日均价 = (100+102+104+106+108)/5 = 104
        assert abs(result['ma5'].iloc[-1] - 104.0) < 0.01

    def test_golden_cross(self):
        """金叉检测测试"""
        # 金叉：前一日快线<慢线，当前快线>=慢线
        # 当前：105 >= 100，前一天：99 < 100
        ma_fast = pd.Series([100, 95, 97, 99, 105])  # 最后一天从下穿上
        ma_slow = pd.Series([100, 100, 100, 100, 100])

        assert golden_cross(ma_fast, ma_slow) == True

    def test_death_cross(self):
        """死叉检测测试"""
        # 死叉：前一日快线>慢线，当前快线<=慢线
        # 当前：95 <= 100，前一天：101 > 100
        ma_fast = pd.Series([100, 105, 103, 101, 95])  # 最后一天从上穿下
        ma_slow = pd.Series([100, 100, 100, 100, 100])

        assert death_cross(ma_fast, ma_slow) == True

    def test_no_cross(self):
        """无交叉测试"""
        ma_fast = pd.Series([100, 101, 102, 103, 104])
        ma_slow = pd.Series([100, 101, 102, 103, 104])

        # 持续多头排列，不应有交叉
        assert golden_cross(ma_fast, ma_slow) == False
        assert death_cross(ma_fast, ma_slow) == False


class TestMACD:
    """MACD 指标测试"""

    def test_calculate_macd(self):
        """MACD 计算测试"""
        df = create_sample_ohlcv(100)
        result = calculate_macd(df)

        assert 'dif' in result.columns
        assert 'dem' in result.columns
        assert 'hist' in result.columns

        # DIF 应该大于 DEA 在 MACD 多头时
        assert result['dif'].notna().sum() > 0
        assert result['dem'].notna().sum() > 0

    def test_macd_bullish(self):
        """MACD 多头测试"""
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'close': [100.0, 102.0, 104.0],
            'volume': [1000, 1000, 1000]
        })

        result = calculate_macd(df)
        # 上涨趋势 DIF 应该 > 0
        assert macd_bullish(result) == True

    def test_macd_bearish(self):
        """MACD 空头测试"""
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'close': [104.0, 102.0, 100.0],
            'volume': [1000, 1000, 1000]
        })

        result = calculate_macd(df)
        # 下跌趋势 DIF 应该 < 0
        assert macd_bearish(result) == True


class TestRSI:
    """RSI 指标测试"""

    def test_calculate_rsi(self):
        """RSI 计算测试"""
        df = create_sample_ohlcv(50)
        result = calculate_rsi(df, periods=[6, 14])

        assert 'rsi6' in result.columns
        assert 'rsi14' in result.columns

    def test_rsi_range(self):
        """RSI 范围测试"""
        df = create_sample_ohlcv(100)
        result = calculate_rsi(df, periods=[14])

        # RSI 应该在 0~100 之间
        valid_rsi = result['rsi14'].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_oversold(self):
        """RSI 超卖测试"""
        assert rsi_oversold(25) == True
        assert rsi_oversold(35) == False

    def test_rsi_overbought(self):
        """RSI 超买测试"""
        assert rsi_overbought(75) == True
        assert rsi_overbought(65) == False


class TestBollinger:
    """布林带指标测试"""

    def test_calculate_bollinger(self):
        """布林带计算测试"""
        df = create_sample_ohlcv(50)
        result = calculate_bollinger(df)

        assert 'bb_upper' in result.columns
        assert 'bb_middle' in result.columns
        assert 'bb_lower' in result.columns
        assert 'bb_bandwidth' in result.columns
        assert 'bb_position' in result.columns

    def test_bollinger_relationship(self):
        """布林带关系测试"""
        df = create_sample_ohlcv(100)
        result = calculate_bollinger(df)

        # 上轨 >= 中轨 >= 下轨
        valid_rows = result.dropna(subset=['bb_upper', 'bb_middle', 'bb_lower'])
        assert (valid_rows['bb_upper'] >= valid_rows['bb_middle']).all()
        assert (valid_rows['bb_middle'] >= valid_rows['bb_lower']).all()

    def test_bollinger_bandwidth(self):
        """布林带宽测试"""
        df = create_sample_ohlcv(100)
        result = calculate_bollinger(df)

        # 带宽应该 >= 0（第一行可能为NaN或0）
        valid_bandwidth = result['bb_bandwidth'].dropna()
        assert (valid_bandwidth >= 0).all()


class TestATR:
    """ATR 指标测试"""

    def test_calculate_atr(self):
        """ATR 计算测试"""
        df = create_sample_ohlcv(50)
        result = calculate_atr(df)

        assert 'tr' in result.columns
        assert 'atr' in result.columns
        assert 'atr_pct' in result.columns

    def test_atr_positive(self):
        """ATR 正值测试"""
        df = create_sample_ohlcv(100)
        result = calculate_atr(df)

        # ATR 应该 > 0
        valid_atr = result['atr'].dropna()
        assert (valid_atr > 0).all()

    def test_atr_stop_loss(self):
        """ATR 止损计算测试"""
        stop = atr_stop_loss(entry_price=100.0, atr=2.0, multiplier=2.0)
        assert stop == 96.0  # 100 - 2*2 = 96

    def test_atr_trailing_stop(self):
        """ATR 追踪止损计算测试"""
        trailing = atr_trailing_stop(highest_price=120.0, atr=2.5, multiplier=3.0)
        assert trailing == 112.5  # 120 - 3*2.5 = 112.5


class TestVolume:
    """成交量指标测试"""

    def test_calculate_volume_ma(self):
        """成交量均线计算测试"""
        df = create_sample_ohlcv(50)
        result = calculate_volume_ma(df)

        assert 'volume_ma' in result.columns

    def test_volume_surge(self):
        """放量检测测试"""
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'close': [100.0, 102.0, 104.0],
            'volume': [1000000, 1000000, 2000000],  # 第3天放量
            'volume_ma': [1000000, 1000000, 1000000]
        })

        assert volume_surge(df, threshold=1.5) == True

    def test_volume_shrink(self):
        """缩量检测测试"""
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'close': [100.0, 102.0, 104.0],
            'volume': [1000000, 500000, 300000],  # 持续缩量
            'volume_ma': [1000000, 1000000, 1000000]
        })

        assert volume_shrink(df, threshold=0.5) == True


class TestSwingSignals:
    """波段信号检测器测试"""

    def setup_method(self):
        """测试初始化"""
        self.signals = SwingSignals()

    def test_calculate_all(self):
        """计算所有指标测试"""
        df = create_sample_ohlcv(100)
        result = self.signals.calculate_all(df)

        # 检查所有指标列
        assert 'ma5' in result.columns
        assert 'dif' in result.columns
        assert 'dem' in result.columns
        assert 'rsi6' in result.columns
        assert 'rsi14' in result.columns
        assert 'bb_upper' in result.columns
        assert 'atr' in result.columns
        assert 'volume_ma' in result.columns

    def test_detect_trend_uptrend(self):
        """检测上涨趋势"""
        df = pd.DataFrame({
            'date': ['2024-01-01'] * 70,
            'close': np.linspace(100, 150, 70),  # 持续上涨
            'open': np.linspace(99, 149, 70),
            'high': np.linspace(101, 151, 70),
            'low': np.linspace(98, 148, 70),
            'volume': [1000000] * 70
        })

        df = self.signals.calculate_all(df)
        trend, confidence = self.signals.detect_trend(df)

        assert trend == "uptrend"
        assert confidence > 0.5

    def test_detect_trend_downtrend(self):
        """检测下跌趋势"""
        df = pd.DataFrame({
            'date': ['2024-01-01'] * 70,
            'close': np.linspace(150, 100, 70),  # 持续下跌
            'open': np.linspace(151, 101, 70),
            'high': np.linspace(152, 102, 70),
            'low': np.linspace(149, 99, 70),
            'volume': [1000000] * 70
        })

        df = self.signals.calculate_all(df)
        trend, confidence = self.signals.detect_trend(df)

        assert trend == "downtrend"
        assert confidence > 0.5

    def test_analyze_with_entry(self):
        """综合分析含入场价格"""
        df = create_sample_ohlcv(100)
        df = self.signals.calculate_all(df)

        entry_price = df['close'].iloc[-10]  # 10天前的价格
        result = self.signals.analyze(df, entry_price=entry_price)

        assert result.date is not None
        assert result.trend in ['uptrend', 'downtrend', 'sideways']
        assert result.entry_signal in ['golden', 'breakout', 'none']
        assert result.exit_signal in ['stop_loss', 'trailing', 'ma_cross', 'rsi_overbought', 'take_profit_1', 'none']


class TestRealDataIndicators:
    """真实数据指标测试（使用真实股票数据）"""

    def test_real_stock_indicators(self):
        """使用真实股票数据验证指标"""
        try:
            from src.data.loader import StockDataLoader
            from src.data.fetcher.price_converter import convert_to_forward_adj

            loader = StockDataLoader("/Users/bruce/workspace/trade/StockData")
            df = loader.load_daily("600519", start_date="2025-01-01")

            if len(df) < 50:
                pytest.skip("Not enough data")

            # 转换前复权
            df = convert_to_forward_adj(df)

            # 计算所有指标
            signals = SwingSignals()
            df = signals.calculate_all(df)

            # 验证 RSI 范围
            valid_rsi = df['rsi14'].dropna()
            assert (valid_rsi >= 0).all()
            assert (valid_rsi <= 100).all()

            # 验证布林带关系
            valid_bb = df.dropna(subset=['bb_upper', 'bb_middle', 'bb_lower'])
            assert (valid_bb['bb_upper'] >= valid_bb['bb_middle']).all()
            assert (valid_bb['bb_middle'] >= valid_bb['bb_lower']).all()

            # 验证 ATR > 0
            valid_atr = df['atr'].dropna()
            assert (valid_atr > 0).all()

            print(f"✓ 600519 指标验证通过: RSI范围={valid_rsi.min():.1f}~{valid_rsi.max():.1f}")

        except ImportError:
            pytest.skip("StockData not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
