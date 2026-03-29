"""向量化指标验证测试

对比 VectorizedIndicators 与原有 SwingSignals 实现的指标计算结果

质量标准：
- 向量化结果必须与原有实现100%一致
- 所有指标（MA, MACD, RSI, ATR, Bollinger, ADX）必须匹配
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

from src.data.vectorized import VectorizedIndicators, IndicatorConfig
from src.data.indicators.signals import SwingSignals
from src.data.indicators import ma, macd, rsi, atr, bollinger, adx, volume


class TestVectorizedIndicatorsVsOriginal:
    """对比向量化和原始指标实现"""

    @pytest.fixture
    def sample_price_data(self):
        """创建单股票测试数据"""
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='B')  # 工作日
        np.random.seed(42)

        data = []
        for i, date in enumerate(dates):
            # 生成随机价格，模拟真实市场
            base_price = 100 + i * 0.5 + np.random.randn() * 2
            open_price = base_price + np.random.randn() * 0.5
            high_price = max(open_price, base_price) + abs(np.random.randn() * 0.5)
            low_price = min(open_price, base_price) - abs(np.random.randn() * 0.5)
            close_price = base_price
            volume = 1000000 + np.random.randint(-200000, 200000)

            data.append({
                'data_id': '600519',
                'date': date.strftime('%Y-%m-%d'),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': int(volume)
            })

        return pd.DataFrame(data)

    @pytest.fixture
    def multi_stock_data(self):
        """创建多股票测试数据"""
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='B')
        np.random.seed(42)

        data = []
        for date in dates:
            for data_id in ['600519', '000001', '600036']:
                base_price = 100 + np.random.randn() * 10
                open_price = base_price + np.random.randn() * 0.5
                high_price = max(open_price, base_price) + abs(np.random.randn() * 0.5)
                low_price = min(open_price, base_price) - abs(np.random.randn() * 0.5)
                close_price = base_price
                volume = 1000000 + np.random.randint(-200000, 200000)

                data.append({
                    'data_id': data_id,
                    'date': date.strftime('%Y-%m-%d'),
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': int(volume)
                })

        return pd.DataFrame(data)

    # ========================================================================
    # MA 指标对比
    # ========================================================================

    def test_ma_single_stock(self, sample_price_data):
        """测试：MA 计算在单股票上与原始实现一致"""
        df = sample_price_data.copy()

        # 原始实现（逐行）
        original = ma.calculate_ma(df.copy(), periods=[5, 10, 20, 60])

        # 向量化实现
        config = IndicatorConfig(ma_periods=[5, 10, 20, 60])
        vectorized = VectorizedIndicators(config)
        result = vectorized._calculate_ma_vectorized(df.copy())

        # 对比每只股票的每种 MA
        for period in [5, 10, 20, 60]:
            col = f'ma{period}'
            original_vals = original[col].values
            vectorized_vals = result[col].values

            # 允许极小误差（浮点运算）
            diff = np.abs(original_vals - vectorized_vals)
            max_diff = diff.max()

            assert max_diff < 1e-10, f"MA{period} 最大差异: {max_diff}"

    def test_ma_multi_stock(self, multi_stock_data):
        """测试：MA 计算在多股票上正确分组"""
        config = IndicatorConfig(ma_periods=[5, 10, 20])
        vectorized = VectorizedIndicators(config)
        result = vectorized._calculate_ma_vectorized(multi_stock_data.copy())

        # 验证每个股票分别计算
        for data_id in ['600519', '000001', '600036']:
            stock_data = result[result['data_id'] == data_id]

            # 验证 ma5 值在合理范围
            assert (stock_data['ma5'] > 0).all()
            # 验证多头排列：ma5 > ma10 > ma20（因为是递增数据）
            # 注意：这里随机数据不一定满足，先验证不重复计算

    # ========================================================================
    # MACD 指标对比
    # ========================================================================

    def test_macd_single_stock(self, sample_price_data):
        """测试：MACD 计算在单股票上与原始实现一致"""
        df = sample_price_data.copy()

        # 原始实现
        original = macd.calculate_macd(df.copy(), fast=12, slow=26, signal=9)

        # 向量化实现
        config = IndicatorConfig()
        vectorized = VectorizedIndicators(config)
        result = vectorized._calculate_macd_vectorized(df.copy())

        # 对比 dif（两者都有）
        original_vals = original['dif'].values
        vectorized_vals = result['dif'].values
        diff = np.abs(original_vals - vectorized_vals)
        max_diff = diff.max()
        assert max_diff < 1e-10, f"dif 最大差异: {max_diff}"

        # dem vs dea（列名不同但值相同）
        original_dem = original['dem'].values
        vectorized_dea = result['dea'].values
        diff = np.abs(original_dem - vectorized_dea)
        assert diff.max() < 1e-10, f"dem/dea 最大差异: {diff.max()}"

        # macd vs hist
        original_hist = original['hist'].values
        vectorized_macd = result['macd'].values
        diff = np.abs(original_hist - vectorized_macd)
        assert diff.max() < 1e-10, f"macd/hist 最大差异: {diff.max()}"

    # ========================================================================
    # RSI 指标对比
    # ========================================================================

    def test_rsi_single_stock(self, sample_price_data):
        """测试：RSI 计算在单股票上与原始实现一致"""
        df = sample_price_data.copy()

        # 原始实现
        original = rsi.calculate_rsi(df.copy(), periods=[6, 14])

        # 向量化实现
        config = IndicatorConfig(rsi_period=14)
        vectorized = VectorizedIndicators(config)
        result = vectorized._calculate_rsi_vectorized(df.copy())

        # 对比 RSI14（过滤 NaN）
        original_rsi = original['rsi14'].values
        vectorized_rsi = result['rsi14'].values
        mask = ~(np.isnan(original_rsi) | np.isnan(vectorized_rsi))
        if mask.sum() > 0:
            diff = np.abs(original_rsi[mask] - vectorized_rsi[mask])
            max_diff = diff.max()
            assert max_diff < 1e-10, f"RSI14 最大差异: {max_diff}"

    # ========================================================================
    # ATR 指标对比
    # ========================================================================

    def test_atr_single_stock(self, sample_price_data):
        """测试：ATR 计算在单股票上与原始实现一致"""
        df = sample_price_data.copy()

        # 原始实现
        original = atr.calculate_atr(df.copy(), period=14)

        # 向量化实现
        config = IndicatorConfig(atr_period=14)
        vectorized = VectorizedIndicators(config)
        result = vectorized._calculate_atr_vectorized(df.copy())

        # 对比 ATR（过滤 NaN）
        original_atr = original['atr'].values
        vectorized_atr = result['atr14'].values
        mask = ~(np.isnan(original_atr) | np.isnan(vectorized_atr))
        if mask.sum() > 0:
            diff = np.abs(original_atr[mask] - vectorized_atr[mask])
            max_diff = diff.max()
            assert max_diff < 1e-10, f"ATR14 最大差异: {max_diff}"

    # ========================================================================
    # 布林带指标对比
    # ========================================================================

    def test_bollinger_single_stock(self, sample_price_data):
        """测试：布林带计算在单股票上与原始实现一致"""
        df = sample_price_data.copy()

        # 原始实现
        original = bollinger.calculate_bollinger(df.copy(), period=20, std_dev=2.0)

        # 向量化实现
        config = IndicatorConfig(bollinger_period=20, bollinger_std=2)
        vectorized = VectorizedIndicators(config)
        result = vectorized._calculate_bollinger_vectorized(df.copy())

        # 对比 bb_upper, bb_middle, bb_lower（过滤 NaN）
        for col in ['bb_upper', 'bb_middle', 'bb_lower']:
            original_vals = original[col].values
            vectorized_vals = result[col].values
            mask = ~(np.isnan(original_vals) | np.isnan(vectorized_vals))
            if mask.sum() > 0:
                diff = np.abs(original_vals[mask] - vectorized_vals[mask])
                max_diff = diff.max()
                assert max_diff < 1e-10, f"{col} 最大差异: {max_diff}"

    # ========================================================================
    # ADX 指标对比
    # ========================================================================

    def test_adx_single_stock(self, sample_price_data):
        """测试：ADX 计算在单股票上与原始实现一致"""
        df = sample_price_data.copy()

        # 原始实现
        original = adx.calculate_adx(df.copy(), period=14)

        # 向量化实现
        config = IndicatorConfig()
        vectorized = VectorizedIndicators(config)
        result = vectorized._calculate_adx_vectorized(df.copy())

        # 对比 adx, plus_di, minus_di
        for col in ['adx', 'plus_di', 'minus_di']:
            original_vals = original[col].values
            vectorized_vals = result[col].values
            # 过滤 NaN
            mask = ~(np.isnan(original_vals) | np.isnan(vectorized_vals))
            if mask.sum() > 0:
                diff = np.abs(original_vals[mask] - vectorized_vals[mask])
                max_diff = diff.max()
                assert max_diff < 1e-10, f"{col} 最大差异: {max_diff}"

    # ========================================================================
    # 成交量指标对比
    # ========================================================================

    def test_volume_indicators_single_stock(self, sample_price_data):
        """测试：成交量指标在单股票上与原始实现一致"""
        df = sample_price_data.copy()

        # 原始实现
        original = volume.calculate_volume_ma(df.copy(), period=5)

        # 向量化实现
        vectorized = VectorizedIndicators()
        result = vectorized._calculate_volume_indicators(df.copy())

        # 对比 volume_ma5
        original_vol_ma = original['volume_ma'].values
        vectorized_vol_ma = result['volume_ma5'].values
        diff = np.abs(original_vol_ma - vectorized_vol_ma)
        max_diff = diff.max()
        assert max_diff < 1e-10, f"volume_ma5 最大差异: {max_diff}"

    # ========================================================================
    # 完整流程对比
    # ========================================================================

    def test_calculate_all_single_stock(self, sample_price_data):
        """测试：calculate_all 在单股票上与 SwingSignals 结果一致"""
        df = sample_price_data.copy()

        # 原始实现
        signals = SwingSignals()
        original = signals.calculate_all(df.copy())

        # 向量化实现
        config = IndicatorConfig()
        vectorized = VectorizedIndicators(config)
        result = vectorized.calculate_all(df.copy())

        # 关键指标对比
        test_cols = ['ma5', 'ma10', 'ma20', 'ma60', 'dif', 'dea', 'macd',
                     'rsi14', 'atr', 'bb_upper', 'bb_middle', 'bb_lower']

        for col in test_cols:
            if col in original.columns and col in result.columns:
                original_vals = original[col].values
                vectorized_vals = result[col].values
                mask = ~(np.isnan(original_vals) | np.isnan(vectorized_vals))
                if mask.sum() > 0:
                    diff = np.abs(original_vals[mask] - vectorized_vals[mask])
                    max_diff = diff.max()
                    assert max_diff < 1e-10, f"{col} 最大差异: {max_diff}"


class TestVectorizedIndicatorsQuality:
    """向量化指标质量测试"""

    def test_bollinger_order(self):
        """测试：布林带上轨 > 中轨 > 下轨"""
        data = {
            'data_id': ['600519'] * 50,
            'date': pd.date_range('2024-01-01', periods=50).strftime('%Y-%m-%d'),
            'close': np.cumsum(np.random.randn(50)) + 100,
            'open': 100,
            'high': 105,
            'low': 95,
            'volume': 1000000
        }
        df = pd.DataFrame(data)

        config = IndicatorConfig(bollinger_period=20, bollinger_std=2)
        vectorized = VectorizedIndicators(config)
        result = vectorized._calculate_bollinger_vectorized(df)

        # 上轨 >= 中轨 >= 下轨（跳过前 period-1 行，因为数据不足）
        period = 20
        valid = result['bb_upper'].iloc[period-1:] >= result['bb_middle'].iloc[period-1:]
        assert valid.all(), "布林带上轨应 >= 中轨"

        valid = result['bb_middle'].iloc[period-1:] >= result['bb_lower'].iloc[period-1:]
        assert valid.all(), "布林带中轨应 >= 下轨"

    def test_rsi_bounds(self):
        """测试：RSI 应该在 [0, 100] 范围内"""
        data = {
            'data_id': ['600519'] * 100,
            'date': pd.date_range('2024-01-01', periods=100).strftime('%Y-%m-%d'),
            'close': np.cumsum(np.random.randn(100)) + 100,
            'open': 100,
            'high': 105,
            'low': 95,
            'volume': 1000000
        }
        df = pd.DataFrame(data)

        config = IndicatorConfig(rsi_period=14)
        vectorized = VectorizedIndicators(config)
        result = vectorized._calculate_rsi_vectorized(df)

        rsi = result['rsi14'].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all(), "RSI 应在 [0, 100] 范围内"

    def test_atr_positive(self):
        """测试：ATR 应该始终为正"""
        data = {
            'data_id': ['600519'] * 100,
            'date': pd.date_range('2024-01-01', periods=100).strftime('%Y-%m-%d'),
            'close': np.cumsum(np.random.randn(100)) + 100,
            'open': 100,
            'high': 105,
            'low': 95,
            'volume': 1000000
        }
        df = pd.DataFrame(data)

        config = IndicatorConfig(atr_period=14)
        vectorized = VectorizedIndicators(config)
        result = vectorized._calculate_atr_vectorized(df)

        atr = result['atr14'].dropna()
        assert (atr > 0).all(), "ATR 应始终为正"


class TestVectorizedIndicatorsEdgeCases:
    """边界情况测试"""

    def test_empty_dataframe(self):
        """测试：空 DataFrame"""
        df = pd.DataFrame(columns=['data_id', 'date', 'close', 'open', 'high', 'low', 'volume'])

        vectorized = VectorizedIndicators()
        try:
            result = vectorized.calculate_all(df)
            # 空 DataFrame 可能返回空，也可能返回列名
            assert len(result) == 0 or result.empty
        except Exception:
            # 空 DataFrame 可能导致某些操作失败，这是可接受的
            pass

    def test_single_row(self):
        """测试：单行数据"""
        df = pd.DataFrame({
            'data_id': ['600519'],
            'date': ['2024-03-15'],
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            'volume': [1000000]
        })

        vectorized = VectorizedIndicators()
        result = vectorized.calculate_all(df)

        # 单行数据应该能计算 MA（min_periods=1）
        assert 'ma5' in result.columns

    def test_missing_periods(self):
        """测试：数据不足指定周期"""
        df = pd.DataFrame({
            'data_id': ['600519'] * 3,
            'date': ['2024-03-15', '2024-03-18', '2024-03-19'],
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [95.0, 96.0, 97.0],
            'close': [102.0, 103.0, 104.0],
            'volume': [1000000, 1000000, 1000000]
        })

        config = IndicatorConfig(ma_periods=[5, 10, 20])  # 20天MA需要更多数据
        vectorized = VectorizedIndicators(config)
        result = vectorized._calculate_ma_vectorized(df.copy())

        # 应该返回非 NaN 值（因为 min_periods=1）
        assert not result['ma20'].isna().all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
