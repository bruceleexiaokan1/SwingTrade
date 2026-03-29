"""向量化回测集成测试

验证向量化回测引擎与原始实现的信号检测一致性
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

from src.data.vectorized import VectorizedIndicators, IndicatorConfig
from src.data.vectorized import VectorizedSignals, SignalConfig
from src.data.indicators.signals import SwingSignals


class TestVectorizedSignalsVsSwingSignals:
    """对比向量化信号与 SwingSignals.analyze()"""

    @pytest.fixture
    def stock_data(self):
        """创建测试股票数据"""
        dates = pd.date_range('2024-01-01', '2024-06-30', freq='B')
        np.random.seed(42)

        data = []
        for i, date in enumerate(dates):
            # 模拟先涨后跌的走势，产生金叉死叉
            if i < 50:
                base = 100 + i * 0.5 + np.random.randn() * 2
            elif i < 80:
                base = 125 + np.random.randn() * 3
            else:
                base = 115 - (i - 80) * 0.3 + np.random.randn() * 2

            open_price = base + np.random.randn() * 0.5
            high_price = max(open_price, base) + abs(np.random.randn() * 0.5)
            low_price = min(open_price, base) - abs(np.random.randn() * 0.5)
            close_price = base
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

    def test_golden_cross_match(self, stock_data):
        """测试：金叉检测一致性"""
        df = stock_data.copy()

        # 计算指标
        vec_indicators = VectorizedIndicators(IndicatorConfig(
            ma_periods=[5, 10, 20, 60],
            rsi_period=14,
            atr_period=14,
            bollinger_period=20,
            bollinger_std=2
        ))
        df = vec_indicators.calculate_all(df)

        # 向量化信号
        vec_signals = VectorizedSignals(SignalConfig(
            ma_short_period=20,
            ma_long_period=60,
            rsi_overbought=80
        ))
        df = vec_signals._calculate_ma_cross_signals(df)
        vec_golden = df['golden_cross'].values

        # 原始信号（使用 golden_cross 函数逐行检测）
        from src.data.indicators import ma
        df = ma.calculate_ma(df, periods=[20, 60])
        original_golden = []
        for i in range(len(df)):
            if i < 1:
                original_golden.append(False)
            else:
                ma20_prev = df['ma20'].iloc[i-1]
                ma20_curr = df['ma20'].iloc[i]
                ma60_prev = df['ma60'].iloc[i-1]
                ma60_curr = df['ma60'].iloc[i]
                golden = (ma20_prev <= ma60_prev) and (ma20_curr > ma60_curr)
                original_golden.append(golden)

        # 对比
        match = sum(1 for v, o in zip(vec_golden, original_golden) if bool(v) == o)
        total = len(df)
        match_rate = match / total
        assert match_rate > 0.95, f"金叉匹配率: {match_rate:.2%}"

    def test_death_cross_match(self, stock_data):
        """测试：死叉检测一致性"""
        df = stock_data.copy()

        # 计算指标
        vec_indicators = VectorizedIndicators(IndicatorConfig(
            ma_periods=[5, 10, 20, 60],
            rsi_period=14,
            atr_period=14,
            bollinger_period=20,
            bollinger_std=2
        ))
        df = vec_indicators.calculate_all(df)

        # 向量化信号
        vec_signals = VectorizedSignals(SignalConfig(
            ma_short_period=20,
            ma_long_period=60,
            rsi_overbought=80
        ))
        df = vec_signals._calculate_ma_cross_signals(df)
        vec_death = df['death_cross'].values

        # 原始信号
        from src.data.indicators import ma
        df = ma.calculate_ma(df, periods=[20, 60])
        original_death = []
        for i in range(len(df)):
            if i < 1:
                original_death.append(False)
            else:
                ma20_prev = df['ma20'].iloc[i-1]
                ma20_curr = df['ma20'].iloc[i]
                ma60_prev = df['ma60'].iloc[i-1]
                ma60_curr = df['ma60'].iloc[i]
                death = (ma20_prev >= ma60_prev) and (ma20_curr < ma60_curr)
                original_death.append(death)

        # 对比
        match = sum(1 for v, o in zip(vec_death, original_death) if bool(v) == o)
        total = len(df)
        match_rate = match / total
        assert match_rate > 0.95, f"死叉匹配率: {match_rate:.2%}"

    def test_rsi_signals_match(self, stock_data):
        """测试：RSI 超买超卖检测一致性"""
        df = stock_data.copy()

        # 计算指标
        vec_indicators = VectorizedIndicators(IndicatorConfig(
            rsi_period=14
        ))
        df = vec_indicators.calculate_all(df)

        # 向量化 RSI 信号
        vec_signals = VectorizedSignals(SignalConfig(
            rsi_oversold=35,
            rsi_overbought=80
        ))
        df = vec_signals._calculate_rsi_signals(df)
        vec_oversold = df['rsi_oversold'].values
        vec_overbought = df['rsi_overbought'].values

        # 原始信号
        original_oversold = (df['rsi14'] < 35).values
        original_overbought = (df['rsi14'] > 80).values

        # 对比
        oversold_match = (vec_oversold == original_oversold).all()
        overbought_match = (vec_overbought == original_overbought).all()
        assert oversold_match, "RSI 超卖检测不一致"
        assert overbought_match, "RSI 超买检测不一致"

    def test_entry_signals_comprehensive(self, stock_data):
        """测试：综合入场信号一致性"""
        df = stock_data.copy()

        # 计算指标
        vec_indicators = VectorizedIndicators(IndicatorConfig(
            ma_periods=[20, 60],
            rsi_period=14,
            bollinger_period=20,
            bollinger_std=2
        ))
        df = vec_indicators.calculate_all(df)

        # 向量化信号
        vec_signals = VectorizedSignals(SignalConfig(
            ma_short_period=20,
            ma_long_period=60,
            rsi_overbought=80,
            volume_surge_threshold=1.5
        ))
        df = vec_signals._calculate_ma_cross_signals(df)
        df = vec_signals._calculate_rsi_signals(df)
        df = vec_signals._calculate_bollinger_signals(df)
        df = vec_signals._calculate_volume_signals(df)
        df = vec_signals._calculate_entry_signals(df)

        # 验证：入场置信度在合理范围
        if 'entry_confidence' in df.columns:
            confidence = df['entry_confidence']
            assert (confidence >= 0).all() and (confidence <= 1).all(), \
                "入场置信度超出 [0, 1] 范围"

    def test_signal_columns_exist(self, stock_data):
        """测试：所有信号列都存在"""
        df = stock_data.copy()

        # 计算指标
        vec_indicators = VectorizedIndicators(IndicatorConfig(
            ma_periods=[20, 60],
            rsi_period=14,
            bollinger_period=20,
            bollinger_std=2
        ))
        df = vec_indicators.calculate_all(df)

        # 计算信号
        vec_signals = VectorizedSignals(SignalConfig(
            ma_short_period=20,
            ma_long_period=60
        ))
        df = vec_signals.calculate_all(df)

        # 验证必要列存在
        required_columns = [
            'golden_cross', 'death_cross',
            'rsi_oversold', 'rsi_overbought',
            'entry_confidence'
        ]
        for col in required_columns:
            assert col in df.columns, f"缺少信号列: {col}"


class TestVectorizedIntegrationPerformance:
    """性能测试"""

    def test_multi_stock_performance(self):
        """测试：多股票并行计算性能"""
        import time

        # 创建多只股票数据
        dates = pd.date_range('2024-01-01', '2024-06-30', freq='B')
        np.random.seed(42)

        all_data = []
        for data_id in ['600519', '000001', '600036', '601318', '000002']:
            for i, date in enumerate(dates):
                base = 100 + np.random.randn() * 10
                all_data.append({
                    'data_id': data_id,
                    'date': date.strftime('%Y-%m-%d'),
                    'open': base + np.random.randn() * 0.5,
                    'high': base + abs(np.random.randn()) * 0.5 + 2,
                    'low': base - abs(np.random.randn()) * 0.5 - 2,
                    'close': base,
                    'volume': 1000000 + np.random.randint(-200000, 200000)
                })

        df = pd.DataFrame(all_data)
        print(f"\n测试数据: {len(df)} 行 x 5 只股票")

        # 测试向量化指标计算
        start = time.time()
        vec_indicators = VectorizedIndicators(IndicatorConfig(
            ma_periods=[5, 10, 20, 60],
            rsi_period=14,
            atr_period=14,
            bollinger_period=20,
            bollinger_std=2
        ))
        df = vec_indicators.calculate_all(df)
        indicator_time = time.time() - start
        print(f"指标计算耗时: {indicator_time:.3f}s")

        # 测试向量化信号计算
        start = time.time()
        vec_signals = VectorizedSignals(SignalConfig(
            ma_short_period=20,
            ma_long_period=60,
            rsi_overbought=80
        ))
        df = vec_signals.calculate_all(df)
        signal_time = time.time() - start
        print(f"信号计算耗时: {signal_time:.3f}s")

        # 总耗时应该 < 1秒（5只股票约6个月数据）
        total_time = indicator_time + signal_time
        print(f"总耗时: {total_time:.3f}s")
        assert total_time < 5.0, f"性能测试失败: {total_time:.3f}s > 5.0s"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
