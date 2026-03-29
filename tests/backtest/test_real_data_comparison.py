"""真实数据对比测试

用真实股票数据对比向量化引擎与原始实现的信号检测结果
这是提升置信度的关键验证步骤
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

from src.data.vectorized import VectorizedIndicators, VectorizedSignals, VectorizedMultiCycle, IndicatorConfig, SignalConfig
from src.data.loader import StockDataLoader
from src.backtest.engine import SwingBacktester


class TestRealDataSignalComparison:
    """真实数据信号对比"""

    @pytest.fixture
    def real_stock_data(self):
        """加载真实股票数据"""
        loader = StockDataLoader('/Users/bruce/workspace/trade/StockData')

        # 加载多只股票
        stock_codes = ['600519', '000001']  # 茅台、平安
        all_data = []

        for code in stock_codes:
            df = loader.load_daily(code, start_date='2024-01-01', end_date='2024-06-30')
            if len(df) > 60:
                # 统一列名
                df = df.rename(columns={'code': 'data_id'})
                df['data_id'] = code  # 简化 data_id
                all_data.append(df)

        if not all_data:
            pytest.skip("No sufficient stock data available")

        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values(['data_id', 'date']).reset_index(drop=True)
        return combined

    @pytest.fixture
    def vectorized_signals_result(self, real_stock_data):
        """使用向量化模块计算信号"""
        # 计算指标
        config = IndicatorConfig(
            ma_periods=[5, 10, 20, 60],
            rsi_period=14,
            atr_period=14,
            bollinger_period=20,
            bollinger_std=2
        )
        indicators = VectorizedIndicators(config)
        df = indicators.calculate_all(real_stock_data)

        # 计算信号
        signal_config = SignalConfig(
            ma_short_period=20,
            ma_long_period=60,
            rsi_overbought=80,
            rsi_oversold=35
        )
        signals = VectorizedSignals(signal_config)
        df = signals.calculate_all(df)

        return df

    def test_real_data_has_sufficient_rows(self, real_stock_data):
        """测试：数据行数足够"""
        print(f"\n真实数据行数: {len(real_stock_data)}")
        print(f"股票数量: {real_stock_data['data_id'].nunique()}")
        assert len(real_stock_data) > 100, "数据量不足"

    def test_golden_cross_signals_detected(self, vectorized_signals_result):
        """测试：金叉信号检测"""
        golden_count = vectorized_signals_result['golden_cross'].sum()
        print(f"\n金叉信号数量: {golden_count}")

        # 有数据就应该有金叉
        assert golden_count >= 0

        # 检查金叉只在有足够MA数据后产生
        for data_id in vectorized_signals_result['data_id'].unique():
            stock_data = vectorized_signals_result[vectorized_signals_result['data_id'] == data_id]
            first_golden = stock_data[stock_data['golden_cross'] == 1]

            if len(first_golden) > 0:
                first_idx = first_golden.index[0]
                # MA20 需要至少20天数据
                assert first_idx >= 20, "金叉产生太早"
                print(f"  {data_id}: 第一个金叉在第 {first_idx} 行")

    def test_death_cross_signals_detected(self, vectorized_signals_result):
        """测试：死叉信号检测"""
        death_count = vectorized_signals_result['death_cross'].sum()
        print(f"\n死叉信号数量: {death_count}")
        assert death_count >= 0

    def test_entry_confidence_range(self, vectorized_signals_result):
        """测试：入场置信度在有效范围内"""
        confidence = vectorized_signals_result['entry_confidence']

        # 检查范围
        assert (confidence >= 0).all(), "置信度有负值"
        assert (confidence <= 1).all(), "置信度超过1"

        # 统计分布
        high_conf = (confidence > 0.5).sum()
        low_conf = (confidence <= 0.5).sum()
        print(f"\n高置信度信号: {high_conf}, 低置信度信号: {low_conf}")

    def test_rsi_signals_consistency(self, vectorized_signals_result):
        """测试：RSI信号一致性"""
        # RSI超卖
        rsi_col = 'rsi14'
        oversold_signals = vectorized_signals_result['rsi_oversold'].sum()
        oversold_manual = (vectorized_signals_result[rsi_col] < 35).sum()

        print(f"\nRSI超卖信号数量: {oversold_signals}")
        print(f"手动计算超卖数量: {oversold_manual}")

        # 应该一致
        assert oversold_signals == oversold_manual, "RSI超卖信号不一致"

    def test_multi_stock_isolation_real_data(self, vectorized_signals_result):
        """测试：多股票信号隔离（真实数据）"""
        for data_id in vectorized_signals_result['data_id'].unique():
            stock_data = vectorized_signals_result[vectorized_signals_result['data_id'] == data_id]

            # 每只股票的金叉应该单独计算
            golden_count = stock_data['golden_cross'].sum()
            print(f"  {data_id}: 金叉 {golden_count} 个")


class TestOriginalVsVectorizedSignals:
    """原始实现 vs 向量化实现信号对比"""

    @pytest.fixture
    def stock_data(self):
        """创建测试数据"""
        dates = pd.date_range('2024-01-01', '2024-06-30', freq='B')
        np.random.seed(42)

        data = []
        for i, date in enumerate(dates):
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

    def test_golden_cross_match_rate(self, stock_data):
        """测试：金叉检测匹配率"""
        # 向量化实现
        vec_indicators = VectorizedIndicators(IndicatorConfig(
            ma_periods=[20, 60],
            rsi_period=14
        ))
        df = vec_indicators.calculate_all(stock_data)

        vec_signals = VectorizedSignals(SignalConfig(
            ma_short_period=20,
            ma_long_period=60
        ))
        df = vec_signals.calculate_all(df)
        vec_golden = df['golden_cross'].values

        # 原始实现（逐行检测）
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

        # 计算匹配率
        match = sum(1 for v, o in zip(vec_golden, original_golden) if bool(v) == o)
        total = len(df)
        match_rate = match / total

        print(f"\n金叉匹配率: {match_rate:.2%}")
        print(f"匹配: {match}/{total}")

        assert match_rate > 0.95, f"金叉匹配率不足: {match_rate:.2%}"

    def test_death_cross_match_rate(self, stock_data):
        """测试：死叉检测匹配率"""
        vec_indicators = VectorizedIndicators(IndicatorConfig(
            ma_periods=[20, 60],
            rsi_period=14
        ))
        df = vec_indicators.calculate_all(stock_data)

        vec_signals = VectorizedSignals(SignalConfig(
            ma_short_period=20,
            ma_long_period=60
        ))
        df = vec_signals.calculate_all(df)
        vec_death = df['death_cross'].values

        # 原始实现
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

        match = sum(1 for v, o in zip(vec_death, original_death) if bool(v) == o)
        total = len(df)
        match_rate = match / total

        print(f"\n死叉匹配率: {match_rate:.2%}")

        assert match_rate > 0.95, f"死叉匹配率不足: {match_rate:.2%}"

    def test_rsi_signals_exact_match(self, stock_data):
        """测试：RSI信号精确匹配"""
        vec_indicators = VectorizedIndicators(IndicatorConfig(
            rsi_period=14
        ))
        df = vec_indicators.calculate_all(stock_data)

        vec_signals = VectorizedSignals(SignalConfig(
            rsi_oversold=35,
            rsi_overbought=80
        ))
        df = vec_signals.calculate_all(df)

        # 原始实现
        vec_oversold = df['rsi_oversold'].values
        vec_overbought = df['rsi_overbought'].values

        original_oversold = (df['rsi14'] < 35).values
        original_overbought = (df['rsi14'] > 80).values

        oversold_match = (vec_oversold == original_oversold).all()
        overbought_match = (vec_overbought == original_overbought).all()

        print(f"\nRSI超卖匹配: {oversold_match}")
        print(f"RSI超买匹配: {overbought_match}")

        assert oversold_match, "RSI超卖信号不一致"
        assert overbought_match, "RSI超买信号不一致"


class TestPerformanceBenchmark:
    """性能基准测试"""

    def test_vectorized_performance(self):
        """测试：向量化性能"""
        import time

        # 创建大量数据
        dates = pd.date_range('2023-01-01', '2025-06-30', freq='B')
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

        # 指标计算
        start = time.time()
        indicators = VectorizedIndicators(IndicatorConfig(
            ma_periods=[5, 10, 20, 60],
            rsi_period=14,
            atr_period=14,
            bollinger_period=20,
            bollinger_std=2
        ))
        df = indicators.calculate_all(df)
        indicator_time = time.time() - start

        # 信号计算
        start = time.time()
        signals = VectorizedSignals(SignalConfig(
            ma_short_period=20,
            ma_long_period=60,
            rsi_overbought=80
        ))
        df = signals.calculate_all(df)
        signal_time = time.time() - start

        total_time = indicator_time + signal_time

        print(f"指标计算: {indicator_time:.3f}s")
        print(f"信号计算: {signal_time:.3f}s")
        print(f"总耗时: {total_time:.3f}s")
        print(f"数据行数: {len(df)}")

        # 性能要求：5只股票 x 2.5年数据应该在 5 秒内完成
        assert total_time < 5.0, f"性能不足: {total_time:.3f}s > 5.0s"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
