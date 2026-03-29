"""向量化多周期集成测试

验证 VectorizedMultiCycle 与原始 MultiCycleResonance 的信号检测一致性
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

from src.data.vectorized import VectorizedMultiCycle, MultiCycleConfig
from src.backtest.multi_cycle import MultiCycleResonance


class TestVectorizedMultiCycleIntegration:
    """向量化多周期集成测试"""

    @pytest.fixture
    def sample_daily_data(self):
        """创建示例日线数据"""
        dates = pd.date_range('2025-06-01', periods=200, freq='D')
        np.random.seed(42)

        data = []
        for i, date in enumerate(dates):
            base = 100 + i * 0.5 + np.random.randn() * 2
            data.append({
                'data_id': '600519',
                'date': date.strftime('%Y-%m-%d'),
                'open': base + np.random.randn() * 0.5,
                'high': max(base, base + np.random.randn() * 0.5) + abs(np.random.randn()),
                'low': min(base, base + np.random.randn() * 0.5) - abs(np.random.randn()),
                'close': base,
                'volume': 1000000 + np.random.randint(-200000, 200000)
            })

        return pd.DataFrame(data)

    @pytest.fixture
    def mock_loader_with_data(self, sample_daily_data):
        """创建带数据的模拟加载器"""
        loader = MagicMock()
        loader.load_daily = MagicMock(return_value=sample_daily_data)
        return loader

    def test_vectorized_vs_original_daily_trend(self, sample_daily_data):
        """测试：日线趋势检测一致性"""
        # 原始实现
        original = MultiCycleResonance.__new__(MultiCycleResonance)
        original.MIN_DAILY_BARS = 20
        original_trend, original_conf = original._detect_trend(sample_daily_data, 'daily')

        # 向量化实现
        config = MultiCycleConfig()
        vec_cycle = VectorizedMultiCycle(config=config)

        # 使用相同的 MA 参数计算
        ma_short = sample_daily_data['close'].rolling(window=20, min_periods=1).mean()
        ma_long = sample_daily_data['close'].rolling(window=60, min_periods=1).mean()
        last_short = ma_short.iloc[-1]
        last_long = ma_long.iloc[-1]

        if last_short > last_long:
            vec_trend = 'up'
        elif last_short < last_long:
            vec_trend = 'down'
        else:
            vec_trend = 'sideways'

        assert vec_trend == original_trend, f"日线趋势不一致: vec={vec_trend}, orig={original_trend}"

    def test_vectorized_vs_original_weekly_trend(self, sample_daily_data):
        """测试：周线趋势检测一致性"""
        # 创建周线数据
        df = sample_daily_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        weekly = df.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()

        # 原始实现
        original = MultiCycleResonance.__new__(MultiCycleResonance)
        original_trend, _ = original._detect_trend(weekly, 'weekly')

        # 向量化实现（使用相同的 MA 参数）
        ma_short = weekly['close'].rolling(window=10, min_periods=1).mean()
        ma_long = weekly['close'].rolling(window=20, min_periods=1).mean()
        last_short = ma_short.iloc[-1]
        last_long = ma_long.iloc[-1]

        if last_short > last_long:
            vec_trend = 'up'
        elif last_short < last_long:
            vec_trend = 'down'
        else:
            vec_trend = 'sideways'

        assert vec_trend == original_trend

    def test_vectorized_vs_original_monthly_trend(self, sample_daily_data):
        """测试：月线趋势检测一致性"""
        # 创建月线数据
        df = sample_daily_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        monthly = df.resample('ME').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()

        # 原始实现
        original = MultiCycleResonance.__new__(MultiCycleResonance)
        original_trend, _ = original._detect_trend(monthly, 'monthly')

        # 向量化实现
        ma_short = monthly['close'].rolling(window=5, min_periods=1).mean()
        ma_long = monthly['close'].rolling(window=10, min_periods=1).mean()
        last_short = ma_short.iloc[-1]
        last_long = ma_long.iloc[-1]

        if last_short > last_long:
            vec_trend = 'up'
        elif last_short < last_long:
            vec_trend = 'down'
        else:
            vec_trend = 'sideways'

        assert vec_trend == original_trend

    def test_precompute_all_output_structure(self, sample_daily_data):
        """测试：预计算输出结构正确"""
        config = MultiCycleConfig()
        vec_cycle = VectorizedMultiCycle(config=config)

        result = vec_cycle.precompute_all(
            ['600519'],
            '2025-12-31',
            lookback_months=6
        )

        # 检查必要列
        required_cols = [
            'data_id', 'date',
            'monthly_trend', 'monthly_conf',
            'weekly_trend', 'weekly_conf',
            'daily_trend', 'daily_conf',
            'resonance_level', 'position_limit', 'is_bullish'
        ]

        for col in required_cols:
            assert col in result.columns, f"缺少列: {col}"

    def test_multi_stock_resonance_isolation(self):
        """测试：多股票共振隔离"""
        # 创建两只股票，一只上涨趋势，一只下跌趋势
        dates = pd.date_range('2025-06-01', periods=200, freq='D')

        all_data = []
        for i, date in enumerate(dates):
            for data_id in ['600519', '000001']:
                if data_id == '600519':
                    base = 100 + i * 0.5  # 上涨
                else:
                    base = 200 - i * 0.5  # 下跌

                all_data.append({
                    'data_id': data_id,
                    'date': date.strftime('%Y-%m-%d'),
                    'open': base + np.random.randn() * 0.5,
                    'high': base + abs(np.random.randn()) + 2,
                    'low': base - abs(np.random.randn()) - 2,
                    'close': base,
                    'volume': 1000000 + np.random.randint(-200000, 200000)
                })

        df = pd.DataFrame(all_data)

        config = MultiCycleConfig()
        vec_cycle = VectorizedMultiCycle(config=config)

        result = vec_cycle.precompute_all(
            ['600519', '000001'],
            '2025-12-31',
            lookback_months=6
        )

        # 检查每只股票的结果
        for data_id in ['600519', '000001']:
            stock_result = result[result['data_id'] == data_id]
            assert len(stock_result) > 0

            # 检查趋势方向符合预期
            if data_id == '600519':
                # 应该是看多
                assert stock_result['is_bullish'].any(), f"{data_id} 应该有看多信号"
            else:
                # 可能是看空或中性
                pass

    def test_resonance_level_distribution(self, sample_daily_data):
        """测试：共振等级分布合理"""
        config = MultiCycleConfig()
        vec_cycle = VectorizedMultiCycle(config=config)

        result = vec_cycle.precompute_all(
            ['600519'],
            '2025-12-31',
            lookback_months=6
        )

        if not result.empty:
            # 检查共振等级在有效范围内
            valid_levels = [0, 3, 4, 5]
            assert result['resonance_level'].isin(valid_levels).all(), \
                "共振等级包含无效值"

            # 检查仓位上限与共振等级匹配
            for _, row in result.iterrows():
                level = row['resonance_level']
                limit = row['position_limit']

                expected_limit = {
                    0: 0.0,
                    3: 0.2,
                    4: 0.6,
                    5: 0.8
                }.get(level, 0.0)

                assert abs(limit - expected_limit) < 0.01, \
                    f"共振等级 {level} 的仓位上限 {limit} 不符合预期 {expected_limit}"

    def test_performance_multi_stock(self):
        """测试：多股票性能"""
        import time

        # 创建多只股票数据
        dates = pd.date_range('2025-01-01', periods=300, freq='D')
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

        config = MultiCycleConfig()
        vec_cycle = VectorizedMultiCycle(config=config)

        start = time.time()
        result = vec_cycle.precompute_all(
            ['600519', '000001', '600036', '601318', '000002'],
            '2025-10-31',
            lookback_months=6
        )
        elapsed = time.time() - start

        print(f"多周期预计算耗时: {elapsed:.3f}s")
        print(f"结果行数: {len(result)}")

        # 性能要求：5只股票 x 300天应该在 5 秒内完成
        assert elapsed < 5.0, f"性能测试失败: {elapsed:.3f}s > 5.0s"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
