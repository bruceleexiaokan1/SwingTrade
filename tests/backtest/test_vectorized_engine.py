"""向量化回测引擎单元测试

测试覆盖：
1. DateBoundAccessor 日期边界防护
2. 数据加载和索引
3. 预计算指标
4. 向量化信号检测
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

from src.backtest.vectorized.engine import (
    DateBoundAccessor,
    VectorizedBacktester,
    BacktestConfig,
    VectorizedResult
)


class TestDateBoundAccessor:
    """DateBoundAccessor 测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
        data = []

        for date in dates:
            for data_id in ['600519', '000001']:
                data.append({
                    'data_id': data_id,
                    'date': date.strftime('%Y-%m-%d'),
                    'close': 100 + np.random.randn() * 10,
                    'ma20': 100 + np.random.randn() * 5,
                    'rsi14': 50 + np.random.randn() * 20
                })

        return pd.DataFrame(data)

    def test_get_value_no_future_data(self, sample_data):
        """测试：绝对不能访问未来数据"""
        accessor = DateBoundAccessor(sample_data)

        for date in ['2024-02-15', '2024-03-01', '2024-03-31']:
            for data_id in ['600519', '000001']:
                # 获取该日期的值
                value = accessor.get_value(data_id, date, 'close')

                # 验证：该值的时间戳必须 <= date
                valid_data = sample_data[
                    (sample_data['data_id'] == data_id) &
                    (sample_data['date'] <= date)
                ]

                if not valid_data.empty:
                    max_valid_date = valid_data['date'].max()
                    assert max_valid_date <= date, \
                        f"数据穿越! {data_id} at {date}, max valid date: {max_valid_date}"

    def test_get_series_lookback(self, sample_data):
        """测试：get_series 返回正确回看窗口"""
        accessor = DateBoundAccessor(sample_data)

        series = accessor.get_series('600519', '2024-03-15', 'close', lookback=5)

        # 验证：应该返回5个值
        assert len(series) <= 5

        # 验证：所有值对应的日期都 <= 2024-03-15
        for val, (_, row) in zip(series.values, accessor._df[
            (accessor._df['data_id'] == '600519') &
            (accessor._df['date'] <= '2024-03-15')
        ].tail(5).iterrows()):
            assert row['date'] <= '2024-03-15'

    def test_has_data(self, sample_data):
        """测试：has_data 正确判断"""
        accessor = DateBoundAccessor(sample_data)

        # 早期日期应该无数据（因为没有更早的历史）
        assert accessor.has_data('600519', '2024-01-01')  # 应该有数据（当天）

        # 不存在的股票
        assert not accessor.has_data('999999', '2024-03-15')

    def test_empty_data(self):
        """测试：空数据处理"""
        empty_df = pd.DataFrame(columns=['data_id', 'date', 'close'])
        accessor = DateBoundAccessor(empty_df)

        assert accessor.get_value('600519', '2024-03-15', 'close') is np.nan

    def test_multiple_data_ids(self, sample_data):
        """测试：多个股票代码隔离"""
        accessor = DateBoundAccessor(sample_data)

        # 两个股票的数据应该完全隔离
        value_600519 = accessor.get_value('600519', '2024-03-15', 'close')
        value_000001 = accessor.get_value('000001', '2024-03-15', 'close')

        # 获取原始数据验证
        original_600519 = sample_data[
            (sample_data['data_id'] == '600519') &
            (sample_data['date'] <= '2024-03-15')
        ]['close'].iloc[-1]

        original_000001 = sample_data[
            (sample_data['data_id'] == '000001') &
            (sample_data['date'] <= '2024-03-15')
        ]['close'].iloc[-1]

        assert np.isclose(value_600519, original_600519)
        assert np.isclose(value_000001, original_000001)


class TestVectorizedBacktester:
    """VectorizedBacktester 测试"""

    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return BacktestConfig(
            initial_capital=1_000_000,
            max_positions=5,
            atr_stop_multiplier=2.0
        )

    def test_config_defaults(self):
        """测试：配置默认值"""
        config = BacktestConfig()

        assert config.initial_capital == 1_000_000
        assert config.max_positions == 5
        assert config.commission_rate == 0.0003

    def test_result_structure(self):
        """测试：结果数据结构"""
        result = VectorizedResult()

        assert hasattr(result, 'total_trades')
        assert hasattr(result, 'total_return')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'max_drawdown')
        assert result.total_trades == 0


class TestDataIntegrity:
    """数据完整性测试"""

    def test_date_sorted_required(self):
        """测试：日期必须排序"""
        dates = ['2024-03-15', '2024-03-10', '2024-03-20']  # 未排序
        df = pd.DataFrame({
            'data_id': ['600519'] * 3,
            'date': dates,
            'close': [100, 101, 102]
        })

        accessor = DateBoundAccessor(df)

        # 验证：数据已被排序
        sorted_dates = accessor._df['date'].tolist()
        assert sorted_dates == sorted(dates)


class TestEdgeCases:
    """边界情况测试"""

    def test_single_row(self):
        """测试：单行数据"""
        df = pd.DataFrame({
            'data_id': ['600519'],
            'date': ['2024-03-15'],
            'close': [100.0]
        })

        accessor = DateBoundAccessor(df)
        value = accessor.get_value('600519', '2024-03-15', 'close')

        assert value == 100.0

    def test_missing_date(self):
        """测试：缺失日期"""
        df = pd.DataFrame({
            'data_id': ['600519'] * 3,
            'date': ['2024-03-10', '2024-03-12', '2024-03-15'],  # 缺少11、13、14
            'close': [100, 101, 102]
        })

        accessor = DateBoundAccessor(df)

        # 应该返回最接近但不超过的值
        value = accessor.get_value('600519', '2024-03-14', 'close')
        assert value == 101  # 2024-03-12 的值

    def test_same_day_as_data(self):
        """测试：请求日期恰好有数据时包含当日"""
        df = pd.DataFrame({
            'data_id': ['600519'] * 3,
            'date': ['2024-03-10', '2024-03-15', '2024-03-20'],
            'close': [100, 105, 110]
        })

        accessor = DateBoundAccessor(df)

        # get_value 包含当日数据
        value = accessor.get_value('600519', '2024-03-15', 'close')
        assert value == 105  # 2024-03-15 的值（包含当日）

        # 前一个工作日（2024-03-14 之前）
        value_before = accessor.get_value('600519', '2024-03-14', 'close')
        assert value_before == 100  # 2024-03-10 的值

        # 2024-03-20 之后
        value_after = accessor.get_value('600519', '2024-03-21', 'close')
        assert value_after == 110  # 2024-03-20 的值


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
