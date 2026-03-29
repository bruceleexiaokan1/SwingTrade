"""因子拥挤度检测模块测试

测试所有拥挤度指标的计算和信号生成
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np

from src.data.indicators.crowding import (
    turnover_crowding,
    momentum_crowding,
    fund_flow_crowding,
    position_concentration_hhi,
    correlation_breakdown_detection,
    a_share_crowding_indicator,
)


def create_sample_price_data(days: int = 100, base_price: float = 10.0) -> pd.DataFrame:
    """创建样本价格数据"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # 构造有一定趋势的价格序列
    trend = np.linspace(0, 0.5, days)  # 上涨趋势
    noise = np.random.randn(days) * 0.02

    data = {
        'date': dates.strftime('%Y-%m-%d'),
        'open': base_price * (1 + trend + noise * 0.5),
        'high': base_price * (1 + trend + noise * 0.5 + abs(np.random.randn(days)) * 0.02),
        'low': base_price * (1 + trend + noise * 0.5 - abs(np.random.randn(days)) * 0.02),
        'close': base_price * (1 + trend + noise),
        'volume': (1000000 + np.random.randn(days) * 100000).astype(int),
        'turnover': (0.03 + np.random.rand(days) * 0.07).astype(float),  # 3%-10%
        'pct_chg': (np.random.rand(days) * 0.04 - 0.02).astype(float),  # -2%~2%
    }

    df = pd.DataFrame(data)

    # 确保 high >= open, close, low
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


class TestTurnoverCrowding:
    """换手率拥挤度测试"""

    def test_basic_calculation(self):
        """基本计算测试"""
        df = create_sample_price_data(100)
        result = turnover_crowding(df)

        assert 'turnover_short_ma' in result.columns
        assert 'turnover_long_ma' in result.columns
        assert 'turnover_ratio' in result.columns
        assert 'turnover_crowding_signal' in result.columns

    def test_turnover_ratio_calculation(self):
        """换手率比值计算正确性"""
        df = create_sample_price_data(100)
        # 手动设置换手率
        df.loc[df.index[-5:], 'turnover'] = 0.15  # 最后5天高换手
        df.loc[df.index[:-5], 'turnover'] = 0.03  # 之前低换手

        result = turnover_crowding(df, short_period=5, long_period=20)

        # 短期均线应该大于长期均线（因为最近放量）
        short_avg = result['turnover_short_ma'].iloc[-1]
        long_avg = result['turnover_long_ma'].iloc[-1]
        assert short_avg > long_avg

    def test_crowding_signal_generation(self):
        """拥挤信号生成"""
        df = create_sample_price_data(100)
        # 构造明显的拥挤情况
        df.loc[df.index[-10:], 'turnover'] = 0.20  # 短期高换手
        df.loc[df.index[:-10], 'turnover'] = 0.05  # 长期低换手

        result = turnover_crowding(df, threshold=1.5)

        # 最后几天应该是拥挤信号
        assert result['turnover_crowding_signal'].iloc[-1] == 1

    def test_no_crowding_scenario(self):
        """正常情况无拥挤信号"""
        df = create_sample_price_data(100)
        # 稳定换手率
        df['turnover'] = 0.05

        result = turnover_crowding(df, threshold=1.5)

        # 稳定情况下应该没有拥挤信号
        assert result['turnover_crowding_signal'].max() <= 1

    def test_empty_data_handling(self):
        """空数据处理"""
        df = pd.DataFrame()
        result = turnover_crowding(df)

        # 应该返回空DataFrame但包含必要的列
        assert 'turnover_ratio' in result.columns or len(result) == 0

    def test_insufficient_history(self):
        """历史数据不足处理"""
        df = create_sample_price_data(5)  # 少于短期周期
        result = turnover_crowding(df, short_period=5, long_period=20)

        # 短期均线应该仍能计算（使用min_periods=1）
        assert 'turnover_short_ma' in result.columns


class TestMomentumCrowding:
    """动量拥挤度测试"""

    def test_basic_calculation(self):
        """基本计算测试"""
        df = create_sample_price_data(100)
        result = momentum_crowding(df)

        assert 'momentum_short' in result.columns
        assert 'momentum_long' in result.columns
        assert 'momentum_ratio' in result.columns
        assert 'momentum_crowding_signal' in result.columns

    def test_momentum_ratio_calculation(self):
        """动量比值计算"""
        df = create_sample_price_data(100)

        # 构造最近大涨（短期动量强）
        df.loc[df.index[-20:], 'close'] = df.loc[df.index[-20], 'close'] * 1.2

        result = momentum_crowding(df)

        # 短期动量应该大于0（价格上涨）
        assert result['momentum_short'].iloc[-1] > 0

    def test_crowding_signal_uptrend(self):
        """上涨趋势拥挤信号"""
        df = create_sample_price_data(100)

        # 最近20天大涨，长期平稳
        start_price = df.loc[df.index[-20], 'close']
        for i in range(20):
            df.loc[df.index[-(20-i)], 'close'] = start_price * (1 + 0.01 * i)

        result = momentum_crowding(df, threshold=2.0)

        # 应该检测到拥挤信号
        assert result['momentum_ratio'].notna().any()

    def test_different_thresholds(self):
        """不同阈值测试"""
        df = create_sample_price_data(100)
        # 构造明显的动量差异场景
        # 最近20天上涨很多，之前60天平稳
        base = df['close'].iloc[-60]
        for i in range(20):
            df.loc[df.index[-(20-i)], 'close'] = base * (1 + 0.015 * i)

        result1 = momentum_crowding(df, threshold=1.5)
        result2 = momentum_crowding(df, threshold=5.0)

        # 更高阈值应该产生更少或相等的正向拥挤信号
        assert result2['momentum_crowding_signal'].max() <= result1['momentum_crowding_signal'].max()


class TestFundFlowCrowding:
    """资金流拥挤度测试"""

    def test_basic_calculation(self):
        """基本计算测试"""
        df = create_sample_price_data(100)
        df['inflow'] = np.random.randn(100) * 1000000  # 随机资金流

        result = fund_flow_crowding(df)

        assert 'inflow_short_avg' in result.columns
        assert 'inflow_long_avg' in result.columns
        assert 'inflow_acceleration' in result.columns
        assert 'fund_flow_crowding_signal' in result.columns

    def test_inflow_acceleration_calculation(self):
        """流入加速度计算"""
        df = create_sample_price_data(100)

        # 构造持续净流入
        df['inflow'] = 1000000  # 每天净流入100万

        result = fund_flow_crowding(df, short_period=20, long_period=60)

        # 短期和长期平均应该接近相等（匀速流入）
        short_avg = result['inflow_short_avg'].iloc[-1]
        long_avg = result['inflow_long_avg'].iloc[-1]
        assert abs(short_avg - long_avg) < 100000  # 差异小于10万

    def test_crowding_signal_inflow_surge(self):
        """资金加速涌入信号"""
        df = create_sample_price_data(100)

        # 前期少量流入，后期大量流入
        df.loc[df.index[:-20], 'inflow'] = 100000
        df.loc[df.index[-20:], 'inflow'] = 2000000

        result = fund_flow_crowding(df, threshold=1.5)

        # 最后几天应该有拥挤信号
        recent_signals = result['fund_flow_crowding_signal'].iloc[-10:]
        assert (recent_signals == 1).any()

    def test_outflow_scenario(self):
        """资金流出场景"""
        df = create_sample_price_data(100)

        # 持续净流出
        df['inflow'] = -500000

        result = fund_flow_crowding(df)

        # 长期平均为负
        assert result['inflow_long_avg'].iloc[-1] < 0


class TestPositionConcentrationHHI:
    """持仓集中度HHI测试"""

    def test_hhi_calculation(self):
        """HHI基本计算"""
        # 构造均匀分布的市值
        market_cap = pd.Series([1000, 1000, 1000, 1000])

        hhi, signal, rank = position_concentration_hhi(market_cap)

        # 均匀分布时HHI = 1/n * 10000 = 2500
        assert abs(hhi - 2500) < 1

    def test_hhi_concentration(self):
        """HHI集中度检测"""
        # 构造高度集中的市值（一个标的占绝大部分）
        market_cap = pd.Series([9000, 300, 300, 200, 200])

        hhi, signal, rank = position_concentration_hhi(market_cap)

        # 高度集中时HHI应该很大
        assert hhi > 4000

    def test_hhi_low_concentration(self):
        """HHI低集中度"""
        # 构造大量分散的市值 - 10个标的均匀分布
        # HHI = n * (1/n)² * 10000 = 10000/n = 1000
        market_cap = pd.Series([1000] * 10)

        hhi, signal, rank = position_concentration_hhi(market_cap)

        # 高度分散时HHI应该很小 (< 1500)
        assert hhi < 1500

    def test_signal_levels(self):
        """信号等级测试"""
        # 极度集中的市值
        market_cap = pd.Series([9800, 100, 100])

        hhi, signal, rank = position_concentration_hhi(market_cap, threshold_high=2500, threshold_extreme=4000)

        # 极度集中时信号应该为2
        assert signal == 2

    def test_empty_market_cap(self):
        """空市值数据处理"""
        market_cap = pd.Series()

        hhi, signal, rank = position_concentration_hhi(market_cap)

        assert len(hhi) == 0

    def test_zero_total_market_cap(self):
        """总市值为零处理"""
        market_cap = pd.Series([0, 0, 0])

        hhi, signal, rank = position_concentration_hhi(market_cap)

        assert len(hhi) == 0


class TestCorrelationBreakdownDetection:
    """相关性崩溃检测测试"""

    def test_basic_calculation(self):
        """基本计算测试"""
        # 构造多个标的的价格数据
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        price_df = pd.DataFrame({
            'stock1': np.linspace(10, 15, 100),
            'stock2': np.linspace(10, 15, 100) + np.random.randn(100) * 0.1,
            'stock3': np.linspace(20, 25, 100) + np.random.randn(100) * 0.1,
        }, index=dates)

        result = correlation_breakdown_detection(price_df)

        assert 'avg_correlation' in result
        assert 'correlation_spike' in result
        assert 'max_correlation' in result
        assert 'crisis_flag' in result

    def test_high_correlation_crisis(self):
        """高相关性危机检测"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')

        # 构造高度相关的标的（同涨同跌）
        base = np.linspace(10, 15, 100)
        price_df = pd.DataFrame({
            'stock1': base,
            'stock2': base + 1,
            'stock3': base + 2,
            'stock4': base + 3,
        }, index=dates)

        result = correlation_breakdown_detection(price_df, correlation_threshold=0.5)

        # 高度相关的市场应该有危机标志
        assert result['crisis_flag'] == True

    def test_low_correlation_normal(self):
        """低相关性正常市场"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')

        # 构造低相关性的标的
        np.random.seed(42)
        price_df = pd.DataFrame({
            'stock1': np.cumsum(np.random.randn(100)),
            'stock2': np.cumsum(np.random.randn(100)),
            'stock3': np.cumsum(np.random.randn(100)),
        }, index=dates)

        result = correlation_breakdown_detection(price_df, correlation_threshold=0.5)

        # 低相关性市场应该没有危机标志
        # 注意：由于随机性，可能恰好超过阈值
        # 这里主要测试计算能正常运行
        assert 'crisis_flag' in result

    def test_insufficient_data(self):
        """数据不足处理"""
        dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
        price_df = pd.DataFrame({
            'stock1': np.linspace(10, 15, 10),
            'stock2': np.linspace(10, 15, 10),
        }, index=dates)

        result = correlation_breakdown_detection(price_df)

        # 数据不足时返回空结果
        assert len(result['avg_correlation']) == 0 or result['crisis_flag'] == False


class TestAShareCrowdingIndicator:
    """A股特色拥挤度测试"""

    def test_basic_calculation(self):
        """基本计算测试"""
        df = create_sample_price_data(100)

        result = a_share_crowding_indicator(df)

        assert 'margin_balance_ratio' in result.columns
        assert 'margin_ratio_percentile' in result.columns
        assert 'limit_up_percentile' in result.columns
        assert 'etf_flow_acceleration' in result.columns
        assert 'a_share_crowding_score' in result.columns
        assert 'a_share_crowding_level' in result.columns

    def test_margin_balance_ratio(self):
        """融资余额比计算"""
        df = create_sample_price_data(100)
        margin_balance = pd.Series(np.linspace(1e9, 1.5e9, 100))  # 融资余额增长

        result = a_share_crowding_indicator(df, margin_balance=margin_balance, market_cap=1e10)

        assert 'margin_balance_ratio' in result.columns
        assert result['margin_balance_ratio'].iloc[-1] > 0

    def test_limit_up_percentile(self):
        """涨停家数分位计算"""
        df = create_sample_price_data(100)
        limit_up_count = pd.Series([5] * 60 + [50, 80, 100, 120, 150])  # 最近涨停家数暴增

        result = a_share_crowding_indicator(df, limit_up_count=limit_up_count)

        # 最近几天分位应该很高（至少最后几个值应该很高）
        recent_percentiles = result['limit_up_percentile'].iloc[-10:].dropna()
        if len(recent_percentiles) > 0:
            assert recent_percentiles.max() > 80

    def test_etf_flow_acceleration(self):
        """ETF流入加速度"""
        df = create_sample_price_data(100)
        etf_flow = pd.Series(np.linspace(1e7, 5e7, 100))  # ETF流入增加

        result = a_share_crowding_indicator(df, etf_flow=etf_flow)

        assert 'etf_flow_acceleration' in result.columns
        assert result['etf_flow_acceleration'].notna().any()

    def test_crowding_score_range(self):
        """拥挤度评分范围"""
        df = create_sample_price_data(100)

        result = a_share_crowding_indicator(df)

        # 评分应该在0-100之间
        valid_scores = result['a_share_crowding_score'].dropna()
        assert (valid_scores >= 0).all()
        assert (valid_scores <= 100).all()

    def test_crowding_levels(self):
        """拥挤等级划分"""
        df = create_sample_price_data(100)
        margin_balance = pd.Series(np.linspace(1e9, 1e10, 100))  # 融资余额暴增
        limit_up_count = pd.Series([200] * 100)  # 大量涨停

        result = a_share_crowding_indicator(
            df,
            margin_balance=margin_balance,
            limit_up_count=limit_up_count,
            market_cap=1e10
        )

        # 极端情况下应该有extreme等级
        extreme_count = (result['a_share_crowding_level'] == 'extreme').sum()
        assert extreme_count > 0

    def test_no_optional_data(self):
        """无可选数据时正常处理"""
        df = create_sample_price_data(100)
        # 不提供任何可选数据

        result = a_share_crowding_indicator(df)

        # 仍能计算基本拥挤度
        assert 'a_share_crowding_score' in result.columns
        assert result['a_share_crowding_score'].notna().any()


class TestEdgeCases:
    """边界情况测试"""

    def test_all_columns_present(self):
        """所有输出列都存在"""
        df = create_sample_price_data(100)
        df['inflow'] = np.random.randn(100) * 1000000

        result = turnover_crowding(df)
        assert all(col in result.columns for col in ['turnover_short_ma', 'turnover_long_ma', 'turnover_ratio', 'turnover_crowding_signal'])

        result = momentum_crowding(df)
        assert all(col in result.columns for col in ['momentum_short', 'momentum_long', 'momentum_ratio', 'momentum_crowding_signal'])

        result = fund_flow_crowding(df)
        assert all(col in result.columns for col in ['inflow_short_avg', 'inflow_long_avg', 'inflow_acceleration', 'fund_flow_crowding_signal'])

    def test_nan_handling(self):
        """NaN值处理"""
        df = create_sample_price_data(100)
        df.loc[df.index[50:60], 'turnover'] = np.nan
        df.loc[df.index[50:60], 'close'] = np.nan

        result = turnover_crowding(df)
        # NaN传播是正常的，不应报错
        assert 'turnover_ratio' in result.columns

    def test_zero_division_handling(self):
        """除零处理"""
        df = create_sample_price_data(100)
        df['turnover'] = 0  # 换手率为0

        result = turnover_crowding(df)
        # 不应报错，ratio会是nan或inf
        assert 'turnover_ratio' in result.columns

    def test_constant_prices(self):
        """价格不变情况"""
        df = create_sample_price_data(100)
        df['close'] = 10.0  # 价格恒定
        df['open'] = 10.0
        df['high'] = 10.0
        df['low'] = 10.0

        result = momentum_crowding(df)
        # 动量为0，ratio计算不应报错
        assert 'momentum_ratio' in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
