"""HMM市场状态识别测试"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np

from src.data.indicators.hmm_model import (
    HMMModel,
    HMMMarketRegime,
    HMMState,
    HMMResult,
    calculate_hmm_regime,
    detect_market_regime,
)


def create_sample_price_data(days: int = 100, base_price: float = 10.0) -> pd.DataFrame:
    """创建样本价格数据"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # 构造有趋势的价格序列
    trend = np.linspace(0, 0.3, days)  # 上涨趋势
    noise = np.random.randn(days) * 0.02

    data = {
        'date': dates.strftime('%Y-%m-%d'),
        'open': base_price * (1 + trend + noise * 0.5),
        'high': base_price * (1 + trend + noise * 0.5 + abs(np.random.randn(days)) * 0.02),
        'low': base_price * (1 + trend + noise * 0.5 - abs(np.random.randn(days)) * 0.02),
        'close': base_price * (1 + trend + noise),
        'volume': (1000000 + np.random.randn(days) * 100000).astype(int),
    }

    df = pd.DataFrame(data)
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


class TestHMMModel:
    """HMM模型测试"""

    def test_prepare_features(self):
        """特征准备测试"""
        df = create_sample_price_data(100)

        model = HMMModel()
        features = model.prepare_features(df)

        # 特征应该是 (n_features, n_samples) 形状
        assert features.shape[1] == len(df)
        # 至少有收益率和波动率两个特征
        assert features.shape[0] >= 2

    def test_prepare_features_with_volume(self):
        """带成交量的特征准备"""
        df = create_sample_price_data(100)

        model = HMMModel()
        features = model.prepare_features(df, volume_col='volume')

        # 应该有3个特征：收益率、波动率、成交量变化
        assert features.shape[0] == 3

    def test_name_states_by_characteristics(self):
        """状态命名测试"""
        # 构造三种不同特征的状态
        means = np.array([
            [0.0001, 0.01],   # 低收益低波动 -> 震荡
            [0.003, 0.02],    # 高收益中等波动 -> 上涨
            [-0.002, 0.025],  # 负收益高波动 -> 下跌
        ])

        model = HMMModel()
        state_names = model._name_states_by_characteristics(means)

        # 应该有3个状态
        assert len(state_names) == 3
        # 应该包含这三种类型
        assert set(state_names.values()) == {'sideways', 'uptrend', 'downtrend'}

    def test_fit_without_hmmlearn(self):
        """无hmmlearn库时的处理"""
        # 临时设置标志
        import src.data.indicators.hmm_model as hmm_module
        original = hmm_module.HAS_HMMLEARN
        hmm_module.HAS_HMMLEARN = False

        try:
            model = HMMModel()
            df = create_sample_price_data(60)
            features = model.prepare_features(df)

            # 尝试拟合应该抛出ImportError
            with pytest.raises(ImportError):
                model.fit(features)
        finally:
            hmm_module.HAS_HMMLEARN = original


class TestHMMMarketRegime:
    """HMM市场状态检测器测试"""

    def test_initialization(self):
        """初始化测试"""
        detector = HMMMarketRegime(n_states=3, lookback=60)

        assert detector.n_states == 3
        assert detector.lookback == 60
        assert detector.min_periods == 30

    def test_default_result(self):
        """默认结果测试"""
        detector = HMMMarketRegime()
        result = detector._default_result()

        assert result.state_name == 'unknown'
        assert result.regime_confidence == 0.0
        assert result.position_limit == 0.0

    def test_empty_data_handling(self):
        """空数据处理"""
        detector = HMMMarketRegime()
        df = pd.DataFrame()

        result = detector.detect_current_regime(df)
        assert result.state_name == 'unknown'

    def test_insufficient_data_handling(self):
        """数据不足处理"""
        detector = HMMMarketRegime(min_periods=60)
        df = create_sample_price_data(30)  # 少于60天

        result = detector.detect_current_regime(df)
        assert result.state_name == 'unknown'

    def test_get_action_and_position_uptrend(self):
        """上涨状态的操作和仓位"""
        detector = HMMMarketRegime()
        action, position = detector._get_action_and_position('uptrend', np.array([0.2, 0.7, 0.1]))

        assert action == '买入'
        assert position > 0.5

    def test_get_action_and_position_downtrend(self):
        """下跌状态的操作和仓位"""
        detector = HMMMarketRegime()
        action, position = detector._get_action_and_position('downtrend', np.array([0.7, 0.2, 0.1]))

        assert action == '卖出/对冲'
        assert position < 0.3

    def test_get_action_and_position_sideways(self):
        """震荡状态的操作和仓位"""
        detector = HMMMarketRegime()
        action, position = detector._get_action_and_position('sideways', np.array([0.1, 0.2, 0.7]))

        assert action == '观望'
        assert position <= 0.3


class TestHMMConvenienceFunctions:
    """便捷函数测试"""

    def test_calculate_hmm_regime_insufficient_data(self):
        """数据不足时返回空DataFrame"""
        df = create_sample_price_data(10)
        result = calculate_hmm_regime(df)

        assert len(result) == 0

    def test_detect_market_regime_insufficient_data(self):
        """数据不足时返回默认结果"""
        df = create_sample_price_data(10)
        result = detect_market_regime(df)

        assert result.state_name == 'unknown'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
