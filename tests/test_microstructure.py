"""市场微观结构指标测试

Tests for market microstructure indicators
"""

import numpy as np
import pandas as pd
import pytest

from src.data.indicators.microstructure import (
    calculate_amihud_illiq,
    calculate_order_imbalance,
    calculate_vpin,
    detect_volume_anomaly,
    liquidity_regime_detection,
    estimate_market_impact,
)


class TestCalculateAmihudIlliq:
    """测试 Amihud ILLIQ 计算"""

    def test_basic_calculation(self):
        """基本计算测试"""
        np.random.seed(42)
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005] * 4)
        volume = pd.Series([1000000] * 20)  # 固定成交额

        illiq = calculate_amihud_illiq(returns, volume, window=5)

        assert len(illiq) == 20
        assert illiq.notna().all()
        assert illiq.iloc[-1] >= 0  # 始终非负

    def test_zero_returns(self):
        """零收益率测试"""
        returns = pd.Series([0.0] * 10)
        volume = pd.Series([1000000] * 10)

        illiq = calculate_amihud_illiq(returns, volume)

        assert illiq.notna().all()
        assert illiq.iloc[-1] == 0.0

    def test_large_volume(self):
        """大成交量测试（流动性好）"""
        returns = pd.Series([0.01] * 10)
        volume = pd.Series([10000000] * 10)  # 大成交额

        illiq = calculate_amihud_illiq(returns, volume)

        # 大成交额应该导致低 ILLIQ
        assert illiq.iloc[-1] < 0.01

    def test_small_volume(self):
        """小成交量测试（流动性差）"""
        returns = pd.Series([0.01] * 10)
        volume = pd.Series([100000] * 10)  # 小成交额

        illiq = calculate_amihud_illiq(returns, volume)

        # 小成交额应该导致高 ILLIQ（相对于大成交额情况）
        assert illiq.iloc[-1] > 0

    def test_insufficient_data(self):
        """数据不足测试"""
        returns = pd.Series([0.01])
        volume = pd.Series([1000000])

        illiq = calculate_amihud_illiq(returns, volume, window=20)

        # 窗口大于数据长度时应返回所有可用值
        assert len(illiq) == 1

    def test_length_mismatch_error(self):
        """长度不匹配错误"""
        returns = pd.Series([0.01] * 10)
        volume = pd.Series([1000000] * 5)

        with pytest.raises(ValueError, match="same length"):
            calculate_amihud_illiq(returns, volume)

    def test_negative_volume_error(self):
        """负成交量错误"""
        returns = pd.Series([0.01] * 10)
        volume = pd.Series([-1000000] * 10)

        with pytest.raises(ValueError, match="positive"):
            calculate_amihud_illiq(returns, volume)


class TestCalculateOrderImbalance:
    """测试订单不平衡度计算"""

    def test_basic_calculation(self):
        """基本计算测试"""
        bid = pd.Series([1000, 1200, 800, 1500])
        ask = pd.Series([800, 1000, 1200, 1000])

        oi = calculate_order_imbalance(bid, ask)

        assert len(oi) == 4
        assert -1 <= oi.iloc[-1] <= 1

    def test_balanced_market(self):
        """平衡市场测试"""
        bid = pd.Series([1000] * 10)
        ask = pd.Series([1000] * 10)

        oi = calculate_order_imbalance(bid, ask)

        assert oi.notna().all()
        np.testing.assert_almost_equal(oi.iloc[-1], 0.0, decimal=10)

    def test_bid_dominant(self):
        """买盘主导测试"""
        bid = pd.Series([2000] * 5)
        ask = pd.Series([1000] * 5)

        oi = calculate_order_imbalance(bid, ask)

        assert oi.iloc[-1] > 0
        np.testing.assert_almost_equal(oi.iloc[-1], 1/3, decimal=5)

    def test_ask_dominant(self):
        """卖盘主导测试"""
        bid = pd.Series([1000] * 5)
        ask = pd.Series([2000] * 5)

        oi = calculate_order_imbalance(bid, ask)

        assert oi.iloc[-1] < 0
        np.testing.assert_almost_equal(oi.iloc[-1], -1/3, decimal=5)

    def test_window_smoothing(self):
        """窗口平滑测试"""
        bid = pd.Series([1000, 2000, 1500, 1800])
        ask = pd.Series([1000, 1000, 1000, 1000])

        oi = calculate_order_imbalance(bid, ask, window=3)

        # 平滑后值应该更接近零
        assert -1 <= oi.iloc[-1] <= 1

    def test_zero_total_volume(self):
        """总成交量为零测试"""
        bid = pd.Series([0] * 5)
        ask = pd.Series([0] * 5)

        oi = calculate_order_imbalance(bid, ask)

        # 应该返回 NaN 而非 inf
        assert oi.isna().all()


class TestCalculateVPIN:
    """测试 VPIN 计算"""

    def test_basic_calculation(self):
        """基本计算测试"""
        volume = pd.Series([1000] * 10)
        buy = pd.Series([600, 400, 700, 300, 500, 500, 800, 200, 550, 450])
        sell = pd.Series([400, 600, 300, 700, 500, 500, 200, 800, 450, 550])

        vpin = calculate_vpin(volume, buy, sell, window=5)

        assert len(vpin) == 10
        assert 0 <= vpin.iloc[-1] <= 1

    def test_all_buy(self):
        """全买单测试"""
        volume = pd.Series([1000] * 10)
        buy = pd.Series([1000] * 10)
        sell = pd.Series([0] * 10)

        vpin = calculate_vpin(volume, buy, sell)

        # 全买时 VPIN 应该接近 1
        assert vpin.iloc[-1] >= 0.9

    def test_all_sell(self):
        """全卖单测试"""
        volume = pd.Series([1000] * 10)
        buy = pd.Series([0] * 10)
        sell = pd.Series([1000] * 10)

        vpin = calculate_vpin(volume, buy, sell)

        # 全卖时 VPIN 应该接近 1
        assert vpin.iloc[-1] >= 0.9

    def test_balanced_trade(self):
        """平衡交易测试"""
        volume = pd.Series([1000] * 10)
        buy = pd.Series([500] * 10)
        sell = pd.Series([500] * 10)

        vpin = calculate_vpin(volume, buy, sell)

        # 平衡时 VPIN 应该接近 0
        assert vpin.iloc[-1] < 0.1

    def test_window_effect(self):
        """窗口效应测试"""
        volume = pd.Series([1000] * 20)
        buy = pd.Series([800] * 10 + [200] * 10)
        sell = pd.Series([200] * 10 + [800] * 10)

        vpin_short = calculate_vpin(volume, buy, sell, window=5)
        vpin_long = calculate_vpin(volume, buy, sell, window=15)

        # 长窗口应该更平滑（标准差更小或相等）
        assert vpin_long.std() <= vpin_short.std()


class TestDetectVolumeAnomaly:
    """测试成交量异常检测"""

    def test_basic_detection(self):
        """基本检测测试"""
        np.random.seed(42)
        volume = pd.Series([1000] * 30)
        volume.iloc[-5:] = 3000  # 放量

        result = detect_volume_anomaly(volume, window=20)

        assert 'z_score' in result.columns
        assert 'is_surge' in result.columns
        assert 'is_extreme' in result.columns
        assert 'is_shrink' in result.columns

    def test_surge_detection(self):
        """放量检测测试"""
        np.random.seed(42)
        volume = pd.Series([1000] * 30)
        volume.iloc[-1] = 5000  # 显著放量

        result = detect_volume_anomaly(volume, window=20, threshold_low=2.0)

        assert result['is_surge'].iloc[-1] == True

    def test_extreme_detection(self):
        """极端放量检测测试"""
        np.random.seed(42)
        volume = pd.Series([1000] * 30)
        volume.iloc[-1] = 10000  # 极端放量

        result = detect_volume_anomaly(volume, window=20, threshold_high=3.0)

        assert result['is_extreme'].iloc[-1] == True

    def test_shrink_detection(self):
        """缩量检测测试"""
        np.random.seed(42)
        volume = pd.Series([1000] * 30)
        volume.iloc[-1] = 100  # 显著缩量

        result = detect_volume_anomaly(volume, window=20)

        assert result['is_shrink'].iloc[-1] == True

    def test_normal_volume(self):
        """正常成交量测试"""
        np.random.seed(42)
        volume = pd.Series([1000] * 50)  # 无异常

        result = detect_volume_anomaly(volume, window=20)

        # 正常情况下不应触发异常
        assert result['is_extreme'].sum() == 0

    def test_insufficient_data(self):
        """数据不足测试"""
        volume = pd.Series([1000])

        with pytest.raises(ValueError, match="at least 2"):
            detect_volume_anomaly(volume)

    def test_custom_thresholds(self):
        """自定义阈值测试"""
        np.random.seed(42)
        # Create volume with z_score between 2 and 3
        # Need some variation in base volume to avoid std = 0
        base_vol = list(np.random.normal(1000, 50, 25))
        last_vol = 1500  # Moderate surge, z_score should be ~2-3
        volume = pd.Series(base_vol + [last_vol])

        result = detect_volume_anomaly(
            volume, window=20,
            threshold_low=2.0,
            threshold_high=3.0
        )

        # 使用较低阈值应该检测到放量
        assert result['is_surge'].iloc[-1] == True


class TestLiquidityRegimeDetection:
    """测试流动性状态检测"""

    def test_basic_detection(self):
        """基本检测测试"""
        np.random.seed(42)
        illiq = pd.Series([0.001] * 30)
        volume = pd.Series([1000000] * 30)
        returns = pd.Series([0.01] * 30)

        result = liquidity_regime_detection(illiq, volume, returns, window=20)

        assert 'regime' in result.columns
        assert 'regime_code' in result.columns
        assert result['regime'].isin(['high', 'medium', 'low']).all()

    def test_regime_values(self):
        """状态值测试"""
        np.random.seed(42)
        # 创建不同流动性的数据 - 确保有足够的分散度
        illiq = pd.Series([0.0001] * 5 + [0.0005] * 5 + [0.001] * 5 + [0.005] * 5 + [0.01] * 5 + [0.02] * 5)
        volume = pd.Series([1000000] * 30)
        returns = pd.Series([0.01] * 30)

        result = liquidity_regime_detection(illiq, volume, returns)

        # 应该包含所有三种状态
        regimes = result['regime'].unique()
        assert 'high' in regimes
        assert 'medium' in regimes
        assert 'low' in regimes

    def test_regime_code_mapping(self):
        """状态编码映射测试"""
        np.random.seed(42)
        # 使用线性递增数据确保三种状态都存在
        illiq = pd.Series(np.linspace(0.0001, 0.02, 30))
        volume = pd.Series([1000000] * 30)
        returns = pd.Series([0.01] * 30)

        result = liquidity_regime_detection(illiq, volume, returns)

        # 验证编码映射
        high_codes = result.loc[result['regime'] == 'high', 'regime_code'].unique()
        low_codes = result.loc[result['regime'] == 'low', 'regime_code'].unique()

        assert 1 in high_codes
        assert -1 in low_codes

    def test_volatility_calculation(self):
        """波动率计算测试"""
        np.random.seed(42)
        illiq = pd.Series([0.001] * 30)
        volume = pd.Series([1000000] * 30)
        returns = pd.Series(np.random.randn(30) * 0.02)

        result = liquidity_regime_detection(illiq, volume, returns, window=20)

        assert 'volatility' in result.columns
        # 窗口内的值应该有波动率（窗口前的值为NaN）
        assert result['volatility'].iloc[20:].notna().all()

    def test_turnover_calculation(self):
        """换手率计算测试"""
        np.random.seed(42)
        illiq = pd.Series([0.001] * 30)
        volume = pd.Series([1000000] * 30)
        returns = pd.Series([0.01] * 30)

        result = liquidity_regime_detection(illiq, volume, returns)

        assert 'turnover' in result.columns
        # 换手率应该接近 1（因为用的是均值比）
        np.testing.assert_almost_equal(
            result['turnover'].iloc[-1],
            1.0,
            decimal=1
        )


class TestEstimateMarketImpact:
    """测试冲击成本估算"""

    def test_basic_calculation(self):
        """基本计算测试"""
        impact = estimate_market_impact(
            order_size=100000,
            avg_daily_volume=1000000
        )

        assert 'impact_pct' in impact
        assert 'participation_rate' in impact
        assert 'advice' in impact
        assert 'risk_level' in impact

    def test_impact_proportional(self):
        """冲击与订单规模成正比测试"""
        impact_small = estimate_market_impact(100000, 1000000)
        impact_large = estimate_market_impact(400000, 1000000)

        # 4倍订单规模应该产生更大的冲击
        assert impact_large['impact_pct'] > impact_small['impact_pct']

    def test_participation_rate(self):
        """参与率计算测试"""
        impact = estimate_market_impact(100000, 1000000)

        assert impact['participation_rate'] == 0.1

    def test_low_risk_scenario(self):
        """低风险场景测试"""
        impact = estimate_market_impact(
            order_size=5000,
            avg_daily_volume=1000000
        )

        assert impact['risk_level'] == 'low'
        assert impact['impact_pct'] < 1.0

    def test_medium_risk_scenario(self):
        """中等风险场景测试"""
        impact = estimate_market_impact(
            order_size=100000,
            avg_daily_volume=1000000
        )

        assert impact['risk_level'] == 'medium'

    def test_high_risk_scenario(self):
        """高风险场景测试"""
        impact = estimate_market_impact(
            order_size=500000,
            avg_daily_volume=1000000
        )

        assert impact['risk_level'] == 'high'
        assert '分批' in impact['advice']

    def test_advice_content(self):
        """建议内容测试"""
        impact = estimate_market_impact(
            order_size=300000,
            avg_daily_volume=1000000
        )

        assert isinstance(impact['advice'], str)
        assert len(impact['advice']) > 0

    def test_zero_adv_error(self):
        """零日均成交错误"""
        with pytest.raises(ValueError, match="positive"):
            estimate_market_impact(100000, 0)

    def test_negative_order_error(self):
        """负订单规模错误"""
        with pytest.raises(ValueError, match="positive"):
            estimate_market_impact(-100000, 1000000)

    def test_large_order_warning(self):
        """大订单警告测试"""
        impact = estimate_market_impact(
            order_size=500000,
            avg_daily_volume=1000000
        )

        # 50% 参与率应该触发警告
        assert impact['participation_rate'] == 0.5
        assert impact['risk_level'] == 'high'


class TestEdgeCases:
    """边界情况测试"""

    def test_all_zero_returns(self):
        """全零收益率测试"""
        returns = pd.Series([0.0] * 20)
        volume = pd.Series([1000000] * 20)

        illiq = calculate_amihud_illiq(returns, volume)

        assert illiq.notna().all()
        assert illiq.iloc[-1] == 0.0

    def test_identical_values(self):
        """相同值测试"""
        volume = pd.Series([1000] * 30)
        buy = pd.Series([500] * 30)
        sell = pd.Series([500] * 30)

        vpin = calculate_vpin(volume, buy, sell)

        # 恒定平衡交易应该产生恒定 VPIN
        assert vpin.iloc[-1] == 0.0

    def test_nan_handling(self):
        """NaN 处理测试"""
        bid = pd.Series([1000, np.nan, 1000, 1000])
        ask = pd.Series([1000, 1000, np.nan, 1000])

        oi = calculate_order_imbalance(bid, ask)

        # NaN 位置应该传播
        assert np.isnan(oi.iloc[1]) or np.isnan(oi.iloc[2])


class TestBackwardCompatibility:
    """向后兼容性别名测试"""

    def test_aliases_exist(self):
        """验证别名存在"""
        from src.data.indicators.microstructure import (
            amihud_illiq,
            order_imbalance,
            vpin,
            volume_anomaly,
            liquidity_regime,
            market_impact,
        )

        assert callable(amihud_illiq)
        assert callable(order_imbalance)
        assert callable(vpin)
        assert callable(volume_anomaly)
        assert callable(liquidity_regime)
        assert callable(market_impact)

    def test_alias_produces_same_result(self):
        """别名产生相同结果测试"""
        from src.data.indicators.microstructure import amihud_illiq

        np.random.seed(42)
        returns = pd.Series([0.01, -0.02, 0.015] * 5)
        volume = pd.Series([1000000] * 15)

        result1 = calculate_amihud_illiq(returns, volume)
        result2 = amihud_illiq(returns, volume)

        pd.testing.assert_series_equal(result1, result2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
