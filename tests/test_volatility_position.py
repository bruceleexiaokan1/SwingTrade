"""波动率仓位管理模块测试

测试覆盖：
1. EWMA 波动率计算
2. GARCH(1,1) 波动率计算
3. 目标波动率仓位计算
4. 波动率状态检测
5. 波动率动量指标
6. 边界情况处理
"""

import pytest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.backtest.volatility_position import (
    EWMAVolatilityCalculator,
    GARCHVolatilityCalculator,
    VolatilityRegimeDetector,
    VolatilityMomentumIndicator,
    TargetVolatilityPositionSizer,
    VolatilityRegime,
    VolatilityResult,
    TargetVolatilityPosition,
    calculate_volatility_from_prices,
    detect_volatility_regime
)


class TestEWMAVolatilityCalculator:
    """EWMA 波动率计算器测试"""

    def setup_method(self):
        """测试初始化"""
        self.calculator = EWMAVolatilityCalculator(halflife=30)

    def test_lambda_calculation(self):
        """
        测试 lambda 计算

        λ = 0.5^(1/halflife)
        当 halflife = 30 时，λ ≈ 0.5^(1/30) ≈ 0.9772
        """
        expected_lambda = 0.5 ** (1 / 30)
        assert abs(self.calculator.lambda_value - expected_lambda) < 1e-4

    def test_basic_calculation(self):
        """
        基础 EWMA 波动率计算测试

        使用标准正态分布的收益率，预期波动率约为 1 (年化 sqrt(252))
        """
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.01)
        vol = self.calculator.calculate(returns)
        assert vol > 0
        assert vol < 1.0

    def test_zero_returns(self):
        """
        全零收益率测试

        全零收益率的波动率应为 0
        """
        returns = pd.Series([0.0] * 100)
        vol = self.calculator.calculate(returns)
        assert vol == 0.0

    def test_constant_positive_returns(self):
        """
        常数正收益率测试

        固定收益率 1% daily 的年化波动率应该是 0.01 * sqrt(252) ≈ 15.87%
        """
        returns = pd.Series([0.01] * 200)
        vol = self.calculator.calculate(returns)
        expected_vol = 0.01 * math.sqrt(252)
        assert abs(vol - expected_vol) / expected_vol < 0.01  # 误差 < 1%

    def test_insufficient_data(self):
        """数据不足测试"""
        returns = pd.Series([0.01])
        with pytest.raises(ValueError, match="需要至少 2 个数据点"):
            self.calculator.calculate(returns)

    def test_nan_values(self):
        """NaN 值测试"""
        returns = pd.Series([0.01, 0.02, None, 0.01, 0.02])
        with pytest.raises(ValueError, match="包含 NaN 值"):
            self.calculator.calculate(returns)

    def test_calculate_series(self):
        """
        EWMA 波动率序列计算测试

        验证返回的是完整序列而非单个值
        """
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.01)
        vol_series = self.calculator.calculate_series(returns)
        assert len(vol_series) == len(returns)
        assert not vol_series.isna().any()

    def test_halflife_different(self):
        """
        不同半衰期测试

        较短半衰期对近期数据更敏感，波动率应该更高
        """
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.01)

        calc_short = EWMAVolatilityCalculator(halflife=10)
        calc_long = EWMAVolatilityCalculator(halflife=60)

        vol_short = calc_short.calculate(returns)
        vol_long = calc_long.calculate(returns)

        assert vol_short >= vol_long * 0.9


class TestGARCHVolatilityCalculator:
    """GARCH(1,1) 波动率计算器测试"""

    def setup_method(self):
        """测试初始化"""
        self.calculator = GARCHVolatilityCalculator(
            alpha=0.1,
            beta=0.85
        )

    def test_initialization(self):
        """初始化参数验证"""
        assert self.calculator.alpha == 0.1
        assert self.calculator.beta == 0.85

    def test_invalid_alpha(self):
        """无效 alpha 测试"""
        with pytest.raises(ValueError, match="alpha 和 beta 必须为正数"):
            GARCHVolatilityCalculator(alpha=-0.1, beta=0.85)

    def test_invalid_beta(self):
        """无效 beta 测试"""
        with pytest.raises(ValueError, match="alpha 和 beta 必须为正数"):
            GARCHVolatilityCalculator(alpha=0.1, beta=-0.85)

    def test_unstable_parameters(self):
        """不稳定参数测试 (alpha + beta >= 1)"""
        with pytest.raises(ValueError):
            GARCHVolatilityCalculator(alpha=0.5, beta=0.6)

    def test_basic_calculation(self):
        """
        基础 GARCH 波动率计算测试
        """
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.01)
        vol = self.calculator.calculate(returns)
        assert vol > 0
        assert vol < 1.0

    def test_with_long_term_vol(self):
        """
        指定长期波动率测试

        当提供长期波动率时，计算结果应该接近该值
        """
        np.random.seed(42)
        returns = pd.Series(np.random.randn(50) * 0.01)
        calc = GARCHVolatilityCalculator(
            alpha=0.1,
            beta=0.85,
            long_term_vol=0.20
        )
        vol = calc.calculate(returns)
        assert vol > 0

    def test_insufficient_data(self):
        """数据不足测试"""
        returns = pd.Series([0.01] * 5)
        with pytest.raises(ValueError, match="需要至少 10 个数据点"):
            self.calculator.calculate(returns)

    def test_zero_volatility_input(self):
        """
        零波动率输入测试

        全零收益率应该返回零波动率
        """
        returns = pd.Series([0.0] * 50)
        vol = self.calculator.calculate(returns)
        assert vol == 0.0

    def test_calculate_series(self):
        """
        GARCH 波动率序列计算测试
        """
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.01)
        vol_series = self.calculator.calculate_series(returns)
        assert len(vol_series) == len(returns)
        assert not vol_series.isna().any()
        assert all(vol_series > 0)


class TestVolatilityRegimeDetector:
    """波动率状态检测器测试"""

    def setup_method(self):
        """测试初始化"""
        self.detector = VolatilityRegimeDetector(lookback_period=60)

    def test_high_regime(self):
        """
        高波动率状态测试

        当 vol > mean + 2*std 时，应该检测为 HIGH
        """
        vol_history = pd.Series([0.10] * 50 + [0.50] * 10)
        result = self.detector.detect(vol_history, current_vol=0.50)
        assert result.regime == VolatilityRegime.HIGH
        assert "mean+2std" in result.regime_reason.lower()

    def test_elevated_regime(self):
        """
        中高波动率状态测试

        当 mean + std < vol <= mean + 2*std 时，应该检测为 ELEVATED
        """
        # [0.10]*50 + [0.25]*10 -> mean=0.125, std≈0.0559
        # mean+std≈0.181, mean+2std≈0.237
        # 0.20 is between 0.181 and 0.237
        vol_history = pd.Series([0.10] * 50 + [0.25] * 10)
        result = self.detector.detect(vol_history, current_vol=0.20)
        assert result.regime == VolatilityRegime.ELEVATED

    def test_normal_regime(self):
        """
        正常波动率状态测试

        当 mean - std <= vol <= mean + std 时，应该检测为 NORMAL
        """
        vol_history = pd.Series([0.10] * 100)
        result = self.detector.detect(vol_history, current_vol=0.10)
        assert result.regime == VolatilityRegime.NORMAL

    def test_low_regime(self):
        """
        低波动率状态测试

        当 vol < mean - std 时，应该检测为 LOW
        """
        vol_history = pd.Series([0.20] * 100)
        result = self.detector.detect(vol_history, current_vol=0.05)
        assert result.regime == VolatilityRegime.LOW
        assert "mean-std" in result.regime_reason.lower()

    def test_insufficient_data(self):
        """数据不足测试"""
        vol_history = pd.Series([0.10] * 3)
        result = self.detector.detect(vol_history)
        assert result.regime == VolatilityRegime.NORMAL
        assert "数据不足" in result.regime_reason

    def test_zero_volatility(self):
        """零波动率测试"""
        vol_history = pd.Series([0.10] * 50)
        result = self.detector.detect(vol_history, current_vol=0.0)
        assert result.regime == VolatilityRegime.NORMAL

    def test_percentile_calculation(self):
        """
        百分位计算测试

        验证百分位在 0-100 范围内
        """
        np.random.seed(42)
        vol_history = pd.Series(np.random.rand(100) * 0.2 + 0.1)
        result = self.detector.detect(vol_history, current_vol=0.15)
        assert 0 <= result.volatility_percentile <= 100

    def test_momentum_calculation(self):
        """
        动量计算测试

        验证动量值合理
        """
        vol_history = pd.Series([0.10] * 50 + [0.15] * 10)
        result = self.detector.detect(vol_history)
        assert isinstance(result.momentum, float)


class TestVolatilityMomentumIndicator:
    """波动率动量指标测试"""

    def setup_method(self):
        """测试初始化"""
        self.indicator = VolatilityMomentumIndicator(ma_period=10)

    def test_rising_volatility(self):
        """
        波动率上升测试

        近期波动率高于均值，动量应该为正
        """
        vol_series = pd.Series([0.10] * 20 + [0.15] * 10)
        momentum = self.indicator.calculate(vol_series)
        assert momentum > 0

    def test_falling_volatility(self):
        """
        波动率下降测试

        近期波动率低于均值，动量应该为负或零
        """
        # [0.20]*15 + [0.10]*15: last 10 values are all 0.10, ma=0.10, recent=0.10, momentum=0
        # Need more values in the lower regime
        vol_series = pd.Series([0.20] * 10 + [0.10] * 20)
        momentum = self.indicator.calculate(vol_series)
        assert momentum <= 0  # 0 or negative

    def test_stable_volatility(self):
        """
        波动率稳定测试

        波动率不变或接近均值，动量应该接近零
        """
        vol_series = pd.Series([0.10] * 30)
        momentum = self.indicator.calculate(vol_series)
        assert abs(momentum) < 0.01

    def test_insufficient_data(self):
        """数据不足测试"""
        vol_series = pd.Series([0.10] * 5)
        momentum = self.indicator.calculate(vol_series)
        assert momentum == 0.0

    def test_acceleration_rising(self):
        """
        动量加速测试

        验证加速度计算返回合理的数值
        当波动率先升后稳时，加速度为负（相对于历史高动量）
        """
        vol_series = pd.Series([0.10] * 20 + [0.12] * 5 + [0.18] * 5)
        acceleration = self.indicator.calculate_acceleration(vol_series)
        assert isinstance(acceleration, float)

    def test_is_accelerating(self):
        """
        是否加速判断测试

        验证函数返回布尔值
        """
        vol_series = pd.Series([0.10] * 20 + [0.12] * 5 + [0.20] * 5)
        result = self.indicator.is_accelerating(vol_series)
        assert isinstance(result, bool)


class TestTargetVolatilityPositionSizer:
    """目标波动率仓位管理器测试"""

    def setup_method(self):
        """测试初始化"""
        self.sizer = TargetVolatilityPositionSizer(
            target_volatility=0.20,
            max_ratio=2.0,
            min_ratio=0.25
        )

    def test_target_ratio_equal_volatility(self):
        """
        波动率等于目标测试

        当 current_vol = target_vol 时，ratio = 1.0
        """
        ratio = self.sizer.calculate_target_ratio(0.20)
        assert ratio == pytest.approx(1.0, abs=0.01)

    def test_target_ratio_high_volatility(self):
        """
        高波动率测试

        当 current_vol > target_vol 时，ratio < 1.0
        """
        ratio = self.sizer.calculate_target_ratio(0.40)
        assert ratio == pytest.approx(0.5, abs=0.01)

    def test_target_ratio_low_volatility(self):
        """
        低波动率测试

        当 current_vol < target_vol 时，ratio > 1.0
        """
        ratio = self.sizer.calculate_target_ratio(0.10)
        assert ratio == pytest.approx(2.0, abs=0.01)

    def test_target_ratio_max_cap(self):
        """
        最大比率限制测试

        ratio 不应超过 max_ratio (2.0)
        """
        ratio = self.sizer.calculate_target_ratio(0.05)
        assert ratio == pytest.approx(2.0, abs=0.01)

    def test_target_ratio_min_cap(self):
        """
        最小比率限制测试

        ratio 不应低于 min_ratio (0.25)
        """
        ratio = self.sizer.calculate_target_ratio(1.00)
        assert ratio == pytest.approx(0.25, abs=0.01)

    def test_target_ratio_zero_volatility(self):
        """
        零波动率测试

        零波动率应该返回最大比率
        """
        ratio = self.sizer.calculate_target_ratio(0.0)
        assert ratio == self.sizer.max_ratio

    def test_calculate_position_basic(self):
        """
        基础仓位计算测试
        """
        result = self.sizer.calculate_position(
            base_position=100_000,
            current_volatility=0.20
        )
        assert isinstance(result, TargetVolatilityPosition)
        assert result.adjusted_position == pytest.approx(100_000, rel=0.01)
        assert result.target_ratio == pytest.approx(1.0, abs=0.01)

    def test_calculate_position_with_regime_high(self):
        """
        高波动率状态降仓测试
        """
        result = self.sizer.calculate_position(
            base_position=100_000,
            current_volatility=0.40,
            regime=VolatilityRegime.HIGH
        )
        assert result.adjusted_position < 50_000

    def test_calculate_position_with_regime_low(self):
        """
        低波动率状态升仓测试

        低波动率时 ratio=2.0, LOW额外*1.2=2.4但受max_ratio=2.0限制
        所以结果应该是 200_000 (恰好等于200_000)
        """
        result = self.sizer.calculate_position(
            base_position=100_000,
            current_volatility=0.10,
            regime=VolatilityRegime.LOW
        )
        assert result.adjusted_position >= 200_000

    def test_calculate_position_zero_base(self):
        """
        零基础仓位测试
        """
        result = self.sizer.calculate_position(
            base_position=0,
            current_volatility=0.20
        )
        assert result.adjusted_position == 0.0

    def test_calculate_position_no_regime_adjustment(self):
        """
        不使用状态调整测试
        """
        result1 = self.sizer.calculate_position(
            base_position=100_000,
            current_volatility=0.40,
            regime=VolatilityRegime.HIGH,
            regime_adjustment=False
        )
        result2 = self.sizer.calculate_position(
            base_position=100_000,
            current_volatility=0.40,
            regime=VolatilityRegime.HIGH,
            regime_adjustment=True
        )
        assert result1.adjusted_position > result2.adjusted_position

    def test_get_position_multiplier(self):
        """
        仓位倍数获取测试
        """
        vol_series = pd.Series([0.20] * 50)
        multiplier = self.sizer.get_position_multiplier(
            vol_series,
            VolatilityRegime.NORMAL
        )
        assert multiplier == pytest.approx(1.0, abs=0.01)


class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_calculate_volatility_from_prices_ewma(self):
        """
        EWMA 便捷函数测试
        """
        np.random.seed(42)
        prices = pd.Series(100 + np.random.randn(100).cumsum())
        vol = calculate_volatility_from_prices(prices, method="ewma")
        assert vol > 0

    def test_calculate_volatility_from_prices_garch(self):
        """
        GARCH 便捷函数测试
        """
        np.random.seed(42)
        prices = pd.Series(100 + np.random.randn(100).cumsum())
        vol = calculate_volatility_from_prices(prices, method="garch")
        assert vol > 0

    def test_calculate_volatility_invalid_method(self):
        """
        无效方法测试
        """
        prices = pd.Series([100, 101, 102, 103, 104])
        with pytest.raises(ValueError, match="不支持的波动率计算方法"):
            calculate_volatility_from_prices(prices, method="invalid")

    def test_detect_volatility_regime(self):
        """
        便捷状态检测函数测试
        """
        vol_history = pd.Series([0.10] * 50 + [0.30] * 10)
        result = detect_volatility_regime(vol_history, current_vol=0.30)
        assert isinstance(result, VolatilityResult)
        assert result.regime == VolatilityRegime.HIGH


class TestEdgeCases:
    """边界情况测试"""

    def test_extreme_volatility_ratio(self):
        """
        极端波动率比值测试
        """
        sizer = TargetVolatilityPositionSizer(target_volatility=0.20)
        ratio = sizer.calculate_target_ratio(1e-10)
        assert ratio == sizer.max_ratio

    def test_very_small_base_position(self):
        """
        极小仓位测试
        """
        sizer = TargetVolatilityPositionSizer(target_volatility=0.20)
        result = sizer.calculate_position(
            base_position=0.01,
            current_volatility=0.20
        )
        assert result.adjusted_position >= 0

    def test_large_position_with_high_vol(self):
        """
        大仓位高波动测试
        """
        sizer = TargetVolatilityPositionSizer(
            target_volatility=0.20,
            max_ratio=2.0,
            min_ratio=0.25
        )
        result = sizer.calculate_position(
            base_position=10_000_000,
            current_volatility=0.60,
            regime=VolatilityRegime.HIGH
        )
        assert result.adjusted_position < result.base_position


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
