"""仓位管理器测试

测试 KellyPositionSizer 的各项功能：
1. 凯利公式计算
2. 波动率调整仓位
3. 综合仓位计算
4. 风险约束验证
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.backtest.position_sizer import KellyPositionSizer


class TestKellyFraction:
    """凯利公式计算测试"""

    def setup_method(self):
        """测试初始化"""
        self.sizer = KellyPositionSizer()

    def test_kelly_basic(self):
        """
        基础凯利公式测试

        场景：
        - 胜率 40%
        - 平均盈利 2000
        - 平均亏损 1000
        - 盈亏比 b = 2.0

        预期：
        f = (2.0 × 0.4 - 0.6) / 2.0 = 0.1 = 10%
        """
        kelly = self.sizer.calculate_kelly_fraction(
            win_rate=0.4,
            avg_win=2000,
            avg_loss=1000
        )
        assert kelly == pytest.approx(0.10, rel=0.01)

    def test_kelly_higher_win_rate(self):
        """
        高胜率测试

        场景：
        - 胜率 60%
        - 平均盈利 1500
        - 平均亏损 1500
        - 盈亏比 b = 1.0

        预期：
        f = (1.0 × 0.6 - 0.4) / 1.0 = 0.2 = 20%
        """
        kelly = self.sizer.calculate_kelly_fraction(
            win_rate=0.6,
            avg_win=1500,
            avg_loss=1500
        )
        assert kelly == pytest.approx(0.20, rel=0.01)

    def test_kelly_high_profit_ratio(self):
        """
        高盈亏比测试

        场景：
        - 胜率 50%
        - 平均盈利 3000
        - 平均亏损 1000
        - 盈亏比 b = 3.0

        预期：
        f = (3.0 × 0.5 - 0.5) / 3.0 = (1.5 - 0.5) / 3.0 = 0.333 = 33.33%
        限制到 25%
        """
        kelly = self.sizer.calculate_kelly_fraction(
            win_rate=0.5,
            avg_win=3000,
            avg_loss=1000
        )
        # 理论值 33.33%，但会被限制到 25%
        assert kelly == pytest.approx(0.25, abs=0.01)

    def test_kelly_negative_expectation(self):
        """
        负期望值测试

        场景：
        - 胜率 30%
        - 平均盈利 1000
        - 平均亏损 2000
        - 盈亏比 b = 0.5

        预期：f = (0.5 × 0.3 - 0.7) / 0.5 = (0.15 - 0.7) / 0.5 = -1.1
        负数应该返回 0
        """
        kelly = self.sizer.calculate_kelly_fraction(
            win_rate=0.3,
            avg_win=1000,
            avg_loss=2000
        )
        assert kelly == 0.0

    def test_kelly_zero_loss(self):
        """零亏损测试（应该返回 0）"""
        kelly = self.sizer.calculate_kelly_fraction(
            win_rate=0.5,
            avg_win=1000,
            avg_loss=0
        )
        assert kelly == 0.0

    def test_kelly_perfect_win_rate(self):
        """100% 胜率测试"""
        kelly = self.sizer.calculate_kelly_fraction(
            win_rate=1.0,
            avg_win=1000,
            avg_loss=1000
        )
        # f = (1.0 × 1.0 - 0) / 1.0 = 1.0，但限制到 0.25
        assert kelly == pytest.approx(0.25, abs=0.01)

    def test_kelly_zero_win_rate(self):
        """0% 胜率测试（应该返回 0）"""
        kelly = self.sizer.calculate_kelly_fraction(
            win_rate=0.0,
            avg_win=1000,
            avg_loss=1000
        )
        assert kelly == 0.0


class TestVolatilityAdjustedSize:
    """波动率调整仓位测试"""

    def setup_method(self):
        """测试初始化"""
        self.sizer = KellyPositionSizer(
            max_risk_pct=0.02,
            max_position_pct=0.15
        )

    def test_low_volatility_high_position(self):
        """
        低波动率 → 高仓位测试

        ATR% = 2% (< LOW_VOLATILITY_THRESHOLD 3%)
        预期：仓位放大 1.5 倍
        """
        # 账户 100 万，入场价 100，止损 98（2% 距离）
        # 单笔风险 2% = 2 万
        # 基础仓位 = 20000 / 2 * 100 = 100 万
        # 但受 max_position_pct 限制 = 15 万
        # 波动率调整：1.5 倍 → 15 万
        position = self.sizer.calculate_volatility_adjusted_size(
            entry_price=100.0,
            stop_loss=98.0,
            account_value=1_000_000,
            atr_pct=0.02  # 2% ATR%
        )
        # 仓位应该接近 15 万（受 max_position_pct 限制）
        assert position == pytest.approx(150_000, rel=0.01)

    def test_high_volatility_low_position(self):
        """
        高波动率 → 低仓位测试

        ATR% = 10% (> HIGH_VOLATILITY_THRESHOLD 8%)
        预期：仓位收缩 0.5 倍
        """
        position = self.sizer.calculate_volatility_adjusted_size(
            entry_price=100.0,
            stop_loss=95.0,  # 5% 止损距离
            account_value=1_000_000,
            atr_pct=0.10  # 10% ATR%
        )
        # 基础仓位 = 20000 / 5 * 100 = 40 万
        # 波动率调整：0.5 倍 → 20 万
        # max_position_pct = 15 万
        # 取 min(20万, 15万) = 15万
        assert position == pytest.approx(150_000, rel=0.01)

    def test_medium_volatility(self):
        """
        中等波动率测试

        ATR% = 5%（在 3%~8% 区间）
        预期：线性过渡
        """
        position = self.sizer.calculate_volatility_adjusted_size(
            entry_price=100.0,
            stop_loss=98.0,  # 2% 止损距离
            account_value=1_000_000,
            atr_pct=0.05  # 5% ATR%
        )
        # 基础仓位 = 20000 / 2 * 100 = 100 万，但受 max_position_pct 限制 = 15 万
        # 波动率调整因子：1.5 - (0.05-0.03)/(0.08-0.03) * (1.5-0.5) = 1.5 - 0.4 = 1.1
        # 调整后：15万 * 1.1 = 16.5万，但受 max_position_pct 限制 = 15万
        assert position == pytest.approx(150_000, rel=0.01)

    def test_zero_entry_price(self):
        """零入场价测试"""
        position = self.sizer.calculate_volatility_adjusted_size(
            entry_price=0,
            stop_loss=98.0,
            account_value=1_000_000,
            atr_pct=0.05
        )
        assert position == 0.0

    def test_stop_loss_higher_than_entry(self):
        """止损价高于入场价测试（无效）"""
        position = self.sizer.calculate_volatility_adjusted_size(
            entry_price=100.0,
            stop_loss=102.0,  # 止损应该在下方
            account_value=1_000_000,
            atr_pct=0.05
        )
        assert position == 0.0


class TestCalculatePosition:
    """综合仓位计算测试"""

    def setup_method(self):
        """测试初始化"""
        self.sizer = KellyPositionSizer(
            max_risk_pct=0.02,
            max_position_pct=0.15
        )

    def test_position_without_kelly(self):
        """仅使用波动率调整（不提供凯利参数）"""
        position = self.sizer.calculate_position(
            account_value=1_000_000,
            entry_price=100.0,
            stop_loss=98.0,
            atr_pct=0.05,
            win_rate=None,
            avg_win=None,
            avg_loss=None
        )
        # 应该返回波动率调整后的仓位
        assert position > 0
        assert position <= 150_000  # 不超过 max_position_pct

    def test_position_with_kelly_low_volatility(self):
        """
        低波动率 + 凯利公式测试
        """
        position = self.sizer.calculate_position(
            account_value=1_000_000,
            entry_price=100.0,
            stop_loss=98.0,
            atr_pct=0.02,  # 低波动率
            win_rate=0.5,
            avg_win=2000,
            avg_loss=1000
        )
        # 凯利公式：f = (2.0 * 0.5 - 0.5) / 2.0 = 0.25 = 25%
        # 凯利仓位 = 25 万
        # 波动率调整（低波动率）：仓位可以放大 1.5 倍，但受 max_position_pct 限制 = 15 万
        assert position == pytest.approx(150_000, rel=0.01)

    def test_position_with_kelly_high_volatility(self):
        """
        高波动率 + 凯利公式测试
        """
        position = self.sizer.calculate_position(
            account_value=1_000_000,
            entry_price=100.0,
            stop_loss=95.0,  # 5% 止损距离
            atr_pct=0.10,    # 高波动率
            win_rate=0.5,
            avg_win=2000,
            avg_loss=1000
        )
        # 凯利公式：25% = 25 万
        # 波动率调整：高波动率 0.5 倍 → 12.5 万
        # 但还要受 max_position_pct 限制 = 15 万
        # 取 min(25万, 12.5万) = 12.5 万
        assert position == pytest.approx(125_000, rel=0.01)

    def test_position_kelly_capped(self):
        """
        凯利公式上限测试

        高盈亏比导致凯利公式建议仓位 > 25%，但被限制
        """
        position = self.sizer.calculate_position(
            account_value=1_000_000,
            entry_price=100.0,
            stop_loss=99.0,  # 1% 止损距离
            atr_pct=0.05,
            win_rate=0.6,
            avg_win=3000,
            avg_loss=1000  # 盈亏比 3.0
        )
        # 凯利公式：f = (3.0 * 0.6 - 0.4) / 3.0 = 1.4 / 3 = 0.467 = 46.7%
        # 限制到 25% = 25 万
        # 波动率调整（5% 中等）：因子约 1.0
        # 单笔风险 2% = 2 万，止损 1%，最大仓位 = 2 万 / 1% * 100 = 200 万
        # 但受 max_position_pct 限制 = 15 万
        assert position == pytest.approx(150_000, rel=0.01)


class TestRiskConstraints:
    """风险约束测试"""

    def setup_method(self):
        """测试初始化"""
        self.sizer = KellyPositionSizer(
            max_risk_pct=0.02,
            max_position_pct=0.15
        )

    def test_single_risk_limit(self):
        """
        单笔风险不超过 2% 测试
        """
        # 账户 100 万，2% 风险 = 2 万
        # 入场价 100，止损 97（3% 距离）
        # 最大仓位 = 20000 / 3% = 66.67 万
        position = self.sizer.calculate_volatility_adjusted_size(
            entry_price=100.0,
            stop_loss=97.0,
            account_value=1_000_000,
            atr_pct=0.05
        )
        # 单笔最大损失：position * 3% = position * 0.03
        # 应该不超过 2 万
        max_loss = position * 0.03
        assert max_loss <= 20_100  # 允许一点精度误差

    def test_max_position_limit(self):
        """
        最大持仓不超过 15% 测试
        """
        position = self.sizer.calculate_volatility_adjusted_size(
            entry_price=100.0,
            stop_loss=50.0,  # 50% 止损距离（大到会让仓位计算很大）
            account_value=1_000_000,
            atr_pct=0.02  # 低波动率
        )
        # 即使止损距离很大，也不应该超过 15%
        assert position <= 150_000

    def test_kelly_max_cap(self):
        """
        凯利公式最大上限 25% 测试
        """
        # 极端盈利比，但胜率一般
        kelly = self.sizer.calculate_kelly_fraction(
            win_rate=0.55,
            avg_win=10000,
            avg_loss=1000  # 盈亏比 10
        )
        # 理论值会很大，但限制到 0.25
        assert kelly <= 0.25


class TestKellyPositionSizerEdgeCases:
    """边界情况测试"""

    def test_initialization_default(self):
        """默认参数初始化"""
        sizer = KellyPositionSizer()
        assert sizer.max_risk_pct == 0.02
        assert sizer.max_position_pct == 0.15

    def test_initialization_custom(self):
        """自定义参数初始化"""
        sizer = KellyPositionSizer(
            max_risk_pct=0.01,
            max_position_pct=0.20
        )
        assert sizer.max_risk_pct == 0.01
        assert sizer.max_position_pct == 0.20

    def test_zero_account_value(self):
        """零账户值测试"""
        sizer = KellyPositionSizer()
        position = sizer.calculate_position(
            account_value=0,
            entry_price=100.0,
            stop_loss=98.0,
            atr_pct=0.05
        )
        assert position == 0.0

    def test_very_small_position(self):
        """极小仓位测试（避免除零）"""
        sizer = KellyPositionSizer()
        position = sizer.calculate_volatility_adjusted_size(
            entry_price=0.01,
            stop_loss=0.0099,
            account_value=1000,
            atr_pct=0.05
        )
        assert position >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
