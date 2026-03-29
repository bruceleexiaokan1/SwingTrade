"""仓位管理模块

基于凯利公式和波动率自适应的仓位管理系统：
1. 凯利公式：f = (b × p - q) / b
2. 波动率调整：根据 ATR% 动态调整仓位
3. 风险约束：单笔风险 ≤ 2%，总持仓 ≤ 15%

凯利公式约束：
- 理论仓位最大不超过 25%
- 实际使用时取凯利和波动率调整的较小值
"""

from typing import Optional


class KellyPositionSizer:
    """
    基于凯利公式的仓位计算器

    凯利公式：f = (b × p - q) / b
    - f: 仓位比例
    - b: 盈亏比 (avg_win / avg_loss)
    - p: 胜率
    - q: 1 - p

    波动率自适应：
    - 高 ATR% → 低仓位（波动大，降低风险）
    - 低 ATR% → 高仓位（波动小，可适当放大）
    """

    # 凯利公式最大仓位上限（25%）
    MAX_KELLY_FRACTION = 0.25

    # ATR% 波动率调整参数
    # ATR% 低于此值时仓位可以放大
    LOW_VOLATILITY_THRESHOLD = 0.03   # 3%
    # ATR% 高于此值时仓位需要收缩
    HIGH_VOLATILITY_THRESHOLD = 0.08  # 8%

    def __init__(
        self,
        max_risk_pct: float = 0.02,
        max_position_pct: float = 0.15
    ):
        """
        初始化仓位计算器

        Args:
            max_risk_pct: 单笔最大风险比例，默认 2%
            max_position_pct: 最大同时持仓比例，默认 15%
        """
        self.max_risk_pct = max_risk_pct
        self.max_position_pct = max_position_pct

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        计算凯利仓位比例

        Args:
            win_rate: 胜率 (0-1)
            avg_win: 平均盈利金额
            avg_loss: 平均亏损金额

        Returns:
            凯利仓位比例 (0-1)

        Formula:
            b = avg_win / avg_loss (盈亏比)
            f = (b × p - q) / b

        Example:
            win_rate = 0.4, avg_win = 2000, avg_loss = 1000
            b = 2.0
            f = (2.0 × 0.4 - 0.6) / 2.0 = (0.8 - 0.6) / 2.0 = 0.1 = 10%
        """
        if avg_loss <= 0 or win_rate < 0 or win_rate > 1:
            return 0.0

        # 盈亏比 b = avg_win / avg_loss
        b = avg_win / avg_loss

        if b <= 0:
            return 0.0

        # 胜率 p，败率 q = 1 - p
        p = win_rate
        q = 1.0 - p

        # 凯利公式：f = (b × p - q) / b
        kelly_fraction = (b * p - q) / b

        # 凯利公式可能为负数（期望为负），取 0
        if kelly_fraction < 0:
            return 0.0

        # 限制最大仓位不超过 25%
        return min(kelly_fraction, self.MAX_KELLY_FRACTION)

    def calculate_volatility_adjusted_size(
        self,
        entry_price: float,
        stop_loss: float,
        account_value: float,
        atr_pct: float
    ) -> float:
        """
        根据波动率调整仓位大小

        高 ATR% → 低仓位（市场波动大，降低风险暴露）
        低 ATR% → 高仓位（市场波动小，可适当放大仓位）

        Args:
            entry_price: 入场价格
            stop_loss: 止损价格
            account_value: 账户总值
            atr_pct: ATR 百分比 (ATR / close * 100)

        Returns:
            调整后的仓位金额
        """
        if entry_price <= 0 or stop_loss <= 0 or account_value <= 0:
            return 0.0

        # 止损距离
        stop_distance = entry_price - stop_loss
        if stop_distance <= 0:
            return 0.0

        # 单笔最大风险金额
        max_risk_amount = account_value * self.max_risk_pct

        # 基础仓位 = 最大风险金额 / 止损距离 × 入场价格
        # 这是确保单笔损失不超过 max_risk_pct 的仓位
        base_position = (max_risk_amount / stop_distance) * entry_price

        # 波动率调整因子
        # 当 ATR% < LOW_VOLATILITY_THRESHOLD 时，可以放大仓位（最多 1.5 倍）
        # 当 ATR% > HIGH_VOLATILITY_THRESHOLD 时，必须收缩仓位（最少 0.5 倍）
        # 中间区间做线性过渡
        if atr_pct <= self.LOW_VOLATILITY_THRESHOLD:
            # 低波动率：仓位可以放大，最多 1.5 倍
            volatility_multiplier = 1.5
        elif atr_pct >= self.HIGH_VOLATILITY_THRESHOLD:
            # 高波动率：仓位必须收缩，最多 0.5 倍
            volatility_multiplier = 0.5
        else:
            # 过渡区间：线性插值
            # 从 1.5 (at LOW) 过渡到 0.5 (at HIGH)
            denominator = self.HIGH_VOLATILITY_THRESHOLD - self.LOW_VOLATILITY_THRESHOLD
            if denominator == 0:
                ratio = 0.5  # 避免除零
            else:
                ratio = (atr_pct - self.LOW_VOLATILITY_THRESHOLD) / denominator
            volatility_multiplier = 1.5 - ratio * (1.5 - 0.5)

        # 波动率调整后的仓位
        adjusted_position = base_position * volatility_multiplier

        # 不能超过最大持仓比例
        max_position_value = account_value * self.max_position_pct

        return min(adjusted_position, max_position_value)

    def calculate_position(
        self,
        account_value: float,
        entry_price: float,
        stop_loss: float,
        atr_pct: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> float:
        """
        综合仓位计算

        综合凯利公式和波动率调整：
        1. 如果提供了凯利参数（win_rate, avg_win, avg_loss），计算凯利仓位
        2. 将凯利仓位乘以波动率调整因子
        3. 最终取 min(调整后凯利仓位, 单笔风险限制仓位)

        Args:
            account_value: 账户总值
            entry_price: 入场价格
            stop_loss: 止损价格
            atr_pct: ATR 百分比
            win_rate: 胜率 (可选)
            avg_win: 平均盈利 (可选)
            avg_loss: 平均亏损 (可选)

        Returns:
            最终仓位金额
        """
        if entry_price <= 0 or stop_loss <= 0 or account_value <= 0:
            return 0.0

        # 如果没有凯利参数，使用波动率调整仓位
        if win_rate is None or avg_win is None or avg_loss is None:
            return self.calculate_volatility_adjusted_size(
                entry_price=entry_price,
                stop_loss=stop_loss,
                account_value=account_value,
                atr_pct=atr_pct
            )

        # 计算凯利仓位
        kelly_fraction = self.calculate_kelly_fraction(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss
        )
        kelly_position = account_value * kelly_fraction

        # 计算波动率调整因子
        if atr_pct <= self.LOW_VOLATILITY_THRESHOLD:
            volatility_multiplier = 1.5
        elif atr_pct >= self.HIGH_VOLATILITY_THRESHOLD:
            volatility_multiplier = 0.5
        else:
            ratio = (atr_pct - self.LOW_VOLATILITY_THRESHOLD) / (
                self.HIGH_VOLATILITY_THRESHOLD - self.LOW_VOLATILITY_THRESHOLD
            )
            volatility_multiplier = 1.5 - ratio * (1.5 - 0.5)

        # 波动率调整：凯利仓位乘以波动率因子
        volatility_adjusted_kelly = kelly_position * volatility_multiplier

        # 最大持仓限制
        max_position_limit = account_value * self.max_position_pct

        # 单笔风险限制仓位
        stop_distance = entry_price - stop_loss
        if stop_distance > 0:
            max_risk_amount = account_value * self.max_risk_pct
            risk_limited_position = (max_risk_amount / stop_distance) * entry_price
        else:
            risk_limited_position = float('inf')

        # 取最小值：凯利仓位，波动率调整凯利仓位，最大持仓限制，单笔风险限制仓位
        return min(kelly_position, volatility_adjusted_kelly, max_position_limit, risk_limited_position)

    def calculate_stop_loss_by_risk(
        self,
        entry_price: float,
        account_value: float,
        risk_pct: Optional[float] = None
    ) -> float:
        """
        根据风险限制反推止损价格

        Args:
            entry_price: 入场价格
            account_value: 账户总值
            risk_pct: 风险比例（默认使用 self.max_risk_pct）

        Returns:
            止损价格
        """
        if risk_pct is None:
            risk_pct = self.max_risk_pct

        if entry_price <= 0 or account_value <= 0:
            return 0.0

        # 最大亏损金额
        max_loss = account_value * risk_pct

        # 止损距离 = 最大亏损金额 / 仓位 × 入场价格
        # 简化：假设使用 max_position_pct 仓位
        position_value = account_value * self.max_position_pct
        stop_distance = max_loss / position_value * entry_price

        return entry_price - stop_distance
