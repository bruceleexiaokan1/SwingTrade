"""共振仓位管理

根据共振等级计算仓位，支持：
1. 分级仓位（S/A/B/C 对应不同仓位）
2. 试探仓升级机制
3. ATR 标准化仓位
"""

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from ..data.indicators.resonance import ResonanceLevel, ResonanceResult

if TYPE_CHECKING:
    from ..backtest.models import Position

logger = logging.getLogger(__name__)


# 共振等级对应仓位比例
LEVEL_TO_POSITION_RATIO = {
    ResonanceLevel.S: 1.0,    # 100% 基础仓位
    ResonanceLevel.A: 0.75,   # 75% 基础仓位
    ResonanceLevel.B: 0.5,    # 50% 基础仓位
    ResonanceLevel.C: 0.25,   # 25% 基础仓位
    ResonanceLevel.INVALID: 0.0  # 不开仓
}

# 试探仓比例
TRIAL_POSITION_RATIO = 0.5  # 试探仓为基础仓位的 50%


@dataclass
class PositionSize:
    """仓位计算结果"""
    base_ratio: float      # 基础仓位比例
    final_ratio: float    # 最终仓位比例（考虑其他因素）
    is_trial: bool         # 是否试探仓
    reason: str            # 计算理由


class ResonancePositionManager:
    """
    共振仓位管理器

    职责：
    1. 根据共振等级计算仓位
    2. 管理试探仓和正式仓
    3. 计算 ATR 标准化仓位
    """

    def __init__(
        self,
        base_position_value: float = 100_000.0,
        max_positions: int = 5,
        trial_upgrade_days: int = 5,
        trial_upgrade_profit: float = 0.05
    ):
        """
        初始化仓位管理器

        Args:
            base_position_value: 基础仓位金额
            max_positions: 最大持仓数
            trial_upgrade_days: 试探仓升级需要持仓天数
            trial_upgrade_profit: 试探仓升级需要盈利比例
        """
        self.base_position_value = base_position_value
        self.max_positions = max_positions
        self.trial_upgrade_days = trial_upgrade_days
        self.trial_upgrade_profit = trial_upgrade_profit

    def calculate_position_size(
        self,
        resonance_result: ResonanceResult,
        current_positions: int = 0,
        is_existing_position: bool = False,
        existing_position_holding_days: int = 0,
        existing_position_profit: float = 0.0
    ) -> PositionSize:
        """
        计算仓位

        Args:
            resonance_result: 共振检测结果
            current_positions: 当前持仓数
            is_existing_position: 是否已有仓位（用于升级判断）
            existing_position_holding_days: 已有持仓天数
            existing_position_profit: 已有持仓盈利比例

        Returns:
            PositionSize
        """
        level = resonance_result.resonance_level
        base_ratio = LEVEL_TO_POSITION_RATIO.get(level, 0.0)

        # 不可开仓
        if base_ratio == 0.0:
            return PositionSize(
                base_ratio=0.0,
                final_ratio=0.0,
                is_trial=False,
                reason="共振等级无效，不开仓"
            )

        # 如果是已有仓位，检查是否需要升级
        if is_existing_position:
            should_upgrade = self._should_upgrade(
                holding_days=existing_position_holding_days,
                profit_pct=existing_position_profit,
                level=level
            )

            if should_upgrade:
                # 升级到正式仓
                return PositionSize(
                    base_ratio=base_ratio,
                    final_ratio=base_ratio,
                    is_trial=False,
                    reason=f"试探仓升级为正式仓（持仓{existing_position_holding_days}天，盈利{existing_position_profit:.1%}）"
                )
            else:
                # 保持试探仓
                return PositionSize(
                    base_ratio=base_ratio,
                    final_ratio=base_ratio * TRIAL_POSITION_RATIO,
                    is_trial=True,
                    reason=f"保持试探仓（持仓{existing_position_holding_days}天，盈利{existing_position_profit:.1%}，未达升级条件）"
                )

        # 新仓位
        # 剩余可开仓数
        remaining_slots = max(0, self.max_positions - current_positions)

        if remaining_slots == 0:
            return PositionSize(
                base_ratio=0.0,
                final_ratio=0.0,
                is_trial=False,
                reason="已达到最大持仓数"
            )

        # 计算最终仓位
        # 试探仓：基础仓位的 50%
        # 正式仓：基础仓位的 100%
        final_ratio = base_ratio
        is_trial = True  # 新仓位默认试探

        # 计算实际金额
        position_value = self.base_position_value * final_ratio / self.max_positions

        return PositionSize(
            base_ratio=base_ratio,
            final_ratio=final_ratio,
            is_trial=is_trial,
            reason=f"{level.value}级共振，{'试探' if is_trial else '正式'}仓"
        )

    def calculate_atr_position(
        self,
        entry_price: float,
        atr: float,
        risk_pct: float = 0.02
    ) -> int:
        """
        基于 ATR 计算标准化仓位

        原理：风险金额 = 账户总额 × risk_pct
              仓位股数 = 风险金额 / (入场价 - 止损价)
              止损价 = 入场价 - 2 × ATR

        Args:
            entry_price: 入场价格
            atr: ATR 值
            risk_pct: 单笔风险比例，默认 2%

        Returns:
            建议股数（手）
        """
        # 止损距离 = 2 × ATR
        stop_distance = atr * 2

        # 每股风险金额
        risk_per_share = stop_distance

        if risk_per_share <= 0:
            return 0

        # 计算股数（不需要精确，这里简化为 100 的倍数）
        shares = int(self.base_position_value * risk_pct / risk_per_share / 100) * 100

        return max(100, shares)  # 最小 1 手

    def _should_upgrade(
        self,
        holding_days: int,
        profit_pct: float,
        level: ResonanceLevel
    ) -> bool:
        """
        判断是否应该升级仓位

        条件：
        1. 持仓天数 >= trial_upgrade_days
        2. 盈利比例 >= trial_upgrade_profit
        3. 共振等级 >= B

        Args:
            holding_days: 持仓天数
            profit_pct: 盈利比例
            level: 共振等级

        Returns:
            是否应该升级
        """
        # 共振等级必须在 B 级以上
        if level == ResonanceLevel.C or level == ResonanceLevel.INVALID:
            return False

        # 持仓天数足够
        if holding_days < self.trial_upgrade_days:
            return False

        # 盈利达标
        if profit_pct < self.trial_upgrade_profit:
            return False

        return True

    def should_add_position(
        self,
        resonance_result: ResonanceResult,
        current_profit: float
    ) -> bool:
        """
        判断是否应该加仓

        用于金字塔加仓策略

        Args:
            resonance_result: 共振检测结果
            current_profit: 当前盈利比例

        Returns:
            是否应该加仓
        """
        # 只在盈利时考虑加仓
        if current_profit < 0.03:  # 盈利小于 3% 不加仓
            return False

        # 共振等级为 S 或 A 才考虑加仓
        if resonance_result.resonance_level not in [ResonanceLevel.S, ResonanceLevel.A]:
            return False

        # 盈利超过 10% 才加仓
        if current_profit < 0.10:
            return False

        return True

    def should_reduce_position(
        self,
        resonance_result: ResonanceResult,
        current_profit: float,
        holding_days: int
    ) -> bool:
        """
        判断是否应该减仓

        Args:
            resonance_result: 共振检测结果
            current_profit: 当前盈利比例
            holding_days: 持仓天数

        Returns:
            是否应该减仓
        """
        # 盈利超过 20% 且共振减弱
        if current_profit > 0.20:
            if resonance_result.resonance_level in [ResonanceLevel.C, ResonanceLevel.INVALID]:
                return True

        # 持仓超过 20 天且 RSI 超买
        if holding_days > 20:
            if resonance_result.stock_rsi > 70:
                return True

        return False

    def get_stop_loss(
        self,
        entry_price: float,
        atr: float,
        is_trial: bool = True
    ) -> float:
        """
        计算止损价

        Args:
            entry_price: 入场价格
            atr: ATR 值
            is_trial: 是否试探仓（试探仓止损更紧）

        Returns:
            止损价格
        """
        multiplier = 1.5 if is_trial else 2.0
        return entry_price - atr * multiplier

    def get_take_profit(
        self,
        entry_price: float,
        atr: float,
        level: ResonanceLevel,
        is_trial: bool = False
    ) -> tuple:
        """
        计算止盈价位

        Args:
            entry_price: 入场价格
            atr: ATR 值
            level: 共振等级
            is_trial: 是否试探仓

        Returns:
            (止盈1, 止盈2, 止盈3)
        """
        # 不同等级不同止盈目标
        if level == ResonanceLevel.S:
            multipliers = (3, 5, 7)
        elif level == ResonanceLevel.A:
            multipliers = (2, 4, 6)
        elif level == ResonanceLevel.B:
            multipliers = (2, 3, 5)
        else:
            multipliers = (1.5, 2.5, 4)

        return tuple(entry_price + atr * m for m in multipliers)
