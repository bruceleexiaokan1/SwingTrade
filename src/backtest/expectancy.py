"""正期望值计算模块

正期望公式：
E = P_win × Avg_Win - P_loss × Avg_Loss - Cost

中长线交易规则：
- 胜率 30%-40%
- 必须盈亏比 > 3:1
- 凡盈亏比 < 3:1 的信号直接过滤
"""

from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExpectancyResult:
    """正期望计算结果"""
    expectancy: float           # 期望值 E = P_win × Avg_Win - P_loss × Avg_Loss - Cost
    win_rate: float             # 胜率 P_win
    avg_win: float              # 平均盈利
    avg_loss: float             # 平均亏损
    profit_loss_ratio: float    # 盈亏比 Avg_Win / |Avg_Loss|
    total_trades: int           # 总交易数
    winning_trades: int         # 盈利交易数
    losing_trades: int          # 亏损交易数
    total_cost: float           # 总交易成本

    @property
    def is_positive(self) -> bool:
        """期望值是否 > 0"""
        return self.expectancy > 0

    @property
    def passes_filter(self) -> bool:
        """是否通过正期望过滤器（盈亏比 >= 3:1）"""
        return self.profit_loss_ratio >= 3.0

    def summary(self) -> str:
        """生成摘要"""
        return f"""=== Expectancy Analysis ===
Trades: {self.total_trades} (Win: {self.winning_trades}, Loss: {self.losing_trades})
Win Rate: {self.win_rate:.2%}
Avg Win: {self.avg_win:.2f} | Avg Loss: {self.avg_loss:.2f}
Profit/Loss Ratio: {self.profit_loss_ratio:.2f}:1
Total Cost: {self.total_cost:.2f}
Expectancy: {self.expectancy:.4f}
Positive: {self.is_positive} | Passes Filter (>=3:1): {self.passes_filter}
"""


def calculate_expectancy(
    trades: List["Trade"],
    commission_rate: float = 0.0003,
    stamp_tax: float = 0.0001,
) -> ExpectancyResult:
    """
    计算正期望值

    E = P_win × Avg_Win - P_loss × Avg_Loss - Cost

    Args:
        trades: 交易列表
        commission_rate: 佣金率，默认 0.03%
        stamp_tax: 印花税率，默认 0.01%（仅卖出时）

    Returns:
        ExpectancyResult: 正期望计算结果
    """
    if not trades:
        return ExpectancyResult(
            expectancy=0.0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_loss_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_cost=0.0,
        )

    # 分离盈利和亏损交易
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl <= 0]

    # 计算基本统计
    total_trades = len(trades)
    winning_count = len(winning_trades)
    losing_count = len(losing_trades)

    # 胜率
    win_rate = winning_count / total_trades if total_trades > 0 else 0.0

    # 平均盈利和平均亏损
    avg_win = sum(t.pnl for t in winning_trades) / winning_count if winning_count > 0 else 0.0
    avg_loss = abs(sum(t.pnl for t in losing_trades) / losing_count) if losing_count > 0 else 0.0

    # 盈亏比
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    # 计算交易成本
    # 买入：佣金 = 成交额 × 佣金率
    # 卖出：佣金 + 印花税 = 成交额 × (佣金率 + 印花税率)
    total_cost = 0.0
    for t in trades:
        entry_cost = t.entry_price * t.shares * commission_rate
        exit_cost = t.exit_price * t.shares * (commission_rate + stamp_tax)
        total_cost += entry_cost + exit_cost

    # 计算期望值
    # E = P_win × Avg_Win - P_loss × Avg_Loss - Avg_Cost
    p_loss = 1 - win_rate
    avg_cost = total_cost / total_trades if total_trades > 0 else 0.0

    expectancy = (
        win_rate * avg_win -
        p_loss * avg_loss -
        avg_cost
    )

    return ExpectancyResult(
        expectancy=expectancy,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_loss_ratio=profit_loss_ratio,
        total_trades=total_trades,
        winning_trades=winning_count,
        losing_trades=losing_count,
        total_cost=total_cost,
    )


def filter_by_expectancy(
    signals: List["EntrySignal"],
    historical_trades: List["Trade"],
    min_ratio: float = 3.0,
) -> List["EntrySignal"]:
    """
    过滤不符合正期望的信号

    基于历史交易的盈亏比进行过滤：
    - 如果历史交易的盈亏比 < min_ratio，则过滤所有信号

    Args:
        signals: 入场信号列表
        historical_trades: 历史交易列表
        min_ratio: 最小盈亏比要求，默认 3.0

    Returns:
        过滤后的信号列表（如果历史数据不足盈亏比要求则返回空）
    """
    if not historical_trades or len(historical_trades) < 10:
        # 历史数据不足，无法可靠估计，保守返回空
        logger.warning("Insufficient historical trades for expectancy filter, blocking all signals")
        return []

    # 计算历史期望
    exp_result = calculate_expectancy(historical_trades)

    if not exp_result.passes_filter:
        # 盈亏比不达标，过滤所有信号
        logger.info(
            f"Historical profit/loss ratio {exp_result.profit_loss_ratio:.2f}:1 < {min_ratio}:1, "
            f"filtering all signals"
        )
        return []

    logger.info(
        f"Expectancy filter passed: ratio={exp_result.profit_loss_ratio:.2f}:1, "
        f"expectancy={exp_result.expectancy:.4f}"
    )

    return signals


def calculate_expectancy_from_stats(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    total_trades: int,
    avg_commission_pct: float = 0.0004,
) -> ExpectancyResult:
    """
    根据统计指标计算正期望（无需原始交易数据）

    Args:
        win_rate: 胜率 (0.0 ~ 1.0)
        avg_win: 平均盈利金额
        avg_loss: 平均亏损金额（正值）
        total_trades: 总交易数
        avg_commission_pct: 平均佣金比例（买卖合计），默认 0.04%

    Returns:
        ExpectancyResult: 正期望计算结果
    """
    losing_trades = int(total_trades * (1 - win_rate))
    winning_trades = total_trades - losing_trades

    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    # 估算平均成本
    avg_cost = (avg_win + avg_loss) * avg_commission_pct if avg_commission_pct else 0.0

    # 计算期望值
    p_loss = 1 - win_rate
    expectancy = win_rate * avg_win - p_loss * avg_loss - avg_cost

    return ExpectancyResult(
        expectancy=expectancy,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_loss_ratio=profit_loss_ratio,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        total_cost=avg_cost * total_trades,
    )


def is_viable_strategy(
    expectancy_result: ExpectancyResult,
    min_win_rate: float = 0.25,
    min_ratio: float = 3.0,
) -> bool:
    """
    判断策略是否可行

    中长线策略要求：
    1. 胜率 >= 25%（保守估计，理想 30-40%）
    2. 盈亏比 >= 3:1
    3. 期望值 > 0

    Args:
        expectancy_result: 正期望计算结果
        min_win_rate: 最低胜率要求
        min_ratio: 最低盈亏比要求

    Returns:
        策略是否可行
    """
    conditions = [
        expectancy_result.win_rate >= min_win_rate,
        expectancy_result.profit_loss_ratio >= min_ratio,
        expectancy_result.is_positive,
    ]

    return all(conditions)
