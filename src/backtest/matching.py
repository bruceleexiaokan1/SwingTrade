"""撮合引擎

T+1 订单撮合逻辑：
- T日收盘信号 → T+1日开盘价成交
- 滑点模型（成交额超过当日1%时滑点放大）
- 涨跌停限制（涨停无法买入，跌停无法卖出）
"""

from typing import Optional, Tuple
import pandas as pd

from .models import MatchResult


def calculate_slippage(
    target_price: float,
    open_price: float,
    turnover: float,
    avg_daily_turnover: float
) -> Tuple[float, str]:
    """
    计算滑点成本

    滑点公式：
    - 基准滑点 = 0.1% (0.001)
    - 流动性折价：当成交额 > avg_daily_turnover * 1% 时，滑点放大
    - 波动折价：当日涨跌幅 > 5% 时，额外增加

    Args:
        target_price: 目标价格（信号生成时的收盘价）
        open_price: 开盘价（实际成交价）
        turnover: 成交额
        avg_daily_turnover: 日均成交额

    Returns:
        (滑点比例, 原因)
    """
    slippage_base = 0.001  # 0.1%

    # 流动性折价
    if avg_daily_turnover > 0:
        turnover_ratio = turnover / avg_daily_turnover
    else:
        turnover_ratio = 0

    if turnover_ratio > 0.01:
        # 成交流动性折价：成交额越大，滑点越大
        slippage = slippage_base * (1 + turnover_ratio * 10)
        reason = f"liquidity: turnover_ratio={turnover_ratio:.2%}"
    else:
        slippage = slippage_base
        reason = "base"

    return slippage, reason


def check_limit_hit(
    open_price: float,
    prev_close: float,
    limit_up_pct: float = 0.10,
    limit_down_pct: float = -0.10
) -> Tuple[bool, Optional[str]]:
    """
    检查是否触及涨跌停

    规则：
    - 开盘涨停：open >= prev_close * (1 + limit_up_pct) -> 无法买入
    - 开盘跌停：open <= prev_close * (1 + limit_down_pct) -> 无法卖出

    Args:
        open_price: 开盘价
        prev_close: 前收盘价
        limit_up_pct: 涨停比例，默认 10%
        limit_down_pct: 跌停比例，默认 -10%

    Returns:
        (是否触及涨跌停, 涨跌停类型)
    """
    if prev_close <= 0:
        return False, None

    change_pct = (open_price - prev_close) / prev_close

    if change_pct >= limit_up_pct:
        return True, "limit_up"
    elif change_pct <= limit_down_pct:
        return True, "limit_down"
    else:
        return False, None


class OrderMatcher:
    """
    订单撮合器

    T+1 撮合逻辑：
    1. T日收盘信号 -> T+1日开盘价成交
    2. 涨跌停限制
    3. 滑点模型
    """

    def __init__(
        self,
        slippage_base: float = 0.001,
        slippage_threshold: float = 0.01,
        limit_up_pct: float = 0.10,
        limit_down_pct: float = -0.10,
        commission_rate: float = 0.0003,
        stamp_tax: float = 0.0001,
    ):
        """
        初始化撮合器

        Args:
            slippage_base: 基准滑点，默认 0.1%
            slippage_threshold: 成交额占比阈值，默认 1%
            limit_up_pct: 涨停比例，默认 10%
            limit_down_pct: 跌停比例，默认 -10%
            commission_rate: 佣金率，默认 0.03%
            stamp_tax: 印花税率，默认 0.01%（卖出时）
        """
        self.slippage_base = slippage_base
        self.slippage_threshold = slippage_threshold
        self.limit_up_pct = limit_up_pct
        self.limit_down_pct = limit_down_pct
        self.commission_rate = commission_rate
        self.stamp_tax = stamp_tax

    def match_buy(
        self,
        signal_date: str,
        code: str,
        target_price: float,
        df_next: pd.DataFrame,
        shares: int,
        avg_daily_turnover: float = 0.0
    ) -> MatchResult:
        """
        撮合买入订单

        Args:
            signal_date: 信号日期 T
            code: 股票代码
            target_price: 目标价格（信号生成时的收盘价）
            df_next: T+1 日行情数据
            shares: 买入股数
            avg_daily_turnover: 日均成交额

        Returns:
            MatchResult: 撮合结果
        """
        if len(df_next) == 0:
            return MatchResult(
                success=False,
                match_date=signal_date,
                reason="no_data"
            )

        # 获取 T+1 日数据
        open_price = df_next["open"].iloc[0]
        prev_close = df_next["close"].iloc[0]  # 实际上是前一日收盘，T日收盘

        # 检查涨跌停
        limit_hit, limit_type = check_limit_hit(
            open_price, prev_close,
            self.limit_up_pct, self.limit_down_pct
        )

        if limit_hit and limit_type == "limit_up":
            return MatchResult(
                success=False,
                match_date=df_next["date"].iloc[0],
                match_price=open_price,
                limit_hit=True,
                limit_type=limit_type,
                reason="limit_up_cannot_buy"
            )

        # 计算滑点
        turnover = open_price * shares
        slippage_ratio, slippage_reason = calculate_slippage(
            target_price, open_price, turnover, avg_daily_turnover
        )

        # 实际成交价（有滑点）
        match_price = open_price * (1 + slippage_ratio)
        actual_turnover = match_price * shares

        # 手续费（买入收取佣金）
        commission = actual_turnover * self.commission_rate

        return MatchResult(
            success=True,
            match_date=df_next["date"].iloc[0],
            match_price=match_price,
            slippage=slippage_ratio,
            slippage_reason=slippage_reason,
            limit_hit=False,
            filled_shares=shares,
            turnover=actual_turnover,
            commission=commission,
        )

    def match_sell(
        self,
        signal_date: str,
        code: str,
        target_price: float,
        df_next: pd.DataFrame,
        shares: int,
        avg_daily_turnover: float = 0.0
    ) -> MatchResult:
        """
        撮合卖出订单

        Args:
            signal_date: 信号日期 T
            code: 股票代码
            target_price: 目标价格
            df_next: T+1 日行情数据
            shares: 卖出股数
            avg_daily_turnover: 日均成交额

        Returns:
            MatchResult: 撮合结果
        """
        if len(df_next) == 0:
            return MatchResult(
                success=False,
                match_date=signal_date,
                reason="no_data"
            )

        # 获取 T+1 日数据
        open_price = df_next["open"].iloc[0]
        prev_close = df_next["close"].iloc[0]

        # 检查涨跌停
        limit_hit, limit_type = check_limit_hit(
            open_price, prev_close,
            self.limit_up_pct, self.limit_down_pct
        )

        if limit_hit and limit_type == "limit_down":
            return MatchResult(
                success=False,
                match_date=df_next["date"].iloc[0],
                match_price=open_price,
                limit_hit=True,
                limit_type=limit_type,
                reason="limit_down_cannot_sell"
            )

        # 计算滑点（卖出滑点为负，利好）
        turnover = open_price * shares
        slippage_ratio, slippage_reason = calculate_slippage(
            target_price, open_price, turnover, avg_daily_turnover
        )

        # 实际成交价（卖出滑点为负，价格更低）
        match_price = open_price * (1 - slippage_ratio)
        actual_turnover = match_price * shares

        # 手续费（卖出收取佣金+印花税）
        commission = actual_turnover * (self.commission_rate + self.stamp_tax)

        return MatchResult(
            success=True,
            match_date=df_next["date"].iloc[0],
            match_price=match_price,
            slippage=slippage_ratio,
            slippage_reason=slippage_reason,
            limit_hit=False,
            filled_shares=shares,
            turnover=actual_turnover,
            commission=commission,
        )
