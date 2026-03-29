"""ATR 指标

ATR (Average True Range) - 平均真实波幅
衡量市场波动性，用于设置止损和仓位管理
"""

import pandas as pd


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14
) -> pd.DataFrame:
    """
    计算 ATR 指标

    Args:
        df: 包含 high, low, close 列的 DataFrame
        period: ATR 周期，默认 14

    Returns:
        添加了 tr, atr, atr_pct 列的 DataFrame

    Formula:
        TR = max(H - L, |H - prev_close|, |L - prev_close|)
        ATR = SMA(TR, period)

        atr_pct = ATR / close * 100  (ATR百分比，用于比较不同价格水平的股票)

    Example:
        >>> df = calculate_atr(df)
        >>> df[['date', 'high', 'low', 'close', 'tr', 'atr']].tail()
    """
    df = df.copy()

    # 前一交易日收盘价
    prev_close = df["close"].shift(1)

    # 计算三个真实波幅分量
    high_low = df["high"] - df["low"]                           # 当日高低价差
    high_close = abs(df["high"] - prev_close)                   # 最高价与前收盘价差
    low_close = abs(df["low"] - prev_close)                     # 最低价与前收盘价差

    # True Range = 三者中的最大值
    df["tr"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # ATR = True Range 的移动平均
    df["atr"] = df["tr"].rolling(window=period, min_periods=1).mean()

    # ATR百分比 = ATR / 收盘价 * 100（用于横向比较）
    df["atr_pct"] = (df["atr"] / df["close"]) * 100

    return df


def atr_stop_loss(
    entry_price: float,
    atr: float,
    multiplier: float = 2.0,
    direction: str = "long"
) -> float:
    """
    计算 ATR 止损价格

    Args:
        entry_price: 入场价格
        atr: ATR 值
        multiplier: ATR 倍数，默认 2.0
        direction: "long" 做多止损，"short" 做空止损

    Returns:
        止损价格

    Formula:
        Long Stop = Entry Price - (multiplier * ATR)
        Short Stop = Entry Price + (multiplier * ATR)

    Example:
        >>> stop = atr_stop_loss(100.0, 2.5, multiplier=2.0)
        >>> print(f"止损价: {stop}")  # 95.0
    """
    if direction == "long":
        return entry_price - (multiplier * atr)
    else:
        return entry_price + (multiplier * atr)


def atr_trailing_stop(
    highest_price: float,
    atr: float,
    multiplier: float = 3.0
) -> float:
    """
    计算 ATR 追踪止损价格（用于多头）

    也称为 Chandelier Exit（吊灯止损）

    Args:
        highest_price: 持仓期间最高价
        atr: ATR 值
        multiplier: ATR 倍数，默认 3.0

    Returns:
        追踪止损价格

    Formula:
        Trailing Stop = Highest Price - (multiplier * ATR)

    Example:
        >>> trailing = atr_trailing_stop(120.0, 2.5, multiplier=3.0)
        >>> print(f"追踪止损: {trailing}")  # 112.5
    """
    return highest_price - (multiplier * atr)


def atr_position_size(
    account_value: float,
    risk_per_trade: float,
    atr: float,
    entry_price: float
) -> int:
    """
    根据 ATR 计算仓位大小

    Args:
        account_value: 账户总值
        risk_per_trade: 每笔交易风险比例（如 0.02 表示 2%）
        atr: ATR 值
        entry_price: 入场价格

    Returns:
        可买入的股数

    Formula:
        Position Size = (Account * Risk%) / (ATR * Entry Price)

    Example:
        >>> shares = atr_position_size(100000, 0.02, 2.5, 100.0)
        >>> print(f"买入股数: {shares}")
    """
    risk_amount = account_value * risk_per_trade
    risk_per_share = atr  # 每股风险 = ATR

    if risk_per_share <= 0:
        return 0

    shares = risk_amount / risk_per_share

    return int(shares)


def atr_risk_reward(
    entry_price: float,
    target_price: float,
    atr: float,
    direction: str = "long"
) -> float:
    """
    计算风险回报比

    Args:
        entry_price: 入场价格
        target_price: 目标价格
        atr: ATR 值
        direction: "long" 或 "short"

    Returns:
        风险回报比（如 2.0 表示 1:2）

    Example:
        >>> rr = atr_risk_reward(100.0, 120.0, 2.0)
        >>> print(f"风险回报比: {rr}")  # 10.0
    """
    if direction == "long":
        potential_profit = target_price - entry_price
        potential_loss = atr  # 使用1倍ATR作为初始止损
    else:
        potential_profit = entry_price - target_price
        potential_loss = atr

    if potential_loss <= 0:
        return 0.0

    return potential_profit / potential_loss


def atr_volatility(atr_pct: float) -> str:
    """
    根据 ATR 百分比判断波动率等级

    Args:
        atr_pct: ATR 百分比

    Returns:
        波动率等级：
        - "very_low" < 2%
        - "low" 2~4%
        - "medium" 4~8%
        - "high" 8~15%
        - "very_high" > 15%
    """
    if atr_pct < 2:
        return "very_low"
    elif atr_pct < 4:
        return "low"
    elif atr_pct < 8:
        return "medium"
    elif atr_pct < 15:
        return "high"
    else:
        return "very_high"
