"""RSI 指标

RSI (Relative Strength Index) - 相对强弱指标
衡量价格变动的速度和幅度，用于判断超买超卖
"""

from typing import List
import pandas as pd


def calculate_rsi(
    df: pd.DataFrame,
    periods: List[int] = [6, 14],
    column: str = "close"
) -> pd.DataFrame:
    """
    计算 RSI 指标

    Args:
        df: 包含收盘价的 DataFrame
        periods: RSI 周期列表，默认 [6, 14]
        column: 用于计算 RSI 的列名，默认 close

    Returns:
        添加了 rsi{N} 列的 DataFrame

    Formula:
        RSI = 100 - (100 / (1 + RS))
        RS = 平均涨幅 / 平均跌幅

    Example:
        >>> df = calculate_rsi(df, periods=[6, 14])
        >>> df[['date', 'close', 'rsi6', 'rsi14']].tail()
    """
    df = df.copy()

    # 计算价格变动
    delta = df[column].diff()

    for period in periods:
        col_name = f"rsi{period}"

        # 分离上涨和下跌
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)

        # 计算平均涨幅和跌幅（使用 EMA）
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

        # 计算 RS 和 RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        df[col_name] = rsi

    return df


def rsi_oversold(rsi: float, period: int = 14) -> bool:
    """
    检测 RSI 超卖信号

    超卖：RSI < 动态阈值（可能预示反弹）
    知识库定义：
    - period=14: RSI < 30 超卖确认
    - 动态调整：短周期更敏感（阈值略高），长周期更保守（阈值略低）

    Args:
        rsi: RSI 值
        period: RSI 周期（用于动态计算阈值）

    Returns:
        True if RSI indicates oversold
    """
    # 动态阈值：period=14 时 threshold=30，短周期更敏感
    threshold = max(20, 30 - (14 - period))
    return rsi < threshold


def rsi_overbought(rsi: float, period: int = 14) -> bool:
    """
    检测 RSI 超买信号

    超买：RSI > 70（可能预示回调）

    Args:
        rsi: RSI 值
        period: RSI 周期（用于判断阈值）

    Returns:
        True if RSI indicates overbought
    """
    threshold = 70
    return rsi > threshold


def rsi_extreme(rsi: float, direction: str = "buy") -> bool:
    """
    检测 RSI 极端信号

    Args:
        rsi: RSI 值
        direction: "buy" 检测极度超卖(RSI<20)，"sell" 检测极度超买(RSI>80)

    Returns:
        True if RSI is at extreme level
    """
    if direction == "buy":
        return rsi < 20
    elif direction == "sell":
        return rsi > 80
    return False


def rsi_divergence(
    prices: pd.Series,
    rsi: pd.Series,
    window: int = 20
) -> str:
    """
    检测 RSI 背离

    顶背离：价格创新高，RSI 未能创新高（看跌）
    底背离：价格创新低，RSI 未能创新低（看涨）

    Args:
        prices: 价格序列
        rsi: RSI 序列
        window: 检测窗口

    Returns:
        "bullish" 底背离（买入信号）
        "bearish" 顶背离（卖出信号）
        "none" 无背离
    """
    if len(prices) < window * 2 or len(rsi) < window * 2:
        return "none"

    # 取最近 window 天的数据
    recent_prices = prices.iloc[-window:]
    recent_rsi = rsi.iloc[-window:]

    # 前期窗口
    prev_prices = prices.iloc[-window * 2:-window]
    prev_rsi = rsi.iloc[-window * 2:-window]

    # 检测顶背离：价格创新高，RSI 未创新高
    price_new_high = recent_prices.max() > prev_prices.max()
    rsi_not_new_high = recent_rsi.max() < prev_rsi.max()

    if price_new_high and rsi_not_new_high:
        return "bearish"

    # 检测底背离：价格创新低，RSI 未创新低
    price_new_low = recent_prices.min() < prev_prices.min()
    rsi_not_new_low = recent_rsi.min() > prev_rsi.min()

    if price_new_low and rsi_not_new_low:
        return "bullish"

    return "none"


def rsi_trend(rsi: pd.Series, period: int = 14) -> str:
    """
    检测 RSI 趋势方向

    Args:
        rsi: RSI 序列
        period: 周期

    Returns:
        "strong_buy" RSI > 70
        "buy" RSI > 50
        "sell" RSI < 50
        "strong_sell" RSI < 30
    """
    if len(rsi) < 1:
        return "neutral"

    current = rsi.iloc[-1]

    if current >= 70:
        return "strong_buy"
    elif current >= 50:
        return "buy"
    elif current >= 30:
        return "sell"
    else:
        return "strong_sell"
