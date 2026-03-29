"""移动平均线指标

MA (Moving Average) - 趋势判断基础指标
"""

from typing import List, Optional
import pandas as pd


def calculate_ma(
    df: pd.DataFrame,
    periods: List[int] = [5, 10, 20, 60],
    column: str = "close"
) -> pd.DataFrame:
    """
    计算移动平均线

    Args:
        df: 包含收盘价的 DataFrame
        periods: MA周期列表，默认 [5, 10, 20, 60]
        column: 用于计算MA的列名，默认 close

    Returns:
        添加了 ma{N} 列的 DataFrame

    Example:
        >>> df = calculate_ma(df, periods=[5, 20, 60])
        >>> df[['date', 'close', 'ma5', 'ma20', 'ma60']].tail()
    """
    df = df.copy()

    for period in periods:
        col_name = f"ma{period}"
        if col_name not in df.columns:
            df[col_name] = df[column].rolling(window=period, min_periods=1).mean()

    return df


def golden_cross(ma_fast: pd.Series, ma_slow: pd.Series) -> bool:
    """
    检测金叉（买入信号）

    金叉条件：快线从下穿上（前一交易日快线 < 慢线，当前快线 >= 慢线）

    Args:
        ma_fast: 快线序列
        ma_slow: 慢线序列

    Returns:
        True if golden cross detected

    Note:
        金叉表示短期趋势向上，可能预示上涨
    """
    if len(ma_fast) < 2 or len(ma_slow) < 2:
        return False

    # 前一交易日：快线在慢线下方
    prev_fast_below = ma_fast.iloc[-2] < ma_slow.iloc[-2]
    # 当前交易日：快线在慢线上方或相等
    curr_fast_above = ma_fast.iloc[-1] >= ma_slow.iloc[-1]

    return prev_fast_below and curr_fast_above


def death_cross(ma_fast: pd.Series, ma_slow: pd.Series) -> bool:
    """
    检测死叉（卖出信号）

    死叉条件：快线从上穿下（前一交易日快线 > 慢线，当前快线 <= 慢线）

    Args:
        ma_fast: 快线序列
        ma_slow: 慢线序列

    Returns:
        True if death cross detected

    Note:
        死叉表示短期趋势向下，可能预示下跌
    """
    if len(ma_fast) < 2 or len(ma_slow) < 2:
        return False

    # 前一交易日：快线在慢线上方
    prev_fast_above = ma_fast.iloc[-2] > ma_slow.iloc[-2]
    # 当前交易日：快线在慢线下方或相等
    curr_fast_below = ma_fast.iloc[-1] <= ma_slow.iloc[-1]

    return prev_fast_above and curr_fast_below


def ma_bullish(ma_fast: pd.Series, ma_slow: pd.Series) -> bool:
    """
    检测 MA 多头排列（上涨趋势）

    多头排列：快线 > 慢线 > 更慢的线

    Args:
        ma_fast: 快线序列（短期）
        ma_slow: 慢线序列（中期）

    Returns:
        True if bullish alignment detected
    """
    if len(ma_fast) < 1 or len(ma_slow) < 1:
        return False

    return ma_fast.iloc[-1] > ma_slow.iloc[-1]


def ma_bearish(ma_fast: pd.Series, ma_slow: pd.Series) -> bool:
    """
    检测 MA 空头排列（下跌趋势）

    空头排列：快线 < 慢线

    Args:
        ma_fast: 快线序列（短期）
        ma_slow: 慢线序列（中期）

    Returns:
        True if bearish alignment detected
    """
    if len(ma_fast) < 1 or len(ma_slow) < 1:
        return False

    return ma_fast.iloc[-1] < ma_slow.iloc[-1]


def price_above_ma(price: float, ma: float) -> bool:
    """
    检测价格是否在 MA 上方

    Args:
        price: 当前价格
        ma: 移动平均线值

    Returns:
        True if price is above MA
    """
    return price > ma


def price_below_ma(price: float, ma: float) -> bool:
    """
    检测价格是否在 MA 下方

    Args:
        price: 当前价格
        ma: 移动平均线值

    Returns:
        True if price is below MA
    """
    return price < ma


def ma_slope(ma_series: pd.Series, period: int = 5) -> float:
    """
    计算 MA 斜率

    Args:
        ma_series: MA 序列
        period: 计算斜率的周期

    Returns:
        斜率（百分比），正值表示向上
    """
    if len(ma_series) < period:
        period = len(ma_series)

    if period < 2:
        return 0.0

    start_val = ma_series.iloc[-period]
    end_val = ma_series.iloc[-1]

    if start_val == 0:
        return 0.0

    return ((end_val - start_val) / start_val) * 100
