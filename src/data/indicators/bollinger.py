"""布林带指标

Bollinger Bands - 价格波动的标准差通道
用于判断波动率变化和潜在的支撑阻力位
"""

import pandas as pd


def calculate_bollinger(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    column: str = "close"
) -> pd.DataFrame:
    """
    计算布林带指标

    Args:
        df: 包含收盘价的 DataFrame
        period: 移动平均周期，默认 20
        std_dev: 标准差倍数，默认 2.0
        column: 用于计算布林带的列名，默认 close

    Returns:
        添加了 bb_upper, bb_middle, bb_lower, bb_bandwidth, bb_position 列的 DataFrame

    Formula:
        Middle = SMA(close, period)
        Upper = Middle + (std_dev * STD(close, period))
        Lower = Middle - (std_dev * STD(close, period))
        Bandwidth = (Upper - Lower) / Middle * 100
        Position = (close - Lower) / (Upper - Lower)  (%B)

    Example:
        >>> df = calculate_bollinger(df)
        >>> df[['date', 'close', 'bb_upper', 'bb_middle', 'bb_lower']].tail()
    """
    df = df.copy()

    # 中轨 = N日简单移动平均
    df["bb_middle"] = df[column].rolling(window=period, min_periods=1).mean()

    # 标准差
    rolling_std = df[column].rolling(window=period, min_periods=1).std()

    # 上轨 = 中轨 + (倍数 * 标准差)
    df["bb_upper"] = df["bb_middle"] + (std_dev * rolling_std)

    # 下轨 = 中轨 - (倍数 * 标准差)
    df["bb_lower"] = df["bb_middle"] - (std_dev * rolling_std)

    # 带宽 = (上轨 - 下轨) / 中轨 * 100
    df["bb_bandwidth"] = ((df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]) * 100

    # %B 位置 = (价格 - 下轨) / (上轨 - 下轨)
    denominator = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df[column] - df["bb_lower"]) / denominator

    # 处理除零情况
    df["bb_position"] = df["bb_position"].fillna(0.5)

    return df


def bollinger_breakout_upper(df: pd.DataFrame, column: str = "close") -> bool:
    """
    检测价格突破布林带上轨

    Args:
        df: 包含 bb_upper 和收盘价的 DataFrame
        column: 价格列名

    Returns:
        True if price broke above upper band
    """
    if "bb_upper" not in df.columns:
        return False

    return df[column].iloc[-1] > df["bb_upper"].iloc[-1]


def bollinger_breakout_lower(df: pd.DataFrame, column: str = "close") -> bool:
    """
    检测价格跌破布林带下轨

    Args:
        df: 包含 bb_lower 和收盘价的 DataFrame
        column: 价格列名

    Returns:
        True if price broke below lower band
    """
    if "bb_lower" not in df.columns:
        return False

    return df[column].iloc[-1] < df["bb_lower"].iloc[-1]


def bollinger_squeeze(df: pd.DataFrame, threshold: float = 3.0) -> bool:
    """
    检测布林带收口（压缩）

    布林带收口通常预示着大幅波动即将到来

    Args:
        df: 包含 bb_bandwidth 列的 DataFrame
        threshold: 带宽阈值（%），低于此值视为收口

    Returns:
        True if bands are squeezed (narrowing)
    """
    if "bb_bandwidth" not in df.columns:
        return False

    # 当前带宽是否低于阈值
    current_bandwidth = df["bb_bandwidth"].iloc[-1]

    if current_bandwidth < threshold:
        return True

    # 或者检测带宽是否收窄（比前期小）
    if len(df) >= 5:
        recent_bandwidth = df["bb_bandwidth"].iloc[-5:-1].mean()
        return current_bandwidth < recent_bandwidth * 0.9  # 收窄10%以上

    return False


def bollinger_expansion(df: pd.DataFrame, threshold: float = 5.0) -> bool:
    """
    检测布林带开口（扩张）

    Args:
        df: 包含 bb_bandwidth 列的 DataFrame
        threshold: 带宽阈值（%），高于此值视为开口

    Returns:
        True if bands are expanding (widening)
    """
    if "bb_bandwidth" not in df.columns:
        return False

    current_bandwidth = df["bb_bandwidth"].iloc[-1]

    if current_bandwidth > threshold:
        return True

    # 或者检测带宽是否扩张（比前期大）
    if len(df) >= 5:
        recent_bandwidth = df["bb_bandwidth"].iloc[-5:-1].mean()
        return current_bandwidth > recent_bandwidth * 1.1  # 扩张10%以上

    return False


def bollinger_position_zone(position: float) -> str:
    """
    判断布林带 %B 位置区域

    Args:
        position: %B 位置值（0~1）

    Returns:
        区域名称：
        - "below_lower" 在下轨下方
        - "lower" 在下半部分（下轨和中轨之间）
        - "middle" 在中间
        - "upper" 在上半部分（中轨和上轨之间）
        - "above_upper" 在上轨上方
    """
    if position < 0:
        return "below_lower"
    elif position < 0.25:
        return "lower"
    elif position < 0.5:
        return "middle"
    elif position < 0.75:
        return "upper"
    elif position <= 1.0:
        return "above_upper"
    else:
        return "above_upper"
