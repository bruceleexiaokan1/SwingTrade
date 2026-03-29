"""成交量指标

Volume - 量能分析
用于确认趋势的可靠性和突破的真实性
"""

import pandas as pd


def calculate_volume_ma(
    df: pd.DataFrame,
    period: int = 20,
    column: str = "volume"
) -> pd.DataFrame:
    """
    计算成交量移动平均线

    Args:
        df: 包含成交量列的 DataFrame
        period: MA 周期，默认 20
        column: 成交量列名，默认 volume

    Returns:
        添加了 volume_ma 列的 DataFrame

    Example:
        >>> df = calculate_volume_ma(df)
        >>> df[['date', 'volume', 'volume_ma']].tail()
    """
    df = df.copy()

    df["volume_ma"] = df[column].rolling(window=period, min_periods=1).mean()

    return df


def volume_surge(
    df: pd.DataFrame,
    threshold: float = 1.5,
    column: str = "volume"
) -> bool:
    """
    检测放量（成交量超过均量的 threshold 倍）

    Args:
        df: 包含 volume 和 volume_ma 列的 DataFrame
        threshold: 倍数阈值，默认 1.5（1.5倍）
        column: 成交量列名

    Returns:
        True if volume > threshold * volume_ma

    Note:
        放量通常发生在：
        - 趋势启动或确认时
        - 突破关键位置时
        - 趋势尾声（出货）
    """
    if "volume_ma" not in df.columns:
        return False

    current_volume = df[column].iloc[-1]
    avg_volume = df["volume_ma"].iloc[-1]

    if avg_volume <= 0:
        return False

    return current_volume > (avg_volume * threshold)


def volume_shrink(
    df: pd.DataFrame,
    threshold: float = 0.5,
    column: str = "volume"
) -> bool:
    """
    检测缩量（成交量低于均量的 threshold 倍）

    Args:
        df: 包含 volume 和 volume_ma 列的 DataFrame
        threshold: 倍数阈值，默认 0.5（一半）
        column: 成交量列名

    Returns:
        True if volume < threshold * volume_ma

    Note:
        缩量通常发生在：
        - 趋势回调/盘整时（正常）
        - 底部区域（即将反转信号）
    """
    if "volume_ma" not in df.columns:
        return False

    current_volume = df[column].iloc[-1]
    avg_volume = df["volume_ma"].iloc[-1]

    if avg_volume <= 0:
        return False

    return current_volume < (avg_volume * threshold)


def volume_ratio(df: pd.DataFrame, column: str = "volume") -> float:
    """
    计算成交量比率（当前量 / 均量）

    Args:
        df: 包含 volume 和 volume_ma 列的 DataFrame
        column: 成交量列名

    Returns:
        成交量比率
    """
    if "volume_ma" not in df.columns:
        return 1.0

    current_volume = df[column].iloc[-1]
    avg_volume = df["volume_ma"].iloc[-1]

    if avg_volume <= 0:
        return 1.0

    return current_volume / avg_volume


def volume_breakout(
    df: pd.DataFrame,
    lookback: int = 20,
    threshold: float = 1.5,
    column: str = "volume"
) -> bool:
    """
    检测量能突破（成交量突破近 N 日平均的 threshold 倍）

    Args:
        df: 包含成交量列的 DataFrame
        lookback: 回溯周期，默认 20
        threshold: 倍数阈值，默认 1.5
        column: 成交量列名

    Returns:
        True if volume broke out
    """
    if len(df) < lookback:
        return False

    recent_volumes = df[column].iloc[-lookback:-1]
    avg_volume = recent_volumes.mean()

    if avg_volume <= 0:
        return False

    current_volume = df[column].iloc[-1]

    return current_volume > (avg_volume * threshold)


def volume_pullback(
    df: pd.DataFrame,
    peak_volume: float,
    threshold: float = 0.33,
    column: str = "volume"
) -> bool:
    """
    检测量能萎缩（回调时量能萎缩至峰值的 threshold 以下）

    Args:
        df: 包含成交量列的 DataFrame
        peak_volume: 前期峰值量
        threshold: 萎缩比例阈值，默认 0.33（1/3）
        column: 成交量列名

    Returns:
        True if volume shrank to below threshold of peak

    Note:
        回调时缩量是健康的表现，表明抛压不重
    """
    current_volume = df[column].iloc[-1]

    if peak_volume <= 0:
        return False

    return current_volume < (peak_volume * threshold)


def obv(df: pd.DataFrame, column: str = "volume") -> pd.DataFrame:
    """
    计算 OBV（能量潮指标）

    Args:
        df: 包含收盘价和成交量列的 DataFrame
        column: 成交量列名

    Returns:
        添加了 obv 列的 DataFrame

    Formula:
        If close > prev_close: OBV = OBV + volume
        If close < prev_close: OBV = OBV - volume
        If close == prev_close: OBV unchanged
    """
    df = df.copy()

    # 价格变动
    price_change = df["close"].diff()

    # 初始化 OBV
    df["obv"] = 0.0

    # 计算 OBV
    for i in range(1, len(df)):
        if price_change.iloc[i] > 0:
            df.loc[df.index[i], "obv"] = df["obv"].iloc[i - 1] + df[column].iloc[i]
        elif price_change.iloc[i] < 0:
            df.loc[df.index[i], "obv"] = df["obv"].iloc[i - 1] - df[column].iloc[i]
        else:
            df.loc[df.index[i], "obv"] = df["obv"].iloc[i - 1]

    return df


def obv_trend(obv: pd.Series, period: int = 10) -> str:
    """
    检测 OBV 趋势

    Args:
        obv: OBV 序列
        period: 检测周期

    Returns:
        "uptrend", "downtrend", "sideways"
    """
    if len(obv) < period:
        period = len(obv)

    if period < 2:
        return "sideways"

    recent_obv = obv.iloc[-period:]

    # 简单判断：斜率
    slope = (recent_obv.iloc[-1] - recent_obv.iloc[0]) / recent_obv.iloc[0] if recent_obv.iloc[0] != 0 else 0

    if slope > 0.05:  # 5% 以上上升
        return "uptrend"
    elif slope < -0.05:  # 5% 以上下降
        return "downtrend"
    else:
        return "sideways"
