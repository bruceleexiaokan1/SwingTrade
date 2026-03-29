"""MACD 指标

MACD (Moving Average Convergence Divergence) - 指数平滑异同移动平均线
动能指标，用于判断趋势强度和方向
"""

from typing import Tuple
import pandas as pd


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close"
) -> pd.DataFrame:
    """
    计算 MACD 指标

    Args:
        df: 包含收盘价的 DataFrame
        fast: 快线 EMA 周期，默认 12
        slow: 慢线 EMA 周期，默认 26
        signal: 信号线周期，默认 9
        column: 用于计算 MACD 的列名，默认 close

    Returns:
        添加了 dif, dem, hist 列的 DataFrame
        - dif: 快线 EMA - 慢线 EMA（DIF线）
        - dem: DIF 的 EMA 信号线（DEA线）
        - hist: DIF - DEM（柱状图）

    Formula:
        DIF = EMA(close, fast) - EMA(close, slow)
        DEM = EMA(DIF, signal)
        HIST = DIF - DEM

    Example:
        >>> df = calculate_macd(df)
        >>> df[['date', 'close', 'dif', 'dem', 'hist']].tail()
    """
    df = df.copy()

    # 计算快线和慢线的 EMA
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()

    # DIF = 快线 EMA - 慢线 EMA
    df["dif"] = ema_fast - ema_slow

    # DEM = DIF 的 EMA（信号线）
    df["dem"] = df["dif"].ewm(span=signal, adjust=False).mean()

    # HIST = DIF - DEM（柱状图）
    df["hist"] = df["dif"] - df["dem"]

    return df


def macd_bullish(df: pd.DataFrame) -> bool:
    """
    检测 MACD 多头信号

    多头信号：DIF > 0（零轴上方）

    Args:
        df: 包含 dif 列的 DataFrame

    Returns:
        True if DIF > 0
    """
    if "dif" not in df.columns or len(df) < 1:
        return False

    return df["dif"].iloc[-1] > 0


def macd_bearish(df: pd.DataFrame) -> bool:
    """
    检测 MACD 空头信号

    空头信号：DIF < 0（零轴下方）

    Args:
        df: 包含 dif 列的 DataFrame

    Returns:
        True if DIF < 0
    """
    if "dif" not in df.columns or len(df) < 1:
        return False

    return df["dif"].iloc[-1] < 0


def macd_histogram_bullish(df: pd.DataFrame) -> bool:
    """
    检测 MACD 柱状图多头信号

    多头信号：柱状图由负转正（绿柱变红柱）

    Args:
        df: 包含 hist 列的 DataFrame

    Returns:
        True if histogram turned positive
    """
    if "hist" not in df.columns or len(df) < 2:
        return False

    prev_hist = df["hist"].iloc[-2]
    curr_hist = df["hist"].iloc[-1]

    return prev_hist <= 0 and curr_hist > 0


def macd_histogram_bearish(df: pd.DataFrame) -> bool:
    """
    检测 MACD 柱状图空头信号

    空头信号：柱状图由正转负（红柱变绿柱）

    Args:
        df: 包含 hist 列的 DataFrame

    Returns:
        True if histogram turned negative
    """
    if "hist" not in df.columns or len(df) < 2:
        return False

    prev_hist = df["hist"].iloc[-2]
    curr_hist = df["hist"].iloc[-1]

    return prev_hist >= 0 and curr_hist < 0


def macd_crossover(df: pd.DataFrame) -> Tuple[str, float]:
    """
    检测 MACD 金叉死叉

    Args:
        df: 包含 dif, dem 列的 DataFrame

    Returns:
        (信号类型, 置信度)
        - ("golden", 0.0~1.0): 金叉，DIF上穿DEM
        - ("death", 0.0~1.0): 死叉，DIF下穿DEM
        - ("none", 0.0): 无信号
    """
    if "dif" not in df.columns or "dem" not in df.columns or len(df) < 2:
        return ("none", 0.0)

    prev_dif = df["dif"].iloc[-2]
    prev_dem = df["dem"].iloc[-2]
    curr_dif = df["dif"].iloc[-1]
    curr_dem = df["dem"].iloc[-1]

    # 金叉：DIF 从下穿上
    if prev_dif <= prev_dem and curr_dif > curr_dem:
        # 置信度基于 DIF 和 DEM 的差距
        diff_pct = abs(curr_dif - curr_dem) / abs(curr_dem) if curr_dem != 0 else 0
        confidence = min(1.0, diff_pct * 10)  # 缩放到 0~1
        return ("golden", confidence)

    # 死叉：DIF 从上穿下
    if prev_dif >= prev_dem and curr_dif < curr_dem:
        diff_pct = abs(curr_dif - curr_dem) / abs(curr_dem) if curr_dem != 0 else 0
        confidence = min(1.0, diff_pct * 10)
        return ("death", confidence)

    return ("none", 0.0)
