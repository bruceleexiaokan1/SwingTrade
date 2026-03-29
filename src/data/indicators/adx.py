"""ADX 指标

ADX (Average Directional Index) - 平均方向指数
衡量趋势强度的指标，配合 +DI 和 -DI 使用

ADX 解读：
- ADX > 25: 趋势确认（强劲）
- ADX < 20: 震荡市场
- ADX 急剧上升: 趋势形成
- +DI > -DI: 多头趋势
- -DI > +DI: 空头趋势
"""

import pandas as pd
import numpy as np


def calculate_adx(
    df: pd.DataFrame,
    period: int = 14
) -> pd.DataFrame:
    """
    计算 ADX, +DI, -DI 指标

    Args:
        df: 包含 high, low, close 列的 DataFrame
        period: 周期，默认 14

    Returns:
        添加了 +dm, -dm, +di, -di, dx, adx 列的 DataFrame

    Formula:
        +DM = max(H - prev_H, 0)  (仅在正值时取)
        -DM = max(prev_L - L, 0)  (仅在正值时取)

        +DI = 100 * EMA(+DM, period) / ATR
        -DI = 100 * EMA(-DM, period) / ATR

        DX = 100 * |(+DI - -DI)| / (+DI + -DI)
        ADX = EMA(DX, period)

    Example:
        >>> df = calculate_adx(df)
        >>> df[['date', 'adx', 'plus_di', 'minus_di']].tail()
    """
    df = df.copy()

    # 确保数据按时间排序
    df = df.sort_values('date').reset_index(drop=True)

    # 计算价格变动
    high = df['high']
    low = df['low']
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    # Directional Movement (方向性移动)
    # +DM: 今日高点高于昨日高点，且低于昨日低点之差
    # -DM: 昨日低点低于今日低点，且高于昨日高点之差
    plus_dm = high - prev_high
    minus_dm = prev_low - low

    # 仅保留正向值，并取两者中较大者（避免重复计算）
    plus_dm = plus_dm.where(plus_dm > 0, 0)
    minus_dm = minus_dm.where(minus_dm > 0, 0)

    # 如果两者同时为正，取较大者
    both_positive = (plus_dm > 0) & (minus_dm > 0)
    plus_dm = plus_dm.where(~both_positive | (plus_dm > minus_dm), 0)
    minus_dm = minus_dm.where(~both_positive | (minus_dm > plus_dm), 0)

    df['plus_dm'] = plus_dm
    df['minus_dm'] = minus_dm

    # 计算 True Range (真实波幅) - ATR 的基础
    prev_close = df['close'].shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 计算 ATR (使用指数移动平均)
    df['atr'] = df['tr'].ewm(span=period, min_periods=1, adjust=False).mean()

    # 计算 +DI 和 -DI
    # 使用 EMA 计算 +DM 和 -DM 的平滑值
    plus_dm_ema = df['plus_dm'].ewm(span=period, min_periods=1, adjust=False).mean()
    minus_dm_ema = df['minus_dm'].ewm(span=period, min_periods=1, adjust=False).mean()

    # +DI = 100 * EMA(+DM) / ATR
    df['plus_di'] = 100 * plus_dm_ema / df['atr']

    # -DI = 100 * EMA(-DM) / ATR
    df['minus_di'] = 100 * minus_dm_ema / df['atr']

    # 计算 DX (Directional Index)
    # DX = 100 * |(+DI - -DI)| / (+DI + -DI)
    di_sum = df['plus_di'] + df['minus_di']
    df['dx'] = np.where(di_sum > 0, 100 * abs(df['plus_di'] - df['minus_di']) / di_sum, np.nan)

    # 计算 ADX (对 DX 进行平滑)
    df['adx'] = df['dx'].ewm(span=period, min_periods=1, adjust=False).mean()

    return df


def adx_strong_trend(adx: float, threshold: float = 25) -> bool:
    """
    判断是否为强劲趋势

    Args:
        adx: ADX 值
        threshold: 阈值，默认 25

    Returns:
        True 如果 ADX > threshold（趋势确认）

    Example:
        >>> adx_strong_trend(30)  # True - 强劲趋势
        >>> adx_strong_trend(15)  # False - 震荡市场
    """
    return adx > threshold


def adx_weak_trend(adx: float, threshold: float = 20) -> bool:
    """
    判断是否为弱势/震荡市场

    Args:
        adx: ADX 值
        threshold: 阈值，默认 20

    Returns:
        True 如果 ADX < threshold（震荡市场）

    Example:
        >>> adx_weak_trend(15)  # True - 震荡市场
        >>> adx_weak_trend(25)  # False - 有趋势
    """
    return adx < threshold


def adx_rising(adx_series: pd.Series, lookback: int = 5) -> bool:
    """
    判断 ADX 是否在上升（趋势形成中）

    Args:
        adx_series: ADX 序列
        lookback: 回溯期，默认 5

    Returns:
        True 如果 ADX 呈上升趋势

    Example:
        >>> adx_rising(df['adx'])  # True if ADX rising
    """
    if len(adx_series) < lookback:
        return False

    recent = adx_series.tail(lookback).values
    # 简单判断：最后值大于第一个值
    return recent[-1] > recent[0]


def adx_bullish_signal(adx: float, plus_di: float, minus_di: float) -> bool:
    """
    判断 ADX 多头信号

    Args:
        adx: ADX 值
        plus_di: +DI 值
        minus_di: -DI 值

    Returns:
        True 如果 ADX > 25 且 +DI > -DI

    Example:
        >>> adx_bullish_signal(30, 35, 20)  # True - 多头信号
    """
    return (adx > 25) and (plus_di > minus_di)


def adx_bearish_signal(adx: float, plus_di: float, minus_di: float) -> bool:
    """
    判断 ADX 空头信号

    Args:
        adx: ADX 值
        plus_di: +DI 值
        minus_di: -DI 值

    Returns:
        True 如果 ADX > 25 且 -DI > +DI

    Example:
        >>> adx_bearish_signal(30, 20, 35)  # True - 空头信号
    """
    return (adx > 25) and (minus_di > plus_di)


def adx_trend_strength(adx: float) -> str:
    """
    根据 ADX 值返回趋势强度描述

    Args:
        adx: ADX 值

    Returns:
        趋势强度描述：
        - "very_weak" < 10
        - "weak" 10~20
        - "moderate" 20~25
        - "strong" 25~50
        - "very_strong" > 50
    """
    if adx < 10:
        return "very_weak"
    elif adx < 20:
        return "weak"
    elif adx < 25:
        return "moderate"
    elif adx < 50:
        return "strong"
    else:
        return "very_strong"
