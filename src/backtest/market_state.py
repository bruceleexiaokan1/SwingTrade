"""市场状态识别模块

识别三种市场状态：
- 趋势市 (TREND): ADX > 25，只做顺势波段，放宽止损
- 震荡市 (VOLATILE): ADX < 20，空仓或切换期权/网格
- 转折市 (TRANSITION): 波动率急剧放大 + 关键位突破

参考知识库：
| 市场状态 | 条件 | 策略动作 |
|----------|------|----------|
| 趋势市 | ADX > 25 | 只做顺势波段，放宽止损 |
| 震荡市 | ADX < 20 | 空仓或切换期权/网格 |
| 转折市 | 波动率急剧放大 + 关键位突破 | 趋势反转信号，重仓区 |
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

from src.data.indicators.adx import calculate_adx


class MarketState(Enum):
    """市场状态枚举"""
    TREND = "trend"           # 趋势市
    VOLATILE = "volatile"     # 震荡市（波动市）
    TRANSITION = "transition" # 转折市


@dataclass
class MarketStateResult:
    """市场状态检测结果"""
    state: MarketState
    adx: float
    volatility_ratio: float   # 当前ATR/20日ATR均值
    trend_direction: str       # "up"/"down"/"sideways"
    confidence: float          # 0.0 ~ 1.0

    # 附加信息
    adx_strong_threshold: float = 25.0
    adx_weak_threshold: float = 20.0
    volatility_spike_threshold: float = 2.0

    @property
    def state_name(self) -> str:
        """状态名称（中文）"""
        names = {
            MarketState.TREND: "趋势市",
            MarketState.VOLATILE: "震荡市",
            MarketState.TRANSITION: "转折市"
        }
        return names.get(self.state, "未知")

    @property
    def should_skip_entry(self) -> bool:
        """是否应该跳过入场（在震荡市缩小仓位）"""
        return self.state == MarketState.VOLATILE

    @property
    def position_size_multiplier(self) -> float:
        """仓位调整倍数"""
        if self.state == MarketState.VOLATILE:
            return 0.5  # 震荡市减半
        elif self.state == MarketState.TRANSITION:
            return 1.2  # 转折市可适当加重
        return 1.0  # 趋势市正常仓位


def detect_market_state(
    df: pd.DataFrame,
    adx_period: int = 14,
    adx_strong_threshold: float = 25.0,
    adx_weak_threshold: float = 20.0,
    volatility_spike_threshold: float = 2.0,
    atr_column: str = "atr"
) -> MarketStateResult:
    """
    检测市场状态

    趋势市: ADX > 25
    震荡市: ADX < 20
    转折市: 波动率急剧放大 + 关键位突破

    Volatility spike detection:
        current_atr = df['atr'].iloc[-1]
        atr_ma = df['atr'].rolling(20).mean().iloc[-1]
        volatility_ratio = current_atr / atr_ma
        # volatility_ratio > 2.0 means volatility spike

    Args:
        df: 包含 OHLCV 数据的 DataFrame（需已计算 ATR）
        adx_period: ADX 计算周期，默认 14
        adx_strong_threshold: 强趋势阈值，默认 25.0
        adx_weak_threshold: 弱趋势/震荡阈值，默认 20.0
        volatility_spike_threshold: 波动率急剧放大阈值，默认 2.0
        atr_column: ATR 列名，默认 "atr"

    Returns:
        MarketStateResult: 市场状态检测结果
    """
    if len(df) < adx_period + 20:
        # 数据不足，返回默认状态
        return MarketStateResult(
            state=MarketState.VOLATILE,
            adx=0.0,
            volatility_ratio=1.0,
            trend_direction="sideways",
            confidence=0.0,
            adx_strong_threshold=adx_strong_threshold,
            adx_weak_threshold=adx_weak_threshold,
            volatility_spike_threshold=volatility_spike_threshold
        )

    # 确保 ADX 已计算
    if 'adx' not in df.columns:
        df = calculate_adx(df, period=adx_period)

    # 获取最新值
    adx = df['adx'].iloc[-1]
    plus_di = df['plus_di'].iloc[-1]
    minus_di = df['minus_di'].iloc[-1]

    # 处理 NaN
    if pd.isna(adx):
        adx = 0.0

    # 计算波动率比率
    if atr_column in df.columns:
        current_atr = df[atr_column].iloc[-1]
        atr_ma = df[atr_column].rolling(20).mean().iloc[-1]

        if pd.isna(atr_ma) or atr_ma <= 0:
            volatility_ratio = 1.0
        else:
            volatility_ratio = current_atr / atr_ma
    else:
        volatility_ratio = 1.0

    # 判断趋势方向
    if plus_di > minus_di:
        trend_direction = "up"
    elif minus_di > plus_di:
        trend_direction = "down"
    else:
        trend_direction = "sideways"

    # 判断市场状态
    # 1. 转折市检测：波动率急剧放大（优先判断）
    is_volatility_spike = volatility_ratio > volatility_spike_threshold

    if is_volatility_spike:
        state = MarketState.TRANSITION
        confidence = min(1.0, (volatility_ratio - volatility_spike_threshold) / volatility_spike_threshold)
    # 2. 趋势市检测：ADX > 25
    elif adx > adx_strong_threshold:
        state = MarketState.TREND
        confidence = min(1.0, (adx - adx_strong_threshold) / (100 - adx_strong_threshold))
    # 3. 震荡市检测：ADX < 20
    elif adx < adx_weak_threshold:
        state = MarketState.VOLATILE
        confidence = min(1.0, (adx_weak_threshold - adx) / adx_weak_threshold)
    # 4. 中间地带：视为震荡
    else:
        state = MarketState.VOLATILE
        confidence = 0.5

    return MarketStateResult(
        state=state,
        adx=float(adx),
        volatility_ratio=float(volatility_ratio),
        trend_direction=trend_direction,
        confidence=float(confidence),
        adx_strong_threshold=adx_strong_threshold,
        adx_weak_threshold=adx_weak_threshold,
        volatility_spike_threshold=volatility_spike_threshold
    )


def detect_breakout(
    df: pd.DataFrame,
    lookback: int = 20,
    breakout_threshold: float = 0.02
) -> bool:
    """
    检测是否发生关键位突破

    Args:
        df: 包含 high, low, close 列的 DataFrame
        lookback: 回溯期，默认 20 天
        breakout_threshold: 突破阈值，默认 2%

    Returns:
        bool: 是否发生突破
    """
    if len(df) < lookback + 1:
        return False

    # 获取前 N 天的最高价和最低价
    recent_high = df['high'].iloc[-lookback-1:-1].max()
    recent_low = df['low'].iloc[-lookback-1:-1].min()

    current_close = df['close'].iloc[-1]

    # 突破上前高
    if current_close > recent_high * (1 + breakout_threshold):
        return True

    # 跌破前低
    if current_close < recent_low * (1 - breakout_threshold):
        return True

    return False
