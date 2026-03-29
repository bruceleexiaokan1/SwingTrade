"""向量化信号检测器

核心设计：
1. 预计算所有信号（无循环）
2. 使用 shift() 实现交叉检测（无状态机）
3. 信号格式：[data_id, date, signal_type, confidence, ...]

性能优势：
- 信号检测 O(1) vs 原有 O(n_stocks × n_dates)
- 预计算后回测只需查表
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class SignalConfig:
    """信号配置"""
    ma_short_period: int = 20
    ma_long_period: int = 60
    rsi_period: int = 14
    rsi_oversold: int = 35
    rsi_overbought: int = 80
    atr_stop_multiplier: float = 2.0
    atr_trailing_multiplier: float = 3.0
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    volume_surge_threshold: float = 1.5


class VectorizedSignals:
    """
    向量化信号检测器

    使用向量化操作预计算所有信号，支持多股票同时处理

    信号类型：
    - golden_cross: MA 金叉
    - death_cross: MA 死叉
    - breakout: 突破布林上轨 + 放量
    - breakdown: 跌破布林下轨 + 放量
    - rsi_oversold: RSI 超卖
    - rsi_overbought: RSI 超买
    - stop_loss: ATR 止损
    - trailing_stop: 追踪止损
    - take_profit: 布林上轨止盈
    """

    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有信号

        Args:
            df: 包含指标数据的 DataFrame（由 VectorizedIndicators 计算）

        Returns:
            添加了信号列的 DataFrame
        """
        result = df.copy()

        # 确保排序
        result = result.sort_values(['data_id', 'date']).reset_index(drop=True)

        # 1. MA 交叉信号
        result = self._calculate_ma_cross_signals(result)

        # 2. RSI 信号
        result = self._calculate_rsi_signals(result)

        # 3. 布林带信号
        result = self._calculate_bollinger_signals(result)

        # 4. 成交量信号
        result = self._calculate_volume_signals(result)

        # 5. 综合入场信号
        result = self._calculate_entry_signals(result)

        # 6. 综合出场信号
        result = self._calculate_exit_signals(result)

        return result

    def _calculate_ma_cross_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """向量化计算 MA 交叉信号"""
        ma_short = f"ma{self.config.ma_short_period}"
        ma_long = f"ma{self.config.ma_long_period}"

        if ma_short not in df.columns or ma_long not in df.columns:
            return df

        # 金叉：前一天 short <= long，今天 short > long
        prev_short = df.groupby('data_id')[ma_short].shift(1)
        prev_long = df.groupby('data_id')[ma_long].shift(1)
        curr_short = df[ma_short]
        curr_long = df[ma_long]

        df['golden_cross'] = (
            (prev_short <= prev_long) &
            (curr_short > curr_long)
        ).astype(int)

        # 死叉：前一天 short >= long，今天 short < long
        df['death_cross'] = (
            (prev_short >= prev_long) &
            (curr_short < curr_long)
        ).astype(int)

        return df

    def _calculate_rsi_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """向量化计算 RSI 信号"""
        rsi_col = f"rsi{self.config.rsi_period}"

        if rsi_col not in df.columns:
            return df

        rsi = df[rsi_col]

        # RSI 超卖：RSI < rsi_oversold
        df['rsi_oversold'] = (rsi < self.config.rsi_oversold).astype(int)

        # RSI 超买：RSI > rsi_overbought
        df['rsi_overbought'] = (rsi > self.config.rsi_overbought).astype(int)

        # RSI 从超卖区域反弹（前日超卖，今日 RSI 回升）
        prev_rsi = df.groupby('data_id')[rsi_col].shift(1)
        df['rsi_oversold_bounce'] = (
            (prev_rsi < self.config.rsi_oversold) &
            (rsi >= self.config.rsi_oversold)
        ).astype(int)

        return df

    def _calculate_bollinger_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """向量化计算布林带信号"""
        if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
            return df

        price = df['close']
        bb_upper = df['bb_upper']
        bb_lower = df['bb_lower']

        # 突破布林上轨
        df['bb_breakout'] = (price > bb_upper).astype(int)

        # 跌破布林下轨
        df['bb_breakdown'] = (price < bb_lower).astype(int)

        return df

    def _calculate_volume_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """向量化计算成交量信号"""
        if 'volume_ma5' not in df.columns:
            return df

        volume = df['volume']
        volume_ma = df['volume_ma5']

        # 放量：当天成交量 > 1.5 倍均量
        df['volume_surge'] = (
            volume > volume_ma * self.config.volume_surge_threshold
        ).astype(int)

        # 缩量：当天成交量 < 0.5 倍均量
        df['volume_shrink'] = (
            volume < volume_ma * 0.5
        ).astype(int)

        return df

    def _calculate_entry_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """向量化计算综合入场信号"""
        # 入场信号 1：MA 金叉 + RSI 未超买
        rsi_col = f"rsi{self.config.rsi_period}"
        if rsi_col in df.columns:
            df['entry_golden_cross'] = (
                (df['golden_cross'] == 1) &
                (df[rsi_col] < self.config.rsi_overbought)
            ).astype(int)
        else:
            df['entry_golden_cross'] = df['golden_cross']

        # 入场信号 2：突破布林上轨 + 放量
        if 'volume_surge' in df.columns:
            df['entry_breakout'] = (
                (df['bb_breakout'] == 1) &
                (df['volume_surge'] == 1)
            ).astype(int)
        else:
            df['entry_breakout'] = df['bb_breakout']

        # 入场信号 3：RSI 超卖反弹
        if 'rsi_oversold_bounce' in df.columns:
            df['entry_rsi_bounce'] = df['rsi_oversold_bounce']
        else:
            df['entry_rsi_bounce'] = 0

        # 综合入场置信度
        df['entry_confidence'] = (
            df['entry_golden_cross'] * 0.7 +
            df['entry_breakout'] * 0.6 +
            df['entry_rsi_bounce'] * 0.4
        )

        return df

    def _calculate_exit_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """向量化计算综合出场信号"""
        # 出场信号 1：MA 死叉
        df['exit_death_cross'] = df['death_cross']

        # 出场信号 2：RSI 超买
        if 'rsi_overbought' in df.columns:
            df['exit_rsi_overbought'] = df['rsi_overbought']
        else:
            df['exit_rsi_overbought'] = 0

        return df


def detect_entry_signals_vectorized(
    df: pd.DataFrame,
    ma_short: str = 'ma20',
    ma_long: str = 'ma60',
    rsi_col: str = 'rsi14',
    rsi_overbought: int = 80
) -> pd.DataFrame:
    """
    向量化入场信号检测

    Args:
        df: 包含指标数据的 DataFrame
        ma_short: 短期均线列名
        ma_long: 长期均线列名
        rsi_col: RSI 列名
        rsi_overbought: RSI 超买阈值

    Returns:
        添加了 entry_signal, entry_confidence 列的 DataFrame
    """
    result = df.copy()

    # 1. 金叉信号
    prev_short = result.groupby('data_id')[ma_short].shift(1)
    prev_long = result.groupby('data_id')[ma_long].shift(1)
    curr_short = result[ma_short]
    curr_long = result[ma_long]

    golden_cross = (prev_short <= prev_long) & (curr_short > curr_long)

    # 2. RSI 未超买
    rsi_ok = result[rsi_col] < rsi_overbought if rsi_col in result.columns else True

    # 综合入场
    result['entry_signal'] = (golden_cross & rsi_ok).astype(int)
    result['entry_confidence'] = result['entry_signal'] * 0.7  # 金叉基础置信度

    return result


def detect_exit_signals_vectorized(
    df: pd.DataFrame,
    ma_short: str = 'ma20',
    ma_long: str = 'ma60',
    rsi_col: str = 'rsi14',
    rsi_overbought: int = 80
) -> pd.DataFrame:
    """
    向量化出场信号检测

    Args:
        df: 包含指标数据的 DataFrame
        ma_short: 短期均线列名
        ma_long: 长期均线列名
        rsi_col: RSI 列名
        rsi_overbought: RSI 超买阈值

    Returns:
        添加了 exit_signal 列的 DataFrame
    """
    result = df.copy()

    # 1. 死叉信号
    prev_short = result.groupby('data_id')[ma_short].shift(1)
    prev_long = result.groupby('data_id')[ma_long].shift(1)
    curr_short = result[ma_short]
    curr_long = result[ma_long]

    death_cross = (prev_short >= prev_long) & (curr_short < curr_long)

    # 2. RSI 超买
    rsi_overbought_signal = result[rsi_col] > rsi_overbought if rsi_col in result.columns else False

    # 综合出场
    result['exit_signal'] = (death_cross | rsi_overbought_signal).astype(int)

    return result


def detect_breakout_signals_vectorized(
    df: pd.DataFrame,
    bb_upper_col: str = 'bb_upper',
    volume_col: str = 'volume',
    volume_ma_col: str = 'volume_ma5',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    向量化突破信号检测

    Args:
        df: 包含指标数据的 DataFrame
        bb_upper_col: 布林上轨列名
        volume_col: 成交量列名
        volume_ma_col: 成交量均线列名
        threshold: 放量倍数阈值

    Returns:
        添加了 breakout_signal 列的 DataFrame
    """
    result = df.copy()

    # 突破布林上轨
    breakout = result['close'] > result[bb_upper_col] if bb_upper_col in result.columns else False

    # 放量
    volume_surge = (
        result[volume_col] > result[volume_ma_col] * threshold
    ) if volume_col in result.columns and volume_ma_col in result.columns else True

    # 综合突破信号
    result['breakout_signal'] = (breakout & volume_surge).astype(int)

    return result
