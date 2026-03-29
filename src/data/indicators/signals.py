"""波段交易信号检测器

综合使用 MA、MACD、RSI、布林带、ATR、成交量等指标
检测入场和出场信号

支持通过 StrategyParams 自定义所有参数。
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Union
import pandas as pd

from .ma import calculate_ma, golden_cross, death_cross, ma_bullish, ma_bearish
from .macd import calculate_macd, macd_bullish, macd_bearish, macd_crossover
from .rsi import calculate_rsi, rsi_oversold, rsi_overbought, rsi_extreme
from .bollinger import (
    calculate_bollinger,
    bollinger_squeeze,
    bollinger_position_zone
)
from .atr import calculate_atr, atr_stop_loss, atr_trailing_stop
from .adx import calculate_adx
from .volume import calculate_volume_ma, volume_surge, volume_shrink, volume_ratio
from .chan_theory import detect_chan_signals

# 导入策略参数（支持前向引用避免循环导入）
import sys
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...backtest.strategy_params import StrategyParams
    from ...backtest.market_state import MarketState


def detect_rsi_divergence(
    df: pd.DataFrame,
    lookback: int = 20,
    min_price_diff_pct: float = 0.05,
    max_time_bars: int = 10
) -> Tuple[bool, str, float]:
    """
    检测 RSI 底背离（看涨背离）

    底背离判定条件：
    - 价格 20 日内创新低
    - RSI 对应低点高于前低（RSI 未随价格创新低）
    - 差值 >= 5%（价格新低 vs RSI 抬高）
    - 时间间隔 <= 10 根 K 线

    Args:
        df: 包含 close, rsi14 列的 DataFrame
        lookback: 检测窗口，默认 20 根 K 线
        min_price_diff_pct: 价格差异百分比阈值，默认 5%
        max_time_bars: 两个低点之间的最大时间间隔，默认 10 根 K 线

    Returns:
        (has_divergence, divergence_type, strength)
        - has_divergence: 是否检测到底背离
        - divergence_type: "bullish"（底背离）或 "bearish"（顶背离）或 ""
        - strength: 背离强度 0.0~1.0

    Example:
        >>> df = calculate_rsi(calculate_ma(df))
        >>> has_div, div_type, strength = detect_rsi_divergence(df)
    """
    if len(df) < lookback:
        return (False, "", 0.0)

    if "close" not in df.columns or "rsi14" not in df.columns:
        return (False, "", 0.0)

    prices = df["close"]
    rsi = df["rsi14"]

    if prices.isna().any() or rsi.isna().any():
        return (False, "", 0.0)

    # 获取最近 lookback 根 K 线的数据
    recent_prices = prices.iloc[-lookback:]
    recent_rsi = rsi.iloc[-lookback:]

    # 找到最近的低点（局部最小值）
    # 使用 rolling window 找局部最小值
    window_size = 3
    price_local_min = recent_prices[(recent_prices < recent_prices.shift(1)) &
                                     (recent_prices < recent_prices.shift(-1))]
    rsi_local_min = recent_rsi[(recent_rsi < recent_rsi.shift(1)) &
                                (recent_rsi < recent_rsi.shift(-1))]

    if len(price_local_min) < 2 or len(rsi_local_min) < 2:
        return (False, "", 0.0)

    # 获取最后两个价格低点的索引
    price_min_indices = price_local_min.index.tolist()
    rsi_min_indices = rsi_local_min.index.tolist()

    # 找到最近的且时间间隔满足条件的两个价格低点
    current_price_idx = price_min_indices[-1]
    prev_price_idx = None

    for idx in price_min_indices[:-1]:
        time_diff = current_price_idx - idx
        if time_diff <= max_time_bars and time_diff > 0:
            prev_price_idx = idx
            break

    if prev_price_idx is None:
        return (False, "", 0.0)

    # 获取对应的 RSI 值
    current_price_low = prices.loc[current_price_idx]
    prev_price_low = prices.loc[prev_price_idx]

    # 找到最接近这两个价格低点的 RSI 低点
    def find_nearest_rsi_low(price_idx, rsi_min_indices_list):
        """找到最近的 RSI 低点索引"""
        for rsi_idx in reversed(rsi_min_indices_list):
            if rsi_idx <= price_idx:
                return rsi_idx
        return None

    current_rsi_idx = find_nearest_rsi_low(current_price_idx, rsi_min_indices)
    prev_rsi_idx = find_nearest_rsi_low(prev_price_idx, rsi_min_indices)

    if current_rsi_idx is None or prev_rsi_idx is None:
        return (False, "", 0.0)

    current_rsi_low = rsi.loc[current_rsi_idx]
    prev_rsi_low = rsi.loc[prev_rsi_idx]

    # 底背离判定：
    # 1. 价格创新低（当前低点 < 前一低点）
    # 2. RSI 未创新低（当前 RSI > 前一 RSI）
    price_new_low = current_price_low < prev_price_low
    rsi_not_new_low = current_rsi_low > prev_rsi_low

    # 计算背离强度
    if price_new_low and rsi_not_new_low:
        # 价格下跌幅度
        price_drop_pct = (prev_price_low - current_price_low) / prev_price_low
        # RSI 上升幅度
        rsi_rise = current_rsi_low - prev_rsi_low

        # 背离强度：综合价格差异和 RSI 差异
        strength = min(1.0, price_drop_pct * 10 + rsi_rise / 50)
        strength = max(0.0, min(1.0, strength))

        return (True, "bullish", strength)

    # 顶背离判定（价格新高但 RSI 未新高）
    # 找到最近的两个价格高点
    price_local_max = recent_prices[(recent_prices > recent_prices.shift(1)) &
                                     (recent_prices > recent_prices.shift(-1))]
    rsi_local_max = recent_rsi[(recent_rsi > recent_rsi.shift(1)) &
                                (recent_rsi > recent_rsi.shift(-1))]

    if len(price_local_max) >= 2 and len(rsi_local_max) >= 2:
        price_max_indices = price_local_max.index.tolist()
        rsi_max_indices = rsi_local_max.index.tolist()

        current_price_idx = price_max_indices[-1]
        prev_price_idx = None

        for idx in price_max_indices[:-1]:
            time_diff = current_price_idx - idx
            if time_diff <= max_time_bars and time_diff > 0:
                prev_price_idx = idx
                break

        if prev_price_idx is not None:
            current_price_high = prices.loc[current_price_idx]
            prev_price_high = prices.loc[prev_price_idx]

            def find_nearest_rsi_high(price_idx, rsi_max_indices_list):
                for rsi_idx in reversed(rsi_max_indices_list):
                    if rsi_idx <= price_idx:
                        return rsi_idx
                return None

            current_rsi_idx = find_nearest_rsi_high(current_price_idx, rsi_max_indices)
            prev_rsi_idx = find_nearest_rsi_high(prev_price_idx, rsi_max_indices)

            if current_rsi_idx is not None and prev_rsi_idx is not None:
                current_rsi_high = rsi.loc[current_rsi_idx]
                prev_rsi_high = rsi.loc[prev_rsi_idx]

                price_new_high = current_price_high > prev_price_high
                rsi_not_new_high = current_rsi_high < prev_rsi_high

                if price_new_high and rsi_not_new_high:
                    price_rise_pct = (current_price_high - prev_price_high) / prev_price_high
                    rsi_fall = prev_rsi_high - current_rsi_high
                    strength = min(1.0, price_rise_pct * 10 + rsi_fall / 50)
                    strength = max(0.0, min(1.0, strength))
                    return (True, "bearish", strength)

    return (False, "", 0.0)


@dataclass
class SignalResult:
    """信号检测结果"""
    date: str
    trend: str                    # uptrend/downtrend/sideways
    trend_confidence: float       # 0.0 ~ 1.0

    entry_signal: str             # golden/breakout/none
    entry_confidence: float       # 0.0 ~ 1.0
    entry_reason: str            # 信号原因

    exit_signal: str              # stop_loss/take_profit_1/take_profit_2/trailing/none
    exit_confidence: float       # 0.0 ~ 1.0
    exit_reason: str             # 出场原因

    # 关键价格
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    trailing_stop: Optional[float] = None

    # 指标值（用于调试）
    ma20: Optional[float] = None
    ma60: Optional[float] = None
    rsi14: Optional[float] = None
    atr: Optional[float] = None

    # RSI 背离
    rsi_divergence: bool = False        # 是否底背离
    rsi_divergence_type: str = ""       # "bullish"/"bearish"/""
    rsi_divergence_strength: float = 0.0  # 背离强度 0.0~1.0

    # 市场状态
    market_state: Optional["MarketState"] = None  # 市场状态


class SwingSignals:
    """
    波段交易信号检测器

    三屏系统：
    1. 方向（趋势）：MA短期/长期, MACD 零轴
    2. 时机（信号）：RSI, 布林带
    3. 确认（量价）：成交量

    支持两种初始化方式：
    1. 传入 StrategyParams（推荐）
    2. 单独传入各参数（向后兼容）
    """

    def __init__(
        self,
        params: Optional["StrategyParams"] = None,
        # 向后兼容参数
        ma_periods: list = None,
        macd_params: tuple = None,
        rsi_periods: list = None,
        bollinger_period: int = None,
        atr_period: int = None,
        adx_period: int = None,
        volume_period: int = None,
        # RSI 阈值（向后兼容）
        rsi_oversold: int = None,
        rsi_overbought: int = None,
    ):
        """
        初始化信号检测器

        Args:
            params: 策略参数对象（推荐方式）
            ma_periods: MA 周期列表（向后兼容）
            macd_params: MACD 参数 (fast, slow, signal)（向后兼容）
            rsi_periods: RSI 周期列表（向后兼容）
            bollinger_period: 布林带周期（向后兼容）
            atr_period: ATR 周期（向后兼容）
            adx_period: ADX 周期（向后兼容）
            volume_period: 成交量均线周期（向后兼容）
            rsi_oversold: RSI 超卖阈值（向后兼容）
            rsi_overbought: RSI 超买阈值（向后兼容）
        """
        if params is not None:
            # 使用 StrategyParams
            self.params = params
            self.ma_periods = [params.ma_short, params.ma_long]
            self.macd_params = (params.macd_fast, params.macd_slow, params.macd_signal)
            self.rsi_periods = [params.rsi_period]
            self.bollinger_period = params.bollinger_period
            self.atr_period = params.atr_period
            self.adx_period = params.adx_period
            self.volume_period = params.volume_period
            self.rsi_oversold = params.rsi_oversold
            self.rsi_overbought = params.rsi_overbought
        else:
            # 使用向后兼容参数（使用默认值）
            self.params = None
            self.ma_periods = ma_periods or [5, 10, 20, 60]
            self.macd_params = macd_params or (12, 26, 9)
            self.rsi_periods = rsi_periods or [6, 14]
            self.bollinger_period = bollinger_period or 20
            self.atr_period = atr_period or 14
            self.adx_period = adx_period or 14
            self.volume_period = volume_period or 20
            self.rsi_oversold = rsi_oversold or 35
            self.rsi_overbought = rsi_overbought or 80

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有指标

        Args:
            df: 包含 OHLCV 数据的 DataFrame

        Returns:
            添加了所有指标列的 DataFrame
        """
        df = df.copy()

        # MA
        df = calculate_ma(df, self.ma_periods)

        # MACD
        df = calculate_macd(df, *self.macd_params)

        # RSI
        df = calculate_rsi(df, self.rsi_periods)

        # 布林带
        df = calculate_bollinger(df, self.bollinger_period)

        # ATR
        df = calculate_atr(df, self.atr_period)

        # ADX (需要 ATR 先计算)
        df = calculate_adx(df, self.adx_period)

        # 成交量均线
        df = calculate_volume_ma(df, self.volume_period)

        return df

    def detect_trend(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        检测趋势方向

        Args:
            df: 包含指标数据的 DataFrame

        Returns:
            (趋势方向, 置信度)
            - ("uptrend", 0.0~1.0): MA多头排列 + MACD > 0
            - ("downtrend", 0.0~1.0): MA空头排列 + MACD < 0
            - ("sideways", 0.0~1.0): 盘整
        """
        if len(df) < 60:
            return ("sideways", 0.0)

        # 获取短期/长期均线的列名
        ma_short_col = f"ma{self.params.ma_short}" if self.params else "ma20"
        ma_long_col = f"ma{self.params.ma_long}" if self.params else "ma60"

        # 检查列是否存在
        if ma_short_col not in df.columns or ma_long_col not in df.columns:
            return ("sideways", 0.0)

        ma_short = df[ma_short_col].iloc[-1]
        ma_long = df[ma_long_col].iloc[-1]
        dif = df["dif"].iloc[-1] if "dif" in df.columns else 0

        # 上涨趋势：短期均线 > 长期均线 + MACD > 0
        if ma_short > ma_long and dif > 0:
            # 置信度基于差距
            ma_diff_pct = (ma_short - ma_long) / ma_long * 100
            confidence = min(1.0, ma_diff_pct / 5)  # 5%差距 = 100%置信度
            return ("uptrend", confidence)

        # 下跌趋势：短期均线 < 长期均线 + MACD < 0
        if ma_short < ma_long and dif < 0:
            ma_diff_pct = (ma_long - ma_short) / ma_long * 100
            confidence = min(1.0, ma_diff_pct / 5)
            return ("downtrend", confidence)

        return ("sideways", 0.5)

    def detect_entry(self, df: pd.DataFrame) -> Tuple[str, float, str]:
        """
        检测入场信号

        Args:
            df: 包含指标数据的 DataFrame

        Returns:
            (信号类型, 置信度, 原因)
            - ("golden", 0.0~1.0, 原因): 金叉回踩（趋势向上 + 回调整理）
            - ("breakout", 0.0~1.0, 原因): 突破（放量突破）
            - ("none", 0.0, ""): 无信号
        """
        if len(df) < 20:
            return ("none", 0.0, "数据不足")

        # 1. 检测金叉回踩信号（主要信号）
        ma_short_col = f"ma{self.params.ma_short}" if self.params else "ma20"
        ma_long_col = f"ma{self.params.ma_long}" if self.params else "ma60"

        if ma_short_col in df.columns and ma_long_col in df.columns:
            ma_short_series = df[ma_short_col]
            ma_long_series = df[ma_long_col]

            # 金叉 + 回调企稳
            if golden_cross(ma_short_series, ma_long_series):
                # 检查 RSI 是否超卖后反弹
                rsi14 = df["rsi14"].iloc[-1] if "rsi14" in df.columns else 50
                rsi6 = df["rsi6"].iloc[-1] if "rsi6" in df.columns else 50

                if rsi14 < self.rsi_overbought:  # 未进入超买区域
                    reason = f"MA金叉 + RSI14={rsi14:.1f}"
                    confidence = 0.7 + (50 - rsi14) / 100  # RSI越低置信度越高
                    return ("golden", min(1.0, confidence), reason)

        # 2. 检测突破信号
        price = df["close"].iloc[-1]
        bb_upper = df["bb_upper"].iloc[-1] if "bb_upper" in df.columns else price
        bb_lower = df["bb_lower"].iloc[-1] if "bb_lower" in df.columns else price

        # 突破布林上轨 + 放量
        if price > bb_upper and volume_surge(df):
            reason = "突破布林上轨 + 放量"
            confidence = min(1.0, volume_ratio(df))
            return ("breakout", confidence, reason)

        # 3. RSI 超卖反弹
        if "rsi14" in df.columns:
            rsi14 = df["rsi14"].iloc[-1]
            if rsi14 < self.rsi_oversold:  # 超卖区域
                reason = f"RSI超卖({rsi14:.1f})"
                confidence = (self.rsi_oversold - rsi14) / self.rsi_oversold * 0.5  # 最高50%置信度
                return ("golden", confidence, reason)

        return ("none", 0.0, "")

    def detect_exit(
        self,
        df: pd.DataFrame,
        entry_price: Optional[float] = None,
        highest_price: Optional[float] = None
    ) -> Tuple[str, float, str]:
        """
        检测出场信号

        Args:
            df: 包含指标数据的 DataFrame
            entry_price: 入场价格（用于计算止损止盈）
            highest_price: 持仓期间最高价（用于追踪止损）

        Returns:
            (信号类型, 置信度, 原因)
        """
        if len(df) < 2:
            return ("none", 0.0, "")

        signals = []

        # 获取止损/止盈倍数
        stop_mult = self.params.atr_stop_multiplier if self.params else 2.0
        trail_mult = self.params.atr_trailing_multiplier if self.params else 3.0

        # 1. ATR 止损信号
        if entry_price is not None and "atr" in df.columns:
            atr = df["atr"].iloc[-1]
            stop = atr_stop_loss(entry_price, atr, multiplier=stop_mult)
            price = df["close"].iloc[-1]

            if price <= stop:
                confidence = min(1.0, (entry_price - price) / entry_price * 5)
                signals.append(("stop_loss", confidence, f"触及ATR止损({stop:.2f})"))

        # 2. 追踪止损信号
        if highest_price is not None and "atr" in df.columns:
            atr = df["atr"].iloc[-1]
            trailing = atr_trailing_stop(highest_price, atr, multiplier=trail_mult)
            price = df["close"].iloc[-1]

            if price <= trailing:
                confidence = 0.8
                signals.append(("trailing", confidence, f"触及追踪止损({trailing:.2f})"))

        # 3. MA 死叉出场
        ma_short_col = f"ma{self.params.ma_short}" if self.params else "ma20"
        ma_long_col = f"ma{self.params.ma_long}" if self.params else "ma60"

        if ma_short_col in df.columns and ma_long_col in df.columns:
            ma_short_series = df[ma_short_col]
            ma_long_series = df[ma_long_col]

            if death_cross(ma_short_series, ma_long_series):
                reason = "MA死叉"
                signals.append(("ma_cross", 0.6, reason))

        # 4. RSI 超买出场
        if "rsi14" in df.columns:
            rsi14 = df["rsi14"].iloc[-1]
            if rsi14 > self.rsi_overbought:
                confidence = (rsi14 - self.rsi_overbought) / (100 - self.rsi_overbought) * 0.5  # 最高50%置信度
                signals.append(("rsi_overbought", confidence, f"RSI超买({rsi14:.1f})"))

        # 5. 布林带上轨止盈
        if "bb_upper" in df.columns and entry_price is not None:
            bb_upper = df["bb_upper"].iloc[-1]
            price = df["close"].iloc[-1]
            profit_pct = (price - entry_price) / entry_price * 100

            if price >= bb_upper and profit_pct > 5:  # 盈利超过5%且触及上轨
                confidence = min(0.7, profit_pct / 20)
                signals.append(("take_profit_1", confidence, f"触及布林上轨({bb_upper:.2f})"))

        # 返回最高置信度的信号
        if signals:
            signals.sort(key=lambda x: x[1], reverse=True)
            return (signals[0][0], signals[0][1], signals[0][2])

        return ("none", 0.0, "")

    def analyze(
        self,
        df: pd.DataFrame,
        entry_price: Optional[float] = None,
        highest_price: Optional[float] = None
    ) -> SignalResult:
        """
        综合分析，生成完整信号报告

        Args:
            df: 包含 OHLCV 数据的 DataFrame
            entry_price: 入场价格（可选）
            highest_price: 持仓期间最高价（可选）

        Returns:
            SignalResult: 综合信号结果
        """
        # 计算所有指标
        df = self.calculate_all(df)

        # 检测趋势
        trend, trend_conf = self.detect_trend(df)

        # 检测入场信号
        entry_signal, entry_conf, entry_reason = self.detect_entry(df)

        # 检测出场信号
        exit_signal, exit_conf, exit_reason = self.detect_exit(df, entry_price, highest_price)

        # 检测 RSI 背离
        rsi_div, rsi_div_type, rsi_div_strength = detect_rsi_divergence(df)

        # 计算止损止盈价格
        stop_loss = None
        take_profit_1 = None
        take_profit_2 = None
        trailing_stop = None

        # 获取止损/止盈倍数
        stop_mult = self.params.atr_stop_multiplier if self.params else 2.0
        trail_mult = self.params.atr_trailing_multiplier if self.params else 3.0
        profit_mult = self.params.profit_target_multiplier if self.params else 3.0

        if entry_price is not None and "atr" in df.columns:
            atr = df["atr"].iloc[-1]
            stop_loss = atr_stop_loss(entry_price, atr, multiplier=stop_mult)
            take_profit_1 = entry_price + (atr * profit_mult)
            take_profit_2 = entry_price + (atr * profit_mult * 2)

        if highest_price is not None and "atr" in df.columns:
            atr = df["atr"].iloc[-1]
            trailing_stop = atr_trailing_stop(highest_price, atr, multiplier=trail_mult)

        # 获取关键指标值
        ma_short_col = f"ma{self.params.ma_short}" if self.params else "ma20"
        ma_long_col = f"ma{self.params.ma_long}" if self.params else "ma60"
        ma20 = df[ma_short_col].iloc[-1] if ma_short_col in df.columns else None
        ma60 = df[ma_long_col].iloc[-1] if ma_long_col in df.columns else None
        rsi14 = df["rsi14"].iloc[-1] if "rsi14" in df.columns else None
        atr = df["atr"].iloc[-1] if "atr" in df.columns else None

        return SignalResult(
            date=df["date"].iloc[-1],
            trend=trend,
            trend_confidence=trend_conf,
            entry_signal=entry_signal,
            entry_confidence=entry_conf,
            entry_reason=entry_reason,
            exit_signal=exit_signal,
            exit_confidence=exit_conf,
            exit_reason=exit_reason,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            trailing_stop=trailing_stop,
            ma20=ma20,
            ma60=ma60,
            rsi14=rsi14,
            atr=atr,
            rsi_divergence=rsi_div,
            rsi_divergence_type=rsi_div_type,
            rsi_divergence_strength=rsi_div_strength
        )
