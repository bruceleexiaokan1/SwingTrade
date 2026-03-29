"""板块信号检测器

基于三屏系统理念，为板块数据提供信号检测：
1. 方向（趋势）：MA20/MA60
2. 动量（RSI）：RSI 健康区间
3. 强度（波动率）：ATR

与 SwingSignals 类似，但针对板块数据进行了简化。
"""

from dataclasses import dataclass
from typing import Tuple, Optional
from typing import TYPE_CHECKING

import pandas as pd

from .ma import calculate_ma
from .rsi import calculate_rsi
from .atr import calculate_atr

if TYPE_CHECKING:
    from ...backtest.strategy_params import StrategyParams


@dataclass
class SectorSignalResult:
    """板块信号结果"""
    date: str
    sector_name: str

    # 趋势指标
    trend: str                    # uptrend / downtrend / sideways
    trend_confidence: float       # 0.0 ~ 1.0

    # 动量指标
    rsi: float                    # RSI14
    rsi_healthy: bool            # RSI < 65 and RSI > 30

    # 波动指标
    atr: float
    momentum_20d: float          # 20日涨幅 %

    # 复合信号
    signal: str                  # strong / neutral / weak / none
    signal_confidence: float     # 0.0 ~ 1.0
    signal_reason: str           # 信号原因


class SectorSignals:
    """
    板块信号检测器

    职责：
    1. 计算板块技术指标
    2. 检测板块趋势
    3. 生成板块信号

    板块信号用于共振检测的第一层过滤。
    """

    # RSI 健康区间
    RSI_LOW = 30
    RSI_HIGH = 65

    # 动量阈值
    MOMENTUM_THRESHOLD = 0  # 20日涨幅 > 0%

    def __init__(self, params: Optional["StrategyParams"] = None):
        """
        初始化板块信号检测器

        Args:
            params: 策略参数（用于确定 MA 周期）
        """
        if params is not None:
            self.ma_short = params.ma_short
            self.ma_long = params.ma_long
            self.rsi_period = params.rsi_period
            self.atr_period = params.atr_period
        else:
            # 默认值
            self.ma_short = 20
            self.ma_long = 60
            self.rsi_period = 14
            self.atr_period = 14

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有板块指标

        Args:
            df: 包含 OHLCV 数据的 DataFrame

        Returns:
            添加了所有指标列的 DataFrame
        """
        df = df.copy()

        # MA
        df = calculate_ma(df, [self.ma_short, self.ma_long, 5, 10])

        # RSI
        df = calculate_rsi(df, [self.rsi_period])

        # ATR
        df = calculate_atr(df, self.atr_period)

        # 20日动量
        df['momentum_20d'] = (df['close'] / df['close'].shift(20) - 1) * 100

        return df

    def detect_trend(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        检测板块趋势

        Args:
            df: 包含指标数据的 DataFrame

        Returns:
            (趋势方向, 置信度)
            - ("uptrend", 0.0~1.0): MA多头排列
            - ("downtrend", 0.0~1.0): MA空头排列
            - ("sideways", 0.0~1.0): 盘整
        """
        if len(df) < self.ma_long:
            return ("sideways", 0.0)

        ma_short_col = f"ma{self.ma_short}"
        ma_long_col = f"ma{self.ma_long}"

        if ma_short_col not in df.columns or ma_long_col not in df.columns:
            return ("sideways", 0.0)

        ma_short = df[ma_short_col].iloc[-1]
        ma_long = df[ma_long_col].iloc[-1]

        # 上涨趋势：短期均线 > 长期均线
        if ma_short > ma_long:
            ma_diff_pct = (ma_short - ma_long) / ma_long * 100
            confidence = min(1.0, ma_diff_pct / 2)  # 2% 差距 = 100% 置信度
            return ("uptrend", confidence)

        # 下跌趋势：短期均线 < 长期均线
        if ma_short < ma_long:
            ma_diff_pct = (ma_long - ma_short) / ma_long * 100
            confidence = min(1.0, ma_diff_pct / 2)
            return ("downtrend", confidence)

        return ("sideways", 0.5)

    def detect_momentum(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        检测板块动量是否健康

        Args:
            df: 包含指标数据的 DataFrame

        Returns:
            (是否健康, 置信度)
            健康条件：
            1. RSI 在 30-65 之间
            2. 20日涨幅 > 0
        """
        if len(df) < 20:
            return (False, 0.0)

        rsi = df['rsi14'].iloc[-1] if 'rsi14' in df.columns else 50
        momentum = df['momentum_20d'].iloc[-1] if 'momentum_20d' in df.columns else 0

        # RSI 健康
        rsi_healthy = self.RSI_LOW < rsi < self.RSI_HIGH

        # 动量为正
        momentum_positive = momentum > self.MOMENTUM_THRESHOLD

        # 综合判断
        is_healthy = rsi_healthy and momentum_positive

        # 置信度
        if is_healthy:
            # RSI 越接近 50 越健康（既不强也不弱）
            rsi_distance = abs(rsi - 50) / 25  # 50 为中心，25 为边界
            rsi_conf = 1 - min(1.0, rsi_distance)

            momentum_conf = min(1.0, abs(momentum) / 5)  # 5% 涨幅 = 100%

            confidence = (rsi_conf * 0.6 + momentum_conf * 0.4)
        else:
            confidence = 0.0

        return (is_healthy, confidence)

    def detect_signal(self, df: pd.DataFrame, sector_name: str = "") -> SectorSignalResult:
        """
        综合分析，生成板块信号

        Args:
            df: 包含 OHLCV 数据的 DataFrame
            sector_name: 板块名称

        Returns:
            SectorSignalResult
        """
        if len(df) < self.ma_long:
            return SectorSignalResult(
                date=df['date'].iloc[-1] if len(df) > 0 else "",
                sector_name=sector_name,
                trend="sideways",
                trend_confidence=0.0,
                rsi=50.0,
                rsi_healthy=False,
                atr=0.0,
                momentum_20d=0.0,
                signal="none",
                signal_confidence=0.0,
                signal_reason="数据不足"
            )

        # 计算所有指标
        df = self.calculate_all(df)

        # 检测趋势
        trend, trend_conf = self.detect_trend(df)

        # 检测动量
        momentum_healthy, momentum_conf = self.detect_momentum(df)

        # 获取指标值
        rsi = df['rsi14'].iloc[-1]
        atr = df['atr'].iloc[-1]
        momentum = df['momentum_20d'].iloc[-1]

        # RSI 是否健康
        rsi_healthy = self.RSI_LOW < rsi < self.RSI_HIGH

        # 生成信号
        if trend == "uptrend" and rsi_healthy and momentum > 0:
            signal = "strong"
            signal_conf = min(1.0, (trend_conf + momentum_conf) / 2)
            signal_reason = f"趋势向上({trend_conf:.0%}) + RSI健康({rsi:.0f}) + 动量正({momentum:.1f}%)"
        elif trend == "uptrend" and (rsi_healthy or momentum > 0):
            signal = "neutral"
            signal_conf = min(0.6, trend_conf * 0.5)
            signal_reason = f"趋势向上，但动量{'不足' if not momentum_positive else 'RSI偏高'}"
        elif trend == "sideways":
            signal = "weak"
            signal_conf = 0.3
            signal_reason = "趋势不明"
        else:
            signal = "none"
            signal_conf = 0.0
            signal_reason = "趋势向下或RSI不健康"

        return SectorSignalResult(
            date=df['date'].iloc[-1],
            sector_name=sector_name,
            trend=trend,
            trend_confidence=trend_conf,
            rsi=rsi,
            rsi_healthy=rsi_healthy,
            atr=atr,
            momentum_20d=momentum,
            signal=signal,
            signal_confidence=signal_conf,
            signal_reason=signal_reason
        )
