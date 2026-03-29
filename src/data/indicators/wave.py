"""波浪理论指标

艾略特波浪理论实现 - 用于定位和辅助共振判断

主要功能：
1. 波浪计数 - 识别推动浪和调整浪
2. 回调比例计算 - 基于斐波那契比例
3. 波浪位置判断 - 当前处于第几浪
4. 波浪强度评估 - 用于共振辅助指标
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class WaveType(Enum):
    """波浪类型"""
    IMPULSE_1 = "impulse_1"      # 推动浪1
    IMPULSE_2 = "impulse_2"      # 推动浪2
    IMPULSE_3 = "impulse_3"      # 推动浪3
    IMPULSE_4 = "impulse_4"      # 推动浪4
    IMPULSE_5 = "impulse_5"      # 推动浪5
    CORRECTIVE_A = "corrective_a"  # 调整浪A
    CORRECTIVE_B = "corrective_b"  # 调整浪B
    CORRECTIVE_C = "corrective_c"  # 调整浪C
    UNKNOWN = "unknown"


class WaveDirection(Enum):
    """波浪方向"""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


@dataclass
class WavePoint:
    """波浪转折点"""
    index: int           # 在DataFrame中的索引位置
    date: str             # 日期
    price: float          # 价格
    wave_type: WaveType   # 波浪类型
    strength: float       # 强度 (0.0 ~ 1.0)


@dataclass
class WaveResult:
    """波浪分析结果"""
    date: str
    direction: WaveDirection
    current_wave: WaveType
    wave_count: int           # 当前波浪序号 (1-5 或 A-C)
    wave_count_confidence: float  # 波浪计数置信度
    fib_retracement: float    # 当前回撤比例
    momentum: float           # 波浪动能 (0.0 ~ 1.0)
    is_correction: bool       # 是否处于调整浪
    wave_reasons: List[str]   # 分析理由
    next_support: Optional[float]  # 下一支撑位
    next_resistance: Optional[float]  # 下一阻力位


class WaveIndicators:
    """
    波浪理论指标计算器

    使用斐波那契回撤和价格形态识别波浪
    """

    # 斐波那契回撤比例
    FIB_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.618]

    # 波浪识别参数
    MIN_WAVE_RATIO = 0.05      # 最小波浪幅度比例（相对于前期波动）
    FIB_TOLERANCE = 0.03       # 斐波那契容差

    def __init__(
        self,
        min_periods: int = 20,
        fib_tolerance: float = 0.03
    ):
        """
        初始化波浪指标

        Args:
            min_periods: 最小计算周期
            fib_tolerance: 斐波那契容差
        """
        self.min_periods = min_periods
        self.fib_tolerance = fib_tolerance

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有波浪指标

        Args:
            df: 包含 OHLCV 数据的 DataFrame

        Returns:
            添加了波浪相关列的 DataFrame
        """
        df = df.copy()

        # 计算价格变化
        df['price_change'] = df['close'].diff()

        # 计算波动幅度（用于波浪强度判断）
        df['wave_amplitude'] = df['high'] - df['low']

        # 计算对数价格（用于百分比计算）
        df['log_close'] = np.log(df['close'])

        # 计算回撤/反弹
        df['retracement'] = self._calculate_retracement(df)

        # 计算波浪动量
        df['wave_momentum'] = self._calculate_wave_momentum(df)

        # 识别局部极值（用于波浪转折点）
        df['is_local_max'] = self._is_local_maximum(df)
        df['is_local_min'] = self._is_local_minimum(df)

        return df

    def _calculate_retracement(self, df: pd.DataFrame) -> pd.Series:
        """
        计算回撤比例

        回撤 = (当前价 - 最低价) / (最高价 - 最低价)
        """
        rolling_max = df['close'].rolling(window=20, min_periods=1).max()
        rolling_min = df['close'].rolling(window=20, min_periods=1).min()
        range_ = rolling_max - rolling_min

        retracement = (df['close'] - rolling_min) / range_.replace(0, np.nan)

        return retracement

    def _calculate_wave_momentum(self, df: pd.DataFrame) -> pd.Series:
        """
        计算波浪动量

        基于价格变化率和成交量
        """
        price_change = df['close'].pct_change()
        volume_ma = df['volume'].rolling(window=5, min_periods=1).mean()
        volume_ratio = df['volume'] / volume_ma.replace(0, 1)

        # 动量 = 价格变化 * 成交量确认
        momentum = price_change * volume_ratio.clip(0.5, 2.0)

        # 归一化到 0 ~ 1
        momentum_norm = (momentum - momentum.min()) / (momentum.max() - momentum.min() + 1e-10)

        return momentum_norm

    def _is_local_maximum(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        """判断是否为局部最高点（使用滚动窗口向量化）"""
        rolling_max = df['high'].rolling(window=window * 2 + 1, center=True, min_periods=window + 1).max()
        is_max = df['high'] == rolling_max
        # 边界处理：只保留 window 到 len-window 的有效区间
        is_max[:window] = False
        is_max[-(window):] = False if len(is_max) > window else False
        return is_max

    def _is_local_minimum(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        """判断是否为局部最低点（使用滚动窗口向量化）"""
        rolling_min = df['low'].rolling(window=window * 2 + 1, center=True, min_periods=window + 1).min()
        is_min = df['low'] == rolling_min
        # 边界处理：只保留 window 到 len-window 的有效区间
        is_min[:window] = False
        is_min[-(window):] = False if len(is_min) > window else False
        return is_min

    def find_wave_points(self, df: pd.DataFrame, lookback: int = 100) -> List[WavePoint]:
        """
        查找波浪转折点

        Args:
            df: 价格数据
            lookback: 回溯周期

        Returns:
            波浪转折点列表
        """
        if len(df) < 20:
            return []

        df_recent = df.tail(lookback).copy()
        wave_points = []

        # 找到所有局部极值
        for i in range(5, len(df_recent) - 5):
            if df_recent['is_local_max'].iloc[i]:
                wave_points.append(WavePoint(
                    index=df_recent.index[i],
                    date=df_recent['date'].iloc[i],
                    price=df_recent['high'].iloc[i],
                    wave_type=WaveType.UNKNOWN,
                    strength=0.5
                ))
            elif df_recent['is_local_min'].iloc[i]:
                wave_points.append(WavePoint(
                    index=df_recent.index[i],
                    date=df_recent['date'].iloc[i],
                    price=df_recent['low'].iloc[i],
                    wave_type=WaveType.UNKNOWN,
                    strength=0.5
                ))

        # 过滤太近的极值点（间隔小于5天）
        if len(wave_points) > 1:
            filtered = [wave_points[0]]
            for wp in wave_points[1:]:
                if wp.index - filtered[-1].index >= 5:
                    filtered.append(wp)
            wave_points = filtered

        # 标记波浪类型
        wave_points = self._label_wave_types(wave_points)

        return wave_points

    def _label_wave_types(self, wave_points: List[WavePoint]) -> List[WavePoint]:
        """
        标记波浪类型

        基于波浪理论和斐波那契比例
        """
        if len(wave_points) < 5:
            return wave_points

        # 计算相邻极值的幅度
        for i in range(len(wave_points)):
            if i == 0:
                continue

            amplitude = abs(wave_points[i].price - wave_points[i-1].price)
            prev_amplitude = abs(wave_points[i-1].price - wave_points[i-2].price) if i >= 2 else amplitude

            # 波浪强度基于幅度比例
            strength = min(1.0, amplitude / (prev_amplitude + 1e-10))
            wave_points[i].strength = strength

            # 标记波浪类型（简化版：交替标记高低点）
            if i % 5 == 1:
                wave_points[i].wave_type = WaveType.IMPULSE_1
            elif i % 5 == 2:
                wave_points[i].wave_type = WaveType.IMPULSE_2
            elif i % 5 == 3:
                wave_points[i].wave_type = WaveType.IMPULSE_3
            elif i % 5 == 4:
                wave_points[i].wave_type = WaveType.IMPULSE_4
            else:
                wave_points[i].wave_type = WaveType.IMPULSE_5

        return wave_points

    def analyze(
        self,
        df: pd.DataFrame,
        date: str = None
    ) -> WaveResult:
        """
        综合波浪分析

        Args:
            df: 价格数据
            date: 分析日期（默认为最后一天）

        Returns:
            WaveResult
        """
        if len(df) < self.min_periods:
            return WaveResult(
                date=df['date'].iloc[-1] if len(df) > 0 else date or "",
                direction=WaveDirection.SIDEWAYS,
                current_wave=WaveType.UNKNOWN,
                wave_count=0,
                wave_count_confidence=0.0,
                fib_retracement=0.5,
                momentum=0.5,
                is_correction=False,
                wave_reasons=["数据不足"],
                next_support=None,
                next_resistance=None
            )

        # 计算所有波浪指标（确保 is_local_max/is_local_min 列存在）
        df = self.calculate_all(df)

        # 找到分析日期的索引
        if date is None:
            idx = len(df) - 1
        else:
            if date not in df['date'].values:
                idx = len(df) - 1
            else:
                idx = df[df['date'] == date].index[0]

        recent_df = df.iloc[:idx+1]

        # 找到波浪点
        wave_points = self.find_wave_points(recent_df)

        # 确定当前方向
        if len(wave_points) >= 2:
            if wave_points[-1].price > wave_points[-2].price:
                direction = WaveDirection.UP
            elif wave_points[-1].price < wave_points[-2].price:
                direction = WaveDirection.DOWN
            else:
                direction = WaveDirection.SIDEWAYS
        else:
            direction = WaveDirection.SIDEWAYS

        # 计算回撤比例
        retracement = recent_df['retracement'].iloc[-1]

        # 计算动量
        momentum = recent_df['wave_momentum'].iloc[-1]

        # 确定当前波浪
        current_wave = WaveType.UNKNOWN
        wave_count = 0
        confidence = 0.5
        reasons = []

        if len(wave_points) >= 3:
            last_points = wave_points[-3:]
            current_price = recent_df['close'].iloc[-1]

            # 检查是否处于调整
            if direction == WaveDirection.DOWN:
                current_wave = WaveType.CORRECTIVE_A
                wave_count = len(wave_points) % 5
                confidence = 0.6
                reasons.append("处于下降调整浪")
            else:
                wave_num = len(wave_points) % 5
                if wave_num == 0:
                    wave_num = 5

                wave_map = {
                    1: WaveType.IMPULSE_1,
                    2: WaveType.IMPULSE_2,
                    3: WaveType.IMPULSE_3,
                    4: WaveType.IMPULSE_4,
                    5: WaveType.IMPULSE_5
                }
                current_wave = wave_map.get(wave_num, WaveType.UNKNOWN)
                wave_count = wave_num
                confidence = 0.5 + wave_points[-1].strength * 0.3
                reasons.append(f"推断为第{wave_num}浪")

        # 计算支撑和阻力
        next_support = None
        next_resistance = None

        if len(wave_points) >= 2:
            last_price = wave_points[-1].price
            atr = recent_df['high'].iloc[-1] - recent_df['low'].iloc[-1]

            if direction == WaveDirection.UP:
                next_support = last_price * 0.95  # 5% 回撤支撑
                next_resistance = last_price * 1.05  # 5% 上涨空间
            else:
                next_support = last_price * 0.95
                next_resistance = last_price * 1.05

        return WaveResult(
            date=recent_df['date'].iloc[-1],
            direction=direction,
            current_wave=current_wave,
            wave_count=wave_count,
            wave_count_confidence=confidence,
            fib_retracement=retracement if not pd.isna(retracement) else 0.5,
            momentum=momentum if not pd.isna(momentum) else 0.5,
            is_correction=direction == WaveDirection.DOWN,
            wave_reasons=reasons if reasons else ["趋势不明确"],
            next_support=next_support,
            next_resistance=next_resistance
        )

    def get_wave_score(
        self,
        df: pd.DataFrame,
        wave_type: str = "uptrend"
    ) -> float:
        """
        获取波浪评分（用于共振辅助）

        Args:
            df: 价格数据
            wave_type: 波浪方向偏好 ("uptrend" 或 "downtrend")

        Returns:
            波浪评分 (0.0 ~ 1.0)
        """
        result = self.analyze(df)

        score = 0.5

        # 方向匹配加分
        if (wave_type == "uptrend" and result.direction == WaveDirection.UP) or \
           (wave_type == "downtrend" and result.direction == WaveDirection.DOWN):
            score += 0.2

        # 动能加分
        score += (result.momentum - 0.5) * 0.3

        # 调整浪减分
        if result.is_correction and wave_type == "uptrend":
            score -= 0.2

        # 回撤比例评估
        fib = result.fib_retracement
        if 0.3 <= fib <= 0.7:  # 健康回撤区间
            score += 0.1

        return max(0.0, min(1.0, score))


def calculate_wave_levels(prices: List[float]) -> Dict[str, float]:
    """
    计算波浪斐波那契水平

    Args:
        prices: 价格列表 [wave1_start, wave1_end, wave2_end, wave3_end, wave4_end, wave5_end]

    Returns:
        斐波那契水平字典
    """
    if len(prices) < 3:
        return {}

    wave1_start, wave1_end = prices[0], prices[1]
    wave1_range = wave1_end - wave1_start

    levels = {}

    # Wave 2 回撤
    for fib in [0.382, 0.500, 0.618]:
        levels[f'wave2_fib_{fib}'] = wave1_end - wave1_range * fib

    # Wave 3 扩展
    levels['wave3_1618'] = wave1_end + wave1_range * 1.618
    levels['wave3_1272'] = wave1_end + wave1_range * 1.272

    # Wave 4 回撤
    for fib in [0.382, 0.500]:
        if 'wave3_1618' in levels:
            wave3_range = levels['wave3_1618'] - wave1_end
            levels[f'wave4_fib_{fib}'] = levels['wave3_1618'] - wave3_range * fib

    # Wave 5 目标
    if 'wave4_fib_0.382' in levels:
        wave4_range = levels['wave3_1618'] - levels['wave4_fib_0.382']
        levels['wave5_100'] = levels['wave4_fib_0.382'] + wave4_range
        levels['wave5_1236'] = levels['wave4_fib_0.382'] + wave4_range * 1.236

    return levels
