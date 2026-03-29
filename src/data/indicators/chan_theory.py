"""缠论指标

Chan Theory (缠论) 实现 - 用于识别笔、线段、中枢及三类买卖点

主要概念：
- 包含关系：两根K线可以合并（取最高high和最低low）
- 笔：连续5根以上无包含关系的K线
- 线段：连续3笔以上同方向
- 中枢：三段以上连续重叠区域
- 三类买卖点：类一买、类二买、类三买
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Direction(Enum):
    """方向枚举"""
    UP = "up"
    DOWN = "down"
    NONE = "none"


@dataclass
class KLineWithDirection:
    """带方向的K线"""
    index: int          # 在原DataFrame中的索引
    date: str           # 日期
    high: float         # 最高价
    low: float          # 最低价
    direction: Direction  # 方向


@dataclass
class Pen:
    """笔 - 连续5根以上无包含关系的K线"""
    start_idx: int      # 起始K线索引
    end_idx: int        # 结束K线索引
    direction: Direction  # 方向（UP/DOWN）
    high: float         # 笔的最高价
    low: float          # 笔的最低价
    kline_count: int    # 包含的K线数


@dataclass
class Segment:
    """线段 - 连续3笔以上同方向"""
    start_idx: int      # 起始笔索引
    end_idx: int        # 结束笔索引
    direction: Direction  # 方向
    pens: List[Pen]     # 包含的笔列表
    high: float         # 线段的最高价
    low: float          # 线段的最低价


@dataclass
class Center:
    """中枢 - 三段以上连续重叠区域"""
    start_idx: int      # 起始段索引
    end_idx: int        # 结束段索引
    segments: List[Segment]  # 包含的段列表
    high: float         # 中枢上沿
    low: float          # 中枢下沿
    zg: float           # 中枢最高点 (ZG)
    zd: float           # 中枢最低点 (ZD)
    gg: float           # 颈线最高点 (GG)
    dd: float           # 颈线最低点 (DD)


@dataclass
class ChanBuySignal:
    """缠论买入信号"""
    signal_type: str          # "类一买", "类二买", "类三买"
    date: str                 # 信号日期
    price: float              # 信号价格
    confidence: float         # 置信度 0.0 ~ 1.0
    reason: str               # 信号原因
    center: Optional[Center] = None  # 相关中枢
    stop_loss: Optional[float] = None  # 止损价


class ChanTheory:
    """
    缠论实战量化

    实现简化的缠论分析：
    1. 处理包含关系
    2. 识别笔（5根以上无包含关系的连续K线）
    3. 识别线段（3笔以上同方向）
    4. 识别中枢（3段以上连续重叠）
    5. 识别三类买卖点
    """

    # 笔的最小K线数
    MIN_PEN_K = 5

    # 线段的最小笔数
    MIN_SEGMENT_PENS = 3

    # 中枢的最少段数
    MIN_CENTER_SEGMENTS = 3

    def __init__(self, min_pen_k: int = 5, min_segment_pens: int = 3, min_center_segments: int = 3):
        """
        初始化缠论指标

        Args:
            min_pen_k: 笔的最小K线数，默认5
            min_segment_pens: 线段的最小笔数，默认3
            min_center_segments: 中枢的最少段数，默认3
        """
        self.MIN_PEN_K = min_pen_k
        self.MIN_SEGMENT_PENS = min_segment_pens
        self.MIN_CENTER_SEGMENTS = min_center_segments

    def _process_containment(self, df: pd.DataFrame) -> List[KLineWithDirection]:
        """
        处理包含关系

        包含关系：两根相邻K线，一根完全在另一根范围内
        处理规则：
        - 向上：取两根K线的高点最高和低点最高
        - 向下：取两根K线的高点最低和低点最低

        Args:
            df: 包含OHLC数据的DataFrame

        Returns:
            处理后的K线列表（无包含关系）
        """
        if len(df) < 3:
            return []

        # 初始化：复制数据并添加方向
        klines = []
        for i in range(len(df)):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]

            # 判断方向：第一根K线无方向
            if i == 0:
                direction = Direction.NONE
            else:
                prev_high = df['high'].iloc[i - 1]
                prev_low = df['low'].iloc[i - 1]

                # 判断是否包含：当前K线完全在前一根K线范围内
                if high <= prev_high and low >= prev_low:
                    # 包含关系 - 延续前一根的方向
                    direction = klines[-1].direction if klines else Direction.NONE
                elif high >= prev_high and low <= prev_low:
                    # 相反包含关系
                    direction = Direction.NONE
                else:
                    # 无包含关系，判断方向
                    if high > prev_high and low > prev_low:
                        direction = Direction.UP
                    elif high < prev_high and low < prev_low:
                        direction = Direction.DOWN
                    else:
                        # 混合关系，参考前一根
                        direction = klines[-1].direction if klines else Direction.NONE

            klines.append(KLineWithDirection(
                index=i,
                date=str(df['date'].iloc[i]) if 'date' in df.columns else "",
                high=high,
                low=low,
                direction=direction
            ))

        # 处理包含关系（合并包含的K线）
        processed = []
        i = 0
        while i < len(klines):
            if len(processed) == 0:
                processed.append(klines[i])
                i += 1
                continue

            prev = processed[-1]
            curr = klines[i]

            # 判断是否包含
            if curr.high <= prev.high and curr.low >= prev.low:
                # 当前K线被前一根包含 - 合并
                # 向上处理：取高点最高、低点最高
                # 向下处理：取高点最低、低点最低
                if prev.direction == Direction.UP:
                    new_high = max(prev.high, curr.high)
                    new_low = max(prev.low, curr.low)
                elif prev.direction == Direction.DOWN:
                    new_high = min(prev.high, curr.high)
                    new_low = min(prev.low, curr.low)
                else:
                    new_high = prev.high
                    new_low = prev.low

                processed[-1] = KLineWithDirection(
                    index=prev.index,
                    date=prev.date,
                    high=new_high,
                    low=new_low,
                    direction=prev.direction
                )
                i += 1
            elif curr.high >= prev.high and curr.low <= prev.low:
                # 前一根K线被当前K线包含
                # 需要判断方向
                direction = Direction.NONE
                if curr.high > prev.high and curr.low > prev.low:
                    direction = Direction.UP
                elif curr.high < prev.high and curr.low < prev.low:
                    direction = Direction.DOWN
                else:
                    direction = prev.direction

                # 合并
                if direction == Direction.UP:
                    new_high = max(prev.high, curr.high)
                    new_low = max(prev.low, curr.low)
                elif direction == Direction.DOWN:
                    new_high = min(prev.high, curr.high)
                    new_low = min(prev.low, curr.low)
                else:
                    new_high = prev.high
                    new_low = prev.low

                processed[-1] = KLineWithDirection(
                    index=prev.index,
                    date=prev.date,
                    high=new_high,
                    low=new_low,
                    direction=direction
                )
                i += 1
            else:
                # 无包含关系
                processed.append(curr)
                i += 1

        return processed

    def _find_fen_xing(self, klines: List[KLineWithDirection]) -> List[Tuple[int, str]]:
        """
        识别分型

        顶分型：连续三根K线，中间最高
        底分型：连续三根K线，中间最低

        Args:
            klines: 处理后的K线列表

        Returns:
            分型列表 [(索引, "顶"/"底")]
        """
        if len(klines) < 3:
            return []

        fen_xings = []

        for i in range(1, len(klines) - 1):
            prev = klines[i - 1]
            curr = klines[i]
            next_k = klines[i + 1]

            # 顶分型：中间最高
            if curr.high > prev.high and curr.high > next_k.high:
                fen_xings.append((i, "顶"))
            # 底分型：中间最低
            elif curr.low < prev.low and curr.low < next_k.low:
                fen_xings.append((i, "底"))

        return fen_xings

    def find_pens(self, df: pd.DataFrame) -> List[Pen]:
        """
        识别笔

        笔 = 连续向上/向下且无包含关系的5+K线

        Args:
            df: 包含OHLC数据的DataFrame

        Returns:
            笔列表
        """
        if len(df) < self.MIN_PEN_K:
            return []

        # 处理包含关系
        klines = self._process_containment(df)

        if len(klines) < self.MIN_PEN_K:
            return []

        # 识别分型
        fen_xings = self._find_fen_xing(klines)

        if len(fen_xings) < 2:
            return []

        # 笔的定义：两个相邻同类型分型之间（顶-底或底-顶）的K线
        pens = []
        i = 0

        while i < len(fen_xings) - 1:
            fx1_idx, fx1_type = fen_xings[i]
            fx2_idx, fx2_type = fen_xings[i + 1]

            # 必须是相邻且类型相反的分型
            if fx2_idx - fx1_idx < self.MIN_PEN_K:
                i += 1
                continue

            # 获取K线
            start_kline = klines[fx1_idx]
            end_kline = klines[fx2_idx]

            # 判断方向
            if fx1_type == "顶":
                direction = Direction.DOWN
            else:
                direction = Direction.UP

            # 获取这段K线的high和low
            segment_klines = klines[fx1_idx:fx2_idx + 1]
            high = max(k.high for k in segment_klines)
            low = min(k.low for k in segment_klines)

            pens.append(Pen(
                start_idx=start_kline.index,
                end_idx=end_kline.index,
                direction=direction,
                high=high,
                low=low,
                kline_count=fx2_idx - fx1_idx + 1
            ))

            i += 1

        return pens

    def find_segments(self, pens: List[Pen]) -> List[Segment]:
        """
        识别线段

        线段 = 连续3+笔以上同方向

        Args:
            pens: 笔列表

        Returns:
            线段列表
        """
        if len(pens) < self.MIN_SEGMENT_PENS:
            return []

        segments = []
        i = 0

        while i <= len(pens) - self.MIN_SEGMENT_PENS:
            # 检查连续同方向的笔
            direction = pens[i].direction
            segment_pens = [pens[i]]

            j = i + 1
            while j < len(pens) and pens[j].direction == direction:
                segment_pens.append(pens[j])
                j += 1

            # 需要至少3笔才能形成线段
            if len(segment_pens) >= self.MIN_SEGMENT_PENS:
                high = max(p.high for p in segment_pens)
                low = min(p.low for p in segment_pens)

                segments.append(Segment(
                    start_idx=segment_pens[0].start_idx,
                    end_idx=segment_pens[-1].end_idx,
                    direction=direction,
                    pens=segment_pens,
                    high=high,
                    low=low
                ))

            i = j if j > i + 1 else i + 1

        return segments

    def find_centers(self, segments: List[Segment]) -> List[Center]:
        """
        识别中枢

        中枢 = 三段以上连续重叠区域

        Args:
            segments: 线段列表

        Returns:
            中枢列表
        """
        if len(segments) < self.MIN_CENTER_SEGMENTS:
            return []

        centers = []
        i = 0

        while i <= len(segments) - self.MIN_CENTER_SEGMENTS:
            # 检查连续重叠的段
            center_segments = [segments[i]]

            j = i + 1
            while j < len(segments):
                prev_seg = center_segments[-1]
                curr_seg = segments[j]

                # 检查是否有重叠（方向相同且价格重叠）
                if curr_seg.direction == prev_seg.direction:
                    # 计算重叠区域
                    overlap_high = min(prev_seg.high, curr_seg.high)
                    overlap_low = max(prev_seg.low, curr_seg.low)

                    # 如果有重叠，加入中枢
                    if overlap_high > overlap_low:
                        center_segments.append(curr_seg)
                        j += 1
                    else:
                        break
                else:
                    # 方向改变，可能形成新中枢
                    break

            # 需要至少3段才能形成中枢
            if len(center_segments) >= self.MIN_CENTER_SEGMENTS:
                # 计算中枢区间（所有段的重叠区域）
                center_high = max(s.high for s in center_segments)
                center_low = min(s.low for s in center_segments)

                # 中枢的ZG是向上的笔的高点，ZD是向下的笔的低点
                zg = 0
                zd = float('inf')
                gg = 0
                dd = float('inf')

                for seg in center_segments:
                    if seg.direction == Direction.UP:
                        zg = max(zg, seg.high)
                        gg = max(gg, seg.high)
                    else:
                        zd = min(zd, seg.low)
                        dd = min(dd, seg.low)

                if zd == float('inf'):
                    zd = center_low
                if gg == 0:
                    gg = center_high
                if dd == float('inf'):
                    dd = center_low

                centers.append(Center(
                    start_idx=center_segments[0].start_idx,
                    end_idx=center_segments[-1].end_idx,
                    segments=center_segments,
                    high=center_high,
                    low=center_low,
                    zg=zg if zg > 0 else center_high,
                    zd=zd if zd < float('inf') else center_low,
                    gg=gg if gg > 0 else center_high,
                    dd=dd if dd < float('inf') else center_low
                ))

            i = j if j > i + 1 else i + 1

        return centers

    def find_buy_signals(
        self,
        df: pd.DataFrame,
        centers: List[Center] = None
    ) -> List[ChanBuySignal]:
        """
        找买卖点

        - 类一买：创新低后的反弹（新低后出现底分型）
        - 类二买：回调不创新低（回调到中枢区间不创新低）
        - 类三买：突破中枢后回踩（突破中枢上沿后回踩不破）

        Args:
            df: 价格数据
            centers: 中枢列表（可选）

        Returns:
            买入信号列表
        """
        if centers is None:
            centers = self.find_centers(self.find_segments(self.find_pens(df)))

        pens = self.find_pens(df)
        if len(pens) < 5:
            return []

        signals = []

        # 获取处理后的K线用于分型判断
        klines = self._process_containment(df)
        fen_xings = self._find_fen_xing(klines)

        if len(fen_xings) < 3:
            return []

        # 类一买：创新低后出现底分型
        # 找到创新低的点，然后检查是否随后出现底分型
        prices = df['close'].values
        recent_low_idx = 0
        recent_low_price = float('inf')

        for i in range(1, len(pens)):
            if pens[i].direction == Direction.DOWN and pens[i].low < recent_low_price:
                recent_low_price = pens[i].low
                recent_low_idx = i

        # 检查是否有反弹后的底分型
        for i in range(recent_low_idx + 1, len(pens)):
            if pens[i].direction == Direction.UP:
                # 检查是否在近期新低后反弹
                if recent_low_idx > 0:
                    signals.append(ChanBuySignal(
                        signal_type="类一买",
                        date=pens[i].start_idx if pens[i].start_idx < len(df) else str(df['date'].iloc[-1]),
                        price=pens[i].low,
                        confidence=0.6,
                        reason=f"创新低后反弹，新低={recent_low_price:.2f}",
                        center=None,
                        stop_loss=recent_low_price * 0.95
                    ))
                    break

        # 类二买：回调到中枢区间不创新低
        for center in centers:
            if len(signals) == 0:
                continue

            # 找最近的向下笔回调
            for i in range(1, len(pens)):
                if pens[i].direction == Direction.DOWN:
                    # 检查是否回调到中枢区间
                    if center.zd <= pens[i].high <= center.zg:
                        # 回调不破前期低点（类二买）
                        prev_low = min(p.low for p in pens[:i] if p.direction == Direction.DOWN)
                        if pens[i].low > prev_low * 0.98:  # 不创新低（允许2%误差）
                            signals.append(ChanBuySignal(
                                signal_type="类二买",
                                date=str(df['date'].iloc[pens[i].end_idx]) if pens[i].end_idx < len(df) else str(df['date'].iloc[-1]),
                                price=pens[i].low,
                                confidence=0.75,
                                reason=f"回调到中枢区间不创新低，中枢zd={center.zd:.2f}",
                                center=center,
                                stop_loss=pens[i].low * 0.97
                            ))
                            break

        # 类三买：突破中枢后回踩不破
        for center in centers:
            # 向上突破中枢
            for i in range(len(pens)):
                if pens[i].direction == Direction.UP and pens[i].high > center.zg:
                    # 突破后的向下回踩
                    for j in range(i + 1, len(pens)):
                        if pens[j].direction == Direction.DOWN:
                            # 回踩不破中枢上沿
                            if pens[j].low > center.zg:
                                signals.append(ChanBuySignal(
                                    signal_type="类三买",
                                    date=str(df['date'].iloc[pens[j].end_idx]) if pens[j].end_idx < len(df) else str(df['date'].iloc[-1]),
                                    price=pens[j].low,
                                    confidence=0.7,
                                    reason=f"突破中枢后回踩不破，突破点={pens[i].high:.2f}",
                                    center=center,
                                    stop_loss=center.zg * 0.97
                                ))
                                break
                    break

        # 按优先级排序：类二买 > 类三买 > 类一买
        # 返回置信度最高的
        return sorted(signals, key=lambda x: (
            {"类二买": 0, "类三买": 1, "类一买": 2}[x.signal_type],
            -x.confidence
        ))

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        综合分析

        Args:
            df: 包含OHLC数据的DataFrame

        Returns:
            分析结果字典
        """
        if len(df) < 20:
            return {
                "pens": [],
                "segments": [],
                "centers": [],
                "buy_signals": [],
                "has_buy_signal": False,
                "primary_buy_signal": None
            }

        pens = self.find_pens(df)
        segments = self.find_segments(pens)
        centers = self.find_centers(segments)
        buy_signals = self.find_buy_signals(df, centers)

        return {
            "pens": pens,
            "segments": segments,
            "centers": centers,
            "buy_signals": buy_signals,
            "has_buy_signal": len(buy_signals) > 0,
            "primary_buy_signal": buy_signals[0] if buy_signals else None
        }


def calculate_chan(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算缠论所有指标（添加到DataFrame）

    Args:
        df: 包含OHLC数据的DataFrame

    Returns:
        添加了缠论相关列的DataFrame
    """
    df = df.copy()

    chan = ChanTheory()
    result = chan.analyze(df)

    # 添加基础列
    df['chan_has_pen'] = len(result['pens']) > 0
    df['chan_has_center'] = len(result['centers']) > 0
    df['chan_has_buy_signal'] = result['has_buy_signal']

    # 添加主要信号信息
    if result['primary_buy_signal']:
        signal = result['primary_buy_signal']
        df['chan_signal_type'] = signal.signal_type
        df['chan_signal_price'] = signal.price
        df['chan_signal_confidence'] = signal.confidence
    else:
        df['chan_signal_type'] = None
        df['chan_signal_price'] = None
        df['chan_signal_confidence'] = 0.0

    return df


def detect_chan_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """
    检测缠论买卖点

    Args:
        df: 包含OHLC数据的DataFrame

    Returns:
        缠论分析结果字典
    """
    if len(df) < 20:
        return {
            "has_buy_signal": False,
            "primary_signal": None,
            "all_signals": [],
            "centers": [],
            "pens": [],
            "segments": []
        }

    chan = ChanTheory()
    result = chan.analyze(df)

    return {
        "has_buy_signal": result['has_buy_signal'],
        "primary_signal": result['primary_buy_signal'],
        "all_signals": result['buy_signals'],
        "centers": result['centers'],
        "pens": result['pens'],
        "segments": result['segments']
    }
