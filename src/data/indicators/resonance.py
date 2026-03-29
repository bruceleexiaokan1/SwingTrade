"""共振检测结果数据类

定义共振检测的数据结构和评分算法。
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import IntEnum


class ResonanceDataError(Exception):
    """板块或个股数据获取失败"""

    def __init__(self, code: str, sector: str, reason: str):
        self.code = code
        self.sector = sector
        self.reason = reason
        super().__init__(f"数据获取失败 {code}/{sector}: {reason}")


class ResonanceLevel(IntEnum):
    """共振等级（按 ordinal 排序：INVALID=0 < C=1 < B=2 < A=3 < S=4）"""
    INVALID = 0  # 无效，<2/8 条件满足
    C = 1       # 观察，2-3/8 条件满足
    B = 2       # 弱共振，4-5/8 条件满足
    A = 3       # 强共振，6-7/8 条件满足
    S = 4       # 完美共振，8/8 条件满足

    @property
    def label(self) -> str:
        """显示用标签（全英文统一）"""
        labels = {
            0: "INVALID",
            1: "C",
            2: "B",
            3: "A",
            4: "S"
        }
        return labels.get(self.value, "INVALID")


@dataclass
class ResonanceCondition:
    """共振条件"""
    name: str           # 条件名称
    met: bool          # 是否满足
    weight: float      # 权重
    value: any         # 实际值
    threshold: str      # 阈值描述


@dataclass
class ResonanceResult:
    """共振检测结果"""

    # 基本信息
    date: str
    stock_code: str
    sector_name: str

    # ===== 板块指标 =====
    sector_trend: str = "sideways"           # uptrend / downtrend / sideways
    sector_trend_conf: float = 0.0
    sector_rsi: float = 50.0
    sector_momentum_20d: float = 0.0        # 20日涨幅 %
    sector_atr: float = 0.0

    # ===== 个股指标 =====
    stock_trend: str = "sideways"
    stock_trend_conf: float = 0.0
    stock_rsi: float = 50.0
    stock_golden_cross: bool = False        # MA5 上穿 MA20
    stock_golden_cross_approaching: bool = False  # MA5 接近 MA20（2%以内）
    stock_rs_rank: float = 0.0             # 板块内相对强度排名 0.0 ~ 1.0
    stock_rs_score: float = 0.0            # RS 原始值（个股20日涨幅 - 板块20日涨幅）
    stock_atr: float = 0.0

    # ===== 市场指标 =====
    market_trend: str = "sideways"         # 沪深300 趋势
    market_trend_conf: float = 0.0

    # ===== 共振评分 =====
    conditions: List[ResonanceCondition] = field(default_factory=list)
    conditions_met: int = 0                 # 满足的条件数
    total_conditions: int = 8
    resonance_level: ResonanceLevel = ResonanceLevel.INVALID
    resonance_confidence: float = 0.0       # 加权置信度

    # 共振理由
    resonance_reasons: List[str] = field(default_factory=list)
    resonance_warnings: List[str] = field(default_factory=list)

    # ===== 便捷属性 =====
    @property
    def is_resonance(self) -> bool:
        """是否有共振（等级 C 或以上）"""
        return self.resonance_level in (
            ResonanceLevel.S,
            ResonanceLevel.A,
            ResonanceLevel.B,
            ResonanceLevel.C
        )

    @property
    def position_ratio(self) -> float:
        """根据共振等级返回仓位比例"""
        ratio_map = {
            ResonanceLevel.S: 1.0,
            ResonanceLevel.A: 0.75,
            ResonanceLevel.B: 0.5,
            ResonanceLevel.C: 0.25,
            ResonanceLevel.INVALID: 0.0
        }
        return ratio_map.get(self.resonance_level, 0.0)


# 共振条件定义
RESONANCE_CONDITIONS = [
    ("板块趋势向上", 1.0, "MA20 > MA60"),
    ("板块RSI健康", 1.0, "30 < RSI < 65"),
    ("板块动量为正", 1.0, "20日涨幅 > 0%"),
    ("个股趋势向上", 1.0, "MA20 > MA60"),
    ("个股RSI健康", 1.0, "RSI < 45"),
    ("个股MA金叉/即将金叉", 0.8, "已金叉或MA5距MA20<2%"),
    ("个股相对强度", 0.8, "RS排名 > 50%"),
    ("市场趋势向上", 1.0, "沪深300 > MA20"),
]


def calculate_resonance_score(
    sector_trend: str,
    sector_rsi: float,
    sector_momentum: float,
    stock_trend: str,
    stock_rsi: float,
    stock_golden_cross: bool,
    stock_gc_approaching: bool,
    stock_rs_rank: float,
    market_trend: str
) -> Tuple[ResonanceLevel, float, List[ResonanceCondition], List[str], List[str]]:
    """
    计算共振等级和置信度

    Args:
        sector_trend: 板块趋势
        sector_rsi: 板块 RSI
        sector_momentum: 板块 20日涨幅
        stock_trend: 个股趋势
        stock_rsi: 个股 RSI
        stock_golden_cross: 个股是否金叉
        stock_gc_approaching: 个股是否即将金叉
        stock_rs_rank: 个股相对强度排名
        market_trend: 市场趋势

    Returns:
        (共振等级, 置信度, 条件列表, 满足理由, 警告理由)
    """
    conditions = []
    reasons = []
    warnings = []
    total_weight = 0.0
    met_weight = 0.0

    # 1. 板块趋势向上
    met = sector_trend == "uptrend"
    weight = 1.0
    conditions.append(ResonanceCondition(
        name="板块趋势向上",
        met=met,
        weight=weight,
        value=sector_trend,
        threshold="MA20 > MA60"
    ))
    total_weight += weight
    if met:
        met_weight += weight
        reasons.append(f"板块趋势向上({sector_trend})")
    else:
        warnings.append(f"板块趋势向下({sector_trend})")

    # 2. 板块 RSI 健康 (30 < RSI < 65)
    met = 30 < sector_rsi < 65
    conditions.append(ResonanceCondition(
        name="板块RSI健康",
        met=met,
        weight=weight,
        value=sector_rsi,
        threshold="30 < RSI < 65"
    ))
    total_weight += weight
    if met:
        met_weight += weight
        reasons.append(f"板块RSI健康({sector_rsi:.0f})")
    else:
        warnings.append(f"板块RSI{'偏低' if sector_rsi <= 30 else '偏高'}({sector_rsi:.0f})")

    # 3. 板块动量为正
    met = sector_momentum > 0
    conditions.append(ResonanceCondition(
        name="板块动量为正",
        met=met,
        weight=weight,
        value=sector_momentum,
        threshold="20日涨幅 > 0%"
    ))
    total_weight += weight
    if met:
        met_weight += weight
        reasons.append(f"板块动量正({sector_momentum:.1f}%)")
    else:
        warnings.append(f"板块动量负({sector_momentum:.1f}%)")

    # 4. 个股趋势向上
    met = stock_trend == "uptrend"
    conditions.append(ResonanceCondition(
        name="个股趋势向上",
        met=met,
        weight=weight,
        value=stock_trend,
        threshold="MA20 > MA60"
    ))
    total_weight += weight
    if met:
        met_weight += weight
        reasons.append(f"个股趋势向上({stock_trend})")
    else:
        warnings.append(f"个股趋势向下({stock_trend})")

    # 5. 个股 RSI 健康 (RSI < 45)
    met = stock_rsi < 45
    conditions.append(ResonanceCondition(
        name="个股RSI健康",
        met=met,
        weight=weight,
        value=stock_rsi,
        threshold="RSI < 45"
    ))
    total_weight += weight
    if met:
        met_weight += weight
        reasons.append(f"个股RSI健康({stock_rsi:.0f})")
    else:
        warnings.append(f"个股RSI偏高({stock_rsi:.0f})")

    # 6. 个股 MA 金叉/即将金叉
    met = stock_golden_cross or stock_gc_approaching
    weight = 0.8  # 略低的权重
    conditions.append(ResonanceCondition(
        name="个股MA金叉/即将金叉",
        met=met,
        weight=weight,
        value=stock_golden_cross or stock_gc_approaching,
        threshold="已金叉或MA5距MA20<2%"
    ))
    total_weight += weight
    if met:
        met_weight += weight
        reasons.append(f"个股MA{'金叉' if stock_golden_cross else '即将金叉'}")
    else:
        warnings.append("个股无金叉信号")

    # 7. 个股相对强度排名 > 50%
    met = stock_rs_rank > 0.5
    weight = 0.8  # 略低的权重
    conditions.append(ResonanceCondition(
        name="个股相对强度",
        met=met,
        weight=weight,
        value=stock_rs_rank,
        threshold="RS排名 > 50%"
    ))
    total_weight += weight
    if met:
        met_weight += weight
        reasons.append(f"个股RS排名靠前({stock_rs_rank:.0%})")
    else:
        warnings.append(f"个股RS排名靠后({stock_rs_rank:.0%})")

    # 8. 市场趋势向上
    met = market_trend == "uptrend"
    conditions.append(ResonanceCondition(
        name="市场趋势向上",
        met=met,
        weight=weight,
        value=market_trend,
        threshold="沪深300 > MA20"
    ))
    total_weight += weight
    if met:
        met_weight += weight
        reasons.append(f"市场趋势向上({market_trend})")
    else:
        warnings.append(f"市场趋势向下({market_trend})")

    # 计算置信度
    confidence = met_weight / total_weight if total_weight > 0 else 0.0

    # 确定等级
    met_count = sum(1 for c in conditions if c.met)

    if met_count >= 8:
        level = ResonanceLevel.S
    elif met_count >= 6:
        level = ResonanceLevel.A
    elif met_count >= 4:
        level = ResonanceLevel.B
    elif met_count >= 2:
        level = ResonanceLevel.C
    else:
        level = ResonanceLevel.INVALID

    return level, confidence, conditions, reasons, warnings


def create_resonance_result(
    date: str,
    stock_code: str,
    sector_name: str,
    # 板块指标
    sector_trend: str,
    sector_rsi: float,
    sector_momentum: float,
    # 个股指标
    stock_trend: str,
    stock_rsi: float,
    stock_golden_cross: bool,
    stock_gc_approaching: bool,
    stock_rs_rank: float,
    stock_rs_score: float = 0.0,
    # 市场指标
    market_trend: str = "sideways"
) -> ResonanceResult:
    """
    创建共振检测结果

    Args:
        date: 日期
        stock_code: 股票代码
        sector_name: 板块名称
        sector_trend: 板块趋势
        sector_rsi: 板块 RSI
        sector_momentum: 板块 20日涨幅
        stock_trend: 个股趋势
        stock_rsi: 个股 RSI
        stock_golden_cross: 个股是否金叉
        stock_gc_approaching: 个股是否即将金叉
        stock_rs_rank: 个股相对强度排名 (0.0 ~ 1.0)
        stock_rs_score: 个股 RS 原始值（个股20日涨幅 - 板块20日涨幅）
        market_trend: 市场趋势

    Returns:
        ResonanceResult
    """
    level, confidence, conditions, reasons, warnings = calculate_resonance_score(
        sector_trend=sector_trend,
        sector_rsi=sector_rsi,
        sector_momentum=sector_momentum,
        stock_trend=stock_trend,
        stock_rsi=stock_rsi,
        stock_golden_cross=stock_golden_cross,
        stock_gc_approaching=stock_gc_approaching,
        stock_rs_rank=stock_rs_rank,
        market_trend=market_trend
    )

    return ResonanceResult(
        date=date,
        stock_code=stock_code,
        sector_name=sector_name,
        sector_trend=sector_trend,
        sector_rsi=sector_rsi,
        sector_momentum_20d=sector_momentum,
        stock_trend=stock_trend,
        stock_rsi=stock_rsi,
        stock_golden_cross=stock_golden_cross,
        stock_golden_cross_approaching=stock_gc_approaching,
        stock_rs_rank=stock_rs_rank,
        stock_rs_score=stock_rs_score,
        market_trend=market_trend,
        conditions=conditions,
        conditions_met=sum(1 for c in conditions if c.met),
        resonance_level=level,
        resonance_confidence=confidence,
        resonance_reasons=reasons,
        resonance_warnings=warnings
    )
