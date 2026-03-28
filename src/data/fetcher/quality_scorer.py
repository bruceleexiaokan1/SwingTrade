"""数据质量评分器

与 scripts/utils/quality.py 集成：
- 本模块负责采集时跨源一致性评分
- 调用 quality.validate_daily() 做基础校验
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import sys
import os

# 添加 scripts 到 path 以导入 quality 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'scripts'))
from utils.quality import validate_daily, QualityScore as BaseQualityScore


@dataclass
class QualityScore:
    """质量评分结果"""
    source_consistency: float      # 跨源一致性（0-100）
    field_completeness: float     # 字段完整性（0-100）
    range_validity: float         # 范围合理性（0-100）
    historical_anomaly: float     # 历史连续性（0-100）
    overall: float               # 综合分（0-100）

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "source_consistency": self.source_consistency,
            "field_completeness": self.field_completeness,
            "range_validity": self.range_validity,
            "historical_anomaly": self.historical_anomaly,
            "overall": self.overall
        }


class QualityScorer:
    """
    数据质量评分器

    评分维度：
    1. source_consistency: 跨源一致性
       - 双源完全一致：100
       - 双源差异在容差内：95
       - 单源（Tushare可靠但不100%）：95
       - 双源差异超容差：50
    2. field_completeness: 字段完整性（每缺失一个字段-10）
    3. range_validity: 范围合理性（每超范围一个字段-20）
    4. historical_anomaly: 历史连续性（涨跌停范围内=100，超出=30）

    综合分 = source_consistency*0.3 + field_completeness*0.2 + range_validity*0.3 + historical_anomaly*0.2
    """

    # 涨跌停范围（主板10%，科创/创业板20%）
    LIMIT_RATIO = 0.107

    # 综合分权重
    WEIGHTS = {
        "source_consistency": 0.30,
        "field_completeness": 0.20,
        "range_validity": 0.30,
        "historical_anomaly": 0.20
    }

    # 质量分阈值
    THRESHOLDS = {
        "write_immediately": 80,   # >= 80 分直接写入
        "need_verification": 60,   # 60-80 分触发验证
        "reject": 0                # < 60 分拒绝
    }

    def __init__(self):
        pass

    def score(
        self,
        record: dict,
        prev_record: Optional[dict] = None,
        verify_record: Optional[dict] = None
    ) -> QualityScore:
        """
        计算单条记录的质量分

        Args:
            record: 主数据源记录
            prev_record: 昨日记录（用于历史连续性检查）
            verify_record: 验证数据源记录（用于跨源一致性检查）

        Returns:
            QualityScore
        """
        # 0. 调用基础校验（来自 scripts/utils/quality.py）
        base_result = self._get_base_quality(record)

        # 1. 跨源一致性（QualityScorer 独有职责）
        source_score = self._score_source_consistency(record, verify_record)

        # 2. 字段完整性（使用原始逻辑，与 base_result 无关）
        completeness_score = self._score_field_completeness(record)

        # 3. 范围合理性（使用 base_result 的严重问题作为否决）
        validity_score = self._score_range_validity(record)
        # 如果基础校验发现 price 或 ohlc 严重问题，直接归零
        if base_result:
            if base_result.price_score == 0 or base_result.ohlc_score == 0 or base_result.adj_score == 0:
                validity_score = 0.0

        # 4. 历史连续性
        anomaly_score = self._score_historical_anomaly(record, prev_record)

        # 5. 综合分
        overall = (
            source_score * self.WEIGHTS["source_consistency"] +
            completeness_score * self.WEIGHTS["field_completeness"] +
            validity_score * self.WEIGHTS["range_validity"] +
            anomaly_score * self.WEIGHTS["historical_anomaly"]
        )

        # 如果基础校验失败，大幅降低总分
        if base_result and not base_result.usable:
            overall = min(overall, 40)

        return QualityScore(
            source_consistency=source_score,
            field_completeness=completeness_score,
            range_validity=validity_score,
            historical_anomaly=anomaly_score,
            overall=round(overall, 2)
        )

    def _score_source_consistency(
        self,
        record: dict,
        verify_record: Optional[dict]
    ) -> float:
        """
        计算跨源一致性分数

        - 双源完全一致：100
        - 双源差异在容差内（0.1%）：95
        - 单源（Tushare可靠但不100%）：95
        - 双源差异超容差：50
        """
        if verify_record is None:
            # 单源，Tushare 可靠但不是100%，给 95
            return 95.0

        # 获取收盘价
        close = record.get("close")
        verify_close = verify_record.get("close")

        if close is None or verify_close is None:
            return 70.0

        if close == 0:
            return 70.0

        # 计算差异百分比
        diff_ratio = abs(close - verify_close) / close

        if diff_ratio < 0.0001:  # 完全一致
            return 100.0
        elif diff_ratio < 0.001:  # 差异 < 0.1%
            return 95.0
        elif diff_ratio < 0.01:  # 差异 < 1%
            return 85.0
        else:  # 差异 >= 1%
            return 50.0

    def _score_field_completeness(self, record: dict) -> float:
        """
        计算字段完整性分数

        每缺失一个必填字段扣10分
        """
        required_fields = ["date", "code", "open", "high", "low", "close", "volume"]
        present_count = 0

        for field in required_fields:
            value = record.get(field)
            if value is not None and not (isinstance(value, float) and pd.isna(value)):
                present_count += 1

        score = (present_count / len(required_fields)) * 100
        return round(score, 2)

    def _score_range_validity(self, record: dict) -> float:
        """
        计算范围合理性分数

        每超范围一个字段扣20分
        """
        # 价格范围
        close = record.get("close", 0)
        if close < 0.01 or close > 10000:
            return 60.0

        # OHLC 关系
        open_price = record.get("open", 0)
        high = record.get("high", 0)
        low = record.get("low", 0)
        close = record.get("close", 0)

        if high < low:
            return 60.0
        if high < open_price or high < close:
            return 60.0
        if low > open_price or low > close:
            return 60.0

        # 涨跌幅范围
        pct_chg = record.get("pct_chg")
        if pct_chg is not None:
            if pct_chg < -0.20 or pct_chg > 0.20:
                return 60.0

        return 100.0

    def _score_historical_anomaly(
        self,
        record: dict,
        prev_record: Optional[dict]
    ) -> float:
        """
        计算历史连续性分数

        - 无历史数据：100（无法比较）
        - 在涨跌停范围内：100
        - 超出涨跌停范围：30（可能是数据问题）
        """
        if prev_record is None:
            # 无历史数据，无法比较
            return 100.0

        prev_close = prev_record.get("close")
        close = record.get("close")

        if prev_close is None or close is None or prev_close == 0:
            return 80.0

        # 计算价格变化百分比
        change_ratio = abs(close - prev_close) / prev_close

        # 在涨跌停范围内
        if change_ratio <= self.LIMIT_RATIO:
            return 100.0
        else:
            # 超出涨跌停，可能是真实波动也可能是数据问题
            # 给一个中等分数，后续需要人工确认
            return 30.0

    def _get_base_quality(self, record: dict) -> Optional[BaseQualityScore]:
        """
        调用 scripts/utils/quality.py 进行基础校验

        Returns:
            BaseQualityScore 或 None（校验失败）
        """
        try:
            # 将 record 转为 DataFrame
            df = pd.DataFrame([record])
            result = validate_daily(df)
            return result['score']
        except Exception:
            return None

    def should_verify(self, score: QualityScore) -> bool:
        """是否需要触发 AkShare 验证"""
        return 60 <= score.overall < 80

    def should_write(self, score: QualityScore) -> bool:
        """是否可以写入"""
        return score.overall >= 60

    def should_reject(self, score: QualityScore) -> bool:
        """是否应该拒绝"""
        return score.overall < 60
