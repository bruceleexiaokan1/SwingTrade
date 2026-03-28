"""质量评分器边界测试"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.fetcher.quality_scorer import QualityScorer, QualityScore


class TestQualityScorerEdgeCases:
    """质量评分器边界情况测试"""

    def setup_method(self):
        self.scorer = QualityScorer()

    def test_close_zero(self):
        """收盘价为0的异常情况"""
        record = {
            "date": "2026-03-28",
            "code": "600519",
            "open": 0,
            "high": 0,
            "low": 0,
            "close": 0,
            "volume": 0
        }

        score = self.scorer.score(record)

        # 价格为0应该导致 range_validity 降低
        assert score.range_validity < 100

    def test_volume_negative(self):
        """负成交量"""
        record = {
            "date": "2026-03-28",
            "code": "600519",
            "open": 1800,
            "high": 1850,
            "low": 1790,
            "close": 1820,
            "volume": -100  # 负数
        }

        score = self.scorer.score(record)

        # 负成交量应该导致拒绝
        assert self.scorer.should_reject(score) or score.overall < 60

    def test_pct_chg_extreme(self):
        """涨跌幅超出正常范围"""
        record = {
            "date": "2026-03-28",
            "code": "600519",
            "open": 1800,
            "high": 1850,
            "low": 1790,
            "close": 2200,  # 涨幅超过20%
            "volume": 1000000,
            "pct_chg": 0.25
        }

        score = self.scorer.score(record)

        assert score.range_validity < 100

    def test_missing_field(self):
        """缺少字段"""
        record = {
            "date": "2026-03-28",
            "code": "600519",
            "close": 1820
            # 缺少 open, high, low, volume
        }

        score = self.scorer.score(record)

        assert score.field_completeness < 100
        # 3/7 * 100 = 42.86
        assert score.field_completeness == pytest.approx(42.86, rel=0.01)

    def test_dual_source_slight_diff(self):
        """双源微小差异（0.1%以内）"""
        record = {
            "date": "2026-03-28",
            "code": "600519",
            "open": 1800,
            "high": 1850,
            "low": 1790,
            "close": 1820,
            "volume": 1000000
        }
        # 差异 0.05%
        verify_record = {"close": 1820.91}

        score = self.scorer.score(record, verify_record=verify_record)

        assert score.source_consistency == 95

    def test_dual_source_1pct_diff(self):
        """双源1%差异"""
        record = {
            "date": "2026-03-28",
            "code": "600519",
            "open": 1800,
            "high": 1850,
            "low": 1790,
            "close": 1820,
            "volume": 1000000
        }
        # 差异约1%
        verify_record = {"close": 1838}

        score = self.scorer.score(record, verify_record=verify_record)

        assert score.source_consistency == 85

    def test_prev_record_none(self):
        """无历史记录"""
        record = {
            "date": "2026-03-28",
            "code": "600519",
            "open": 1800,
            "high": 1850,
            "low": 1790,
            "close": 1820,
            "volume": 1000000
        }

        score = self.scorer.score(record, prev_record=None)

        assert score.historical_anomaly == 100

    def test_threshold_boundary_60(self):
        """阈值边界：刚好60分"""
        score = QualityScore(
            source_consistency=95,
            field_completeness=100,
            range_validity=100,
            historical_anomaly=100,
            overall=60
        )

        assert self.scorer.should_write(score) is True
        assert self.scorer.should_verify(score) is False
        assert self.scorer.should_reject(score) is False

    def test_threshold_boundary_80(self):
        """阈值边界：刚好80分"""
        score = QualityScore(
            source_consistency=95,
            field_completeness=100,
            range_validity=100,
            historical_anomaly=100,
            overall=80
        )

        assert self.scorer.should_write(score) is True
        assert self.scorer.should_verify(score) is False  # 80 不需要验证

    def test_ohlc_invalid_h_less_than_l(self):
        """OHLC 严重错误：最高价 < 最低价"""
        record = {
            "date": "2026-03-28",
            "code": "600519",
            "open": 1800,
            "high": 1700,  # 错误
            "low": 1790,
            "close": 1820,
            "volume": 1000000
        }

        score = self.scorer.score(record)

        assert score.range_validity == 60  # OHLC关系错误

    def test_price_out_of_range_low(self):
        """价格过低"""
        record = {
            "date": "2026-03-28",
            "code": "600519",
            "open": 0.001,
            "high": 0.005,
            "low": 0.001,
            "close": 0.002,
            "volume": 1000000
        }

        score = self.scorer.score(record)

        assert score.range_validity == 60  # 价格过低


class TestQualityScorerThresholdCalculation:
    """综合分计算测试"""

    def setup_method(self):
        self.scorer = QualityScorer()

    def test_weighted_score_calculation(self):
        """验证加权计算"""
        # 95 * 0.30 + 100 * 0.20 + 100 * 0.30 + 100 * 0.20
        # = 28.5 + 20 + 30 + 20 = 98.5
        record = {
            "date": "2026-03-28",
            "code": "600519",
            "open": 1800,
            "high": 1850,
            "low": 1790,
            "close": 1820,
            "volume": 1000000
        }

        score = self.scorer.score(record, prev_record={"close": 1800})

        assert score.overall == 98.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
