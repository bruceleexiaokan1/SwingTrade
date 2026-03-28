"""质量评分器测试"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.fetcher.quality_scorer import QualityScorer, QualityScore


class TestQualityScorer:
    """质量评分器测试"""

    def setup_method(self):
        self.scorer = QualityScorer()

    def test_single_source_score(self):
        """单源数据评分"""
        record = {
            "date": "2026-03-28",
            "code": "600519",
            "open": 1800.0,
            "high": 1850.0,
            "low": 1790.0,
            "close": 1820.0,
            "volume": 1000000,
            "pct_chg": 0.011
        }

        score = self.scorer.score(record)

        assert isinstance(score, QualityScore)
        assert score.source_consistency == 95.0  # 单源，Tushare 可靠给 95
        assert score.field_completeness == 100.0  # 全部字段
        assert score.range_validity == 100.0     # 范围正常
        assert score.historical_anomaly == 100.0  # 无历史数据
        assert score.overall >= 80  # 综合分应该足够高

    def test_dual_source_consistent(self):
        """双源完全一致"""
        record = {
            "date": "2026-03-28",
            "code": "600519",
            "open": 1800.0,
            "high": 1850.0,
            "low": 1790.0,
            "close": 1820.0,
            "volume": 1000000
        }
        verify_record = {
            "close": 1820.0
        }

        score = self.scorer.score(record, verify_record=verify_record)

        assert score.source_consistency == 100.0

    def test_dual_source_mismatch(self):
        """双源差异大"""
        record = {
            "date": "2026-03-28",
            "code": "600519",
            "open": 1800.0,
            "high": 1850.0,
            "low": 1790.0,
            "close": 1820.0,
            "volume": 1000000
        }
        verify_record = {
            "close": 1900.0  # 差异 4.4%
        }

        score = self.scorer.score(record, verify_record=verify_record)

        assert score.source_consistency == 50.0

    def test_historical_anomaly_normal(self):
        """历史连续性正常（涨跌停范围内）"""
        record = {
            "date": "2026-03-28",
            "code": "600519",
            "close": 1820.0,
            "open": 1800.0,
            "high": 1850.0,
            "low": 1790.0,
            "volume": 1000000
        }
        prev_record = {
            "close": 1800.0
        }

        score = self.scorer.score(record, prev_record=prev_record)

        assert score.historical_anomaly == 100.0

    def test_historical_anomaly_exceed_limit(self):
        """历史连续性异常（超出涨跌停）"""
        record = {
            "date": "2026-03-28",
            "code": "600519",
            "close": 2000.0,  # 涨幅 11%
            "open": 1800.0,
            "high": 2000.0,
            "low": 1790.0,
            "volume": 1000000
        }
        prev_record = {
            "close": 1800.0
        }

        score = self.scorer.score(record, prev_record=prev_record)

        assert score.historical_anomaly == 30.0

    def test_should_verify(self):
        """是否需要验证"""
        score = QualityScore(
            source_consistency=75,
            field_completeness=100,
            range_validity=100,
            historical_anomaly=100,
            overall=70
        )

        assert self.scorer.should_verify(score) is True
        assert self.scorer.should_write(score) is True
        assert self.scorer.should_reject(score) is False

    def test_should_reject(self):
        """是否应该拒绝"""
        score = QualityScore(
            source_consistency=40,
            field_completeness=50,
            range_validity=30,
            historical_anomaly=30,
            overall=50
        )

        assert self.scorer.should_verify(score) is False
        assert self.scorer.should_write(score) is False
        assert self.scorer.should_reject(score) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
