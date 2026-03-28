"""
StockData 数据质量评分测试
"""

import pytest
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List

# 导入被测试的模块
import sys
sys.path.insert(0, 'scripts')
from utils.quality import QualityScore, calculate_quality_score, validate_daily

from tests.fixtures import ANOMALY_TEST_CASES


class TestQualityScore:
    """数据质量评分测试"""

    def _calc_score_with_anomalies(self, anomalies: list) -> QualityScore:
        """辅助方法：根据异常计算分数"""
        # 创建测试数据
        df = pd.DataFrame({
            'date': pd.date_range('2026-03-01', periods=3),
            'code': '000001',
            'open': [10.0, 10.2, 10.4],
            'high': [10.5, 10.7, 10.8],
            'low': [9.8, 10.0, 10.2],
            'close': [10.2, 10.5, 10.6],
            'volume': [1000000, 1100000, 1200000],
            'amount': [10200000, 11550000, 12600000],
            'adj_factor': [1.0, 1.0, 1.0],
            'turnover': [0.05, 0.055, 0.06],
            'is_halt': [False, False, False],
            'pct_chg': [0.02, 0.029, 0.01],
        })
        return calculate_quality_score(df, anomalies)

    def test_perfect_data_100(self):
        """完美数据得100分"""
        df, anomalies = ANOMALY_TEST_CASES['perfect']()
        score = calculate_quality_score(df, anomalies)

        assert score.total == 100, f"完美数据应为100分，实际: {score.total}"
        assert score.grade == "完美"
        assert score.usable is True

    def test_price_anomaly_deduction(self):
        """价格异常扣分"""
        anomalies = [{'reason': 'price_out_of_range', 'count': 1}]
        score = self._calc_score_with_anomalies(anomalies)

        # 价格25分，扣10分
        assert 10 <= score.price_score <= 20, f"价格分数异常: {score.price_score}"

    def test_multiple_price_anomalies(self):
        """多个价格异常"""
        anomalies = [
            {'reason': 'price_out_of_range', 'count': 1},
            {'reason': 'price_out_of_range', 'count': 1},
        ]
        score = self._calc_score_with_anomalies(anomalies)

        # 价格25分，扣20分，剩余5分
        assert 0 <= score.price_score <= 10, f"价格分数异常: {score.price_score}"

    def test_ohlc_anomaly_deduction(self):
        """OHLC异常扣分"""
        anomalies = [{'reason': 'ohlc_invalid', 'count': 1}]
        score = self._calc_score_with_anomalies(anomalies)

        # OHLC 25分，扣15分
        assert 5 <= score.ohlc_score <= 15, f"OHLC分数异常: {score.ohlc_score}"

    def test_adj_continuity_break_penalty(self):
        """复权连续性断裂 - 极低分"""
        anomalies = [{'reason': 'adj_continuity_break', 'count': 1}]
        score = self._calc_score_with_anomalies(anomalies)

        # 复权连续性 25分，扣20分，剩余5分
        assert 0 <= score.adj_score <= 10, f"复权分数异常: {score.adj_score}"

    def test_adj_factor_negative(self):
        """复权因子为负 - 极低分"""
        anomalies = [{'reason': 'adj_factor_invalid', 'count': 1}]
        score = self._calc_score_with_anomalies(anomalies)

        assert score.adj_score <= 10, f"因子为负应得极低分: {score.adj_score}"

    def test_volume_anomaly(self):
        """成交量异常扣分 - 归类到价格异常"""
        df, anomalies = ANOMALY_TEST_CASES['volume_negative']()
        score = calculate_quality_score(df, anomalies)

        # 成交量异常会导致 completeness_score 降低
        assert score.total < 100, f"成交量异常应扣分: {score.total}"

    def test_multi_anomalies(self):
        """多维度异常"""
        df, anomalies = ANOMALY_TEST_CASES['multi_anomalies']()
        score = calculate_quality_score(df, anomalies)

        # 总分应该较低
        assert score.total < 100, f"多异常总分应低于100: {score.total}"
        assert score.usable is False

    def test_grade_boundaries(self):
        """评分等级边界"""
        # 测试完美
        df, anomalies = ANOMALY_TEST_CASES['perfect']()
        score = calculate_quality_score(df, anomalies)
        assert score.grade == "完美"

        # 测试多异常
        df, anomalies = ANOMALY_TEST_CASES['multi_anomalies']()
        score = calculate_quality_score(df, anomalies)
        assert score.grade in ["可疑", "危险", "废弃"], f"应为可疑以下: {score.grade}"

    def test_quarantine_threshold(self):
        """隔离阈值 - 低于50分不可用"""
        df, anomalies = ANOMALY_TEST_CASES['multi_anomalies']()
        score = calculate_quality_score(df, anomalies)

        # 多异常总分应该低于50
        assert score.total < 50, f"多异常总分应低于50: {score.total}"
        assert score.usable is False

    def test_warning_threshold(self):
        """告警阈值 - 低于80分告警"""
        df, anomalies = ANOMALY_TEST_CASES['price_out_of_range']()
        score = calculate_quality_score(df, anomalies)

        # 价格异常总分应该在某个合理范围
        assert score.total < 100, f"价格异常应扣分: {score.total}"

    def test_single_row_data(self):
        """单行数据 - 应该正常处理"""
        df, anomalies = ANOMALY_TEST_CASES['single_row']()
        score = calculate_quality_score(df, anomalies)

        # 单行数据只要字段完整，应该是高分
        assert score.total >= 80, f"单行正常数据应得高分: {score.total}"

    def test_empty_data(self):
        """空数据 - 得0分"""
        df, anomalies = ANOMALY_TEST_CASES['empty']()
        score = calculate_quality_score(df, anomalies)

        assert score.total == 0, f"空数据应得0分: {score.total}"
        assert score.usable is False

    def test_limit_up_normal(self):
        """涨停数据 - 正常，不应扣分"""
        df, anomalies = ANOMALY_TEST_CASES['limit_up']()
        score = calculate_quality_score(df, anomalies)

        assert score.total >= 80, f"涨停是正常数据: {score.total}"

    def test_limit_down_normal(self):
        """跌停数据 - 正常，不应扣分"""
        df, anomalies = ANOMALY_TEST_CASES['limit_down']()
        score = calculate_quality_score(df, anomalies)

        assert score.total >= 80, f"跌停是正常数据: {score.total}"


class TestQualityValidation:
    """数据校验测试"""

    def test_validate_perfect_data(self):
        """完美数据校验通过"""
        df, _ = ANOMALY_TEST_CASES['perfect']()
        result = validate_daily(df)
        assert result['valid'] is True

    def test_validate_price_out_of_range(self):
        """价格超范围校验失败"""
        df, _ = ANOMALY_TEST_CASES['price_out_of_range']()
        result = validate_daily(df)
        assert bool(result['valid']) is False
        assert len(result['anomalies']) > 0

    def test_validate_ohlc_invalid(self):
        """OHLC不合法校验失败"""
        df, _ = ANOMALY_TEST_CASES['ohlc_invalid']()
        result = validate_daily(df)
        assert bool(result['valid']) is False

    def test_validate_adj_continuity_break(self):
        """复权连续性断裂校验"""
        df, _ = ANOMALY_TEST_CASES['adj_continuity_break']()
        result = validate_daily(df)
        assert bool(result['valid']) is False
        assert any('adj' in a['reason'] for a in result['anomalies'])

    def test_validate_multi_anomalies(self):
        """多异常数据"""
        df, _ = ANOMALY_TEST_CASES['multi_anomalies']()
        result = validate_daily(df)
        assert bool(result['valid']) is False
        assert len(result['anomalies']) >= 2

    def test_validate_suspended_stock(self):
        """停牌股票 - 正常"""
        df, anomalies = ANOMALY_TEST_CASES['suspended']()
        result = validate_daily(df)
        assert result['valid'] is True

    def test_validate_resume_trading(self):
        """复牌股票 - 正常"""
        df, anomalies = ANOMALY_TEST_CASES['resume_trading']()
        result = validate_daily(df)
        assert result['valid'] is True
