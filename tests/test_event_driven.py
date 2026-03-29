"""事件驱动策略测试"""

import pytest
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from src.data.indicators.event_driven import (
    EarningsSurpriseResult,
    JiejinRiskResult,
    RebalanceResult,
    ShareholderBuyResult,
    EventSignal,
    calculate_earnings_surprise,
    earnings_trend_analysis,
    calculate_jiejin_risk,
    scan_jiejin_calendar,
    calculate_rebalance_effect,
    analyze_shareholder_buying,
    calculate_event_score,
    get_event_calendar,
    is_policy_sensitive_period,
    batch_calculate_jiejin_risk,
)


class TestEarningsSurprise:
    """财报预期差测试"""

    def test_calculate_earnings_surprise_beat(self):
        """超预期情况"""
        result = calculate_earnings_surprise(
            actual_eps=1.2,
            consensus_eps=1.0,
            pre_return=0.05,
            post_return=0.03
        )

        assert result.category == 'BEAT'
        assert result.surprise_pct == pytest.approx(20.0, abs=0.001)
        assert result.is_significant == True

    def test_calculate_earnings_surprise_miss(self):
        """低于预期情况"""
        result = calculate_earnings_surprise(
            actual_eps=0.8,
            consensus_eps=1.0,
            pre_return=0.05,
            post_return=-0.02
        )

        assert result.category == 'MISS'
        assert result.surprise_pct == pytest.approx(-20.0, abs=0.001)
        assert result.is_significant == True

    def test_calculate_earnings_surprise_meet(self):
        """符合预期情况"""
        result = calculate_earnings_surprise(
            actual_eps=1.01,
            consensus_eps=1.0,
            pre_return=0.05,
            post_return=0.01
        )

        assert result.category == 'MEET'
        assert result.surprise_pct == pytest.approx(1.0, abs=0.001)
        assert result.is_significant == False

    def test_calculate_earnings_surprise_zero_consensus(self):
        """一致预期为0的处理"""
        result = calculate_earnings_surprise(
            actual_eps=1.0,
            consensus_eps=0,
        )

        assert result.surprise_pct == 0.0


class TestEarningsTrendAnalysis:
    """业绩趋势分析测试"""

    def test_trend_accelerating(self):
        """业绩加速"""
        history = [
            {'period': '2024Q1', 'actual_eps': 1.0, 'consensus_eps': 0.9},
            {'period': '2024Q2', 'actual_eps': 1.1, 'consensus_eps': 0.95},
            {'period': '2024Q3', 'actual_eps': 1.2, 'consensus_eps': 1.0},
            {'period': '2024Q4', 'actual_eps': 1.4, 'consensus_eps': 1.1},
            {'period': '2023Q4', 'actual_eps': 0.9, 'consensus_eps': 0.85},
            {'period': '2023Q3', 'actual_eps': 0.88, 'consensus_eps': 0.85},
            {'period': '2023Q2', 'actual_eps': 0.85, 'consensus_eps': 0.83},
            {'period': '2023Q1', 'actual_eps': 0.82, 'consensus_eps': 0.8},
        ]

        result = earnings_trend_analysis(history)

        assert result['trend'] == '加速'
        assert 'interpretation' in result

    def test_trend_stable(self):
        """业绩稳定"""
        history = [
            {'period': '2024Q1', 'actual_eps': 1.0, 'consensus_eps': 0.95},
            {'period': '2024Q2', 'actual_eps': 1.02, 'consensus_eps': 0.98},
            {'period': '2024Q3', 'actual_eps': 0.99, 'consensus_eps': 0.97},
            {'period': '2024Q4', 'actual_eps': 1.01, 'consensus_eps': 0.99},
            {'period': '2023Q4', 'actual_eps': 0.98, 'consensus_eps': 0.95},
            {'period': '2023Q3', 'actual_eps': 0.97, 'consensus_eps': 0.94},
            {'period': '2023Q2', 'actual_eps': 0.96, 'consensus_eps': 0.93},
            {'period': '2023Q1', 'actual_eps': 0.95, 'consensus_eps': 0.92},
        ]

        result = earnings_trend_analysis(history)

        assert result['trend'] == '稳定'

    def test_insufficient_data(self):
        """数据不足"""
        history = [
            {'period': '2024Q1', 'actual_eps': 1.0, 'consensus_eps': 0.9},
            {'period': '2024Q2', 'actual_eps': 1.1, 'consensus_eps': 1.0},
        ]

        result = earnings_trend_analysis(history)

        assert result['status'] == '数据不足'


class TestJiejinRisk:
    """限售股解禁风险测试"""

    def test_high_risk_ipo(self):
        """高风险IPO解禁"""
        result = calculate_jiejin_risk(
            unlock_shares=10000000,
            float_shares=20000000,  # 解禁比例50%
            holder_type='IPO原始股',
            avg_daily_volume=500000,  # 需要20天消化
        )

        assert result.unlock_ratio == 0.5
        assert result.risk_level == '高危'
        assert result.risk_score >= 5

    def test_medium_risk_dingzeng(self):
        """中等风险定增解禁"""
        result = calculate_jiejin_risk(
            unlock_shares=3000000,
            float_shares=20000000,  # 解禁比例15%
            holder_type='定增',
            avg_daily_volume=500000,
        )

        assert result.unlock_ratio == 0.15
        assert result.risk_level == '中危'
        assert result.risk_score >= 3

    def test_low_risk_incentive(self):
        """低风险股权激励"""
        result = calculate_jiejin_risk(
            unlock_shares=500000,
            float_shares=20000000,  # 解禁比例2.5%
            holder_type='股权激励',
            avg_daily_volume=500000,
        )

        assert result.unlock_ratio == 0.025
        assert result.risk_level in ['低危', '安全']  # 根据实际评分可能是低危或安全

    def test_safe_scenario(self):
        """安全情况"""
        result = calculate_jiejin_risk(
            unlock_shares=100000,
            float_shares=20000000,  # 解禁比例0.5%
            holder_type='股权激励',
            avg_daily_volume=1000000,
        )

        assert result.risk_level == '安全'


class TestScanJiejinCalendar:
    """解禁日历扫描测试"""

    def test_scan_high_risk(self):
        """扫描高风险解禁"""
        today = datetime.now()
        future_date = (today + timedelta(days=15)).strftime('%Y-%m-%d')

        unlock_data = [
            {
                'stock_code': '000001',
                'unlock_date': future_date,
                'unlock_shares': 10000000,
                'float_shares': 20000000,
                'holder_type': 'IPO原始股',
                'avg_daily_volume': 500000,
            }
        ]

        high_risk, opportunities = scan_jiejin_calendar(
            unlock_data,
            today.strftime('%Y-%m-%d'),
            lookforward_days=30
        )

        assert len(high_risk) == 1
        assert high_risk[0].stock_code == '000001'
        assert high_risk[0].risk_level == '高危'

    def test_scan_no_risk(self):
        """扫描无风险解禁"""
        today = datetime.now()

        unlock_data = []

        high_risk, opportunities = scan_jiejin_calendar(
            unlock_data,
            today.strftime('%Y-%m-%d'),
            lookforward_days=30
        )

        assert len(high_risk) == 0


class TestRebalanceEffect:
    """指数调样效应测试"""

    def test_rebalance_in(self):
        """调入信号"""
        result = calculate_rebalance_effect(
            stock_code='000001',
            index_name='沪深300',
            direction='调入',
            stock_weight=0.005,
            etf_aum=10000000000,  # 100亿
            adv=500000000  # 日均5亿
        )

        assert result.direction == '调入'
        assert result.passive_flow == 50000000  # 5000万
        assert result.opportunity == '存在被动买入托底'
        assert result.recommended_action == '买入'

    def test_rebalance_out(self):
        """调出信号"""
        result = calculate_rebalance_effect(
            stock_code='000002',
            index_name='沪深300',
            direction='调出',
            stock_weight=0.003,
            etf_aum=10000000000,
            adv=500000000
        )

        assert result.direction == '调出'
        assert result.opportunity == '被动卖出压力'
        assert result.recommended_action == '回避'


class TestShareholderBuying:
    """大股东增持分析测试"""

    def test_strong_buy_signal(self):
        """强烈买入信号"""
        buy_records = [
            {'date': '2024-01-15', 'amount': 10000000, 'shares': 1000000},
            {'date': '2024-01-20', 'amount': 5000000, 'shares': 500000},
        ]

        result = analyze_shareholder_buying(
            buy_records=buy_records,
            market_cap=500000000,  # 5亿
            current_price=10.0,
            last_buy_date='2024-01-20'
        )

        assert result.buy_ratio > 0.02
        assert '买入' in result.signal
        assert result.recommended_action == '买入'

    def test_buy_signal(self):
        """买入信号"""
        buy_records = [
            {'date': '2024-01-10', 'amount': 2000000, 'shares': 200000},
        ]

        result = analyze_shareholder_buying(
            buy_records=buy_records,
            market_cap=500000000,
            current_price=10.0,
            last_buy_date='2024-01-10'
        )

        assert '买入' in result.signal or '中性' in result.signal

    def test_no_data(self):
        """无增持数据"""
        result = analyze_shareholder_buying(
            buy_records=[],
            market_cap=500000000,
            current_price=10.0
        )

        assert result.signal == '无数据'
        assert result.recommended_action == '观望'


class TestEventScore:
    """综合事件评分测试"""

    def test_positive_score(self):
        """正向评分"""
        earnings = calculate_earnings_surprise(
            actual_eps=1.2,
            consensus_eps=1.0
        )

        result = calculate_event_score(
            earnings=earnings,
            jiejin_risk=None,
            shareholder_buy=None,
            rebalance=None
        )

        assert result.total_score > 0

    def test_negative_score(self):
        """负向评分（解禁风险）"""
        jiejin = calculate_jiejin_risk(
            unlock_shares=10000000,
            float_shares=20000000,
            holder_type='IPO原始股',
            avg_daily_volume=500000
        )

        result = calculate_event_score(
            earnings=None,
            jiejin_risk=jiejin,
            shareholder_buy=None,
            rebalance=None
        )

        assert result.total_score < 0
        assert result.action == '回避'

    def test_buy_signal(self):
        """买入信号"""
        shareholder = analyze_shareholder_buying(
            buy_records=[{'date': '2024-01-15', 'amount': 20000000, 'shares': 2000000}],
            market_cap=500000000,
            current_price=10.0,
            last_buy_date='2024-01-15'
        )

        result = calculate_event_score(
            earnings=None,
            jiejin_risk=None,
            shareholder_buy=shareholder,
            rebalance=None
        )

        assert result.action in ['买入', '观望']  # 取决于具体分数


class TestEventCalendar:
    """事件日历测试"""

    def test_get_event_calendar(self):
        """获取事件日历"""
        events = get_event_calendar(3)

        assert '两会' in events or '年报+一季报披露期开始' in events

    def test_is_policy_sensitive_march(self):
        """3月是政策敏感期"""
        result = is_policy_sensitive_period('2024-03-10')
        assert result == True

    def test_is_policy_sensitive_july(self):
        """7月是政策敏感期"""
        result = is_policy_sensitive_period('2024-07-25')
        assert result == True

    def test_is_not_policy_sensitive(self):
        """非敏感期"""
        result = is_policy_sensitive_period('2024-06-15')
        assert result == False


class TestBatchJiejinRisk:
    """批量解禁风险计算测试"""

    def test_batch_calculate(self):
        """批量计算解禁风险"""
        unlock_data = pd.DataFrame([
            {
                'stock_code': '000001',
                'unlock_date': '2024-02-01',
                'unlock_shares': 10000000,
                'float_shares': 20000000,
                'holder_type': 'IPO原始股',
            },
            {
                'stock_code': '000002',
                'unlock_date': '2024-02-15',
                'unlock_shares': 500000,
                'float_shares': 20000000,
                'holder_type': '股权激励',
            }
        ])

        avg_volumes = {
            '000001': 500000,
            '000002': 1000000,
        }

        result = batch_calculate_jiejin_risk(unlock_data, avg_volumes)

        assert len(result) == 2
        assert result.iloc[0]['risk_level'] == '高危'
        assert result.iloc[1]['risk_level'] in ['低危', '安全']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
