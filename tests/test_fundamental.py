"""基本面指标模块测试

测试基本面指标计算的所有函数
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from src.data.indicators.fundamental import (
    # Profitability
    calculate_roe,
    calculate_roa,
    calculate_gross_margin,
    calculate_net_margin,
    # Growth
    calculate_revenue_growth,
    calculate_profit_growth,
    calculate_growth_score,
    # Valuation
    calculate_pe,
    calculate_pb,
    calculate_ps,
    calculate_valuation_score,
    # Financial Health
    calculate_debt_ratio,
    calculate_current_ratio,
    calculate_cash_flow_ratio,
    assess_financial_health,
    FinancialHealthResult,
    # Composite
    composite_fundamental_score,
)


class TestProfitability:
    """盈利能力指标测试"""

    def test_calculate_roe_normal(self):
        """ROE 正常计算"""
        roe = calculate_roe(net_income=1500, shareholders_equity=10000)
        assert abs(roe - 15.0) < 0.01

    def test_calculate_roe_high_quality(self):
        """ROE 优质公司 (>15%)"""
        roe = calculate_roe(net_income=2000, shareholders_equity=10000)
        assert roe == 20.0
        assert roe > 15  # 优质标准

    def test_calculate_roe_low_quality(self):
        """ROE 较差公司 (<5%)"""
        roe = calculate_roe(net_income=300, shareholders_equity=10000)
        assert roe == 3.0
        assert roe < 5  # 较差标准

    def test_calculate_roe_zero_equity(self):
        """ROE 分母为零"""
        roe = calculate_roe(net_income=1500, shareholders_equity=0)
        assert roe == 0.0

    def test_calculate_roe_nan_equity(self):
        """ROE 分子为 NaN"""
        roe = calculate_roe(net_income=np.nan, shareholders_equity=10000)
        assert roe == 0.0

    def test_calculate_roe_negative_income(self):
        """ROE 负净利润"""
        roe = calculate_roe(net_income=-500, shareholders_equity=10000)
        assert roe == -5.0

    def test_calculate_roa_normal(self):
        """ROA 正常计算"""
        roa = calculate_roa(net_income=800, total_assets=20000)
        assert abs(roa - 4.0) < 0.01

    def test_calculate_roa_zero_assets(self):
        """ROA 资产为零"""
        roa = calculate_roa(net_income=800, total_assets=0)
        assert roa == 0.0

    def test_calculate_roa_nan(self):
        """ROA NaN 输入"""
        roa = calculate_roa(net_income=np.nan, total_assets=20000)
        assert roa == 0.0

    def test_calculate_gross_margin_normal(self):
        """毛利率正常计算"""
        margin = calculate_gross_margin(revenue=10000, cogs=6000)
        assert abs(margin - 40.0) < 0.01

    def test_calculate_gross_margin_excellent(self):
        """毛利率优秀 (>30%)"""
        margin = calculate_gross_margin(revenue=10000, cogs=5000)
        assert margin == 50.0
        assert margin > 30  # 优秀标准

    def test_calculate_gross_margin_outstanding(self):
        """毛利率极佳 (>50%)"""
        margin = calculate_gross_margin(revenue=10000, cogs=4000)
        assert margin == 60.0
        assert margin > 50  # 极佳标准

    def test_calculate_gross_margin_zero_revenue(self):
        """毛利率收入为零"""
        margin = calculate_gross_margin(revenue=0, cogs=6000)
        assert margin == 0.0

    def test_calculate_gross_margin_negative_profit(self):
        """毛利率负值（成本大于收入）"""
        margin = calculate_gross_margin(revenue=5000, cogs=6000)
        assert margin == 0.0  # 负毛利截断为0

    def test_calculate_gross_margin_nan_cogs(self):
        """毛利率 COGS 为 NaN"""
        margin = calculate_gross_margin(revenue=10000, cogs=np.nan)
        assert margin == 0.0

    def test_calculate_net_margin_normal(self):
        """净利率正常计算"""
        margin = calculate_net_margin(net_income=1000, revenue=10000)
        assert abs(margin - 10.0) < 0.01

    def test_calculate_net_margin_high_quality(self):
        """净利率优质 (>10%)"""
        margin = calculate_net_margin(net_income=1500, revenue=10000)
        assert margin == 15.0
        assert margin > 10  # 优质标准

    def test_calculate_net_margin_zero_revenue(self):
        """净利率收入为零"""
        margin = calculate_net_margin(net_income=1000, revenue=0)
        assert margin == 0.0

    def test_calculate_net_margin_nan(self):
        """净利率 NaN 输入"""
        margin = calculate_net_margin(net_income=np.nan, revenue=10000)
        assert margin == 0.0


class TestGrowth:
    """成长能力指标测试"""

    def test_calculate_revenue_growth_normal(self):
        """营收增速正常计算"""
        growth = calculate_revenue_growth(current_revenue=12000, previous_revenue=10000)
        assert abs(growth - 20.0) < 0.01

    def test_calculate_revenue_growth_negative(self):
        """营收增速下降"""
        growth = calculate_revenue_growth(current_revenue=8000, previous_revenue=10000)
        assert growth == -20.0

    def test_calculate_revenue_growth_zero_previous(self):
        """营收增速上期为0"""
        growth = calculate_revenue_growth(current_revenue=12000, previous_revenue=0)
        assert growth == 0.0

    def test_calculate_revenue_growth_nan(self):
        """营收增速 NaN"""
        growth = calculate_revenue_growth(current_revenue=np.nan, previous_revenue=10000)
        assert growth == 0.0

    def test_calculate_profit_growth_normal(self):
        """利润增速正常计算"""
        growth = calculate_profit_growth(current_profit=1200, previous_profit=1000)
        assert abs(growth - 20.0) < 0.01

    def test_calculate_profit_growth_negative(self):
        """利润增速下降"""
        growth = calculate_profit_growth(current_profit=800, previous_profit=1000)
        assert growth == -20.0

    def test_calculate_profit_growth_zero_previous(self):
        """利润增速上期为0"""
        growth = calculate_profit_growth(current_profit=1200, previous_profit=0)
        assert growth == 0.0

    def test_calculate_growth_score_normal(self):
        """成长评分正常计算"""
        score = calculate_growth_score(revenue_growth=20, profit_growth=15, roe=12)
        # 营收: 20*2.5*0.4=20, 利润: 15*2.5*0.4=15, ROE: (12-5)*10*0.2=14
        assert abs(score - 49.0) < 0.1

    def test_calculate_growth_score_high_growth(self):
        """成长评分高增长公司"""
        score = calculate_growth_score(revenue_growth=50, profit_growth=40, roe=20)
        # 营收: min(50*2.5,100)*0.4=40, 利润: min(40*2.5,100)*0.4=40, ROE: min((20-5)*10,100)*0.2=20
        # total = 40 + 40 + 20 = 100
        assert score == 100.0

    def test_calculate_growth_score_zero_growth(self):
        """成长评分零增长"""
        score = calculate_growth_score(revenue_growth=0, profit_growth=0, roe=5)
        assert score == 0.0

    def test_calculate_growth_score_negative_growth(self):
        """成长评分负增长"""
        score = calculate_growth_score(revenue_growth=-10, profit_growth=-5, roe=3)
        # 负增长截断为0
        assert score == 0.0

    def test_calculate_growth_score_nan_values(self):
        """成长评分 NaN 输入"""
        score = calculate_growth_score(revenue_growth=np.nan, profit_growth=np.nan, roe=np.nan)
        assert score == 0.0


class TestValuation:
    """估值指标测试"""

    def test_calculate_pe_normal(self):
        """市盈率正常计算"""
        pe = calculate_pe(price=50, eps=2.5)
        assert abs(pe - 20.0) < 0.01

    def test_calculate_pe_zero_eps(self):
        """市盈率 EPS 为零"""
        pe = calculate_pe(price=50, eps=0)
        assert pe == 0.0

    def test_calculate_pe_negative_eps(self):
        """市盈率负 EPS"""
        pe = calculate_pe(price=50, eps=-2.5)
        assert pe == -20.0

    def test_calculate_pe_nan(self):
        """市盈率 NaN"""
        pe = calculate_pe(price=np.nan, eps=2.5)
        assert pe == 0.0

    def test_calculate_pe_zero_price(self):
        """市盈率零价格"""
        pe = calculate_pe(price=0, eps=2.5)
        assert pe == 0.0

    def test_calculate_pb_normal(self):
        """市净率正常计算"""
        pb = calculate_pb(price=10, book_value_per_share=8)
        assert abs(pb - 1.25) < 0.01

    def test_calculate_pb_below_one(self):
        """市净率破净 (<1)"""
        pb = calculate_pb(price=8, book_value_per_share=10)
        assert abs(pb - 0.8) < 0.01
        assert pb < 1

    def test_calculate_pb_zero_book_value(self):
        """市净率净资产为零"""
        pb = calculate_pb(price=10, book_value_per_share=0)
        assert pb == 0.0

    def test_calculate_pb_nan(self):
        """市净率 NaN"""
        pb = calculate_pb(price=np.nan, book_value_per_share=8)
        assert pb == 0.0

    def test_calculate_ps_normal(self):
        """市销率正常计算"""
        ps = calculate_ps(price=50, revenue_per_share=10)
        assert abs(ps - 5.0) < 0.01

    def test_calculate_ps_zero_revenue(self):
        """市销率营收为零"""
        ps = calculate_ps(price=50, revenue_per_share=0)
        assert ps == 0.0

    def test_calculate_ps_nan(self):
        """市销率 NaN"""
        ps = calculate_ps(price=np.nan, revenue_per_share=10)
        assert ps == 0.0

    def test_calculate_valuation_score_normal(self):
        """估值评分正常计算"""
        score = calculate_valuation_score(pe=15, pb=1.2, ps=3, industry_pe=20)
        # PE: 20/15 = 1.33, ratio = 0.83, (0.83-0.5)*66.67 = 22, *0.5 = 11
        # PB: (5-1.2)/4*100 = 95, *0.25 = 23.75
        # PS: (10-3)/8*100 = 87.5, *0.25 = 21.875
        assert score > 0
        assert score < 100

    def test_calculate_valuation_score_low_pe(self):
        """估值评分低 PE（低估）"""
        score = calculate_valuation_score(pe=10, pb=1, ps=2, industry_pe=20)
        # PE: 20/10 = 2, ratio = 1.5, (1.5-0.5)*66.67 = 66.67, *0.5 = 33.33
        # PB: (5-1)/4*100 = 100, *0.25 = 25
        # PS: (10-2)/8*100 = 100, *0.25 = 25
        assert score > 50  # 低估应该得高分

    def test_calculate_valuation_score_high_pe(self):
        """估值评分高 PE（高估）"""
        score = calculate_valuation_score(pe=60, pb=5, ps=10, industry_pe=20)
        # PE: 20/60 = 0.33, ratio = -0.17, 0
        # PB: (5-5)/4*100 = 0
        # PS: (10-10)/8*100 = 0
        assert score < 20  # 高估应该得低分

    def test_calculate_valuation_score_nan_values(self):
        """估值评分 NaN 输入"""
        score = calculate_valuation_score(pe=np.nan, pb=np.nan, ps=np.nan, industry_pe=20)
        assert score > 0  # 亏损公司有基础分


class TestFinancialHealth:
    """财务健康指标测试"""

    def test_calculate_debt_ratio_normal(self):
        """资产负债率正常计算"""
        ratio = calculate_debt_ratio(total_liabilities=4000, total_assets=10000)
        assert abs(ratio - 40.0) < 0.01

    def test_calculate_debt_ratio_excellent(self):
        """资产负债率优秀 (<50%)"""
        ratio = calculate_debt_ratio(total_liabilities=3000, total_assets=10000)
        assert ratio == 30.0
        assert ratio < 50  # 优秀标准

    def test_calculate_debt_ratio_high_risk(self):
        """资产负债率高风险 (>70%)"""
        ratio = calculate_debt_ratio(total_liabilities=8000, total_assets=10000)
        assert ratio == 80.0
        assert ratio > 70  # 高风险标准

    def test_calculate_debt_ratio_zero_assets(self):
        """资产负债率资产为零"""
        ratio = calculate_debt_ratio(total_liabilities=4000, total_assets=0)
        assert ratio == 0.0

    def test_calculate_debt_ratio_nan(self):
        """资产负债率 NaN"""
        ratio = calculate_debt_ratio(total_liabilities=np.nan, total_assets=10000)
        assert ratio == 0.0

    def test_calculate_debt_ratio_negative_liabilities(self):
        """资产负债率负负债"""
        ratio = calculate_debt_ratio(total_liabilities=-1000, total_assets=10000)
        assert ratio == 0.0

    def test_calculate_current_ratio_normal(self):
        """流动比率正常计算"""
        ratio = calculate_current_ratio(current_assets=5000, current_liabilities=2500)
        assert abs(ratio - 2.0) < 0.01

    def test_calculate_current_ratio_excellent(self):
        """流动比率优秀 (>2)"""
        ratio = calculate_current_ratio(current_assets=6000, current_liabilities=2500)
        assert ratio == 2.4
        assert ratio > 2

    def test_calculate_current_ratio_risk(self):
        """流动比率风险 (<1)"""
        ratio = calculate_current_ratio(current_assets=2000, current_liabilities=3000)
        assert abs(ratio - 0.667) < 0.01
        assert ratio < 1

    def test_calculate_current_ratio_zero_liabilities(self):
        """流动比率负债为零"""
        ratio = calculate_current_ratio(current_assets=5000, current_liabilities=0)
        assert ratio == 0.0

    def test_calculate_current_ratio_nan(self):
        """流动比率 NaN"""
        ratio = calculate_current_ratio(current_assets=np.nan, current_liabilities=2500)
        assert ratio == 0.0

    def test_calculate_current_ratio_negative_assets(self):
        """流动比率为负资产"""
        ratio = calculate_current_ratio(current_assets=-1000, current_liabilities=2500)
        assert ratio == 0.0

    def test_calculate_cash_flow_ratio_normal(self):
        """现金流比率正常计算"""
        ratio = calculate_cash_flow_ratio(operating_cash_flow=1200, net_income=1000)
        assert abs(ratio - 1.2) < 0.01

    def test_calculate_cash_flow_ratio_high_quality(self):
        """现金流比率优质 (>1)"""
        ratio = calculate_cash_flow_ratio(operating_cash_flow=1500, net_income=1000)
        assert ratio == 1.5
        assert ratio > 1

    def test_calculate_cash_flow_ratio_suspicious(self):
        """现金流比率可疑 (<0.5)"""
        ratio = calculate_cash_flow_ratio(operating_cash_flow=400, net_income=1000)
        assert ratio == 0.4
        assert ratio < 0.5

    def test_calculate_cash_flow_ratio_negative(self):
        """现金流比率为负"""
        ratio = calculate_cash_flow_ratio(operating_cash_flow=-500, net_income=1000)
        assert ratio == -0.5

    def test_calculate_cash_flow_ratio_zero_income(self):
        """现金流比率净利润为零"""
        ratio = calculate_cash_flow_ratio(operating_cash_flow=1200, net_income=0)
        assert ratio == 0.0

    def test_calculate_cash_flow_ratio_nan(self):
        """现金流比率 NaN"""
        ratio = calculate_cash_flow_ratio(operating_cash_flow=np.nan, net_income=1000)
        assert ratio == 0.0

    def test_assess_financial_health_excellent(self):
        """财务健康评估优秀"""
        result = assess_financial_health(
            total_liabilities=2000,
            total_assets=10000,
            current_assets=6000,
            current_liabilities=2500,
            operating_cash_flow=1500,
            net_income=1000
        )
        assert isinstance(result, FinancialHealthResult)
        assert result.level in ["excellent", "good", "fair", "poor"]
        assert 0 <= result.debt_score <= 100
        assert 0 <= result.liquidity_score <= 100
        assert 0 <= result.cash_flow_score <= 100

    def test_assess_financial_health_good(self):
        """财务健康评估良好"""
        result = assess_financial_health(
            total_liabilities=4000,
            total_assets=10000,
            current_assets=3500,
            current_liabilities=2500,
            operating_cash_flow=900,
            net_income=1000
        )
        # debt_ratio=40%, debt_score=100, current_ratio=1.4, liquidity_score=40,
        # cash_flow_ratio=0.9, cash_flow_score=75
        # overall = 100*0.4 + 40*0.3 + 75*0.3 = 40 + 12 + 22.5 = 74.5
        assert result.level == "good"
        assert result.overall_score >= 60
        assert result.overall_score < 80

    def test_assess_financial_health_poor(self):
        """财务健康评估差"""
        result = assess_financial_health(
            total_liabilities=9000,
            total_assets=10000,
            current_assets=1000,
            current_liabilities=3000,
            operating_cash_flow=-500,
            net_income=1000
        )
        assert result.level == "poor"
        assert result.overall_score < 40

    def test_assess_financial_health_boundaries(self):
        """财务健康评估边界值"""
        # 资产负债率刚好50%（及格线）
        result = assess_financial_health(
            total_liabilities=5000,
            total_assets=10000,
            current_assets=2000,
            current_liabilities=2000,
            operating_cash_flow=1000,
            net_income=1000
        )
        assert result.debt_ratio == 50.0
        assert result.current_ratio == 1.0


class TestCompositeScore:
    """综合评分测试"""

    def test_composite_fundamental_score_normal(self):
        """综合评分正常计算"""
        result = composite_fundamental_score(
            roe=15, pe=20, pb=2, ps=4,
            revenue_growth=20, profit_growth=15,
            debt_ratio=40, current_ratio=2, cash_flow_ratio=1.2,
            analyst_score=70
        )
        assert isinstance(result, dict)
        assert "composite_score" in result
        assert "roe_score" in result
        assert "valuation_score" in result
        assert "growth_score" in result
        assert "health_score" in result
        assert "analyst_score" in result
        assert 0 <= result["composite_score"] <= 100

    def test_composite_fundamental_score_high_quality(self):
        """综合评分优质公司"""
        result = composite_fundamental_score(
            roe=20, pe=15, pb=1.5, ps=3,
            revenue_growth=30, profit_growth=25,
            debt_ratio=30, current_ratio=2.5, cash_flow_ratio=1.5,
            analyst_score=80
        )
        assert result["composite_score"] > 60
        assert result["health_level"] in ["excellent", "good"]

    def test_composite_fundamental_score_low_quality(self):
        """综合评分劣质公司"""
        result = composite_fundamental_score(
            roe=3, pe=100, pb=8, ps=15,
            revenue_growth=-10, profit_growth=-15,
            debt_ratio=85, current_ratio=0.5, cash_flow_ratio=0.2,
            analyst_score=20
        )
        assert result["composite_score"] < 40

    def test_composite_fundamental_score_analyst_default(self):
        """综合评分分析师默认50分"""
        result = composite_fundamental_score(
            roe=15, pe=20, pb=2, ps=4,
            revenue_growth=20, profit_growth=15,
            debt_ratio=40, current_ratio=2, cash_flow_ratio=1.2
            # 未提供 analyst_score，使用默认值 50
        )
        assert result["analyst_score"] == 50.0

    def test_composite_fundamental_score_analyst_clamped(self):
        """综合评分分析师分数截断"""
        result = composite_fundamental_score(
            roe=15, pe=20, pb=2, ps=4,
            revenue_growth=20, profit_growth=15,
            debt_ratio=40, current_ratio=2, cash_flow_ratio=1.2,
            analyst_score=150  # 超出范围
        )
        assert result["analyst_score"] == 50  # 截断到 50

    def test_composite_fundamental_score_returns_all_fields(self):
        """综合评分返回所有字段"""
        result = composite_fundamental_score(
            roe=15, pe=20, pb=2, ps=4,
            revenue_growth=20, profit_growth=15,
            debt_ratio=40, current_ratio=2, cash_flow_ratio=1.2,
            analyst_score=70
        )
        assert "roe" in result
        assert "pe" in result
        assert "pb" in result
        assert "ps" in result
        assert "revenue_growth" in result
        assert "profit_growth" in result
        assert "debt_ratio" in result
        assert "current_ratio" in result
        assert "cash_flow_ratio" in result
        assert "health_level" in result


class TestEdgeCases:
    """边界条件测试"""

    def test_all_zeros(self):
        """全部零值输入"""
        roe = calculate_roe(0, 0)
        roa = calculate_roa(0, 0)
        gross = calculate_gross_margin(0, 0)
        net = calculate_net_margin(0, 0)
        rev_growth = calculate_revenue_growth(0, 0)
        prof_growth = calculate_profit_growth(0, 0)
        pe = calculate_pe(0, 0)
        pb = calculate_pb(0, 0)
        ps = calculate_ps(0, 0)
        debt = calculate_debt_ratio(0, 0)
        current = calculate_current_ratio(0, 0)
        cash_flow = calculate_cash_flow_ratio(0, 0)

        # 全部应该是 0 或有定义的默认值
        assert roe == 0.0
        assert roa == 0.0
        assert gross == 0.0
        assert net == 0.0
        assert rev_growth == 0.0
        assert prof_growth == 0.0
        assert pe == 0.0
        assert pb == 0.0
        assert ps == 0.0
        assert debt == 0.0
        assert current == 0.0
        assert cash_flow == 0.0

    def test_extreme_values(self):
        """极端值输入"""
        # 极大值
        roe = calculate_roe(net_income=1e15, shareholders_equity=1e12)
        assert roe > 0

        # 极小值
        roe = calculate_roe(net_income=1e-15, shareholders_equity=1e-12)
        assert roe > 0

    def test_negative_values(self):
        """负值输入"""
        roe = calculate_roe(-1000, 10000)
        assert roe < 0

        rev_growth = calculate_revenue_growth(8000, 10000)
        assert rev_growth < 0

        debt = calculate_debt_ratio(-1000, 10000)
        assert debt == 0.0  # 负负债截断为0

    def test_growth_score_caps_at_100(self):
        """成长评分上限100"""
        score = calculate_growth_score(revenue_growth=100, profit_growth=100, roe=30)
        assert score <= 100

    def test_valuation_score_caps(self):
        """估值评分为零或满分"""
        # 极度低估
        score_low = calculate_valuation_score(pe=5, pb=0.5, ps=1, industry_pe=20)
        assert score_low > 80

        # 极度高估
        score_high = calculate_valuation_score(pe=200, pb=10, ps=20, industry_pe=20)
        assert score_high < 10


class TestDataIntegrity:
    """数据完整性测试"""

    def test_no_import_errors(self):
        """无导入错误"""
        from src.data.indicators import fundamental
        assert fundamental is not None

    def test_all_exports_available(self):
        """所有导出可用"""
        from src.data.indicators.fundamental import (
            calculate_roe, calculate_roa, calculate_gross_margin, calculate_net_margin,
            calculate_revenue_growth, calculate_profit_growth, calculate_growth_score,
            calculate_pe, calculate_pb, calculate_ps, calculate_valuation_score,
            calculate_debt_ratio, calculate_current_ratio, calculate_cash_flow_ratio,
            assess_financial_health, FinancialHealthResult,
            composite_fundamental_score
        )
        assert callable(calculate_roe)
        assert callable(calculate_roa)
        assert callable(calculate_gross_margin)
        assert callable(calculate_net_margin)
        assert callable(calculate_revenue_growth)
        assert callable(calculate_profit_growth)
        assert callable(calculate_growth_score)
        assert callable(calculate_pe)
        assert callable(calculate_pb)
        assert callable(calculate_ps)
        assert callable(calculate_valuation_score)
        assert callable(calculate_debt_ratio)
        assert callable(calculate_current_ratio)
        assert callable(calculate_cash_flow_ratio)
        assert callable(assess_financial_health)
        assert callable(composite_fundamental_score)

    def test_result_types(self):
        """返回类型正确"""
        result = assess_financial_health(
            total_liabilities=3000, total_assets=10000,
            current_assets=5000, current_liabilities=2500,
            operating_cash_flow=1200, net_income=1000
        )
        assert isinstance(result, FinancialHealthResult)
        assert isinstance(result.debt_ratio, float)
        assert isinstance(result.current_ratio, float)
        assert isinstance(result.cash_flow_ratio, float)
        assert isinstance(result.level, str)

    def test_composite_returns_dict(self):
        """综合评分返回字典"""
        result = composite_fundamental_score(
            roe=15, pe=20, pb=2, ps=4,
            revenue_growth=20, profit_growth=15,
            debt_ratio=40, current_ratio=2, cash_flow_ratio=1.2
        )
        assert isinstance(result, dict)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
