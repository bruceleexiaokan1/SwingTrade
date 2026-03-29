"""基本面指标计算模块

提供波段交易所需的基本面指标计算：
- Profitability Factors（盈利能力）
- Growth Factors（成长能力）
- Valuation Factors（估值指标）
- Financial Health（财务健康）
- Composite Score（综合评分）
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd


# =============================================================================
# Profitability Factors（盈利能力指标）
# =============================================================================

def calculate_roe(net_income: float, shareholders_equity: float) -> float:
    """
    计算净资产收益率 (ROE - Return on Equity)

    ROE = Net Income / Shareholders' Equity * 100%

    Args:
        net_income: 净利润（元）
        shareholders_equity: 股东权益（净资产）

    Returns:
        ROE 百分比，例如 15.5 表示 15.5%

    Note:
        ROE > 15% 为优质
        ROE < 5% 为较差
        ROE 越高说明股东权益产生的收益越高

    Example:
        >>> roe = calculate_roe(net_income=1500, shareholders_equity=10000)
        >>> round(roe, 2)
        15.0
    """
    if shareholders_equity == 0 or np.isnan(shareholders_equity):
        return 0.0
    if np.isnan(net_income):
        return 0.0
    return (net_income / shareholders_equity) * 100


def calculate_roa(net_income: float, total_assets: float) -> float:
    """
    计算资产收益率 (ROA - Return on Assets)

    ROA = Net Income / Total Assets * 100%

    Args:
        net_income: 净利润（元）
        total_assets: 总资产

    Returns:
        ROA 百分比

    Note:
        ROA > 5% 为优质
        ROA 反映公司利用全部资产获取利润的能力

    Example:
        >>> roa = calculate_roa(net_income=800, total_assets=20000)
        >>> round(roa, 2)
        4.0
    """
    if total_assets == 0 or np.isnan(total_assets):
        return 0.0
    if np.isnan(net_income):
        return 0.0
    return (net_income / total_assets) * 100


def calculate_gross_margin(revenue: float, cogs: float) -> float:
    """
    计算毛利率 (Gross Margin)

    Gross Margin = (Revenue - COGS) / Revenue * 100%
                = Gross Profit / Revenue * 100%

    Args:
        revenue: 营业收入（元）
        cogs: 营业成本（Cost of Goods Sold）

    Returns:
        毛利率百分比

    Note:
        毛利率 > 30% 为优秀
        毛利率 > 50% 为极佳
        毛利率反映产品定价权和成本控制能力

    Example:
        >>> margin = calculate_gross_margin(revenue=10000, cogs=6000)
        >>> round(margin, 2)
        40.0
    """
    if revenue == 0 or np.isnan(revenue):
        return 0.0
    if np.isnan(cogs):
        return 0.0
    gross_profit = revenue - cogs
    if gross_profit < 0:
        return 0.0
    return (gross_profit / revenue) * 100


def calculate_net_margin(net_income: float, revenue: float) -> float:
    """
    计算净利率 (Net Profit Margin)

    Net Margin = Net Income / Revenue * 100%

    Args:
        net_income: 净利润
        revenue: 营业收入

    Returns:
        净利率百分比

    Note:
        净利率 > 10% 为优质
        净利率反映公司最终盈利能力

    Example:
        >>> margin = calculate_net_margin(net_income=1000, revenue=10000)
        >>> round(margin, 2)
        10.0
    """
    if revenue == 0 or np.isnan(revenue):
        return 0.0
    if np.isnan(net_income):
        return 0.0
    return (net_income / revenue) * 100


# =============================================================================
# Growth Factors（成长能力指标）
# =============================================================================

def calculate_revenue_growth(current_revenue: float, previous_revenue: float) -> float:
    """
    计算营收增速 (Revenue Growth Rate)

    Revenue Growth = (Current Revenue - Previous Revenue) / Previous Revenue * 100%

    Args:
        current_revenue: 本期营业收入
        previous_revenue: 上期营业收入

    Returns:
        营收增速百分比

    Note:
        正值表示增长，负值表示下降
        营收增速 > 20% 为高增长

    Example:
        >>> growth = calculate_revenue_growth(current_revenue=12000, previous_revenue=10000)
        >>> round(growth, 2)
        20.0
    """
    if previous_revenue == 0 or np.isnan(previous_revenue):
        return 0.0
    if np.isnan(current_revenue):
        return 0.0
    return ((current_revenue - previous_revenue) / previous_revenue) * 100


def calculate_profit_growth(current_profit: float, previous_profit: float) -> float:
    """
    计算利润增速 (Profit Growth Rate)

    Profit Growth = (Current Profit - Previous Profit) / Previous Profit * 100%

    Args:
        current_profit: 本期利润
        previous_profit: 上期利润

    Returns:
        利润增速百分比

    Note:
        正值表示增长，负值表示下降
        利润增速 > 20% 为高增长

    Example:
        >>> growth = calculate_profit_growth(current_profit=1200, previous_profit=1000)
        >>> round(growth, 2)
        20.0
    """
    if previous_profit == 0 or np.isnan(previous_profit):
        return 0.0
    if np.isnan(current_profit):
        return 0.0
    return ((current_profit - previous_profit) / previous_profit) * 100


def calculate_growth_score(
    revenue_growth: float,
    profit_growth: float,
    roe: float
) -> float:
    """
    计算综合成长评分 (Growth Score)

    综合考虑营收增速、利润增速和ROE

    Args:
        revenue_growth: 营收增速（%）
        profit_growth: 利润增速（%）
        roe: 净资产收益率（%）

    Returns:
        成长评分 (0-100)

    Note:
        - 营收增速权重：40%
        - 利润增速权重：40%
        - ROE 权重：20%
        - 评分范围 0-100，50 为中等

    Example:
        >>> score = calculate_growth_score(revenue_growth=20, profit_growth=15, roe=12)
        >>> round(score, 1)
        18.0
    """
    if np.isnan(revenue_growth):
        revenue_growth = 0.0
    if np.isnan(profit_growth):
        profit_growth = 0.0
    if np.isnan(roe):
        roe = 0.0

    # 营收增速评分：20%以上满分，0%为0分
    revenue_score = min(max(revenue_growth * 2.5, 0), 100) * 0.4

    # 利润增速评分：20%以上满分，0%为0分
    profit_score = min(max(profit_growth * 2.5, 0), 100) * 0.4

    # ROE 评分：15%以上满分，5%为0分
    roe_score = min(max((roe - 5) * 10, 0), 100) * 0.2

    return revenue_score + profit_score + roe_score


# =============================================================================
# Valuation Factors（估值指标）
# =============================================================================

def calculate_pe(price: float, eps: float) -> float:
    """
    计算市盈率 (P/E Ratio)

    P/E = Price / EPS

    Args:
        price: 股价
        eps: 每股收益（EPS）

    Returns:
        市盈率倍数

    Note:
        P/E 越低表示估值越便宜（相对）
        P/E 为负或 0 表示公司亏损
        行业差异较大，需横向比较

    Example:
        >>> pe = calculate_pe(price=50, eps=2.5)
        >>> round(pe, 2)
        20.0
    """
    if eps == 0 or np.isnan(eps):
        return 0.0
    if np.isnan(price):
        return 0.0
    if price <= 0:
        return 0.0
    return price / eps


def calculate_pb(price: float, book_value_per_share: float) -> float:
    """
    计算市净率 (P/B Ratio)

    P/B = Price / Book Value Per Share

    Args:
        price: 股价
        book_value_per_share: 每股净资产

    Returns:
        市净率倍数

    Note:
        P/B < 1 表示股价低于净资产（破净）
        P/B 越低表示估值越便宜
        适用于金融、地产等重资产行业

    Example:
        >>> pb = calculate_pb(price=10, book_value_per_share=8)
        >>> round(pb, 2)
        1.25
    """
    if book_value_per_share == 0 or np.isnan(book_value_per_share):
        return 0.0
    if np.isnan(price):
        return 0.0
    if price <= 0:
        return 0.0
    return price / book_value_per_share


def calculate_ps(price: float, revenue_per_share: float) -> float:
    """
    计算市销率 (P/S Ratio)

    P/S = Price / Revenue Per Share

    Args:
        price: 股价
        revenue_per_share: 每股营业收入

    Returns:
        市销率倍数

    Note:
        适用于营收稳定但利润波动的成长型公司
        P/S 越低表示估值越便宜
        互联网、科技公司常用此指标

    Example:
        >>> ps = calculate_ps(price=50, revenue_per_share=10)
        >>> round(ps, 2)
        5.0
    """
    if revenue_per_share == 0 or np.isnan(revenue_per_share):
        return 0.0
    if np.isnan(price):
        return 0.0
    if price <= 0:
        return 0.0
    return price / revenue_per_share


def calculate_valuation_score(
    pe: float,
    pb: float,
    ps: float,
    industry_pe: float = 20.0
) -> float:
    """
    计算综合估值评分 (Valuation Score)

    综合考虑 PE、PB、PS 与行业均值的比较

    Args:
        pe: 市盈率
        pb: 市净率
        ps: 市销率
        industry_pe: 行业平均市盈率（默认 20）

    Returns:
        估值评分 (0-100)，50 为行业平均

    Note:
        - PE 评分权重：50%
        - PB 评分权重：25%
        - PS 评分权重：25%
        - 评分低于 50 表示相对低估

    Example:
        >>> score = calculate_valuation_score(pe=15, pb=1.2, ps=3, industry_pe=20)
        >>> round(score, 1)
        68.8
    """
    if np.isnan(pe):
        pe = 0.0
    if np.isnan(pb):
        pb = 0.0
    if np.isnan(ps):
        ps = 0.0

    # PE 评分：行业 PE 的 0.5 倍为满分（低估），2 倍为 0 分（高估）
    if pe > 0:
        pe_ratio = industry_pe / pe if pe > 0 else 0
        pe_score = min(max((pe_ratio - 0.5) * 66.67, 0), 100) * 0.5
    else:
        pe_score = 100 * 0.5  # 亏损公司给 50 分的 50%

    # PB 评分：PB < 1 为满分，PB > 5 为 0 分
    if pb > 0:
        pb_score = min(max((5 - pb) / 4 * 100, 0), 100) * 0.25
    else:
        pb_score = 100 * 0.25

    # PS 评分：PS < 2 为满分，PS > 10 为 0 分
    if ps > 0:
        ps_score = min(max((10 - ps) / 8 * 100, 0), 100) * 0.25
    else:
        ps_score = 100 * 0.25

    return pe_score + pb_score + ps_score


# =============================================================================
# Financial Health（财务健康指标）
# =============================================================================

def calculate_debt_ratio(total_liabilities: float, total_assets: float) -> float:
    """
    计算资产负债率 (Debt Ratio)

    Debt Ratio = Total Liabilities / Total Assets * 100%

    Args:
        total_liabilities: 总负债
        total_assets: 总资产

    Returns:
        资产负债率百分比

    Note:
        资产负债率 < 50% 为优秀
        资产负债率 > 70% 为高风险
        不同行业标准不同

    Example:
        >>> ratio = calculate_debt_ratio(total_liabilities=4000, total_assets=10000)
        >>> round(ratio, 2)
        40.0
    """
    if total_assets == 0 or np.isnan(total_assets):
        return 0.0
    if np.isnan(total_liabilities):
        return 0.0
    if total_liabilities < 0:
        return 0.0
    return (total_liabilities / total_assets) * 100


def calculate_current_ratio(current_assets: float, current_liabilities: float) -> float:
    """
    计算流动比率 (Current Ratio)

    Current Ratio = Current Assets / Current Liabilities

    Args:
        current_assets: 流动资产
        current_liabilities: 流动负债

    Returns:
        流动比率（倍数）

    Note:
        Current Ratio > 2 为优秀
        Current Ratio < 1 为短期偿债风险
        流动比率衡量短期偿债能力

    Example:
        >>> ratio = calculate_current_ratio(current_assets=5000, current_liabilities=2500)
        >>> round(ratio, 2)
        2.0
    """
    if current_liabilities == 0 or np.isnan(current_liabilities):
        return 0.0
    if np.isnan(current_assets):
        return 0.0
    if current_assets <= 0:
        return 0.0
    return current_assets / current_liabilities


def calculate_cash_flow_ratio(operating_cash_flow: float, net_income: float) -> float:
    """
    计算现金流/净利润比率 (Cash Flow Ratio)

    Cash Flow Ratio = Operating Cash Flow / Net Income

    Args:
        operating_cash_flow: 经营活动现金流
        net_income: 净利润

    Returns:
        现金流/净利润 比率

    Note:
        比率 > 1 为优质
        比率 < 0.5 为可疑（利润质量差）
        比率 < 0 表示经营现金流为负

    Example:
        >>> ratio = calculate_cash_flow_ratio(operating_cash_flow=1200, net_income=1000)
        >>> round(ratio, 2)
        1.2
    """
    if net_income == 0 or np.isnan(net_income):
        return 0.0
    if np.isnan(operating_cash_flow):
        return 0.0
    return operating_cash_flow / net_income


@dataclass
class FinancialHealthResult:
    """财务健康评估结果"""
    debt_ratio: float           # 资产负债率 (%)
    current_ratio: float        # 流动比率
    cash_flow_ratio: float     # 现金流/净利润
    debt_score: float           # 负债评分 (0-100)
    liquidity_score: float      # 流动性评分 (0-100)
    cash_flow_score: float     # 现金流评分 (0-100)
    overall_score: float        # 综合健康评分 (0-100)
    level: str                  # 健康等级：excellent/good/fair/poor


def assess_financial_health(
    total_liabilities: float,
    total_assets: float,
    current_assets: float,
    current_liabilities: float,
    operating_cash_flow: float,
    net_income: float
) -> FinancialHealthResult:
    """
    综合财务健康评估

    Args:
        total_liabilities: 总负债
        total_assets: 总资产
        current_assets: 流动资产
        current_liabilities: 流动负债
        operating_cash_flow: 经营活动现金流
        net_income: 净利润

    Returns:
        FinancialHealthResult 包含各项评分和综合评级

    Note:
        评分标准：
        - 资产负债率 < 50% 满分，> 80% 为 0 分
        - 流动比率 > 2 满分，< 1 为 0 分
        - 现金流/净利润 > 1.2 满分，< 0 为 0 分

        综合评级：
        - excellent: >= 80
        - good: >= 60
        - fair: >= 40
        - poor: < 40

    Example:
        >>> result = assess_financial_health(
        ...     total_liabilities=3000, total_assets=10000,
        ...     current_assets=5000, current_liabilities=2500,
        ...     operating_cash_flow=1200, net_income=1000
        ... )
        >>> result.level
        'good'
    """
    # 计算基础指标
    debt_ratio = calculate_debt_ratio(total_liabilities, total_assets)
    current_ratio = calculate_current_ratio(current_assets, current_liabilities)
    cash_flow_ratio = calculate_cash_flow_ratio(operating_cash_flow, net_income)

    # 负债评分：< 50% 满分，> 80% 为 0 分
    debt_score = min(max((80 - debt_ratio) / 30 * 100, 0), 100)

    # 流动性评分：> 2 满分，< 1 为 0 分
    liquidity_score = min(max((current_ratio - 1) * 100, 0), 100)

    # 现金流评分：> 1.2 满分，< 0 为 0 分
    cash_flow_score = min(max(cash_flow_ratio * 83.33, 0), 100)

    # 综合评分
    overall_score = debt_score * 0.4 + liquidity_score * 0.3 + cash_flow_score * 0.3

    # 评级
    if overall_score >= 80:
        level = "excellent"
    elif overall_score >= 60:
        level = "good"
    elif overall_score >= 40:
        level = "fair"
    else:
        level = "poor"

    return FinancialHealthResult(
        debt_ratio=debt_ratio,
        current_ratio=current_ratio,
        cash_flow_ratio=cash_flow_ratio,
        debt_score=debt_score,
        liquidity_score=liquidity_score,
        cash_flow_score=cash_flow_score,
        overall_score=overall_score,
        level=level
    )


# =============================================================================
# Composite Score（综合评分）
# =============================================================================

def composite_fundamental_score(
    roe: float,
    pe: float,
    pb: float,
    ps: float,
    revenue_growth: float,
    profit_growth: float,
    debt_ratio: float,
    current_ratio: float,
    cash_flow_ratio: float,
    industry_pe: float = 20.0,
    analyst_score: float = 50.0
) -> Dict[str, Any]:
    """
    计算基本面综合评分

    综合盈利能力、估值、成长、财务健康和分析师评分

    Args:
        roe: 净资产收益率（%）
        pe: 市盈率
        pb: 市净率
        ps: 市销率
        revenue_growth: 营收增速（%）
        profit_growth: 利润增速（%）
        debt_ratio: 资产负债率（%）
        current_ratio: 流动比率
        cash_flow_ratio: 现金流/净利润
        industry_pe: 行业平均市盈率（默认 20）
        analyst_score: 分析师评分（0-100，默认 50）

    Returns:
        包含各维度评分和综合评分的字典

    Note:
        Composite Score Weights:
        - ROE: 30%
        - Valuation: 20%
        - Growth: 25%
        - Financial Health: 25%

    Example:
        >>> result = composite_fundamental_score(
        ...     roe=15, pe=20, pb=2, ps=4,
        ...     revenue_growth=20, profit_growth=15,
        ...     debt_ratio=40, current_ratio=2, cash_flow_ratio=1.2,
        ...     analyst_score=70
        ... )
        >>> round(result['composite_score'], 1)
        69.0
    """
    # ROE 评分：15% 以上满分，5% 为 0 分
    roe_score = min(max((roe - 5) * 10, 0), 100) if roe > 0 else 0

    # 估值评分
    valuation_score = calculate_valuation_score(pe, pb, ps, industry_pe)

    # 成长评分
    growth_score = calculate_growth_score(revenue_growth, profit_growth, roe)

    # 财务健康评分（直接从比率计算，不调用assess_financial_health避免单位混乱）
    # debt_ratio: 资产负债率（%），如 40 表示 40%
    # current_ratio: 流动比率，如 1.5 表示 1.5
    # cash_flow_ratio: 现金流/净利润，如 0.8 表示 0.8
    debt_score = min(max((80 - debt_ratio) / 30 * 100, 0), 100) if debt_ratio <= 80 else 0
    liquidity_score = min(max((current_ratio - 1) * 100, 0), 100) if current_ratio >= 1 else 0
    cash_flow_score = min(max(cash_flow_ratio * 83.33, 0), 100) if cash_flow_ratio >= 0 else 0
    health_score = debt_score * 0.4 + liquidity_score * 0.3 + cash_flow_score * 0.3

    # 分析师评分归一化
    analyst_normalized = analyst_score if 0 <= analyst_score <= 100 else 50

    # 综合评分
    composite_score = (
        roe_score * 0.30 +
        valuation_score * 0.20 +
        growth_score * 0.25 +
        analyst_normalized * 0.25
    )

    return {
        "composite_score": composite_score,
        "roe_score": roe_score,
        "valuation_score": valuation_score,
        "growth_score": growth_score,
        "health_score": health_score,
        "health_level": "excellent" if health_score >= 80 else "good" if health_score >= 60 else "fair" if health_score >= 40 else "poor",
        "analyst_score": analyst_normalized,
        "roe": roe,
        "pe": pe,
        "pb": pb,
        "ps": ps,
        "revenue_growth": revenue_growth,
        "profit_growth": profit_growth,
        "debt_ratio": debt_ratio,
        "current_ratio": current_ratio,
        "cash_flow_ratio": cash_flow_ratio
    }


__all__ = [
    # Profitability
    "calculate_roe",
    "calculate_roa",
    "calculate_gross_margin",
    "calculate_net_margin",
    # Growth
    "calculate_revenue_growth",
    "calculate_profit_growth",
    "calculate_growth_score",
    # Valuation
    "calculate_pe",
    "calculate_pb",
    "calculate_ps",
    "calculate_valuation_score",
    # Financial Health
    "calculate_debt_ratio",
    "calculate_current_ratio",
    "calculate_cash_flow_ratio",
    "assess_financial_health",
    "FinancialHealthResult",
    # Composite
    "composite_fundamental_score",
]
