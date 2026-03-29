"""事件驱动策略：A股特殊事件的量化规律

实现A股重要事件检测和信号生成：
- 财报预期差策略
- 限售股解禁检测
- 指数调样效应
- 大股东增减持分析
- 政策事件日历
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class EarningsSurpriseResult:
    """财报预期差结果"""
    date: str
    stock_code: str
    actual_eps: float
    consensus_eps: float
    surprise_pct: float
    category: str  # BEAT/MEET/MISS
    pre_earnings_return: float
    post_earnings_return: float
    is_significant: bool  # 偏离是否超过10%


@dataclass
class JiejinRiskResult:
    """解禁风险检测结果"""
    date: str
    stock_code: str
    unlock_date: str
    unlock_shares: float
    float_shares: float
    unlock_ratio: float
    holder_type: str  # 定增/IPO原始股/股权激励
    risk_score: int
    risk_level: str  # 高危/中危/低危/安全
    days_to_absorb: float
    recommended_action: str


@dataclass
class RebalanceResult:
    """指数调样结果"""
    date: str
    stock_code: str
    index_name: str
    direction: str  # 调入/调出
    passive_flow: float
    days_for_passive: float
    opportunity: str
    recommended_action: str


@dataclass
class ShareholderBuyResult:
    """大股东增持分析结果"""
    date: str
    stock_code: str
    total_buy_amount: float
    buy_count: int
    buy_ratio: float
    days_since_last_buy: int
    signal: str  # 强烈买入/买入/中性/观望
    recommended_action: str


@dataclass
class EventSignal:
    """综合事件信号"""
    date: str
    stock_code: str
    events: List[Dict]
    total_score: float
    action: str  # 买入/观望/回避
    confidence: float


# ==================== 财报预期差策略 ====================

def calculate_earnings_surprise(
    actual_eps: float,
    consensus_eps: float,
    pre_return: float = 0.0,
    post_return: float = 0.0
) -> EarningsSurpriseResult:
    """
    计算业绩超预期幅度

    Args:
        actual_eps: 实际EPS
        consensus_eps: 一致预期EPS
        pre_return: 财报前收益
        post_return: 财报后收益

    Returns:
        EarningsSurpriseResult
    """
    if consensus_eps == 0:
        surprise_pct = 0.0
    else:
        surprise_pct = (actual_eps - consensus_eps) / abs(consensus_eps) * 100

    # 分类
    if surprise_pct > 5:
        category = 'BEAT'
    elif surprise_pct < -5:
        category = 'MISS'
    else:
        category = 'MEET'

    # 显著判断：偏离超过10%
    is_significant = abs(surprise_pct) > 10

    return EarningsSurpriseResult(
        date='',
        stock_code='',
        actual_eps=actual_eps,
        consensus_eps=consensus_eps,
        surprise_pct=surprise_pct,
        category=category,
        pre_earnings_return=pre_return,
        post_earnings_return=post_return,
        is_significant=is_significant
    )


def earnings_trend_analysis(
    earnings_history: List[Dict],
    lookback_quarters: int = 8
) -> Dict:
    """
    业绩趋势分析

    Args:
        earnings_history: 历史业绩列表 [{'period', 'actual_eps', 'consensus_eps'}]
        lookback_quarters: 回看季度数

    Returns:
        趋势分析结果
    """
    if len(earnings_history) < 4:
        return {'status': '数据不足'}

    # 计算各期超预期幅度
    surprises = []
    for e in earnings_history[:lookback_quarters]:
        if e.get('consensus_eps', 0) != 0:
            surprise = (e['actual_eps'] - e['consensus_eps']) / abs(e['consensus_eps'])
            surprises.append(surprise * 100)  # 转为百分比

    if len(surprises) < 4:
        return {'status': '数据不足'}

    # 计算趋势
    recent_surprises = surprises[:4]
    older_surprises = surprises[4:] if len(surprises) > 4 else []

    avg_recent = np.mean(recent_surprises)
    avg_older = np.mean(older_surprises) if older_surprises else avg_recent

    if avg_recent > avg_older + 5:
        trend = '加速'
    elif avg_recent < avg_older - 5:
        trend = '减速'
    else:
        trend = '稳定'

    # 计算环比
    sequential = []
    for i in range(len(surprises) - 1):
        if surprises[i + 1] != 0:
            seq_growth = (surprises[i] - surprises[i + 1]) / abs(surprises[i + 1])
            sequential.append(seq_growth)

    avg_seq_growth = np.mean(sequential) if sequential else 0

    return {
        'recent_quarters': recent_surprises,
        'older_quarters': older_surprises,
        'trend': trend,
        'avg_surprise_recent': avg_recent,
        'avg_surprise_older': avg_older,
        'sequential_growth': avg_seq_growth,
        'interpretation': f'业绩{trend}，平均超预期{avg_recent:.1f}%'
    }


# ==================== 限售股解禁策略 ====================

def calculate_jiejin_risk(
    unlock_shares: float,
    float_shares: float,
    holder_type: str,
    avg_daily_volume: float,
    unlock_date: Optional[str] = None
) -> JiejinRiskResult:
    """
    计算解禁风险

    Args:
        unlock_shares: 解禁股数
        float_shares: 流通股本
        holder_type: 解禁股东类型（定增/IPO原始股/股权激励）
        avg_daily_volume: 日均成交量（股数）
        unlock_date: 解禁日期

    Returns:
        JiejinRiskResult
    """
    # 计算解禁比例
    unlock_ratio = unlock_shares / float_shares if float_shares > 0 else 0

    # 估算减持压力
    if holder_type == '定增':
        expected_sell_ratio = 0.3
    elif holder_type == 'IPO原始股':
        expected_sell_ratio = 0.5
    else:  # 股权激励
        expected_sell_ratio = 0.2

    potential_sell_shares = unlock_shares * expected_sell_ratio

    # 需要多少天消化
    days_to_absorb = potential_sell_shares / avg_daily_volume if avg_daily_volume > 0 else 999

    # 风险评分
    risk_score = 0
    if unlock_ratio > 0.3:
        risk_score += 3
    elif unlock_ratio > 0.15:
        risk_score += 2
    elif unlock_ratio > 0.05:
        risk_score += 1

    if holder_type in ['IPO原始股', '定增']:
        risk_score += 2

    if days_to_absorb > 20:
        risk_score += 2
    elif days_to_absorb > 10:
        risk_score += 1

    # 风险等级
    if risk_score >= 5:
        risk_level = '高危'
        action = '减持'
    elif risk_score >= 3:
        risk_level = '中危'
        action = '回避'
    elif risk_score >= 1:
        risk_level = '低危'
        action = '正常持有'
    else:
        risk_level = '安全'
        action = '正常持有'

    return JiejinRiskResult(
        date=unlock_date or '',
        stock_code='',
        unlock_date=unlock_date or '',
        unlock_shares=unlock_shares,
        float_shares=float_shares,
        unlock_ratio=unlock_ratio,
        holder_type=holder_type,
        risk_score=risk_score,
        risk_level=risk_level,
        days_to_absorb=days_to_absorb,
        recommended_action=action
    )


def scan_jiejin_calendar(
    unlock_data: List[Dict],
    current_date: str,
    lookforward_days: int = 30
) -> Tuple[List[JiejinRiskResult], List[JiejinRiskResult]]:
    """
    扫描解禁日历，识别高风险和机会

    Args:
        unlock_data: 解禁数据列表 [{'stock_code', 'unlock_date', 'unlock_shares', 'float_shares', 'holder_type'}]
        current_date: 当前日期
        lookforward_days: 向前查看天数

    Returns:
        (high_risk_stocks, opportunities)
    """
    current = datetime.strptime(current_date, '%Y-%m-%d')
    cutoff = current + timedelta(days=lookforward_days)

    high_risk = []
    opportunities = []

    for unlock in unlock_data:
        try:
            unlock_date = datetime.strptime(unlock['unlock_date'], '%Y-%m-%d')
        except (ValueError, KeyError):
            continue

        # 只看未来30天
        if not (current <= unlock_date <= cutoff):
            continue

        risk = calculate_jiejin_risk(
            unlock_shares=unlock.get('unlock_shares', 0),
            float_shares=unlock.get('float_shares', 1),
            holder_type=unlock.get('holder_type', '其他'),
            avg_daily_volume=unlock.get('avg_daily_volume', float('inf')),
            unlock_date=unlock['unlock_date']
        )
        risk.stock_code = unlock.get('stock_code', '')

        if risk.risk_level == '高危':
            high_risk.append(risk)
        elif risk.risk_level == '低危':
            opportunities.append(risk)

    return high_risk, opportunities


# ==================== 指数调样策略 ====================

def calculate_rebalance_effect(
    stock_code: str,
    index_name: str,
    direction: str,
    stock_weight: float,
    etf_aum: float,
    adv: float
) -> RebalanceResult:
    """
    计算指数调样的交易机会

    Args:
        stock_code: 股票代码
        index_name: 指数名称
        direction: 调入/调出
        stock_weight: 股票在指数中的权重
        etf_aum: 相关ETF总规模
        adv: 日均成交量（金额）

    Returns:
        RebalanceResult
    """
    # 估算被动买入/卖出金额
    passive_flow = etf_aum * stock_weight

    # 需要多少天被动买入
    days_for_passive = passive_flow / adv if adv > 0 else 999

    if direction == '调入':
        opportunity = '存在被动买入托底'
        action = '买入' if days_for_passive < 10 else '观望'
    else:
        opportunity = '被动卖出压力'
        action = '回避'

    return RebalanceResult(
        date='',
        stock_code=stock_code,
        index_name=index_name,
        direction=direction,
        passive_flow=passive_flow,
        days_for_passive=days_for_passive,
        opportunity=opportunity,
        recommended_action=action
    )


# ==================== 大股东增减持策略 ====================

def analyze_shareholder_buying(
    buy_records: List[Dict],
    market_cap: float,
    current_price: float,
    last_buy_date: Optional[str] = None
) -> ShareholderBuyResult:
    """
    分析股东增持行为

    Args:
        buy_records: 增持记录列表 [{'date', 'amount', 'shares'}]
        market_cap: 总市值
        current_price: 当前价格
        last_buy_date: 最后增持日期

    Returns:
        ShareholderBuyResult
    """
    if not buy_records:
        return ShareholderBuyResult(
            date=last_buy_date or '',
            stock_code='',
            total_buy_amount=0.0,
            buy_count=0,
            buy_ratio=0.0,
            days_since_last_buy=999,
            signal='无数据',
            recommended_action='观望'
        )

    # 计算总增持金额
    total_buy_amount = sum(r.get('amount', 0) for r in buy_records)

    # 计算增持次数
    buy_count = len(buy_records)

    # 计算增持市值占比
    buy_ratio = total_buy_amount / market_cap if market_cap > 0 else 0

    # 计算距最后增持天数
    if last_buy_date:
        try:
            last_date = datetime.strptime(last_buy_date, '%Y-%m-%d')
            days_since_last_buy = (datetime.now() - last_date).days
        except ValueError:
            days_since_last_buy = 999
    else:
        days_since_last_buy = 999

    # 信号判断
    if buy_ratio > 0.02 and days_since_last_buy < 30:
        signal = '强烈买入'
        action = '买入'
    elif buy_ratio > 0.005:
        signal = '买入'
        action = '买入'
    elif buy_ratio > 0:
        signal = '中性'
        action = '观望'
    else:
        signal = '观望'
        action = '观望'

    return ShareholderBuyResult(
        date=last_buy_date or '',
        stock_code='',
        total_buy_amount=total_buy_amount,
        buy_count=buy_count,
        buy_ratio=buy_ratio,
        days_since_last_buy=days_since_last_buy,
        signal=signal,
        recommended_action=action
    )


# ==================== 综合事件驱动信号 ====================

def calculate_event_score(
    earnings: Optional[EarningsSurpriseResult],
    jiejin_risk: Optional[JiejinRiskResult],
    shareholder_buy: Optional[ShareholderBuyResult],
    rebalance: Optional[RebalanceResult]
) -> EventSignal:
    """
    综合评估所有事件信号，计算总分

    Args:
        earnings: 财报预期差结果
        jiejin_risk: 解禁风险结果
        shareholder_buy: 股东增持结果
        rebalance: 指数调样结果

    Returns:
        EventSignal
    """
    events = []
    total_score = 0.0

    # 财报预期
    if earnings and earnings.is_significant:
        events.append({
            'type': 'earnings',
            'category': earnings.category,
            'surprise_pct': earnings.surprise_pct
        })
        if earnings.category == 'BEAT':
            total_score += earnings.surprise_pct * 0.3
        elif earnings.category == 'MISS':
            total_score -= 5  # 低于预期直接扣分

    # 解禁风险
    if jiejin_risk:
        events.append({
            'type': 'jiejin',
            'risk_level': jiejin_risk.risk_level,
            'risk_score': jiejin_risk.risk_score
        })
        if jiejin_risk.risk_score >= 3:
            total_score -= jiejin_risk.risk_score

    # 股东增持
    if shareholder_buy:
        events.append({
            'type': 'shareholder_buy',
            'signal': shareholder_buy.signal,
            'buy_ratio': shareholder_buy.buy_ratio
        })
        if '买入' in shareholder_buy.signal:
            total_score += 3

    # 指数调样
    if rebalance:
        events.append({
            'type': 'rebalance',
            'direction': rebalance.direction,
            'days_for_passive': rebalance.days_for_passive
        })
        if rebalance.direction == '调入' and rebalance.days_for_passive < 10:
            total_score += 2

    # 综合判断
    if total_score > 3:
        action = '买入'
        confidence = min(abs(total_score) / 10, 1.0)
    elif total_score < -3:
        action = '回避'
        confidence = min(abs(total_score) / 10, 1.0)
    else:
        action = '观望'
        confidence = 0.5

    return EventSignal(
        date='',
        stock_code='',
        events=events,
        total_score=total_score,
        action=action,
        confidence=confidence
    )


# ==================== 事件日历 ====================

def get_event_calendar(month: int) -> List[str]:
    """
    获取A股事件日历

    Args:
        month: 月份 (1-12)

    Returns:
        事件列表
    """
    calendar = {
        1: ['年报业绩预告密集期'],
        2: ['年报业绩预告密集期'],
        3: ['年报+一季报披露期开始', '两会'],
        4: ['年报+一季报披露期', '年报披露截止', '政治局会议'],
        5: ['业绩空窗期'],
        6: ['中报业绩预告'],
        7: ['中报业绩预告密集期', '年中政治局会议'],
        8: ['中报披露期'],
        9: ['三季报业绩预告', '二十大'],
        10: ['三季报披露截止', '国庆'],
        11: ['估值切换', '高送转预期'],
        12: ['中央经济工作会议', '年底结算']
    }

    return calendar.get(month, [])


def is_policy_sensitive_period(date_str: str) -> bool:
    """
    判断是否为政策敏感期

    Args:
        date_str: 日期字符串 (YYYY-MM-DD)

    Returns:
        是否为政策敏感期
    """
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        month = date.month

        # 政策敏感期
        sensitive_months = [3, 4, 7, 10, 12]

        # 每月重要会议期间
        if month == 3 and date.day <= 15:  # 两会
            return True
        if month == 4 and date.day >= 25:  # 政治局会议
            return True
        if month == 7 and date.day >= 20:  # 年中政治局会议
            return True
        if month == 10 and date.day >= 15:  # 重要会议
            return True
        if month == 12 and date.day >= 10:  # 中央经济工作会议
            return True

        return month in sensitive_months

    except ValueError:
        return False


# ==================== 便捷函数 ====================

def batch_calculate_jiejin_risk(
    unlock_data: pd.DataFrame,
    avg_volumes: Dict[str, float]
) -> pd.DataFrame:
    """
    批量计算解禁风险

    Args:
        unlock_data: 解禁数据DataFrame
        avg_volumes: 股票日均成交量字典 {stock_code: volume}

    Returns:
        包含风险评估的DataFrame
    """
    results = []

    for _, row in unlock_data.iterrows():
        stock_code = row.get('stock_code', '')
        avg_vol = avg_volumes.get(stock_code, float('inf'))

        risk = calculate_jiejin_risk(
            unlock_shares=row.get('unlock_shares', 0),
            float_shares=row.get('float_shares', 1),
            holder_type=row.get('holder_type', '其他'),
            avg_daily_volume=avg_vol,
            unlock_date=row.get('unlock_date')
        )
        risk.stock_code = stock_code
        risk.date = row.get('date', '')

        results.append({
            'stock_code': stock_code,
            'unlock_date': risk.unlock_date,
            'unlock_ratio': risk.unlock_ratio,
            'risk_level': risk.risk_level,
            'risk_score': risk.risk_score,
            'days_to_absorb': risk.days_to_absorb,
            'action': risk.recommended_action
        })

    return pd.DataFrame(results)
