"""
StockData 数据质量评估模块
基于知识库三原则：复权、标准化、位移
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """数据质量评分"""
    total: float = 100.0
    price_score: float = 25.0      # 价格质量
    ohlc_score: float = 25.0       # OHLC质量
    adj_score: float = 25.0         # 复权连续性
    completeness_score: float = 25.0 # 完整性

    @property
    def grade(self) -> str:
        if self.total >= 100: return "完美"
        if self.total >= 80: return "良好"
        if self.total >= 50: return "可疑"
        if self.total >= 1: return "危险"
        return "废弃"

    @property
    def usable(self) -> bool:
        return self.total >= 50

    def to_dict(self) -> dict:
        return {
            'total': self.total,
            'grade': self.grade,
            'usable': self.usable,
            'price_score': self.price_score,
            'ohlc_score': self.ohlc_score,
            'adj_score': self.adj_score,
            'completeness_score': self.completeness_score,
        }


def calculate_quality_score(df: pd.DataFrame, anomalies: list) -> QualityScore:
    """
    计算数据质量评分

    知识库三原则 → 质量评分映射：

    1. 复权 → 保证价格连续性
       → adj_score，复权断裂给极低分

    2. 标准化 → 让指标跨标的可比
       → completeness_score，数据完整是标准化的基础

    3. 位移 → 杜绝未来函数
       → price_score，价格合理性校验

    Args:
        df: 日线数据
        anomalies: 异常列表

    Returns:
        QualityScore: 质量评分
    """
    score = QualityScore()

    if df.empty:
        score.total = 0
        return score

    # 1. 价格合理性 (满分25)
    price_anomalies = [a for a in anomalies if 'price' in a.get('reason', '').lower()]
    if price_anomalies:
        deduction = sum(a.get('count', 1) for a in price_anomalies) * 10
        score.price_score = max(0, 25 - deduction)

    # 2. OHLC关系 (满分25)
    ohlc_anomalies = [a for a in anomalies if 'ohlc' in a.get('reason', '').lower()]
    if ohlc_anomalies:
        deduction = sum(a.get('count', 1) for a in ohlc_anomalies) * 15
        score.ohlc_score = max(0, 25 - deduction)

    # 3. 复权连续性 (满分25) - 核心指标，断裂直接扣20/处
    adj_anomalies = [a for a in anomalies if 'adj' in a.get('reason', '').lower()]
    if adj_anomalies:
        deduction = sum(a.get('count', 1) for a in adj_anomalies) * 20
        score.adj_score = max(0, 25 - deduction)

    # 4. 完整性 (满分25)
    missing_count = 0
    required_fields = ['date', 'open', 'high', 'low', 'close', 'volume']
    for field in required_fields:
        if field not in df.columns:
            missing_count += 1
        elif df[field].isnull().any():
            missing_count += df[field].isnull().sum()

    if missing_count > 0:
        score.completeness_score = max(0, 25 - missing_count * 5)

    # 计算总分
    score.total = (
        score.price_score +
        score.ohlc_score +
        score.adj_score +
        score.completeness_score
    )

    return score


def validate_daily(df: pd.DataFrame) -> dict:
    """
    日线数据校验

    知识库三原则实现：
    1. 复权 → 保证价格连续性
    2. 标准化 → 字段完整
    3. 位移 → 价格合理

    Args:
        df: 日线数据

    Returns:
        dict: {
            'valid': bool,
            'anomalies': list,
            'score': QualityScore
        }
    """
    anomalies = []

    if df.empty:
        anomalies.append({'reason': 'empty_data', 'severity': 'error'})

    # 1. 必填字段检查
    required_fields = ['date', 'open', 'high', 'low', 'close', 'volume']
    for field in required_fields:
        if field not in df.columns:
            anomalies.append({
                'reason': f'missing_required_field:{field}',
                'severity': 'error'
            })

    # 2. 价格合理性检查 (位移原则)
    if 'close' in df.columns:
        # 价格范围检查
        out_of_range = (df['close'] < 0.01) | (df['close'] > 10000)
        if out_of_range.any():
            anomalies.append({
                'reason': 'price_out_of_range',
                'count': out_of_range.sum(),
                'severity': 'error'
            })

        # 价格为0
        zero_price = df['close'] == 0
        if zero_price.any():
            anomalies.append({
                'reason': 'price_zero',
                'count': zero_price.sum(),
                'severity': 'error'
            })

    # 3. OHLC 关系检查 (位移原则)
    if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
        # high >= low
        high_lt_low = df['high'] < df['low']
        if high_lt_low.any():
            anomalies.append({
                'reason': 'ohlc_low_gt_high',
                'count': high_lt_low.sum(),
                'severity': 'error'
            })

        # close 在 [low, high] 范围内
        close_out = (df['close'] < df['low']) | (df['close'] > df['high'])
        if close_out.any():
            anomalies.append({
                'reason': 'ohlc_close_out',
                'count': close_out.sum(),
                'severity': 'error'
            })

        # open 在 [low, high] 范围内
        open_out = (df['open'] < df['low']) | (df['open'] > df['high'])
        if open_out.any():
            anomalies.append({
                'reason': 'ohlc_open_out',
                'count': open_out.sum(),
                'severity': 'error'
            })

    # 4. 成交量检查
    if 'volume' in df.columns:
        negative_volume = df['volume'] < 0
        if negative_volume.any():
            anomalies.append({
                'reason': 'volume_invalid',
                'count': negative_volume.sum(),
                'severity': 'error'
            })

    # 5. 复权连续性检查 (复权原则)
    if 'adj_factor' in df.columns and len(df) >= 2:
        # 因子必须为正
        negative_factor = df['adj_factor'] <= 0
        if negative_factor.any():
            anomalies.append({
                'reason': 'adj_factor_invalid',
                'count': negative_factor.sum(),
                'severity': 'error'
            })

        # 检查因子连续性（相邻两天比值应接近1）
        # 注意：这里简化处理，实际应该用后复权价格连续性判断
        if len(df) >= 2:
            adj_ratios = df['adj_factor'].iloc[1:].values / df['adj_factor'].iloc[:-1].values
            # 允许因子变化在 0.99-1.01 范围内（正常分红送配）
            # 或者因子反向大幅变化同时价格也反向调整（保持连续性）
            abnormal = (adj_ratios < 0.99) | (adj_ratios > 1.01)
            if abnormal.any():
                # 检查是否是正常业务导致的因子变化
                # 通过检查后复权价格是否连续来判断
                adj_close = df['close'] * df['adj_factor']
                adj_ratios_close = adj_close.iloc[1:].values / adj_close.iloc[:-1].values
                close_continuous = (0.95 <= adj_ratios_close) & (adj_ratios_close <= 1.05)

                # 如果因子变化但价格连续，是正常业务；否则是异常
                business_break = abnormal & close_continuous
                if business_break.any():
                    # 正常分红送配，不算异常
                    pass
                else:
                    anomalies.append({
                        'reason': 'adj_continuity_break',
                        'count': abnormal.sum(),
                        'severity': 'warning'
                    })

    # 6. 涨跌幅检查
    if 'pct_chg' in df.columns:
        # 合理涨跌幅范围（-20% 到 +20%）
        abnormal_pct = (df['pct_chg'] < -0.2) | (df['pct_chg'] > 0.2)
        if abnormal_pct.any():
            anomalies.append({
                'reason': 'pct_chg_exceed',
                'count': abnormal_pct.sum(),
                'severity': 'warning'
            })

    # 计算质量分
    score = calculate_quality_score(df, anomalies)

    return {
        'valid': score.usable,
        'anomalies': anomalies,
        'score': score
    }
