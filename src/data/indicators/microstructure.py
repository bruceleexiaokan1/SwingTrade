"""市场微观结构指标

Market Microstructure Indicators

包含用于衡量市场流动性、订单流和交易成本的指标。

主要指标:
    - Amihud ILLIQ: 非流动性指标
    - Order Imbalance: 订单不平衡度
    - VPIN: 知情交易概率
    - Volume Anomaly: 成交量异常检测
    - Liquidity Regime: 流动性状态检测
    - Market Impact: 冲击成本估算
"""

import numpy as np
import pandas as pd


def calculate_amihud_illiq(
    returns: pd.Series,
    volume: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    计算 Amihud 非流动性指标 (ILLIQ)

    Amihud (2002) 提出的非流动性指标，衡量单位成交额引起的價格變動。

    Args:
        returns: 日收益率序列
        volume: 日成交额序列（必须是正值）
        window: 计算窗口，默认 20

    Returns:
        Amihud ILLIQ 序列，值越大表示流动性越差

    Formula:
        ILLIQ = |日收益率| / 日成交额

    Note:
        - 该指标反映的是价格对成交量的敏感性
        - 值越高表示流动性越差（，价格受成交量影响大）
        - 常用于衡量股票的流动性风险

    Example:
        >>> illiq = calculate_amihud_illiq(df['pct_change'], df['volume'] * df['close'])
        >>> print(f"当前ILLIQ: {illiq.iloc[-1]:.6f}")
    """
    if len(returns) != len(volume):
        raise ValueError("returns and volume must have the same length")

    if (volume <= 0).any():
        raise ValueError("volume must be positive")

    # 计算 |收益率| / 成交额
    illiq = returns.abs() / volume

    # 计算移动平均
    illiq_ma = illiq.rolling(window=window, min_periods=1).mean()

    return illiq_ma


def calculate_order_imbalance(
    bid_volume: pd.Series,
    ask_volume: pd.Series,
    window: int = 1
) -> pd.Series:
    """
    计算订单不平衡度 (Order Imbalance)

    衡量买卖盘力量的差异。

    Args:
        bid_volume: 买盘成交量序列
        ask_volume: 卖盘成交量序列
        window: 计算窗口，默认 1（单点）

    Returns:
        订单不平衡度序列，范围 [-1, 1]
        - 正值表示买盘强势
        - 负值表示卖盘强势
        - 0 表示平衡

    Formula:
        OI = (bid_vol - ask_vol) / (bid_vol + ask_vol)

    Note:
        - 常用于高频交易和做市商策略
        - 极端值（接近 ±1）可能预示价格移动

    Example:
        >>> oi = calculate_order_imbalance(df['bid_vol'], df['ask_vol'])
        >>> print(f"当前OI: {oi.iloc[-1]:.4f}")
    """
    if len(bid_volume) != len(ask_volume):
        raise ValueError("bid_volume and ask_volume must have the same length")

    total = bid_volume + ask_volume

    # 避免除以零
    total = total.replace(0, np.nan)

    oi = (bid_volume - ask_volume) / total

    if window > 1:
        oi = oi.rolling(window=window, min_periods=1).mean()

    return oi


def calculate_vpin(
    volume: pd.Series,
    buy_volume: pd.Series,
    sell_volume: pd.Series,
    window: int = 50
) -> pd.Series:
    """
    计算 VPIN (Volume-Synchronized Probability of Informed Trading)

    基于交易方向信息不对称性的流动性指标。

    Args:
        volume: 总成交量序列
        buy_volume: 主动买成交量序列
        sell_volume: 主动卖成交量序列
        window: 计算窗口，默认 50

    Returns:
        VPIN 序列，范围 [0, 1]
        - 值越高表示知情交易概率越高
        - 值越低表示流动性越好

    Formula:
        VPIN = |买量 - 卖量| / 总成交量

        使用滚动窗口计算期望 VPIN

    Note:
        - VPIN > 0.3 通常被视为高知情交易风险
        - 常用于检测内幕交易和流动性变化
        - 需要高频数据支持

    Example:
        >>> vpin = calculate_vpin(df['volume'], df['buy_vol'], df['sell_vol'])
        >>> print(f"当前VPIN: {vpin.iloc[-1]:.4f}")
    """
    if len(volume) != len(buy_volume) or len(volume) != len(sell_volume):
        raise ValueError("All volume series must have the same length")

    if (volume <= 0).any():
        raise ValueError("volume must be positive")

    if len(buy_volume) != len(sell_volume):
        raise ValueError("buy_volume and sell_volume must have the same length")

    # 计算单笔 VPIN
    vpin_single = (buy_volume - sell_volume).abs() / volume

    # 滚动平均
    vpin = vpin_single.rolling(window=window, min_periods=1).mean()

    return vpin


def detect_volume_anomaly(
    volume: pd.Series,
    window: int = 20,
    threshold_low: float = 2.0,
    threshold_high: float = 3.0
) -> pd.DataFrame:
    """
    检测成交量异常

    使用 Z-score 方法检测异常放量或缩量。

    Args:
        volume: 成交量序列
        window: 计算基准窗口，默认 20
        threshold_low: 放量阈值（Z-score），默认 2.0
        threshold_high: 极端放量阈值（Z-score），默认 3.0

    Returns:
        DataFrame 包含:
        - volume: 原始成交量
        - volume_ma: 移动平均成交量
        - volume_std: 移动标准差
        - z_score: Z-score
        - is_surge: 是否放量 (z_score > 2)
        - is_extreme: 是否极端放量 (z_score > 3)
        - is_shrink: 是否缩量 (z_score < -2)

    Note:
        - 放量通常与趋势确认或反转相关
        - 极端放量需警惕主力行为
        - 缩量可能表示趋势暂停或盘整

    Example:
        >>> result = detect_volume_anomaly(df['volume'])
        >>> if result['is_surge'].iloc[-1]:
        ...     print("检测到放量信号")
    """
    if len(volume) < 2:
        raise ValueError("volume must have at least 2 data points")

    result = pd.DataFrame(index=volume.index)
    result['volume'] = volume

    # 计算移动平均和标准差
    result['volume_ma'] = volume.rolling(window=window, min_periods=1).mean()
    result['volume_std'] = volume.rolling(window=window, min_periods=1).std()

    # 避免除以零
    result['volume_std'] = result['volume_std'].replace(0, np.nan)

    # 计算 Z-score
    result['z_score'] = (volume - result['volume_ma']) / result['volume_std']

    # 检测异常
    result['is_surge'] = result['z_score'] > threshold_low
    result['is_extreme'] = result['z_score'] > threshold_high
    result['is_shrink'] = result['z_score'] < -threshold_low

    return result


def liquidity_regime_detection(
    illiq: pd.Series,
    volume: pd.Series,
    returns: pd.Series,
    window: int = 20,
    thresholds: tuple = (0.25, 0.75)
) -> pd.DataFrame:
    """
    流动性状态检测

    结合非流动性指标、成交量和波动率检测市场流动性状态。

    Args:
        illiq: Amihud ILLIQ 序列
        volume: 成交量序列
        returns: 收益率序列
        window: 计算窗口，默认 20
        thresholds: 流动性分位阈值，默认 (0.25, 0.75)
            - illiq < 0.25 分位: 高流动性
            - 0.25 <= illiq < 0.75: 中等流动性
            - illiq >= 0.75 分位: 低流动性

    Returns:
        DataFrame 包含:
        - regime: 流动性状态 (high/medium/low)
        - regime_code: 状态编码 (1=high, 0=medium, -1=low)
        - volatility: 波动率
        - turnover: 换手率

    Note:
        - 高流动性状态: 价差小、冲击成本低
        - 低流动性状态: 价差大、冲击成本高
        - 策略应避开低流动性时段

    Example:
        >>> regimes = liquidity_regime_detection(illiq, df['volume'], df['pct_change'])
        >>> print(f"当前状态: {regimes['regime'].iloc[-1]}")
    """
    if len(illiq) != len(volume) or len(illiq) != len(returns):
        raise ValueError("All series must have the same length")

    result = pd.DataFrame(index=illiq.index)

    # 计算波动率
    result['volatility'] = returns.rolling(window=window, min_periods=1).std()

    # 计算换手率（假设成交量/流通股本）
    avg_volume = volume.rolling(window=window, min_periods=1).mean()
    result['turnover'] = volume / avg_volume

    # 计算流动性分位
    illiq_quantile_low = illiq.quantile(thresholds[0])
    illiq_quantile_high = illiq.quantile(thresholds[1])

    # 确定流动性状态
    def get_regime(illiq_val):
        if illiq_val <= illiq_quantile_low:
            return 'high'
        elif illiq_val <= illiq_quantile_high:
            return 'medium'
        else:
            return 'low'

    result['regime'] = illiq.apply(get_regime)

    # 状态编码
    regime_map = {'high': 1, 'medium': 0, 'low': -1}
    result['regime_code'] = result['regime'].map(regime_map)

    return result


def estimate_market_impact(
    order_size: float,
    avg_daily_volume: float,
    participation_rate: float = 0.1
) -> dict:
    """
    估算冲击成本

    基于订单规模估算对价格的潜在冲击。

    Args:
        order_size: 订单规模（股数或金额）
        avg_daily_volume: 日均成交量
        participation_rate: 参与率目标，默认 0.1（10%）
            - 建议不超过 20% 以避免过度冲击

    Returns:
        dict 包含:
        - impact_pct: 预期冲击成本（%）
        - participation_rate: 实际参与率
        - advice: 建议（字符串）
        - risk_level: 风险等级 (low/medium/high)

    Formula:
        Impact ∝ √(订单规模 / 日均成交)

        简化公式: Impact = k * sqrt(order_size / ADV)

    Note:
        - 冲击成本是非线性增长的
        - 大订单应分批执行
        - 参与率越高，冲击越大

    Example:
        >>> impact = estimate_market_impact(1000000, 5000000)
        >>> print(f"预计冲击: {impact['impact_pct']:.2f}%")
        >>> print(f"建议: {impact['advice']}")
    """
    if avg_daily_volume <= 0:
        raise ValueError("avg_daily_volume must be positive")

    if order_size <= 0:
        raise ValueError("order_size must be positive")

    # 计算实际参与率
    actual_participation = order_size / avg_daily_volume

    # 冲击成本系数（经验值）
    # 实际应用中需根据市场特性调整
    k = 0.1  # 基础系数

    # 计算冲击成本
    if actual_participation <= 0:
        impact_pct = 0.0
    else:
        impact_pct = k * np.sqrt(actual_participation) * 100

    # 生成建议
    if actual_participation > 0.25:
        advice = "订单规模过大，建议分批执行（至少4批）"
        risk_level = "high"
    elif actual_participation > 0.10:
        advice = "订单规模较大，建议分批执行（至少2批）"
        risk_level = "medium"
    elif actual_participation > 0.05:
        advice = "订单规模适中，注意市场波动"
        risk_level = "medium"
    else:
        advice = "订单规模较小，冲击成本可控"
        risk_level = "low"

    return {
        'impact_pct': round(impact_pct, 4),
        'participation_rate': round(actual_participation, 4),
        'advice': advice,
        'risk_level': risk_level
    }


# 向后兼容性别名
amihud_illiq = calculate_amihud_illiq
order_imbalance = calculate_order_imbalance
vpin = calculate_vpin
volume_anomaly = detect_volume_anomaly
liquidity_regime = liquidity_regime_detection
market_impact = estimate_market_impact
