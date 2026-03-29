"""执行算法模块

提供多种高级订单执行算法：
- TWAP: 时间加权平均执行
- VWAP: 成交量加权平均执行
- Iceberg: 冰山订单（隐藏大单）
- Adaptive: 自适应执行算法
- 冲击成本估算
- 执行质量监控
- 大单拆分策略

冲击成本公式:
    Impact = spread_bps × √(participation_rate)
    Participation = 订单金额 / (日均成交 × 5)

自适应执行策略:
    - urgency < 0.3: VWAP慢慢执行
    - urgency < 0.7: TWAP/VWAP混合
    - urgency >= 0.7: 快速执行
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd


class ExecutionStatus(Enum):
    """执行状态"""
    PENDING = "pending"           # 待执行
    PARTIAL = "partial"          # 部分成交
    FILLED = "filled"             # 全部成交
    CANCELLED = "cancelled"       # 已取消
    REJECTED = "rejected"         # 被拒绝


@dataclass
class ExecutionSlice:
    """订单切片"""
    slice_id: int                 # 切片序号
    timestamp: str                # 执行时间
    price: float                  # 成交价格
    shares: int                   # 成交数量
    turnover: float               # 成交金额
    slippage: float               # 滑点
    urgency: float                # 执行时的紧急程度


@dataclass
class ExecutionResult:
    """执行结果"""
    order_id: str                 # 订单ID
    code: str                     # 股票代码
    direction: str                # "buy" / "sell"
    total_shares: int             # 总股数
    filled_shares: int            # 已成交股数
    avg_price: float              # 平均成交价
    total_turnover: float         # 总成交金额
    total_slippage: float         # 总滑点
    execution_time: float         # 执行耗时（分钟）
    status: ExecutionStatus       # 执行状态
    slices: List[ExecutionSlice] = field(default_factory=list)  # 切片列表

    @property
    def remaining_shares(self) -> int:
        """剩余未成交股数"""
        return self.total_shares - self.filled_shares

    @property
    def fill_rate(self) -> float:
        """成交率"""
        if self.total_shares == 0:
            return 0.0
        return self.filled_shares / self.total_shares


@dataclass
class MarketImpact:
    """市场冲击"""
    spread_bps: float            # 买卖价差（基点）
    participation_rate: float     # 参与率
    impact_bps: float            # 冲击成本（基点）
    reason: str                   # 原因


@dataclass
class ExecutionQuality:
    """执行质量指标"""
    arrival_price: float          # 到达价格
    avg_execution_price: float    # 平均执行价格
    market_impact_bps: float      # 市场冲击（基点）
    timing_risk_bps: float        # 时机风险（基点）
    total_cost_bps: float         # 总成本（基点）
    fill_rate: float              # 成交率
    vwap_diff_bps: float          # 与VWAP差异（基点）


def twap_execution(
    total_shares: int,
    num_slices: int,
    price_series: pd.Series,
    urgency: float = 0.5,
    timestamp_format: str = "%Y-%m-%d %H:%M"
) -> List[ExecutionSlice]:
    """
    TWAP（时间加权平均价格）执行算法

    将订单均匀分配到多个时间窗口，每个窗口执行相同数量的股数。
    适用于对市场冲击较小、不急于成交的场景。

    Args:
        total_shares: 总股数
        num_slices: 切片数量（时间窗口数）
        price_series: 价格序列（DataFrame的Series或类似结构，包含price列）
                      索引应为时间戳
        urgency: 紧急程度 0.0~1.0，越高越快执行完
                 0.3以下会分散得更均匀，0.7以上会集中执行
        timestamp_format: 时间戳格式

    Returns:
        ExecutionSlice列表

    Raises:
        ValueError: 参数无效时

    Example:
        >>> prices = pd.Series([10.0, 10.1, 10.2, 10.3],
        ...                    index=pd.date_range('2024-01-01', periods=4, freq='h'))
        >>> slices = twap_execution(10000, 4, prices)
        >>> len(slices)
        4
    """
    if total_shares <= 0:
        raise ValueError(f"total_shares must be positive, got {total_shares}")
    if num_slices <= 0:
        raise ValueError(f"num_slices must be positive, got {num_slices}")
    if urgency < 0 or urgency > 1:
        raise ValueError(f"urgency must be between 0 and 1, got {urgency}")

    # 计算每片股数（均匀分配）
    shares_per_slice = total_shares // num_slices
    remainder = total_shares % num_slices

    # 根据紧急程度调整时间分布
    # urgency越高，前几个切片执行越多
    if urgency < 0.3:
        # 均匀分布
        weights = np.ones(num_slices)
    elif urgency >= 0.7:
        # 前重后轻，快速执行
        weights = np.array([num_slices - i for i in range(num_slices)])
        weights = weights / weights.sum()
    else:
        # 混合策略，稍向前倾斜
        weights = np.linspace(0.8, 1.2, num_slices)
        weights = weights / weights.sum()

    slices = []
    prices = price_series.values if hasattr(price_series, 'values') else price_series

    if len(prices) < num_slices:
        # 价格数据不足，重复使用最后一个价格
        prices = np.pad(prices, (0, num_slices - len(prices)), mode='edge')

    base_time = price_series.index[0] if hasattr(price_series, 'index') else None

    # 先按权重计算初步分配
    raw_shares = [shares_per_slice * w for w in weights]

    # 分配余数到前remainder个切片
    for i in range(remainder):
        raw_shares[i] += 1

    # 确保所有值有效并计算总和
    raw_shares = [max(0, int(rs)) for rs in raw_shares]
    actual_total = sum(raw_shares)

    # 调整使总和等于total_shares
    diff = total_shares - actual_total
    if diff != 0:
        # 添加到第一个切片来修正
        raw_shares[0] += diff

    for i in range(num_slices):
        adjusted_shares = raw_shares[i]

        price = float(prices[i])
        turnover = price * adjusted_shares

        # 计算滑点（基于到达价格）
        slippage = 0.0  # TWAP通常假设理想执行

        # 生成时间戳
        if base_time is not None and hasattr(base_time, 'strftime'):
            ts = base_time + pd.Timedelta(minutes=i * 30)
            ts_str = ts.strftime(timestamp_format)
        else:
            ts_str = f"slice_{i}"

        slice_obj = ExecutionSlice(
            slice_id=i,
            timestamp=ts_str,
            price=price,
            shares=adjusted_shares,
            turnover=turnover,
            slippage=slippage,
            urgency=urgency
        )
        slices.append(slice_obj)

    return slices


def vwap_execution(
    total_shares: int,
    volume_profile: pd.Series,
    price_series: pd.Series,
    urgency: float = 0.5,
    timestamp_format: str = "%Y-%m-%d %H:%M"
) -> List[ExecutionSlice]:
    """
    VWAP（成交量加权平均价格）执行算法

    根据历史成交量分布来分配订单，在高成交量时段分配更多订单。
    目标是在执行完成后，整体成交价接近VWAP。

    Args:
        total_shares: 总股数
        volume_profile: 成交量分布序列（每时间段成交量）
        price_series: 价格序列
        urgency: 紧急程度 0.0~1.0
        timestamp_format: 时间戳格式

    Returns:
        ExecutionSlice列表

    Raises:
        ValueError: 参数无效时

    Example:
        >>> volumes = pd.Series([1000, 2000, 1500, 2500],
        ...                     index=pd.date_range('2024-01-01', periods=4, freq='h'))
        >>> prices = pd.Series([10.0, 10.1, 10.2, 10.3],
        ...                    index=pd.date_range('2024-01-01', periods=4, freq='h'))
        >>> slices = vwap_execution(10000, volumes, prices)
        >>> len(slices)
        4
    """
    if total_shares <= 0:
        raise ValueError(f"total_shares must be positive, got {total_shares}")
    if urgency < 0 or urgency > 1:
        raise ValueError(f"urgency must be between 0 and 1, got {urgency}")

    volumes = volume_profile.values if hasattr(volume_profile, 'values') else volume_profile
    prices = price_series.values if hasattr(price_series, 'values') else price_series

    if len(volumes) == 0:
        raise ValueError("volume_profile cannot be empty")

    # 计算成交量权重
    total_volume = np.sum(volumes)
    if total_volume <= 0:
        weights = np.ones(len(volumes)) / len(volumes)
    else:
        weights = volumes / total_volume

    # 根据紧急程度调整权重
    if urgency < 0.3:
        # 偏向VWAP，不急于成交
        adjusted_weights = weights * 0.5 + (1.0 / len(weights)) * 0.5
    elif urgency >= 0.7:
        # 快速执行，集中在前几个高成交量时段
        sorted_indices = np.argsort(-weights)
        adjusted_weights = np.zeros(len(weights))
        for rank, idx in enumerate(sorted_indices):
            adjusted_weights[idx] = (len(weights) - rank) / np.sum(range(1, len(weights) + 1))
    else:
        # 标准VWAP
        adjusted_weights = weights

    adjusted_weights = adjusted_weights / np.sum(adjusted_weights)

    # 计算每片股数
    slices = []
    base_time = volume_profile.index[0] if hasattr(volume_profile, 'index') else None

    # 计算初步分配
    raw_shares = [total_shares * w for w in adjusted_weights]
    raw_shares = [max(0, int(rs)) for rs in raw_shares]

    # 调整使总和等于total_shares
    actual_total = sum(raw_shares)
    diff = total_shares - actual_total
    if diff != 0:
        # 添加到第一个切片来修正
        raw_shares[0] += diff

    for i in range(len(volumes)):
        shares = raw_shares[i]

        price = float(prices[i]) if i < len(prices) else float(prices[-1])
        turnover = price * shares

        slippage = 0.0

        if base_time is not None and hasattr(base_time, 'strftime'):
            ts = base_time + pd.Timedelta(minutes=i * 30)
            ts_str = ts.strftime(timestamp_format)
        else:
            ts_str = f"slice_{i}"

        slice_obj = ExecutionSlice(
            slice_id=i,
            timestamp=ts_str,
            price=price,
            shares=shares,
            turnover=turnover,
            slippage=slippage,
            urgency=urgency
        )
        slices.append(slice_obj)

    return slices


def iceberg_order(
    total_shares: int,
    visible_ratio: float,
    price_series: pd.Series,
    num_iterations: int = 10,
    timestamp_format: str = "%Y-%m-%d %H:%M"
) -> List[ExecutionSlice]:
    """
    冰山订单（Iceberg Order）执行算法

    只显示部分订单（ iceberg iceberg_slice），隐藏真实订单总量。
    每次成交后补充显示下一个切片，直到全部成交。

    Args:
        total_shares: 总股数
        visible_ratio: 可见比例（0.0~1.0），如0.1表示每次只显示10%
        price_series: 价格序列
        num_iterations: 最大迭代次数（防止无限循环）
        timestamp_format: 时间戳格式

    Returns:
        ExecutionSlice列表

    Raises:
        ValueError: 参数无效时

    Example:
        >>> prices = pd.Series([10.0, 10.1, 10.2],
        ...                    index=pd.date_range('2024-01-01', periods=3, freq='h'))
        >>> slices = iceberg_order(10000, 0.1, prices)
        >>> # 将有约10个切片，每个切片约1000股
    """
    if total_shares <= 0:
        raise ValueError(f"total_shares must be positive, got {total_shares}")
    if visible_ratio <= 0 or visible_ratio > 1:
        raise ValueError(f"visible_ratio must be between 0 and 1, got {visible_ratio}")

    slices = []
    remaining = total_shares
    iteration = 0
    slice_id = 0

    prices = price_series.values if hasattr(price_series, 'values') else price_series
    base_time = price_series.index[0] if hasattr(price_series, 'index') else None

    while remaining > 0 and iteration < num_iterations:
        # 计算本次可见数量（基于原始订单总量，每次显示固定比例）
        # 初始可见量 = total_shares * visible_ratio
        initial_visible = int(total_shares * visible_ratio)
        initial_visible = max(1, initial_visible)  # 至少1股

        # 每次显示 initial_visible 或剩余数量（取较小）
        visible_shares = min(remaining, initial_visible)

        price = float(prices[min(iteration, len(prices) - 1)])
        turnover = price * visible_shares

        if base_time is not None and hasattr(base_time, 'strftime'):
            ts = base_time + pd.Timedelta(minutes=iteration * 30)
            ts_str = ts.strftime(timestamp_format)
        else:
            ts_str = f"iceberg_{iteration}"

        slice_obj = ExecutionSlice(
            slice_id=slice_id,
            timestamp=ts_str,
            price=price,
            shares=visible_shares,
            turnover=turnover,
            slippage=0.0,
            urgency=0.5
        )
        slices.append(slice_obj)

        remaining -= visible_shares
        slice_id += 1
        iteration += 1

    return slices


def adaptive_execution(
    total_shares: int,
    volume_profile: pd.Series,
    price_series: pd.Series,
    urgency: float,
    timestamp_format: str = "%Y-%m-%d %H:%M"
) -> List[ExecutionSlice]:
    """
    自适应执行算法

    根据紧急程度自动选择执行策略：
    - urgency < 0.3: VWAP慢慢执行，减少市场冲击
    - urgency < 0.7: TWAP/VWAP混合策略
    - urgency >= 0.7: 快速执行，接受更高冲击

    Args:
        total_shares: 总股数
        volume_profile: 成交量分布
        price_series: 价格序列
        urgency: 紧急程度 0.0~1.0
        timestamp_format: 时间戳格式

    Returns:
        ExecutionSlice列表

    Example:
        >>> volumes = pd.Series([1000, 2000, 1500],
        ...                     index=pd.date_range('2024-01-01', periods=3, freq='h'))
        >>> prices = pd.Series([10.0, 10.1, 10.2],
        ...                    index=pd.date_range('2024-01-01', periods=3, freq='h'))
        >>> # 慢速执行
        >>> slices = adaptive_execution(10000, volumes, prices, urgency=0.2)
        >>> # 快速执行
        >>> slices = adaptive_execution(10000, volumes, prices, urgency=0.8)
    """
    if total_shares <= 0:
        raise ValueError(f"total_shares must be positive, got {total_shares}")
    if urgency < 0 or urgency > 1:
        raise ValueError(f"urgency must be between 0 and 1, got {urgency}")

    if urgency < 0.3:
        # VWAP慢慢执行
        return vwap_execution(total_shares, volume_profile, price_series, urgency=urgency)

    elif urgency < 0.7:
        # TWAP/VWAP混合：平均分配但根据成交量加权
        num_slices = len(volume_profile)
        shares_per_slice = total_shares // num_slices
        remainder = total_shares % num_slices

        volumes = volume_profile.values if hasattr(volume_profile, 'values') else volume_profile
        prices = price_series.values if hasattr(price_series, 'values') else price_series

        # 混合权重：50% TWAP + 50% VWAP
        vwap_weights = volumes / np.sum(volumes) if np.sum(volumes) > 0 else np.ones(num_slices) / num_slices
        twap_weights = np.ones(num_slices) / num_slices
        mixed_weights = 0.5 * twap_weights + 0.5 * vwap_weights
        mixed_weights = mixed_weights / np.sum(mixed_weights)

        slices = []
        base_time = volume_profile.index[0] if hasattr(volume_profile, 'index') else None

        # 计算初步分配
        raw_shares = [total_shares * w for w in mixed_weights]
        raw_shares = [max(0, int(rs)) for rs in raw_shares]

        # 分配余数
        for i in range(remainder):
            raw_shares[i] += 1

        # 调整使总和等于total_shares
        actual_total = sum(raw_shares)
        diff = total_shares - actual_total
        if diff != 0:
            raw_shares[0] += diff

        for i in range(num_slices):
            shares = raw_shares[i]

            price = float(prices[i]) if i < len(prices) else float(prices[-1])
            turnover = price * shares

            if base_time is not None and hasattr(base_time, 'strftime'):
                ts = base_time + pd.Timedelta(minutes=i * 30)
                ts_str = ts.strftime(timestamp_format)
            else:
                ts_str = f"adaptive_{i}"

            slice_obj = ExecutionSlice(
                slice_id=i,
                timestamp=ts_str,
                price=price,
                shares=shares,
                turnover=turnover,
                slippage=0.0,
                urgency=urgency
            )
            slices.append(slice_obj)

        return slices

    else:
        # 快速执行：使用更少的时间窗口，集中成交
        num_slices = max(2, len(volume_profile) // 2)
        num_slices = min(num_slices, total_shares)  # 不能超过总股数

        shares_per_slice = total_shares // num_slices
        remainder = total_shares % num_slices

        prices = price_series.values if hasattr(price_series, 'values') else price_series
        if len(prices) < num_slices:
            prices = np.pad(prices, (0, num_slices - len(prices)), mode='edge')

        base_time = price_series.index[0] if hasattr(price_series, 'index') else None

        # 前重后轻的权重
        weights = np.array([num_slices - i for i in range(num_slices)])
        weights = weights / np.sum(weights)

        # 计算初步分配
        raw_shares = [shares_per_slice * w for w in weights]
        raw_shares = [max(0, int(rs)) for rs in raw_shares]

        # 分配余数
        for i in range(remainder):
            raw_shares[i] += 1

        # 调整使总和等于total_shares
        actual_total = sum(raw_shares)
        diff = total_shares - actual_total
        if diff != 0:
            raw_shares[0] += diff

        slices = []
        for i in range(num_slices):
            shares = raw_shares[i]

            price = float(prices[i])
            turnover = price * shares

            if base_time is not None and hasattr(base_time, 'strftime'):
                ts = base_time + pd.Timedelta(minutes=i * 15)  # 更短间隔
                ts_str = ts.strftime(timestamp_format)
            else:
                ts_str = f"fast_{i}"

            slice_obj = ExecutionSlice(
                slice_id=i,
                timestamp=ts_str,
                price=price,
                shares=shares,
                turnover=turnover,
                slippage=0.0,
                urgency=urgency
            )
            slices.append(slice_obj)

        return slices


def estimate_market_impact(
    order_amount: float,
    daily_avg_volume: float,
    spread_bps: float = 10.0,
    volatility_bps: float = 50.0
) -> MarketImpact:
    """
    估算市场冲击成本

    使用平方根公式估算订单对价格的影响：
    Impact = spread_bps × √(participation_rate)
    Participation = 订单金额 / (日均成交 × 5)

    Args:
        order_amount: 订单金额（元）
        daily_avg_volume: 日均成交额（元）
        spread_bps: 买卖价差（基点），默认10bps
        volatility_bps: 历史波动率（基点），默认50bps

    Returns:
        MarketImpact对象，包含冲击估算详情

    Example:
        >>> impact = estimate_market_impact(
        ...     order_amount=1000000,
        ...     daily_avg_volume=10000000
        ... )
        >>> print(f"冲击成本: {impact.impact_bps:.2f} bps")
    """
    if order_amount < 0:
        raise ValueError(f"order_amount must be non-negative, got {order_amount}")
    if daily_avg_volume <= 0:
        raise ValueError(f"daily_avg_volume must be positive, got {daily_avg_volume}")

    # 计算参与率（假设5天完成）
    participation_rate = order_amount / (daily_avg_volume * 5)

    # 平方根冲击模型
    # Impact = spread_bps × √(participation_rate) + volatility_bps × √(participation_rate) × 0.1
    impact_from_spread = spread_bps * np.sqrt(participation_rate)
    impact_from_vol = volatility_bps * np.sqrt(participation_rate) * 0.1
    total_impact = impact_from_spread + impact_from_vol

    # 确定原因描述
    if participation_rate > 0.2:
        reason = "high_participation"
    elif participation_rate > 0.1:
        reason = "medium_participation"
    elif participation_rate > 0.05:
        reason = "low_participation"
    else:
        reason = "minimal_participation"

    return MarketImpact(
        spread_bps=spread_bps,
        participation_rate=participation_rate,
        impact_bps=total_impact,
        reason=reason
    )


def monitor_execution_quality(
    slices: List[ExecutionSlice],
    arrival_price: float,
    current_vwap: float,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> ExecutionQuality:
    """
    监控执行质量

    计算执行过程中的各项质量指标：
    - 市场冲击（与到达价相比）
    - 时机风险
    - 与VWAP的差异
    - 成交率

    Args:
        slices: 执行切片列表
        arrival_price: 订单到达时的价格
        current_vwap: 当前市场VWAP
        start_time: 开始时间（可选）
        end_time: 结束时间（可选）

    Returns:
        ExecutionQuality对象

    Raises:
        ValueError: slices为空时

    Example:
        >>> quality = monitor_execution_quality(slices, arrival_price=10.0, current_vwap=10.05)
        >>> print(f"总成本: {quality.total_cost_bps:.2f} bps")
    """
    if not slices:
        raise ValueError("slices cannot be empty")

    total_shares = sum(s.shares for s in slices)
    total_turnover = sum(s.turnover for s in slices)

    if total_shares == 0:
        raise ValueError("total shares cannot be zero")

    # 平均执行价格
    avg_price = total_turnover / total_shares

    # 与到达价相比的价格变化（基点）
    if arrival_price > 0:
        price_change = (avg_price - arrival_price) / arrival_price * 10000
    else:
        price_change = 0.0

    # 时机风险（基于价格波动）
    if len(slices) > 1:
        prices = [s.price for s in slices]
        timing_risk = np.std(prices) / arrival_price * 10000 if arrival_price > 0 else 0.0
    else:
        timing_risk = 0.0

    # 与VWAP差异
    if current_vwap > 0:
        vwap_diff = (avg_price - current_vwap) / current_vwap * 10000
    else:
        vwap_diff = 0.0

    # 总成本约等于市场冲击 + 时机风险
    total_cost = abs(price_change) + timing_risk * 0.5

    return ExecutionQuality(
        arrival_price=arrival_price,
        avg_execution_price=avg_price,
        market_impact_bps=abs(price_change),
        timing_risk_bps=timing_risk,
        total_cost_bps=total_cost,
        fill_rate=1.0,  # 假设全部成交
        vwap_diff_bps=vwap_diff
    )


def order_slicer(
    total_shares: int,
    max_slice_size: int,
    volume_profile: Optional[pd.Series] = None,
    strategy: str = "equal"
) -> List[int]:
    """
    大单拆分策略

    将大单拆分为多个小单，根据不同策略分配：
    - "equal": 均匀拆分
    - "vwap": 按成交量加权拆分
    - "time": 按时间均匀拆分（需要volume_profile）

    Args:
        total_shares: 总股数
        max_slice_size: 每片最大股数
        volume_profile: 成交量分布（用于vwap和time策略）
        strategy: 拆分策略，"equal", "vwap", 或 "time"

    Returns:
        每片股数列表

    Raises:
        ValueError: 参数无效时

    Example:
        >>> # 均匀拆分为每片最多1000股
        >>> slices = order_slicer(10000, 1000)
        >>> print(slices)
        [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]

        >>> # 按成交量加权拆分
        >>> volumes = pd.Series([1000, 2000, 1500, 2500])
        >>> slices = order_slicer(10000, 5000, volume_profile=volumes, strategy="vwap")
    """
    if total_shares <= 0:
        raise ValueError(f"total_shares must be positive, got {total_shares}")
    if max_slice_size <= 0:
        raise ValueError(f"max_slice_size must be positive, got {max_slice_size}")
    if strategy not in ("equal", "vwap", "time"):
        raise ValueError(f"strategy must be 'equal', 'vwap', or 'time', got {strategy}")

    if strategy == "equal":
        # 均匀拆分
        num_slices = (total_shares + max_slice_size - 1) // max_slice_size
        slices = []
        remaining = total_shares

        for _ in range(num_slices):
            slice_size = min(remaining, max_slice_size)
            slices.append(slice_size)
            remaining -= slice_size

        return slices

    else:
        # 需要volume_profile
        if volume_profile is None:
            raise ValueError(f"volume_profile required for strategy '{strategy}'")

        volumes = volume_profile.values if hasattr(volume_profile, 'values') else volume_profile

        if len(volumes) == 0:
            raise ValueError("volume_profile cannot be empty")

        # 计算权重
        if strategy == "vwap":
            total_volume = np.sum(volumes)
            if total_volume <= 0:
                weights = np.ones(len(volumes)) / len(volumes)
            else:
                weights = volumes / total_volume
        else:  # time
            weights = np.ones(len(volumes)) / len(volumes)

        # 计算实际需要的片数
        avg_shares_per_unit = total_shares
        num_slices = len(volumes)

        # 根据权重分配，但每片不超过max_slice_size
        raw_shares = [int(avg_shares_per_unit * w) for w in weights]

        # 调整使总和等于total_shares
        diff = total_shares - sum(raw_shares)
        if diff > 0:
            # 添加到最大的片中
            max_idx = raw_shares.index(max(raw_shares))
            raw_shares[max_idx] += diff
        elif diff < 0:
            # 从最小的片中扣除
            min_idx = raw_shares.index(min(raw_shares))
            raw_shares[min_idx] += diff  # diff是负数

        # 确保不超过max_slice_size
        slices = []
        for rs in raw_shares:
            while rs > max_slice_size:
                slices.append(max_slice_size)
                rs -= max_slice_size
            if rs > 0:
                slices.append(rs)

        return slices


# ============================================================================
# 便捷函数：执行订单并返回结果
# ============================================================================

def execute_order(
    order_id: str,
    code: str,
    direction: str,
    total_shares: int,
    price_series: pd.Series,
    volume_profile: Optional[pd.Series] = None,
    algorithm: str = "twap",
    urgency: float = 0.5,
    arrival_price: float = 0.0
) -> ExecutionResult:
    """
    执行订单的便捷函数

    Args:
        order_id: 订单ID
        code: 股票代码
        direction: "buy" 或 "sell"
        total_shares: 总股数
        price_series: 价格序列
        volume_profile: 成交量分布（VWAP/自适应算法需要）
        algorithm: 执行算法 "twap", "vwap", "iceberg", "adaptive"
        urgency: 紧急程度
        arrival_price: 到达价格

    Returns:
        ExecutionResult对象
    """
    # 选择执行算法
    if algorithm == "twap":
        num_slices = len(price_series)
        slices = twap_execution(total_shares, num_slices, price_series, urgency)
    elif algorithm == "vwap":
        if volume_profile is None:
            volume_profile = pd.Series(np.ones(len(price_series)) * 1000)
        slices = vwap_execution(total_shares, volume_profile, price_series, urgency)
    elif algorithm == "iceberg":
        slices = iceberg_order(total_shares, visible_ratio=0.1, price_series=price_series)
    elif algorithm == "adaptive":
        if volume_profile is None:
            volume_profile = pd.Series(np.ones(len(price_series)) * 1000)
        slices = adaptive_execution(total_shares, volume_profile, price_series, urgency)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # 计算统计
    filled_shares = sum(s.shares for s in slices)
    total_turnover = sum(s.turnover for s in slices)
    total_slippage = sum(s.slippage for s in slices)
    avg_price = total_turnover / filled_shares if filled_shares > 0 else 0.0

    # 计算执行时间
    if slices:
        execution_time = len(slices) * 30  # 假设每片30分钟
    else:
        execution_time = 0.0

    return ExecutionResult(
        order_id=order_id,
        code=code,
        direction=direction,
        total_shares=total_shares,
        filled_shares=filled_shares,
        avg_price=avg_price,
        total_turnover=total_turnover,
        total_slippage=total_slippage,
        execution_time=execution_time,
        status=ExecutionStatus.FILLED if filled_shares >= total_shares else ExecutionStatus.PARTIAL,
        slices=slices
    )
