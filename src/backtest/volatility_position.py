"""波动率仓位管理模块

基于目标波动率的仓位管理系统，支持：
1. EWMA 波动率计算
2. GARCH(1,1) 波动率计算（简化版）
3. 目标波动率仓位调整
4. 波动率状态检测
5. 波动率动量指标

Target Position = target_volatility / current_volatility × base_position

EWMA: λ = 0.5^(1/halflife)
GARCH(1,1): σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}

Volatility Regime:
- HIGH: vol > mean + 2*std
- ELEVATED: vol > mean + std
- LOW: vol < mean - std
- NORMAL: otherwise
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class VolatilityRegime(Enum):
    """波动率状态枚举"""
    HIGH = "high"      # 高波动率状态
    ELEVATED = "elevated"  # 中高波动率状态
    NORMAL = "normal"  # 正常波动率状态
    LOW = "low"       # 低波动率状态


@dataclass
class VolatilityResult:
    """波动率计算结果"""
    current_vol: float           # 当前波动率
    regime: VolatilityRegime     # 波动率状态
    regime_reason: str           # 状态判断原因
    volatility_percentile: float # 波动率百分位 (0-100)
    momentum: float             # 波动率动量


@dataclass
class TargetVolatilityPosition:
    """目标波动率仓位结果"""
    target_ratio: float         # 目标仓位比例 (0-1)
    adjusted_position: float    # 调整后仓位金额
    base_position: float        # 基础仓位金额
    current_volatility: float   # 当前波动率
    target_volatility: float    # 目标波动率
    volatility_regime: VolatilityRegime  # 波动率状态
    reason: str                 # 调整原因


class EWMAVolatilityCalculator:
    """
    EWMA 波动率计算器

    指数加权移动平均 (Exponentially Weighted Moving Average) 波动率

    Formula:
        λ = 0.5^(1/halflife)
        EWMA_vol = sqrt(EWMA(r²))

    Example:
        >>> calc = EWMAVolatilityCalculator(halflife=30)
        >>> vol = calc.calculate(pandas_series_of_returns)
    """

    def __init__(self, halflife: int = 30):
        """
        初始化 EWMA 计算器

        Args:
            halflife: 半衰期，表示权重衰减到一半所需的周期数
        """
        self.halflife = halflife
        self._lambda = 0.5 ** (1 / halflife)

    @property
    def lambda_value(self) -> float:
        """返回 lambda 值"""
        return self._lambda

    def calculate(self, returns: pd.Series) -> float:
        """
        计算 EWMA 波动率

        Args:
            returns: 收益率序列

        Returns:
            年化波动率

        Raises:
            ValueError: 如果数据不足或包含无效值
        """
        if len(returns) < 2:
            raise ValueError("需要至少 2 个数据点计算波动率")

        if returns.isna().any():
            raise ValueError("收益率序列包含 NaN 值")

        squared_returns = returns ** 2

        ewma_var = squared_returns.ewm(
            halflife=self.halflife,
            adjust=False
        ).mean().iloc[-1]

        if ewma_var <= 0:
            return 0.0

        daily_vol = math.sqrt(ewma_var)

        annualized_vol = daily_vol * math.sqrt(252)

        return annualized_vol

    def calculate_series(self, returns: pd.Series) -> pd.Series:
        """
        计算 EWMA 波动率序列

        Args:
            returns: 收益率序列

        Returns:
            年化波动率序列
        """
        squared_returns = returns ** 2

        ewma_var = squared_returns.ewm(
            halflife=self.halflife,
            adjust=False
        ).mean()

        daily_vol = ewma_var.apply(math.sqrt)

        annualized_vol = daily_vol * math.sqrt(252)

        return annualized_vol


class GARCHVolatilityCalculator:
    """
    GARCH(1,1) 波动率计算器（简化版）

    GARCH 模型假设波动率有持久性，用条件方差描述

    Formula:
        σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}

    其中：
        ω = 长期方差 (1 - α - β) × σ²_long
        α = 短期冲击系数 (ARCH 系数)
        β = 波动率持久性系数 (GARCH 系数)
        ε²_{t-1} = 上一期收益率平方
        σ²_{t-1} = 上一期条件方差

    Example:
        >>> calc = GARCHVolatilityCalculator()
        >>> vol = calc.calculate(pandas_series_of_returns)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.85,
        long_term_vol: Optional[float] = None
    ):
        """
        初始化 GARCH 计算器

        Args:
            alpha: ARCH 系数，代表短期冲击对波动率的影响
            beta: GARCH 系数，代表波动率的持久性
            long_term_vol: 长期波动率估计（年化），如果为 None 则从数据估计

        Note:
            alpha + beta < 1 确保模型平稳
            典型值：alpha ≈ 0.1, beta ≈ 0.85, alpha + beta ≈ 0.95
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha 和 beta 必须为正数")
        if alpha + beta >= 1:
            raise ValueError("alpha + beta 必须小于 1 以确保模型平稳")

        self.alpha = alpha
        self.beta = beta
        self.long_term_vol = long_term_vol

    def calculate(self, returns: pd.Series) -> float:
        """
        计算 GARCH(1,1) 波动率

        Args:
            returns: 收益率序列

        Returns:
            年化波动率

        Raises:
            ValueError: 如果数据不足
        """
        if len(returns) < 10:
            raise ValueError("GARCH 计算需要至少 10 个数据点")

        returns = returns.dropna()

        if len(returns) < 10:
            raise ValueError("GARCH 计算需要至少 10 个有效数据点")

        daily_returns = returns.values

        if self.long_term_vol is None:
            sample_vol = np.std(daily_returns, ddof=0)
            if sample_vol == 0:
                return 0.0
            long_term_var = sample_vol ** 2
        else:
            long_term_var = (self.long_term_vol / math.sqrt(252)) ** 2

        omega = (1 - self.alpha - self.beta) * long_term_var

        cond_var = long_term_var

        for r in daily_returns[:-1]:
            cond_var = omega + self.alpha * (r ** 2) + self.beta * cond_var

        if cond_var <= 0:
            return 0.0

        daily_vol = math.sqrt(cond_var)
        annualized_vol = daily_vol * math.sqrt(252)

        return annualized_vol

    def calculate_series(self, returns: pd.Series) -> pd.Series:
        """
        计算 GARCH(1,1) 波动率序列

        Args:
            returns: 收益率序列

        Returns:
            年化波动率序列
        """
        if len(returns) < 10:
            raise ValueError("GARCH 计算需要至少 10 个数据点")

        returns = returns.dropna().reset_index(drop=True)
        daily_returns = returns.values

        if self.long_term_vol is None:
            sample_vol = np.std(daily_returns, ddof=0)
            if sample_vol == 0:
                sample_vol = 1e-6
            long_term_var = sample_vol ** 2
        else:
            long_term_var = (self.long_term_vol / math.sqrt(252)) ** 2

        omega = (1 - self.alpha - self.beta) * long_term_var

        cond_var_series = np.zeros(len(daily_returns))
        cond_var_series[0] = long_term_var

        for t in range(1, len(daily_returns)):
            cond_var_series[t] = (
                omega +
                self.alpha * (daily_returns[t - 1] ** 2) +
                self.beta * cond_var_series[t - 1]
            )

        cond_var_series = np.maximum(cond_var_series, 1e-10)

        daily_vol = np.sqrt(cond_var_series)

        annualized_vol = daily_vol * math.sqrt(252)

        return pd.Series(annualized_vol, index=returns.index)


class VolatilityRegimeDetector:
    """
    波动率状态检测器

    基于历史波动率分布判断当前波动率状态

    Regime:
        HIGH: vol > mean + 2*std (高波动率)
        ELEVATED: vol > mean + std (中高波动率)
        NORMAL: mean - std <= vol <= mean + std (正常波动率)
        LOW: vol < mean - std (低波动率)
    """

    def __init__(self, lookback_period: int = 60):
        """
        初始化波动率状态检测器

        Args:
            lookback_period: 回溯期，用于计算波动率均值和标准差
        """
        self.lookback_period = lookback_period

    def detect(
        self,
        volatility_series: pd.Series,
        current_vol: Optional[float] = None
    ) -> VolatilityResult:
        """
        检测当前波动率状态

        Args:
            volatility_series: 历史波动率序列
            current_vol: 当前波动率，如果为 None 则使用序列最后一个值

        Returns:
            VolatilityResult 包含当前波动率、状态、百分位和动量
        """
        if len(volatility_series) < 5:
            return VolatilityResult(
                current_vol=current_vol if current_vol is not None else 0.0,
                regime=VolatilityRegime.NORMAL,
                regime_reason="数据不足，使用默认状态",
                volatility_percentile=50.0,
                momentum=0.0
            )

        if current_vol is None:
            current_vol = float(volatility_series.iloc[-1])

        if current_vol <= 0:
            return VolatilityResult(
                current_vol=current_vol,
                regime=VolatilityRegime.NORMAL,
                regime_reason="波动率为零或负数，使用默认状态",
                volatility_percentile=50.0,
                momentum=0.0
            )

        vol_array = volatility_series.values
        vol_mean = np.mean(vol_array)
        vol_std = np.std(vol_array, ddof=0)

        if vol_std == 0:
            vol_std = 1e-6

        upper_2std = vol_mean + 2 * vol_std
        upper_1std = vol_mean + vol_std
        lower_1std = vol_mean - vol_std

        if current_vol > upper_2std:
            regime = VolatilityRegime.HIGH
            reason = f"波动率 {current_vol:.4f} > mean+2std ({upper_2std:.4f})"
        elif current_vol > upper_1std:
            regime = VolatilityRegime.ELEVATED
            reason = f"波动率 {current_vol:.4f} > mean+std ({upper_1std:.4f})"
        elif current_vol < lower_1std:
            regime = VolatilityRegime.LOW
            reason = f"波动率 {current_vol:.4f} < mean-std ({lower_1std:.4f})"
        else:
            regime = VolatilityRegime.NORMAL
            reason = f"波动率 {current_vol:.4f} 在正常范围内"

        percentile = self._calculate_percentile(volatility_series, current_vol)

        momentum = self._calculate_momentum(volatility_series)

        return VolatilityResult(
            current_vol=current_vol,
            regime=regime,
            regime_reason=reason,
            volatility_percentile=percentile,
            momentum=momentum
        )

    def _calculate_percentile(
        self,
        volatility_series: pd.Series,
        current_vol: float
    ) -> float:
        """
        计算波动率百分位

        Args:
            volatility_series: 历史波动率序列
            current_vol: 当前波动率

        Returns:
            百分位 (0-100)
        """
        vol_array = volatility_series.values
        count_below = np.sum(vol_array < current_vol)
        percentile = (count_below / len(vol_array)) * 100
        return min(100.0, max(0.0, percentile))

    def _calculate_momentum(self, volatility_series: pd.Series) -> float:
        """
        计算波动率动量

        波动率动量 = 当前波动率 / 移动平均波动率 - 1

        Args:
            volatility_series: 历史波动率序列

        Returns:
            波动率动量 (正数表示波动率上升趋势)
        """
        if len(volatility_series) < 10:
            return 0.0

        recent_vol = volatility_series.iloc[-1]
        ma_vol = volatility_series.iloc[-10:].mean()

        if ma_vol <= 0:
            return 0.0

        momentum = (recent_vol / ma_vol) - 1
        return momentum


class VolatilityMomentumIndicator:
    """
    波动率动量指标

    用于判断波动率趋势：
    - 正动量：波动率上升趋势（市场可能进入高波动状态）
    - 负动量：波动率下降趋势（市场可能进入低波动状态）
    - 零动量：波动率平稳

    Formula:
        momentum = vol_current / vol_ma - 1
        acceleration = momentum_current / momentum_historical - 1
    """

    def __init__(self, ma_period: int = 10):
        """
        初始化波动率动量指标

        Args:
            ma_period: 移动平均周期
        """
        self.ma_period = ma_period

    def calculate(self, volatility_series: pd.Series) -> float:
        """
        计算波动率动量

        Args:
            volatility_series: 波动率序列

        Returns:
            波动率动量值
        """
        if len(volatility_series) < self.ma_period:
            return 0.0

        recent_vol = float(volatility_series.iloc[-1])
        ma_vol = float(volatility_series.iloc[-self.ma_period:].mean())

        if ma_vol <= 0:
            return 0.0

        return (recent_vol / ma_vol) - 1

    def calculate_acceleration(
        self,
        volatility_series: pd.Series,
        momentum_period: int = 5
    ) -> float:
        """
        计算波动率动量加速度

        动量加速度 = 当前动量 / 历史平均动量 - 1

        Args:
            volatility_series: 波动率序列
            momentum_period: 动量计算周期

        Returns:
            动量加速度
        """
        if len(volatility_series) < self.ma_period + momentum_period:
            return 0.0

        recent_vol = float(volatility_series.iloc[-1])
        ma_vol = float(volatility_series.iloc[-self.ma_period:].mean())

        historical_vol = float(volatility_series.iloc[
            -(self.ma_period + momentum_period):-self.ma_period
        ].mean())

        if historical_vol <= 0 or ma_vol <= 0:
            return 0.0

        current_momentum = (recent_vol / ma_vol) - 1
        historical_momentum = (ma_vol / historical_vol) - 1

        if abs(historical_momentum) < 1e-10:
            return 0.0

        return (current_momentum / historical_momentum) - 1

    def is_accelerating(self, volatility_series: pd.Series) -> bool:
        """
        判断波动率是否正在加速

        Args:
            volatility_series: 波动率序列

        Returns:
            True 如果波动率正在加速
        """
        acceleration = self.calculate_acceleration(volatility_series)
        return acceleration > 0.1


class TargetVolatilityPositionSizer:
    """
    目标波动率仓位管理器

    根据当前波动率与目标波动率的比率调整仓位

    Formula:
        target_ratio = target_volatility / current_volatility
        adjusted_position = base_position × target_ratio

    Example:
        >>> sizer = TargetVolatilityPositionSizer(target_volatility=0.20)
        >>> result = sizer.calculate_position(
        ...     base_position=100000,
        ...     current_volatility=0.25,
        ...     regime=VolatilityRegime.HIGH
        ... )
    """

    def __init__(
        self,
        target_volatility: float = 0.20,
        max_ratio: float = 2.0,
        min_ratio: float = 0.25
    ):
        """
        初始化目标波动率仓位管理器

        Args:
            target_volatility: 目标年化波动率，默认 20%
            max_ratio: 最大仓位调整比率，默认 2.0（仓位最多翻倍）
            min_ratio: 最小仓位调整比率，默认 0.25（仓位最多降到1/4）
        """
        self.target_volatility = target_volatility
        self.max_ratio = max_ratio
        self.min_ratio = min_ratio

    def calculate_target_ratio(self, current_volatility: float) -> float:
        """
        计算目标仓位比率

        Args:
            current_volatility: 当前年化波动率

        Returns:
            仓位调整比率 (0-1 表示降仓，>1 表示升仓)
        """
        if current_volatility <= 0:
            return self.max_ratio

        ratio = self.target_volatility / current_volatility

        ratio = max(self.min_ratio, min(self.max_ratio, ratio))

        return ratio

    def calculate_position(
        self,
        base_position: float,
        current_volatility: float,
        regime: Optional[VolatilityRegime] = None,
        regime_adjustment: bool = True
    ) -> TargetVolatilityPosition:
        """
        计算调整后的目标波动率仓位

        Args:
            base_position: 基础仓位金额
            current_volatility: 当前年化波动率
            regime: 波动率状态（可选，用于额外调整）
            regime_adjustment: 是否根据波动率状态额外调整

        Returns:
            TargetVolatilityPosition 包含完整仓位计算结果
        """
        if base_position <= 0:
            return TargetVolatilityPosition(
                target_ratio=0.0,
                adjusted_position=0.0,
                base_position=base_position,
                current_volatility=current_volatility,
                target_volatility=self.target_volatility,
                volatility_regime=regime if regime else VolatilityRegime.NORMAL,
                reason="基础仓位为零或负数"
            )

        if current_volatility <= 0:
            return TargetVolatilityPosition(
                target_ratio=0.0,
                adjusted_position=0.0,
                base_position=base_position,
                current_volatility=current_volatility,
                target_volatility=self.target_volatility,
                volatility_regime=regime if regime else VolatilityRegime.NORMAL,
                reason="波动率为零或负数，无法计算仓位"
            )

        base_ratio = self.calculate_target_ratio(current_volatility)

        final_ratio = base_ratio
        reason = f"目标波动率 {self.target_volatility:.2%} / 当前波动率 {current_volatility:.2%} = {base_ratio:.2f}"

        if regime_adjustment and regime is not None:
            if regime == VolatilityRegime.HIGH:
                final_ratio = final_ratio * 0.7
                reason += "，高波动率状态额外降仓 30%"
            elif regime == VolatilityRegime.ELEVATED:
                final_ratio = final_ratio * 0.85
                reason += "，中高波动率状态额外降仓 15%"
            elif regime == VolatilityRegime.LOW:
                final_ratio = final_ratio * 1.2
                reason += "，低波动率状态额外升仓 20%"

        final_ratio = max(self.min_ratio, min(self.max_ratio, final_ratio))

        adjusted_position = base_position * final_ratio

        return TargetVolatilityPosition(
            target_ratio=final_ratio,
            adjusted_position=adjusted_position,
            base_position=base_position,
            current_volatility=current_volatility,
            target_volatility=self.target_volatility,
            volatility_regime=regime if regime else VolatilityRegime.NORMAL,
            reason=reason
        )

    def get_position_multiplier(
        self,
        volatility_series: pd.Series,
        regime: VolatilityRegime
    ) -> float:
        """
        根据波动率序列获取仓位倍数

        用于在回测中快速获取仓位调整因子

        Args:
            volatility_series: 波动率序列
            regime: 当前波动率状态

        Returns:
            仓位倍数
        """
        if len(volatility_series) == 0:
            return 1.0

        current_vol = float(volatility_series.iloc[-1])
        target_ratio = self.calculate_target_ratio(current_vol)

        if regime == VolatilityRegime.HIGH:
            return target_ratio * 0.7
        elif regime == VolatilityRegime.ELEVATED:
            return target_ratio * 0.85
        elif regime == VolatilityRegime.LOW:
            return target_ratio * 1.2
        else:
            return target_ratio


def calculate_volatility_from_prices(
    prices: pd.Series,
    method: str = "ewma",
    **kwargs
) -> float:
    """
    从价格序列计算波动率的便捷函数

    Args:
        prices: 价格序列
        method: 计算方法 "ewma" 或 "garch"
        **kwargs: 传递给具体计算器的参数

    Returns:
        年化波动率

    Raises:
        ValueError: 如果方法不支持或参数无效
    """
    if len(prices) < 2:
        raise ValueError("价格序列需要至少 2 个数据点")

    returns = prices.pct_change().dropna()

    if len(returns) < 2:
        raise ValueError("有效收益率数据不足")

    if method == "ewma":
        halflife = kwargs.get("halflife", 30)
        calc = EWMAVolatilityCalculator(halflife=halflife)
        return calc.calculate(returns)
    elif method == "garch":
        alpha = kwargs.get("alpha", 0.1)
        beta = kwargs.get("beta", 0.85)
        long_term_vol = kwargs.get("long_term_vol", None)
        calc = GARCHVolatilityCalculator(
            alpha=alpha,
            beta=beta,
            long_term_vol=long_term_vol
        )
        return calc.calculate(returns)
    else:
        raise ValueError(f"不支持的波动率计算方法: {method}")


def detect_volatility_regime(
    volatility_history: pd.Series,
    current_vol: Optional[float] = None
) -> VolatilityResult:
    """
    检测波动率状态的便捷函数

    Args:
        volatility_history: 历史波动率序列
        current_vol: 当前波动率，如果为 None 则使用序列最后一个值

    Returns:
        VolatilityResult
    """
    detector = VolatilityRegimeDetector()
    return detector.detect(volatility_history, current_vol)
