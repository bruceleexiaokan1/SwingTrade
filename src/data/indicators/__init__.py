"""技术指标计算模块

提供波段交易所需的技术指标计算：
- MA（移动平均线）
- MACD（指数平滑异同移动平均线）
- RSI（相对强弱指标）
- Bollinger Bands（布林带）
- ATR（平均真实波幅）
- ADX（平均方向指数）
- Volume（成交量指标）
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import pandas as pd


# 导出所有指标计算函数
from .ma import calculate_ma, golden_cross, death_cross, ma_bullish, ma_bearish
from .macd import calculate_macd, macd_bullish, macd_bearish, macd_crossover
from .rsi import calculate_rsi, rsi_oversold, rsi_overbought, rsi_extreme
from .bollinger import calculate_bollinger, bollinger_squeeze, bollinger_breakout_upper, bollinger_breakout_lower
from .atr import calculate_atr, atr_stop_loss, atr_trailing_stop
from .adx import calculate_adx, adx_strong_trend, adx_weak_trend, adx_rising
from .adx import adx_bullish_signal, adx_bearish_signal, adx_trend_strength
from .volume import calculate_volume_ma, volume_surge, volume_shrink, volume_ratio
from .chan_theory import calculate_chan, detect_chan_signals, ChanTheory, ChanBuySignal
from .signals import SwingSignals, detect_rsi_divergence
from .crowding import (
    turnover_crowding,
    momentum_crowding,
    fund_flow_crowding,
    position_concentration_hhi,
    correlation_breakdown_detection,
    a_share_crowding_indicator,
)
from .hmm_model import (
    HMMModel,
    HMMMarketRegime,
    HMMState,
    HMMResult,
    calculate_hmm_regime,
    detect_market_regime,
)
from .event_driven import (
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
from .options_volatility import (
    BSPriceResult,
    ImpliedVolResult,
    VolSurfacePoint,
    VolSurfaceResult,
    StrategyResult,
    bs_call_price,
    bs_put_price,
    calculate_bs_greeks,
    calculate_full_bs,
    implied_volatility,
    implied_volatility_brent,
    build_volatility_surface,
    analyze_volatility_smile,
    long_straddle,
    short_straddle,
    bull_call_spread,
    iron_condor,
    risk_reversal,
    vol_mean_reversion_signal,
    calendar_spread,
    calculate_portfolio_greeks,
    hedge_delta,
    estimate_iv_index,
    pnl_at_expiry,
)
from .fama_french import (
    FactorValues,
    FactorRegressionResult,
    FactorPortfolioResult,
    build_mkt_factor,
    build_smb_factor,
    build_hml_factor,
    build_rmw_factor,
    build_cma_factor,
    build_ff5_factors,
    factor_regression,
    batch_factor_regression,
    factor_validity_test,
    factor_rotation_weights,
    barra_style_factors,
    calculate_style_exposure,
    ff5_portfolio_optimization,
    calculate_factor_exposures,
    rolling_factor_analysis,
)
from .wave import (
    WaveType,
    WaveDirection,
    WavePoint,
    WaveResult,
    WaveIndicators,
    calculate_wave_levels,
)
from .resonance import (
    ResonanceLevel,
    ResonanceCondition,
    ResonanceResult,
    calculate_resonance_score,
)
from .sector_signals import SectorSignalResult, SectorSignals
from .sector_rs import SectorRelativeStrength
from .fundamental import (
    calculate_roe,
    calculate_roa,
    calculate_gross_margin,
    calculate_net_margin,
    calculate_revenue_growth,
    calculate_pe,
    calculate_pb,
    calculate_ps,
    calculate_debt_ratio,
    calculate_current_ratio,
    assess_financial_health,
)
from .microstructure import (
    calculate_amihud_illiq,
    calculate_order_imbalance,
    calculate_vpin,
    detect_volume_anomaly,
    liquidity_regime_detection,
)


@dataclass
class MAResult:
    """MA计算结果（单行）"""
    date: str
    ma5: Optional[float] = None
    ma10: Optional[float] = None
    ma20: Optional[float] = None
    ma60: Optional[float] = None


@dataclass
class MACDResult:
    """MACD计算结果（单行）"""
    date: str
    dif: float       # DIF线（快线）
    dem: float       # DEA线（慢线）
    hist: float      # 柱状图


@dataclass
class RSIResult:
    """RSI计算结果（单行）"""
    date: str
    rsi6: Optional[float] = None
    rsi14: Optional[float] = None


@dataclass
class BollingerResult:
    """布林带计算结果（单行）"""
    date: str
    upper: float      # 上轨
    middle: float     # 中轨（MA）
    lower: float     # 下轨
    bandwidth: float # 带宽
    position: float  # %B位置


@dataclass
class ATRResult:
    """ATR计算结果（单行）"""
    date: str
    tr: float         # 真实波幅
    atr: float       # ATR值
    atr_pct: float  # ATR百分比（ATR/收盘价）


@dataclass
class ADXResult:
    """ADX计算结果（单行）"""
    date: str
    adx: float        # ADX 值（趋势强度）
    plus_di: float   # +DI（多头方向指标）
    minus_di: float  # -DI（空头方向指标）


@dataclass
class SignalResult:
    """综合信号结果"""
    date: str
    trend: str           # uptrend/downtrend/sideways
    entry_signal: str    # golden/breakout/none
    entry_confidence: float
    exit_signal: str     # stop_loss/take_profit/trailing/none
    exit_confidence: float
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None


__all__ = [
    # 数据类
    "MAResult",
    "MACDResult",
    "RSIResult",
    "BollingerResult",
    "ATRResult",
    "ADXResult",
    "SignalResult",
    # MA函数
    "calculate_ma",
    "golden_cross",
    "death_cross",
    "ma_bullish",
    "ma_bearish",
    # MACD函数
    "calculate_macd",
    "macd_bullish",
    "macd_bearish",
    "macd_crossover",
    # RSI函数
    "calculate_rsi",
    "rsi_oversold",
    "rsi_overbought",
    "rsi_extreme",
    # 布林带函数
    "calculate_bollinger",
    "bollinger_squeeze",
    "bollinger_breakout_upper",
    "bollinger_breakout_lower",
    # ATR函数
    "calculate_atr",
    "atr_stop_loss",
    "atr_trailing_stop",
    # ADX函数
    "calculate_adx",
    "adx_strong_trend",
    "adx_weak_trend",
    "adx_rising",
    "adx_bullish_signal",
    "adx_bearish_signal",
    "adx_trend_strength",
    # 成交量函数
    "calculate_volume_ma",
    "volume_surge",
    "volume_shrink",
    "volume_ratio",
    # 信号检测
    "SwingSignals",
    "detect_rsi_divergence",
    # 缠论
    "calculate_chan",
    "detect_chan_signals",
    "ChanTheory",
    "ChanBuySignal",
    # 拥挤度
    "turnover_crowding",
    "momentum_crowding",
    "fund_flow_crowding",
    "position_concentration_hhi",
    "correlation_breakdown_detection",
    "a_share_crowding_indicator",
    # HMM市场状态
    "HMMModel",
    "HMMMarketRegime",
    "HMMState",
    "HMMResult",
    "calculate_hmm_regime",
    "detect_market_regime",
    # 事件驱动
    "EarningsSurpriseResult",
    "JiejinRiskResult",
    "RebalanceResult",
    "ShareholderBuyResult",
    "EventSignal",
    "calculate_earnings_surprise",
    "earnings_trend_analysis",
    "calculate_jiejin_risk",
    "scan_jiejin_calendar",
    "calculate_rebalance_effect",
    "analyze_shareholder_buying",
    "calculate_event_score",
    "get_event_calendar",
    "is_policy_sensitive_period",
    "batch_calculate_jiejin_risk",
    # 期权波动率
    "BSPriceResult",
    "ImpliedVolResult",
    "VolSurfacePoint",
    "VolSurfaceResult",
    "StrategyResult",
    "bs_call_price",
    "bs_put_price",
    "calculate_bs_greeks",
    "calculate_full_bs",
    "implied_volatility",
    "implied_volatility_brent",
    "build_volatility_surface",
    "analyze_volatility_smile",
    "long_straddle",
    "short_straddle",
    "bull_call_spread",
    "iron_condor",
    "risk_reversal",
    "vol_mean_reversion_signal",
    "calendar_spread",
    "calculate_portfolio_greeks",
    "hedge_delta",
    "estimate_iv_index",
    "pnl_at_expiry",
    # Fama-French因子
    "FactorValues",
    "FactorRegressionResult",
    "FactorPortfolioResult",
    "build_mkt_factor",
    "build_smb_factor",
    "build_hml_factor",
    "build_rmw_factor",
    "build_cma_factor",
    "build_ff5_factors",
    "factor_regression",
    "batch_factor_regression",
    "factor_validity_test",
    "factor_rotation_weights",
    "barra_style_factors",
    "calculate_style_exposure",
    "ff5_portfolio_optimization",
    "calculate_factor_exposures",
    "rolling_factor_analysis",
    # 波浪理论
    "WaveType",
    "WaveDirection",
    "WavePoint",
    "WaveResult",
    "WaveIndicators",
    "calculate_wave_levels",
    # 共振检测
    "ResonanceLevel",
    "ResonanceCondition",
    "ResonanceResult",
    "calculate_resonance_score",
    # 板块信号
    "SectorSignalResult",
    "SectorSignals",
    # 板块相对强度
    "SectorRelativeStrength",
    # 基本面量化
    "calculate_roe",
    "calculate_roa",
    "calculate_gross_margin",
    "calculate_net_margin",
    "calculate_revenue_growth",
    "calculate_pe",
    "calculate_pb",
    "calculate_ps",
    "calculate_debt_ratio",
    "calculate_current_ratio",
    "assess_financial_health",
    # 市场微观结构
    "calculate_amihud_illiq",
    "calculate_order_imbalance",
    "calculate_vpin",
    "detect_volume_anomaly",
    "liquidity_regime_detection",
]
