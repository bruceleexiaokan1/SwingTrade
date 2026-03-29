"""期权波动率进阶：完整波动率曲面与套利

实现内容：
- Black-Scholes期权定价
- 隐含波动率计算
- 波动率曲面构建
- 波动率交易策略（跨式、价差、铁秃鹰等）
- 希腊字母计算
- 波动率套利逻辑
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import norm
from scipy.optimize import brentq, newton
import logging

logger = logging.getLogger(__name__)


@dataclass
class BSPriceResult:
    """BS定价结果"""
    call_price: float
    put_price: float
    call_delta: float
    put_delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


@dataclass
class ImpliedVolResult:
    """隐含波动率结果"""
    iv: float
    vega: float  # 用于迭代
    converged: bool
    iterations: int


@dataclass
class VolSurfacePoint:
    """波动率曲面上的点"""
    strike: float
    expiry: float
    moneyness: float  # K/S
    iv: float
    option_type: str  # call/put


@dataclass
class VolSurfaceResult:
    """波动率曲面结果"""
    surface: List[VolSurfacePoint]
    term_structure: Dict[float, float]  # expiry -> atm_vol
    skew: float  # put skew
    term_shape: str  # contango/backwardation


@dataclass
class StrategyResult:
    """期权策略结果"""
    strategy_name: str
    net_cost: float
    max_profit: float
    max_loss: float
    upper_breakeven: Optional[float]
    lower_breakeven: Optional[float]
    delta: float
    gamma: float
    theta: float
    vega: float


# ==================== Black-Scholes定价 ====================

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes看涨期权定价

    Args:
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率

    Returns:
        期权价格
    """
    if T <= 0:
        return max(S - K, 0)

    if sigma <= 0:
        # sigma=0时，BS简化为内在价值
        return max(S - K * np.exp(-r * T), 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes看跌期权定价

    Args:
        S: 标的价格
        K:行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率

    Returns:
        期权价格
    """
    if T <= 0:
        return max(K - S, 0)

    if sigma <= 0:
        # sigma=0时，BS简化为内在价值
        return max(K * np.exp(-r * T) - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def calculate_bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call'
) -> Dict[str, float]:
    """
    计算BS希腊字母

    Args:
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        option_type: call/put

    Returns:
        希腊字母字典
    """
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma（对call和put相同）
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == 'call':
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        theta = (term1 + term2) / 365  # 转为每天
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = (term1 + term2) / 365

    # Vega（对call和put相同）
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # 归一化到1%波动率变化

    # Rho
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }


def calculate_full_bs(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> BSPriceResult:
    """
    计算完整的BS价格和希腊字母

    Args:
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率

    Returns:
        BSPriceResult
    """
    call_price = bs_call_price(S, K, T, r, sigma)
    put_price = bs_put_price(S, K, T, r, sigma)

    call_greeks = calculate_bs_greeks(S, K, T, r, sigma, 'call')
    put_greeks = calculate_bs_greeks(S, K, T, r, sigma, 'put')

    return BSPriceResult(
        call_price=call_price,
        put_price=put_price,
        call_delta=call_greeks['delta'],
        put_delta=put_greeks['delta'],
        gamma=call_greeks['gamma'],
        theta=call_greeks['theta'],
        vega=call_greeks['vega'],
        rho=call_greeks['rho']
    )


# ==================== 隐含波动率计算 ====================

def implied_volatility(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = 'call',
    max_iterations: int = 100,
    tol: float = 1e-6
) -> ImpliedVolResult:
    """
    计算隐含波动率（牛顿法）

    Args:
        option_price: 期权市场价格
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        option_type: call/put
        max_iterations: 最大迭代次数
        tol: 收敛容差

    Returns:
        ImpliedVolResult
    """
    if T <= 0:
        return ImpliedVolResult(iv=0, vega=0, converged=False, iterations=0)

    # 初始猜测
    sigma = 0.3

    for i in range(max_iterations):
        if option_type == 'call':
            price = bs_call_price(S, K, T, r, sigma)
        else:
            price = bs_put_price(S, K, T, r, sigma)

        # Greeks
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # vega归一化

        # 误差
        error = price - option_price

        if abs(error) < tol:
            return ImpliedVolResult(
                iv=sigma,
                vega=vega,
                converged=True,
                iterations=i + 1
            )

        # 更新（牛顿法）
        if vega != 0:
            sigma -= error / vega / 100

        # 边界检查
        sigma = max(0.01, min(sigma, 2.0))

    return ImpliedVolResult(
        iv=sigma,
        vega=vega if 'vega' in locals() else 0,
        converged=False,
        iterations=max_iterations
    )


def implied_volatility_brent(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = 'call'
) -> float:
    """
    使用Brent方法计算隐含波动率（更稳健）

    Args:
        option_price: 期权市场价格
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        option_type: call/put

    Returns:
        隐含波动率
    """
    if T <= 0:
        return 0

    def objective(sigma):
        if option_type == 'call':
            return bs_call_price(S, K, T, r, sigma) - option_price
        else:
            return bs_put_price(S, K, T, r, sigma) - option_price

    try:
        iv = brentq(objective, 0.01, 2.0)
        return iv
    except ValueError:
        return 0


# ==================== 波动率曲面 ====================

def build_volatility_surface(
    option_chain: pd.DataFrame,
    S: float,
    r: float,
    price_col: str = 'price',
    strike_col: str = 'strike',
    expiry_col: str = 'expiry',
    type_col: str = 'type'
) -> VolSurfaceResult:
    """
    构建波动率曲面

    Args:
        option_chain: 期权链数据
        S: 标的资产价格
        r: 无风险利率
        price_col: 价格列名
        strike_col: 行权价列名
        expiry_col: 到期时间列名
        type_col: 类型列名 (call/put)

    Returns:
        VolSurfaceResult
    """
    surface = []

    for _, row in option_chain.iterrows():
        K = row[strike_col]
        T = row[expiry_col]
        market_price = row[price_col]
        opt_type = row.get(type_col, 'call')

        # 计算隐含波动率
        if T > 0:
            iv_result = implied_volatility(market_price, S, K, T, r, opt_type)
            iv = iv_result.iv
        else:
            iv = 0

        # 计算货币程度
        moneyness = K / S

        surface.append(VolSurfacePoint(
            strike=K,
            expiry=T,
            moneyness=moneyness,
            iv=iv,
            option_type=opt_type
        ))

    # 构建期限结构
    term_structure = {}
    for point in surface:
        if point.expiry not in term_structure:
            # 找ATM波动率
            atm_points = [p for p in surface
                         if abs(p.expiry - point.expiry) < 0.001
                         and abs(p.moneyness - 1.0) < 0.05]
            if atm_points:
                term_structure[point.expiry] = np.mean([p.iv for p in atm_points])

    # 计算put skew（ATM put IV - OTM put IV）
    atm_vol = term_structure.get(sorted(term_structure.keys())[0], 0) if term_structure else 0
    otm_puts = [p.iv for p in surface if p.moneyness < 0.95 and p.option_type == 'put']
    otm_put_vol = np.mean(otm_puts) if otm_puts else atm_vol
    skew = atm_vol - otm_put_vol

    # 判断期限结构形态
    if len(term_structure) >= 2:
        sorted_terms = sorted(term_structure.keys())
        short_term = np.mean([term_structure[t] for t in sorted_terms[:2]])
        long_term = np.mean([term_structure[t] for t in sorted_terms[-2:]])
        term_shape = '贴水（Backwardation）' if short_term > long_term else '升水（Contango）'
    else:
        term_shape = '未知'

    return VolSurfaceResult(
        surface=surface,
        term_structure=term_structure,
        skew=skew,
        term_shape=term_shape
    )


def analyze_volatility_smile(surface: List[VolSurfacePoint]) -> Dict:
    """
    分析波动率微笑

    Args:
        surface: 波动率曲面点列表

    Returns:
        微笑分析结果
    """
    # 按到期分组
    by_expiry = {}
    for point in surface:
        if point.expiry not in by_expiry:
            by_expiry[point.expiry] = []
        by_expiry[point.expiry].append(point)

    results = {}
    for expiry, points in by_expiry.items():
        # ATM附近
        atm_points = [p for p in points if abs(p.moneyness - 1.0) < 0.05]
        atm_vol = np.mean([p.iv for p in atm_points]) if atm_points else 0

        # OTM put
        otm_puts = [p.iv for p in points if p.moneyness < 0.95 and p.option_type == 'put']
        otm_put_vol = np.mean(otm_puts) if otm_puts else atm_vol

        # OTM call
        otm_calls = [p.iv for p in points if p.moneyness > 1.05 and p.option_type == 'call']
        otm_call_vol = np.mean(otm_calls) if otm_calls else atm_vol

        results[f'{expiry:.2f}'] = {
            'atm_vol': atm_vol,
            'put_skew': atm_vol - otm_put_vol,
            'call_skew': otm_call_vol - atm_vol,
            'smile': '左偏' if otm_put_vol > otm_call_vol else '右偏'
        }

    return results


# ==================== 期权交易策略 ====================

def long_straddle(
    S: float,
    K: float,
    T: float,
    r: float,
    vol: float
) -> StrategyResult:
    """
    买入跨式期权

    适用：预期大幅波动，但不确定方向

    Args:
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        vol: 隐含波动率

    Returns:
        StrategyResult
    """
    call_price = bs_call_price(S, K, T, r, vol)
    put_price = bs_put_price(S, K, T, r, vol)
    total_cost = call_price + put_price

    # 盈亏平衡点
    upper_breakeven = K + total_cost
    lower_breakeven = K - total_cost

    # 希腊字母
    call_greeks = calculate_bs_greeks(S, K, T, r, vol, 'call')
    put_greeks = calculate_bs_greeks(S, K, T, r, vol, 'put')

    net_delta = call_greeks['delta'] + put_greeks['delta']
    net_gamma = call_greeks['gamma'] + put_greeks['gamma']
    net_theta = call_greeks['theta'] + put_greeks['theta']
    net_vega = call_greeks['vega'] + put_greeks['vega']

    return StrategyResult(
        strategy_name='Long Straddle',
        net_cost=total_cost,
        max_profit=float('inf'),
        max_loss=total_cost,
        upper_breakeven=upper_breakeven,
        lower_breakeven=lower_breakeven,
        delta=net_delta,
        gamma=net_gamma,
        theta=net_theta,
        vega=net_vega
    )


def short_straddle(
    S: float,
    K: float,
    T: float,
    r: float,
    vol: float
) -> StrategyResult:
    """
    卖出跨式期权

    适用：预期震荡，不知道会震荡多少
    风险：方向大幅波动会亏损

    Args:
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        vol: 隐含波动率

    Returns:
        StrategyResult
    """
    call_price = bs_call_price(S, K, T, r, vol)
    put_price = bs_put_price(S, K, T, r, vol)
    premium = call_price + put_price

    # 盈亏平衡点
    upper_breakeven = K + premium
    lower_breakeven = K - premium

    # 希腊字母（取负）
    call_greeks = calculate_bs_greeks(S, K, T, r, vol, 'call')
    put_greeks = calculate_bs_greeks(S, K, T, r, vol, 'put')

    return StrategyResult(
        strategy_name='Short Straddle',
        net_cost=-premium,
        max_profit=premium,
        max_loss=float('inf'),
        upper_breakeven=upper_breakeven,
        lower_breakeven=lower_breakeven,
        delta=-(call_greeks['delta'] + put_greeks['delta']),
        gamma=-(call_greeks['gamma'] + put_greeks['gamma']),
        theta=-(call_greeks['theta'] + put_greeks['theta']),
        vega=-(call_greeks['vega'] + put_greeks['vega'])
    )


def bull_call_spread(
    S: float,
    K1: float,
    K2: float,
    T: float,
    r: float,
    vol: float
) -> StrategyResult:
    """
    牛市看涨价差

    买入低行权价call，卖出高行权价call
    适用：温和看多，想降低成本

    Args:
        S: 标的价格
        K1: 买入行权价（较低）
        K2: 卖出行权价（较高）
        T: 到期时间（年）
        r: 无风险利率
        vol: 隐含波动率

    Returns:
        StrategyResult
    """
    call1 = bs_call_price(S, K1, T, r, vol)  # 买入
    call2 = bs_call_price(S, K2, T, r, vol)  # 卖出

    net_cost = call1 - call2
    max_profit = (K2 - K1) - net_cost if net_cost > 0 else (K2 - K1) + abs(net_cost)
    max_loss = net_cost

    # 希腊字母
    greeks1 = calculate_bs_greeks(S, K1, T, r, vol, 'call')
    greeks2 = calculate_bs_greeks(S, K2, T, r, vol, 'call')

    return StrategyResult(
        strategy_name='Bull Call Spread',
        net_cost=net_cost,
        max_profit=max_profit,
        max_loss=max_loss,
        upper_breakeven=K1 + net_cost,
        lower_breakeven=None,
        delta=greeks1['delta'] - greeks2['delta'],
        gamma=greeks1['gamma'] - greeks2['gamma'],
        theta=greeks1['theta'] - greeks2['theta'],
        vega=greeks1['vega'] - greeks2['vega']
    )


def iron_condor(
    S: float,
    K1: float,
    K2: float,
    K3: float,
    K4: float,
    T: float,
    r: float,
    vol: float
) -> StrategyResult:
    """
    铁秃鹰策略

    卖出put spread + 卖出call spread
    适用：预期股价在一定区间震荡

    Args:
        S: 标的价格
        K1: 买入put行权价（最低）
        K2: 卖出put行权价
        K3: 卖出call行权价
        K4: 买入call行权价（最高）
        T: 到期时间（年）
        r: 无风险利率
        vol: 隐含波动率

    Returns:
        StrategyResult
    """
    # Put spread
    put_buy = bs_put_price(S, K1, T, r, vol)
    put_sell = bs_put_price(S, K2, T, r, vol)

    # Call spread
    call_buy = bs_call_price(S, K4, T, r, vol)
    call_sell = bs_call_price(S, K3, T, r, vol)

    net_credit = (put_sell - put_buy) + (call_sell - call_buy)

    # 最大亏损
    max_loss = (K2 - K1) + (K4 - K3) - net_credit

    # 希腊字母
    put_buy_greeks = calculate_bs_greeks(S, K1, T, r, vol, 'put')
    put_sell_greeks = calculate_bs_greeks(S, K2, T, r, vol, 'put')
    call_buy_greeks = calculate_bs_greeks(S, K4, T, r, vol, 'call')
    call_sell_greeks = calculate_bs_greeks(S, K3, T, r, vol, 'call')

    net_delta = (put_sell_greeks['delta'] - put_buy_greeks['delta'] +
                 call_sell_greeks['delta'] - call_buy_greeks['delta'])
    net_gamma = (put_sell_greeks['gamma'] - put_buy_greeks['gamma'] +
                 call_sell_greeks['gamma'] - call_buy_greeks['gamma'])
    net_theta = (put_sell_greeks['theta'] - put_buy_greeks['theta'] +
                 call_sell_greeks['theta'] - call_buy_greeks['theta'])
    net_vega = (put_sell_greeks['vega'] - put_buy_greeks['vega'] +
                call_sell_greeks['vega'] - call_buy_greeks['vega'])

    return StrategyResult(
        strategy_name='Iron Condor',
        net_cost=-net_credit,
        max_profit=net_credit,
        max_loss=max_loss,
        upper_breakeven=K3 + net_credit,
        lower_breakeven=K2 - net_credit,
        delta=net_delta,
        gamma=net_gamma,
        theta=net_theta,
        vega=net_vega
    )


def risk_reversal(
    S: float,
    T: float,
    r: float,
    vol: float,
    skew_adjustment: float = 0.05
) -> StrategyResult:
    """
    风险逆转策略（看涨）

    卖出低行权价put，买入高行权价call
    适用：强烈看多，想降低买入成本

    Args:
        S: 标的价格
        T: 到期时间（年）
        r: 无风险利率
        vol: 隐含波动率
        skew_adjustment: 波动率偏度调整

    Returns:
        StrategyResult
    """
    K_put = S * 0.95   # 虚值put
    K_call = S * 1.05  # 虚值call

    # 波动率（考虑skew）
    vol_put = vol + skew_adjustment
    vol_call = vol - skew_adjustment

    put_price = bs_put_price(S, K_put, T, r, vol_put)
    call_price = bs_call_price(S, K_call, T, r, vol_call)

    net_cost = call_price - put_price

    # 希腊字母
    put_greeks = calculate_bs_greeks(S, K_put, T, r, vol_put, 'put')
    call_greeks = calculate_bs_greeks(S, K_call, T, r, vol_call, 'call')

    return StrategyResult(
        strategy_name='Risk Reversal (Bullish)',
        net_cost=net_cost,
        max_profit=float('inf'),
        max_loss=float('-inf') if net_cost > 0 else abs(net_cost),
        upper_breakeven=K_call + net_cost if net_cost > 0 else None,
        lower_breakeven=K_put - net_cost if net_cost < 0 else None,
        delta=call_greeks['delta'] - put_greeks['delta'],
        gamma=call_greeks['gamma'] - put_greeks['gamma'],
        theta=call_greeks['theta'] - put_greeks['theta'],
        vega=call_greeks['vega'] - put_greeks['vega']
    )


# ==================== 波动率套利 ====================

def vol_mean_reversion_signal(
    iv: float,
    hv: float,
    iv_history_mean: float,
    hv_history_mean: float
) -> Dict:
    """
    波动率均值回归信号

    原理：
      IV > HV → 期权被高估，卖出期权
      IV < HV → 期权被低估，买入期权

    Args:
        iv: 当前隐含波动率
        hv: 当前历史波动率
        iv_history_mean: 历史IV均值
        hv_history_mean: 历史HV均值

    Returns:
        信号字典
    """
    ivhv_ratio = iv / hv if hv > 0 else 1

    if ivhv_ratio > 1.3:
        return {
            'signal': 'SELL_VOL',
            'action': '卖出期权/跨式',
            'reason': f'IV({iv:.1%})远大于HV({hv:.1%})',
            'target_iv': hv_history_mean,
            'potential_profit': f'预期IV回归到{hv_history_mean:.1%}'
        }
    elif ivhv_ratio < 0.8:
        return {
            'signal': 'BUY_VOL',
            'action': '买入期权/跨式',
            'reason': f'IV({iv:.1%})小于HV({hv:.1%})',
            'target_iv': hv_history_mean,
            'potential_profit': f'预期IV上升到{hv_history_mean:.1%}'
        }
    else:
        return {
            'signal': 'NEUTRAL',
            'action': '观望',
            'reason': 'IV/HV比值正常'
        }


def calendar_spread(
    S: float,
    K: float,
    T1: float,
    T2: float,
    r: float,
    vol1: float,
    vol2: float
) -> StrategyResult:
    """
    日历价差

    卖出近月期权，买入远月期权
    适用：波动率期限结构异常

    Args:
        S: 标的价格
        K: 行权价
        T1: 近月到期时间（年）
        T2: 远月到期时间（年）
        r: 无风险利率
        vol1: 近月IV
        vol2: 远月IV

    Returns:
        StrategyResult
    """
    call_short = bs_call_price(S, K, T1, r, vol1)
    call_long = bs_call_price(S, K, T2, r, vol2)

    net_cost = call_long - call_short

    greeks_short = calculate_bs_greeks(S, K, T1, r, vol1, 'call')
    greeks_long = calculate_bs_greeks(S, K, T2, r, vol2, 'call')

    return StrategyResult(
        strategy_name='Calendar Spread',
        net_cost=net_cost,
        max_profit=float('inf'),
        max_loss=net_cost,
        upper_breakeven=None,
        lower_breakeven=None,
        delta=greeks_long['delta'] - greeks_short['delta'],
        gamma=greeks_long['gamma'] - greeks_short['gamma'],
        theta=greeks_long['theta'] - greeks_short['theta'],
        vega=greeks_long['vega'] - greeks_short['vega']
    )


# ==================== 组合希腊字母 ====================

def calculate_portfolio_greeks(
    positions: List[Dict],
    S: float,
    r: float
) -> Dict[str, float]:
    """
    计算组合希腊字母

    Args:
        positions: 持仓列表
                  [{'type': 'call/put', 'direction': 1/-1, 'strike': K,
                    'T': T, 'vol': sigma, 'size': contracts}]
        S: 标的资产价格
        r: 无风险利率

    Returns:
        组合希腊字母
    """
    total_delta = 0
    total_gamma = 0
    total_theta = 0
    total_vega = 0

    for pos in positions:
        multiplier = pos['size'] * pos['direction']
        contract_multiplier = 100  # 每张合约对应100股

        if pos['type'] == 'call':
            greeks = calculate_bs_greeks(S, pos['strike'], pos['T'], r, pos['vol'], 'call')
        else:
            greeks = calculate_bs_greeks(S, pos['strike'], pos['T'], r, pos['vol'], 'put')

        total_delta += greeks['delta'] * multiplier * contract_multiplier
        total_gamma += greeks['gamma'] * multiplier * contract_multiplier
        total_theta += greeks['theta'] * multiplier * contract_multiplier
        total_vega += greeks['vega'] * multiplier * contract_multiplier

    return {
        'delta': total_delta,
        'gamma': total_gamma,
        'theta': total_theta,
        'vega': total_vega
    }


def hedge_delta(portfolio_delta: float, S: float, target_delta: float = 0) -> Tuple[float, str]:
    """
    计算Delta对冲需求

    Args:
        portfolio_delta: 组合Delta
        S: 标的资产价格
        target_delta: 目标Delta（默认0）

    Returns:
        (需要买入/卖出的股数, 方向)
    """
    delta_to_hedge = target_delta - portfolio_delta

    if abs(delta_to_hedge) < 1:
        return 0, 'none'

    shares_needed = delta_to_hedge  # 正delta需要卖出股票对冲

    if shares_needed > 0:
        return shares_needed, 'buy'
    else:
        return abs(shares_needed), 'sell'


# ==================== 便捷函数 ====================

def estimate_iv_index(
    option_chain: pd.DataFrame,
    S: float,
    r: float,
    price_col: str = 'price',
    strike_col: str = 'strike',
    expiry_col: str = 'expiry',
    type_col: str = 'type'
) -> float:
    """
    估算A股IV指数

    方法：ATM期权加权平均

    Args:
        option_chain: 期权链数据
        S: 标的资产价格
        r: 无风险利率
        price_col: 价格列名
        strike_col: 行权价列名
        expiry_col: 到期时间列名
        type_col: 类型列名

    Returns:
        IV指数
    """
    atm_options = []

    for _, row in option_chain.iterrows():
        K = row[strike_col]
        T = row[expiry_col]

        if 0.9 < K / S < 1.1 and T > 0:
            market_price = row[price_col]
            opt_type = row.get(type_col, 'call')
            iv = implied_volatility_brent(market_price, S, K, T, r, opt_type)
            weight = 1 / T  # 短期权重更大
            atm_options.append((iv, weight))

    if not atm_options:
        return 0

    weighted_sum = sum(iv * w for iv, w in atm_options)
    total_weight = sum(w for _, w in atm_options)

    return weighted_sum / total_weight


def pnl_at_expiry(
    strategy: StrategyResult,
    S_T: float,
    K: float
) -> float:
    """
    计算策略到期盈亏

    Args:
        strategy: 策略结果
        S_T: 到期标的价格
        K: 行权价

    Returns:
        盈亏金额
    """
    if strategy.strategy_name == 'Long Straddle':
        if S_T > strategy.upper_breakeven:
            return S_T - K - abs(strategy.net_cost)
        elif S_T < strategy.lower_breakeven:
            return K - S_T - abs(strategy.net_cost)
        else:
            return -abs(strategy.net_cost)
    elif strategy.strategy_name == 'Short Straddle':
        if S_T > strategy.upper_breakeven:
            return -(S_T - K - strategy.max_profit)
        elif S_T < strategy.lower_breakeven:
            return -(strategy.lower_breakeven - S_T - strategy.max_profit)
        else:
            return strategy.max_profit
    else:
        # 其他策略简化计算
        return strategy.net_cost * (S_T - K) / K if K > 0 else 0
