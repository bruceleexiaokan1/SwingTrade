"""Fama-French因子模型：从学术到实战

实现Fama-French五因子模型：
- MKT（市场因子）
- SMB（规模因子）
- HML（价值因子）
- RMW（盈利因子）
- CMA（投资因子）

支持：
- 因子构建
- 因子有效性检验
- 因子择时
- Barra风格因子
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FactorValues:
    """因子值结果"""
    date: str
    stock_code: str
    mkt: float
    smb: float
    hml: float
    rmw: float
    cma: float
    alpha: float
    r_squared: float


@dataclass
class FactorRegressionResult:
    """因子回归结果"""
    stock_code: str
    period_end: str
    alpha: float
    beta_mkt: float
    beta_smb: float
    beta_hml: float
    beta_rmw: float
    beta_cma: float
    r_squared: float


@dataclass
class FactorPortfolioResult:
    """因子组合结果"""
    factor_name: str
    long_return: float
    short_return: float
    spread: float
    n_long: int
    n_short: int


# ==================== 因子构建 ====================

def build_mkt_factor(
    stock_returns: pd.DataFrame,
    market_cap: pd.Series,
    risk_free_rate: float
) -> pd.Series:
    """
    构建市场因子 MKT

    MKT = 市值加权市场收益 - 无风险利率

    Args:
        stock_returns: 股票收益率DataFrame (index=date, columns=stock_code)
        market_cap: 市值序列 (index=stock_code)
        risk_free_rate: 日无风险利率

    Returns:
        MKT因子序列
    """
    # 市值加权市场收益
    total_cap = market_cap.sum()
    if total_cap == 0:
        return pd.Series(0, index=stock_returns.index)

    weights = market_cap / total_cap
    market_return = (stock_returns * weights).sum(axis=1)

    # 超额收益
    mkt = market_return - risk_free_rate

    return mkt


def build_smb_factor(
    stock_returns: pd.DataFrame,
    market_cap: pd.Series,
    book_to_market: pd.Series,
    lookback: int = 20
) -> pd.Series:
    """
    构建SMB因子（规模因子）

    SMB = 小市值组合 - 大市值组合

    方法：
      1. 把股票按市值分成大（B）和小（S）两组
      2. 再按账面市值比分成高（H）、中（M）、低（L）三组
      3. 计算各组收益率
      4. SMB = (SH + SM + SL) / 3 - (BH + BM + BL) / 3

    Args:
        stock_returns: 股票收益率DataFrame
        market_cap: 市值序列
        book_to_market: 账面市值比序列
        lookback: 计算收益率的回看天数

    Returns:
        SMB因子序列
    """
    n_periods = min(lookback, len(stock_returns))
    if n_periods < 5:
        return pd.Series(0, index=stock_returns.index[-n_periods:] if n_periods > 0 else stock_returns.index)

    returns = stock_returns.iloc[-n_periods:]

    # 按市值分组（中位数）
    median_mcap = market_cap.median()
    big_stocks = market_cap > median_mcap
    small_stocks = ~big_stocks

    # 按账面市值比分组（30%和70%分位）
    if book_to_market.dropna().empty:
        bm_30 = 0
        bm_70 = 0
    else:
        bm_30 = book_to_market.quantile(0.3)
        bm_70 = book_to_market.quantile(0.7)

    high_bm = book_to_market > bm_70
    low_bm = book_to_market < bm_30
    medium_bm = ~(high_bm | low_bm)

    # 计算各组收益率
    def portfolio_return(stocks_mask, bm_mask):
        """计算组合收益率"""
        selected = stocks_mask & bm_mask
        if selected.sum() == 0:
            return 0
        # 获取选中的股票代码
        selected_codes = selected[selected].index.tolist()
        if not selected_codes:
            return 0
        selected_returns = returns[selected_codes]
        if selected_returns.empty or selected_returns.shape[1] == 0:
            return 0
        return selected_returns.mean(axis=1).mean()

    # 6个组合（排除medium）
    small_high = portfolio_return(small_stocks, high_bm)
    small_low = portfolio_return(small_stocks, low_bm)
    big_high = portfolio_return(big_stocks, high_bm)
    big_low = portfolio_return(big_stocks, low_bm)

    # SMB = 小市值平均 - 大市值平均
    small_avg = (small_high + small_low) / 2
    big_avg = (big_high + big_low) / 2
    smb = small_avg - big_avg

    return pd.Series(smb, index=returns.index[-1:])


def build_hml_factor(
    stock_returns: pd.DataFrame,
    book_to_market: pd.Series,
    lookback: int = 20
) -> pd.Series:
    """
    构建HML因子（价值因子）

    HML = 高账面市值比组合 - 低账面市值比组合

    Args:
        stock_returns: 股票收益率DataFrame
        book_to_market: 账面市值比序列
        lookback: 回看天数

    Returns:
        HML因子序列
    """
    n_periods = min(lookback, len(stock_returns))
    if n_periods < 5:
        return pd.Series(0, index=stock_returns.index[-n_periods:] if n_periods > 0 else stock_returns.index)

    returns = stock_returns.iloc[-n_periods:]

    # 按账面市值比分组
    if book_to_market.dropna().empty:
        return pd.Series(0, index=returns.index[-1:])

    bm_30 = book_to_market.quantile(0.3)
    bm_70 = book_to_market.quantile(0.7)

    high_bm = book_to_market > bm_70
    low_bm = book_to_market < bm_30

    # 计算高/低组合收益率
    high_return = returns[high_bm[high_bm].index].mean(axis=1).mean() if high_bm.any() else 0
    low_return = returns[low_bm[low_bm].index].mean(axis=1).mean() if low_bm.any() else 0

    hml = high_return - low_return

    return pd.Series(hml, index=returns.index[-1:])


def build_rmw_factor(
    stock_returns: pd.DataFrame,
    profitability: pd.Series,  # ROE或其他盈利指标
    lookback: int = 20
) -> pd.Series:
    """
    构建RMW因子（盈利因子）

    RMW = 高盈利组合 - 低盈利组合

    Args:
        stock_returns: 股票收益率DataFrame
        profitability: 盈利能力序列（如ROE）
        lookback: 回看天数

    Returns:
        RMW因子序列
    """
    n_periods = min(lookback, len(stock_returns))
    if n_periods < 5:
        return pd.Series(0, index=stock_returns.index[-n_periods:] if n_periods > 0 else stock_returns.index)

    returns = stock_returns.iloc[-n_periods:]

    # 按盈利能力分组
    if profitability.dropna().empty:
        return pd.Series(0, index=returns.index[-1:])

    roe_30 = profitability.quantile(0.3)
    roe_70 = profitability.quantile(0.7)

    robust_roe = profitability > roe_70
    weak_roe = profitability < roe_30

    # 计算组合收益
    robust_return = returns[robust_roe[robust_roe].index].mean(axis=1).mean() if robust_roe.any() else 0
    weak_return = returns[weak_roe[weak_roe].index].mean(axis=1).mean() if weak_roe.any() else 0

    rmw = robust_return - weak_return

    return pd.Series(rmw, index=returns.index[-1:])


def build_cma_factor(
    stock_returns: pd.DataFrame,
    investment: pd.Series,  # 资产增长率
    lookback: int = 20
) -> pd.Series:
    """
    构建CMA因子（投资因子）

    CMA = 保守投资组合 - 积极投资组合

    Args:
        stock_returns: 股票收益率DataFrame
        investment: 投资序列（如资产增长率）
        lookback: 回看天数

    Returns:
        CMA因子序列
    """
    n_periods = min(lookback, len(stock_returns))
    if n_periods < 5:
        return pd.Series(0, index=stock_returns.index[-n_periods:] if n_periods > 0 else stock_returns.index)

    returns = stock_returns.iloc[-n_periods:]

    # 按投资分组
    if investment.dropna().empty:
        return pd.Series(0, index=returns.index[-1:])

    inv_30 = investment.quantile(0.3)
    inv_70 = investment.quantile(0.7)

    conservative_inv = investment < inv_30
    aggressive_inv = investment > inv_70

    # 计算组合收益
    conservative_return = returns[conservative_inv[conservative_inv].index].mean(axis=1).mean() if conservative_inv.any() else 0
    aggressive_return = returns[aggressive_inv[aggressive_inv].index].mean(axis=1).mean() if aggressive_inv.any() else 0

    cma = conservative_return - aggressive_return

    return pd.Series(cma, index=returns.index[-1:])


def build_ff5_factors(
    stock_returns: pd.DataFrame,
    market_cap: pd.Series,
    book_to_market: pd.Series,
    profitability: pd.Series,
    investment: pd.Series,
    risk_free_rate: float = 0.0
) -> pd.DataFrame:
    """
    构建完整的Fama-French五因子

    Args:
        stock_returns: 股票收益率DataFrame
        market_cap: 市值序列
        book_to_market: 账面市值比序列
        profitability: 盈利能力序列
        investment: 投资序列
        risk_free_rate: 日无风险利率

    Returns:
        五因子DataFrame
    """
    results = pd.DataFrame(index=stock_returns.index)

    # 1. MKT
    results['MKT'] = build_mkt_factor(stock_returns, market_cap, risk_free_rate)

    # 2. SMB
    results['SMB'] = build_smb_factor(stock_returns, market_cap, book_to_market)

    # 3. HML
    results['HML'] = build_hml_factor(stock_returns, book_to_market)

    # 4. RMW
    results['RMW'] = build_rmw_factor(stock_returns, profitability)

    # 5. CMA
    results['CMA'] = build_cma_factor(stock_returns, investment)

    return results


# ==================== 因子有效性检验 ====================

def factor_regression(
    stock_return: pd.Series,
    factors: pd.DataFrame,
    n_periods: int = 12
) -> Tuple[float, np.ndarray, float]:
    """
    单只股票的因子回归

    R_i = α + β·MKT + s·SMB + h·HML + r·RMW + c·CMA + ε

    Args:
        stock_return: 股票收益率序列
        factors: 因子值DataFrame
        n_periods: 回看期数

    Returns:
        (alpha, betas, r_squared)
    """
    if len(stock_return) < n_periods or len(factors) < n_periods:
        return 0, np.zeros(5), 0

    # 对齐数据
    common_idx = stock_return.index.intersection(factors.index)
    if len(common_idx) < n_periods:
        return 0, np.zeros(5), 0

    y = stock_return[common_idx].values
    X = factors.loc[common_idx, ['MKT', 'SMB', 'HML', 'RMW', 'CMA']].values

    # 简单线性回归（无截距）
    try:
        # 最小二乘法
        X_with_const = np.column_stack([np.ones(len(y)), X])
        coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]

        alpha = coeffs[0]
        betas = coeffs[1:]

        # R²
        y_pred = X_with_const @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    except np.linalg.LinAlgError:
        alpha = 0
        betas = np.zeros(5)
        r_squared = 0

    return alpha, betas, r_squared


def batch_factor_regression(
    stock_returns: pd.DataFrame,
    factors: pd.DataFrame,
    n_periods: int = 12
) -> pd.DataFrame:
    """
    批量因子回归

    Args:
        stock_returns: 股票收益率DataFrame
        factors: 因子值DataFrame
        n_periods: 回看期数

    Returns:
        回归结果DataFrame
    """
    results = []

    for stock in stock_returns.columns:
        try:
            alpha, betas, r_squared = factor_regression(
                stock_returns[stock], factors, n_periods
            )

            results.append({
                'stock': stock,
                'alpha': alpha,
                'beta_mkt': betas[0] if len(betas) > 0 else 0,
                'beta_smb': betas[1] if len(betas) > 1 else 0,
                'beta_hml': betas[2] if len(betas) > 2 else 0,
                'beta_rmw': betas[3] if len(betas) > 3 else 0,
                'beta_cma': betas[4] if len(betas) > 4 else 0,
                'r_squared': r_squared
            })
        except Exception as e:
            logger.debug(f"Regression failed for {stock}: {e}")
            continue

    return pd.DataFrame(results)


def factor_validity_test(
    regression_results: pd.DataFrame
) -> Dict:
    """
    因子有效性检验

    Args:
        regression_results: 回归结果DataFrame

    Returns:
        检验结果
    """
    return {
        'avg_alpha': regression_results['alpha'].mean(),
        'avg_r_squared': regression_results['r_squared'].mean(),
        'significant_betas': {
            'SMB': float((regression_results['beta_smb'] > 0).mean()),
            'HML': float((regression_results['beta_hml'] > 0).mean()),
            'RMW': float((regression_results['beta_rmw'] > 0).mean()),
            'CMA': float((regression_results['beta_cma'] > 0).mean())
        },
        'interpretation': 'α越小越好，R²越高因子解释力越强'
    }


# ==================== 因子择时 ====================

def factor_rotation_weights(
    growth: float,
    inflation: float
) -> Dict[str, float]:
    """
    根据宏观经济状态动态调整因子权重

    Args:
        growth: 经济增长指标（如GDP增速偏离）
        inflation: 通胀指标

    Returns:
        各因子权重
    """
    # 判断经济状态（4种情况）
    if growth > 0 and inflation < 0.03:
        # 复苏期：经济向上，通胀向下
        return {
            'MKT': 1.0,
            'SMB': 0.8,
            'HML': 0.3,
            'RMW': 1.0,
            'CMA': 0.3
        }

    elif growth > 0 and inflation > 0.03:
        # 过热期：经济向上，通胀向上
        return {
            'MKT': 1.0,
            'SMB': 0.5,
            'HML': 1.0,
            'RMW': 0.8,
            'CMA': 0.8
        }

    elif growth < 0 and inflation > 0.03:
        # 滞胀期：经济向下，通胀向上
        return {
            'MKT': 0.5,
            'SMB': 0.5,
            'HML': 0.5,
            'RMW': 0.8,
            'CMA': 0.3
        }

    elif growth < 0:
        # 衰退期：经济向下，通胀向下
        return {
            'MKT': 0.5,
            'SMB': 0.3,
            'HML': 1.0,
            'RMW': 0.3,
            'CMA': 1.0
        }

    else:
        # 平稳期
        return {
            'MKT': 0.8,
            'SMB': 0.5,
            'HML': 0.5,
            'RMW': 0.5,
            'CMA': 0.5
        }


# ==================== Barra风格因子 ====================

def barra_style_factors() -> Dict[str, str]:
    """
    Barra风格因子定义

    Returns:
        因子名称映射
    """
    return {
        'Market Sensitivity': '市场Beta',
        'Size': '市值（对数）',
        'Nonlinear Size': '非线性市值',
        'Value': '账面市值比',
        'Short-term Momentum': '过去12个月动量',
        'Mid-term Momentum': '中期动量',
        'Beta': '市场Beta',
        'Volatility': '残差波动率',
        'Dividend Yield': '股息率',
        'Earnings Yield': '盈利收益率（EP倒数）',
        'Growth': '盈利增长',
        'Leverage': '杠杆率'
    }


def calculate_style_exposure(
    stock_data: pd.Series
) -> Dict[str, float]:
    """
    计算 Barra 风格因子暴露度

    Args:
        stock_data: 包含以下字段的Series:
                   - market_cap: 市值
                   - book_to_market: 账面市值比
                   - roe: ROE
                   - debt_ratio: 资产负债率
                   - momentum_12m: 12个月动量
                   - volatility: 波动率
                   - dividend_yield: 股息率

    Returns:
        各风格因子的暴露度
    """
    exposure = {}

    # Size（对数市值）
    if 'market_cap' in stock_data.index and stock_data['market_cap'] > 0:
        exposure['Size'] = np.log(stock_data['market_cap'])
    else:
        exposure['Size'] = 0

    # Nonlinear Size
    exposure['Nonlinear Size'] = exposure['Size'] ** 2 if exposure['Size'] != 0 else 0

    # Value
    exposure['Value'] = stock_data.get('book_to_market', 0)

    # Profitability/RMW
    exposure['RMW'] = stock_data.get('roe', 0)

    # Leverage
    exposure['Leverage'] = stock_data.get('debt_ratio', 0)

    # Momentum
    exposure['Short-term Momentum'] = stock_data.get('momentum_1m', 0)
    exposure['Mid-term Momentum'] = stock_data.get('momentum_12m', 0)

    # Volatility
    exposure['Volatility'] = stock_data.get('volatility', 0)

    # Dividend Yield
    exposure['Dividend Yield'] = stock_data.get('dividend_yield', 0)

    # Earnings Yield
    exposure['Earnings Yield'] = 1 / stock_data.get('pe', 1) if stock_data.get('pe', 0) > 0 else 0

    # Growth
    exposure['Growth'] = stock_data.get('earnings_growth', 0)

    return exposure


# ==================== 多因子组合优化 ====================

def ff5_portfolio_optimization(
    expected_returns: np.ndarray,
    factor_cov: Dict[str, float],
    factor_exposures: np.ndarray,
    specific_risks: np.ndarray,
    risk_aversion: float = 1.0
) -> Dict:
    """
    基于Fama-French因子的组合优化

    目标：最大化收益 - λ×风险

    Args:
        expected_returns: 预期收益率数组
        factor_cov: 因子协方差字典 {'MKT': var, 'SMB': var, ...}
        factor_exposures: 因子暴露矩阵 (n_assets, n_factors)
        specific_risks: 特异性风险数组 (n_assets,)
        risk_aversion: 风险厌恶系数

    Returns:
        优化结果
    """
    n_assets = len(expected_returns)

    # 因子协方差矩阵
    factor_names = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
    factor_cov_matrix = np.diag([factor_cov.get(f, 0) for f in factor_names])

    # 优化问题
    def objective(w):
        # 收益
        ret = w @ expected_returns

        # 因子风险
        factor_risk = w @ factor_exposures @ factor_cov_matrix @ factor_exposures.T @ w

        # 特异性风险
        spec_risk = w @ np.diag(specific_risks ** 2) @ w

        # 目标函数
        return -(ret - risk_aversion * (factor_risk + spec_risk))

    # 约束：权重和=1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    # 边界
    bounds = [(0, 0.1) for _ in range(n_assets)]

    # 初始权重
    w0 = np.ones(n_assets) / n_assets

    try:
        from scipy.optimize import minimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        expected_return = result.x @ expected_returns

        # 因子暴露
        portfolio_factor_exposure = result.x @ factor_exposures

        return {
            'optimal_weights': optimal_weights,
            'expected_return': expected_return,
            'portfolio_factor_exposure': portfolio_factor_exposure,
            'optimization_status': 'success'
        }

    except Exception as e:
        logger.warning(f"Optimization failed: {e}")
        return {
            'optimal_weights': w0,
            'expected_return': w0 @ expected_returns,
            'portfolio_factor_exposure': w0 @ factor_exposures,
            'optimization_status': 'failed'
        }


# ==================== 便捷函数 ====================

def calculate_factor_exposures(
    stock_returns: pd.Series,
    factors: pd.DataFrame
) -> Dict[str, float]:
    """
    计算单只股票的五因子暴露度

    Args:
        stock_returns: 股票收益率序列
        factors: 因子DataFrame

    Returns:
        各因子的beta
    """
    alpha, betas, r_squared = factor_regression(stock_returns, factors)

    return {
        'alpha': alpha,
        'beta_mkt': betas[0] if len(betas) > 0 else 0,
        'beta_smb': betas[1] if len(betas) > 1 else 0,
        'beta_hml': betas[2] if len(betas) > 2 else 0,
        'beta_rmw': betas[3] if len(betas) > 3 else 0,
        'beta_cma': betas[4] if len(betas) > 4 else 0,
        'r_squared': r_squared
    }


def rolling_factor_analysis(
    stock_returns: pd.DataFrame,
    factors: pd.DataFrame,
    window: int = 60
) -> pd.DataFrame:
    """
    滚动因子分析

    Args:
        stock_returns: 股票收益率DataFrame
        factors: 因子DataFrame
        window: 滚动窗口大小

    Returns:
        滚动分析结果
    """
    results = []

    for i in range(window, len(stock_returns)):
        window_returns = stock_returns.iloc[i-window:i]
        window_factors = factors.iloc[i-window:i]

        for stock in stock_returns.columns:
            try:
                alpha, betas, r_squared = factor_regression(
                    window_returns[stock], window_factors
                )

                results.append({
                    'date': stock_returns.index[i],
                    'stock': stock,
                    'alpha': alpha,
                    'beta_mkt': betas[0] if len(betas) > 0 else 0,
                    'beta_smb': betas[1] if len(betas) > 1 else 0,
                    'beta_hml': betas[2] if len(betas) > 2 else 0,
                    'beta_rmw': betas[3] if len(betas) > 3 else 0,
                    'beta_cma': betas[4] if len(betas) > 4 else 0,
                    'r_squared': r_squared
                })
            except Exception:
                continue

    return pd.DataFrame(results)
