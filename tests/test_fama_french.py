"""Fama-French因子模型测试"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from src.data.indicators.fama_french import (
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


def create_sample_stock_data(days: int = 60, n_stocks: int = 20) -> tuple:
    """创建样本股票数据"""
    np.random.seed(42)
    dates = pd.date_range(end='2024-01-01', periods=days, freq='D')

    # 生成股票代码
    stock_codes = [f'00{i:04d}' for i in range(n_stocks)]

    # 生成收益率
    stock_returns = pd.DataFrame(
        np.random.randn(days, n_stocks) * 0.02,
        index=dates,
        columns=stock_codes
    )

    # 生成市值
    market_cap = pd.Series(
        np.random.uniform(1e9, 1e11, n_stocks),
        index=stock_codes
    )

    # 生成账面市值比
    book_to_market = pd.Series(
        np.random.uniform(0.2, 5.0, n_stocks),
        index=stock_codes
    )

    # 生成盈利能力
    profitability = pd.Series(
        np.random.uniform(-0.1, 0.3, n_stocks),
        index=stock_codes
    )

    # 生成投资率
    investment = pd.Series(
        np.random.uniform(-0.2, 0.5, n_stocks),
        index=stock_codes
    )

    return stock_returns, market_cap, book_to_market, profitability, investment


class TestBuildMKT:
    """MKT因子构建测试"""

    def test_build_mkt_factor(self):
        """MKT因子构建"""
        stock_returns, market_cap, *_ = create_sample_stock_data(60)

        mkt = build_mkt_factor(stock_returns, market_cap, 0.0001)

        assert len(mkt) == len(stock_returns)
        assert isinstance(mkt, pd.Series)


class TestBuildSMB:
    """SMB因子构建测试"""

    def test_build_smb_factor(self):
        """SMB因子构建"""
        stock_returns, market_cap, book_to_market, *_ = create_sample_stock_data(60)

        smb = build_smb_factor(stock_returns, market_cap, book_to_market)

        assert len(smb) > 0
        assert isinstance(smb, pd.Series)


class TestBuildHML:
    """HML因子构建测试"""

    def test_build_hml_factor(self):
        """HML因子构建"""
        stock_returns, market_cap, book_to_market, *_ = create_sample_stock_data(60)

        hml = build_hml_factor(stock_returns, book_to_market)

        assert len(hml) > 0


class TestBuildRMW:
    """RMW因子构建测试"""

    def test_build_rmw_factor(self):
        """RMW因子构建"""
        stock_returns, market_cap, _, profitability, _ = create_sample_stock_data(60)

        rmw = build_rmw_factor(stock_returns, profitability)

        assert len(rmw) > 0


class TestBuildCMA:
    """CMA因子构建测试"""

    def test_build_cma_factor(self):
        """CMA因子构建"""
        stock_returns, market_cap, _, _, investment = create_sample_stock_data(60)

        cma = build_cma_factor(stock_returns, investment)

        assert len(cma) > 0


class TestBuildFF5:
    """FF5因子构建测试"""

    def test_build_ff5_factors(self):
        """构建完整FF5因子"""
        stock_returns, market_cap, book_to_market, profitability, investment = create_sample_stock_data(60)

        factors = build_ff5_factors(
            stock_returns,
            market_cap,
            book_to_market,
            profitability,
            investment,
            risk_free_rate=0.0001
        )

        assert 'MKT' in factors.columns
        assert 'SMB' in factors.columns
        assert 'HML' in factors.columns
        assert 'RMW' in factors.columns
        assert 'CMA' in factors.columns


class TestFactorRegression:
    """因子回归测试"""

    def test_factor_regression(self):
        """单因子回归"""
        stock_returns, market_cap, book_to_market, profitability, investment = create_sample_stock_data(60)

        factors = build_ff5_factors(
            stock_returns,
            market_cap,
            book_to_market,
            profitability,
            investment
        )

        stock_return = stock_returns.iloc[:, 0]

        alpha, betas, r_squared = factor_regression(stock_return, factors, n_periods=20)

        assert isinstance(alpha, (int, float))
        assert len(betas) == 5
        assert 0 <= r_squared <= 1

    def test_batch_factor_regression(self):
        """批量因子回归"""
        stock_returns, market_cap, book_to_market, profitability, investment = create_sample_stock_data(60)

        factors = build_ff5_factors(
            stock_returns,
            market_cap,
            book_to_market,
            profitability,
            investment
        )

        results = batch_factor_regression(stock_returns, factors, n_periods=20)

        assert len(results) <= len(stock_returns.columns)
        assert 'alpha' in results.columns
        assert 'beta_mkt' in results.columns
        assert 'r_squared' in results.columns


class TestFactorValidity:
    """因子有效性检验测试"""

    def test_factor_validity_test(self):
        """因子有效性检验"""
        # 创建模拟的回归结果
        regression_results = pd.DataFrame({
            'alpha': np.random.randn(100) * 0.01,
            'beta_smb': np.random.randn(100) * 0.5 + 0.3,
            'beta_hml': np.random.randn(100) * 0.5,
            'beta_rmw': np.random.randn(100) * 0.5 + 0.2,
            'beta_cma': np.random.randn(100) * 0.5 - 0.1,
            'r_squared': np.random.uniform(0.1, 0.6, 100)
        })

        result = factor_validity_test(regression_results)

        assert 'avg_alpha' in result
        assert 'avg_r_squared' in result
        assert 'significant_betas' in result


class TestFactorRotation:
    """因子轮动测试"""

    def test_factor_rotation_recovery(self):
        """复苏期因子权重"""
        weights = factor_rotation_weights(growth=0.02, inflation=0.02)

        assert weights['SMB'] > weights['HML']  # 小市值占优
        assert weights['RMW'] > weights['CMA']  # 盈利因子强

    def test_factor_rotation_overheat(self):
        """过热期因子权重"""
        weights = factor_rotation_weights(growth=0.05, inflation=0.04)

        assert weights['HML'] > 0.5  # 价值因子强

    def test_factor_rotation_recession(self):
        """衰退期因子权重"""
        weights = factor_rotation_weights(growth=-0.02, inflation=0.02)

        assert weights['HML'] == 1.0  # 价值防御
        assert weights['CMA'] == 1.0  # 投资因子

    def test_factor_rotation_stagflation(self):
        """滞胀期因子权重"""
        weights = factor_rotation_weights(growth=-0.01, inflation=0.05)

        assert weights['RMW'] > weights['CMA']  # 盈利因子相对重要


class TestBarraStyleFactors:
    """Barra风格因子测试"""

    def test_barra_style_factors(self):
        """Barra风格因子定义"""
        factors = barra_style_factors()

        assert 'Size' in factors
        assert 'Value' in factors
        assert 'Volatility' in factors
        assert 'Momentum' in factors or 'Short-term Momentum' in factors


class TestStyleExposure:
    """风格暴露度测试"""

    def test_calculate_style_exposure(self):
        """计算风格暴露度"""
        stock_data = pd.Series({
            'market_cap': 1e10,
            'book_to_market': 2.0,
            'roe': 0.15,
            'debt_ratio': 0.5,
            'momentum_1m': 0.05,
            'momentum_12m': 0.20,
            'volatility': 0.25,
            'dividend_yield': 0.03,
            'pe': 15.0,
            'earnings_growth': 0.10,
        })

        exposure = calculate_style_exposure(stock_data)

        assert 'Size' in exposure
        assert 'Value' in exposure
        assert 'RMW' in exposure
        assert exposure['Size'] > 0  # 对数市值


class TestFF5PortfolioOptimization:
    """FF5组合优化测试"""

    def test_ff5_portfolio_optimization(self):
        """组合优化"""
        n_assets = 10
        expected_returns = np.random.randn(n_assets) * 0.01

        factor_cov = {
            'MKT': 0.04,
            'SMB': 0.02,
            'HML': 0.015,
            'RMW': 0.012,
            'CMA': 0.01
        }

        factor_exposures = np.random.randn(n_assets, 5) * 0.5
        specific_risks = np.abs(np.random.randn(n_assets)) * 0.02

        result = ff5_portfolio_optimization(
            expected_returns,
            factor_cov,
            factor_exposures,
            specific_risks
        )

        assert 'optimal_weights' in result
        assert len(result['optimal_weights']) == n_assets
        assert abs(sum(result['optimal_weights']) - 1) < 0.01  # 权重和为1


class TestFactorExposures:
    """因子暴露度测试"""

    def test_calculate_factor_exposures(self):
        """计算单只股票的因子暴露度"""
        stock_returns, market_cap, book_to_market, profitability, investment = create_sample_stock_data(60)

        factors = build_ff5_factors(
            stock_returns,
            market_cap,
            book_to_market,
            profitability,
            investment
        )

        stock_return = stock_returns.iloc[:, 0]
        exposures = calculate_factor_exposures(stock_return, factors)

        assert 'alpha' in exposures
        assert 'beta_mkt' in exposures
        assert 'beta_smb' in exposures
        assert 'beta_hml' in exposures


class TestRollingFactorAnalysis:
    """滚动因子分析测试"""

    def test_rolling_factor_analysis(self):
        """滚动因子分析"""
        stock_returns, market_cap, book_to_market, profitability, investment = create_sample_stock_data(100)

        factors = build_ff5_factors(
            stock_returns,
            market_cap,
            book_to_market,
            profitability,
            investment
        )

        results = rolling_factor_analysis(stock_returns, factors, window=30)

        # 应该有很多结果（每天每个股票）
        assert len(results) > 0
        assert 'alpha' in results.columns
        assert 'beta_mkt' in results.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
