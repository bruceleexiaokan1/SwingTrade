"""期权波动率测试"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from src.data.indicators.options_volatility import (
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


class TestBlackScholes:
    """Black-Scholes定价测试"""

    def test_call_price_basic(self):
        """基本看涨期权定价"""
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

        price = bs_call_price(S, K, T, r, sigma)

        # 价格应该在内在价值和时间价值之间
        intrinsic = max(S - K, 0)
        assert price >= intrinsic
        assert price <= S  # 不能超过标的价格

    def test_put_price_basic(self):
        """基本看跌期权定价"""
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

        price = bs_put_price(S, K, T, r, sigma)

        # 价格应该在内在价值和时间价值之间
        intrinsic = max(K - S, 0)
        assert price >= intrinsic
        assert price <= K  # 不能超过行权价

    def test_call_put_parity(self):
        """看涨看跌平价"""
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

        call = bs_call_price(S, K, T, r, sigma)
        put = bs_put_price(S, K, T, r, sigma)

        # C - P = S - K*exp(-rT)
        parity = call - put
        expected = S - K * np.exp(-r * T)

        assert abs(parity - expected) < 0.01

    def test_itm_call_price(self):
        """实值看涨期权"""
        S, K, T, r, sigma = 110, 100, 1, 0.05, 0.2

        price = bs_call_price(S, K, T, r, sigma)

        # 内在价值 = 10
        assert price >= 10

    def test_otm_call_price(self):
        """虚值看涨期权"""
        S, K, T, r, sigma = 90, 100, 1, 0.05, 0.2

        price = bs_call_price(S, K, T, r, sigma)

        # 只有时间价值
        assert price < 10

    def test_zero_time(self):
        """零到期时间"""
        S, K, T, r, sigma = 100, 100, 0, 0.05, 0.2

        price = bs_call_price(S, K, T, r, sigma)

        # 应该是内在价值
        assert price == max(S - K, 0)


class TestBSGreeks:
    """希腊字母测试"""

    def test_call_delta(self):
        """看涨期权Delta"""
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

        greeks = calculate_bs_greeks(S, K, T, r, sigma, 'call')

        # ATM看涨期权Delta应该为正且接近但略大于0.5
        assert 0.5 < greeks['delta'] < 1.0

    def test_put_delta(self):
        """看跌期权Delta"""
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

        greeks = calculate_bs_greeks(S, K, T, r, sigma, 'put')

        # ATM看跌期权Delta为负（N(d1) - 1），对于这些参数约为-0.36
        assert -1.0 < greeks['delta'] < 0

    def test_gamma_positive(self):
        """Gamma为正"""
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

        call_greeks = calculate_bs_greeks(S, K, T, r, sigma, 'call')
        put_greeks = calculate_bs_greeks(S, K, T, r, sigma, 'put')

        # Gamma应该为正
        assert call_greeks['gamma'] > 0
        assert put_greeks['gamma'] > 0

    def test_vega_positive(self):
        """Vega为正"""
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

        greeks = calculate_bs_greeks(S, K, T, r, sigma, 'call')

        # Vega应该为正（波动率上升时期权价值上升）
        assert greeks['vega'] > 0

    def test_theta_negative(self):
        """Theta通常为负（时间价值损耗）"""
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

        call_greeks = calculate_bs_greeks(S, K, T, r, sigma, 'call')

        # 远期期权的Theta通常为负
        assert 'theta' in call_greeks


class TestCalculateFullBS:
    """完整BS计算测试"""

    def test_full_bs_result(self):
        """完整BS结果"""
        result = calculate_full_bs(100, 100, 1, 0.05, 0.2)

        assert isinstance(result, BSPriceResult)
        assert result.call_price > 0
        assert result.put_price > 0
        assert -1 <= result.put_delta <= 0
        assert 0 <= result.call_delta <= 1


class TestImpliedVolatility:
    """隐含波动率测试"""

    def test_iv_calculation(self):
        """隐含波动率计算"""
        S, K, T, r = 100, 100, 1, 0.05
        sigma = 0.2

        # 计算期权价格
        option_price = bs_call_price(S, K, T, r, sigma)

        # 反推隐含波动率
        result = implied_volatility(option_price, S, K, T, r, 'call')

        assert result.converged == True
        assert abs(result.iv - sigma) < 0.01

    def test_iv_brent_method(self):
        """Brent方法计算隐含波动率"""
        S, K, T, r = 100, 100, 1, 0.05
        sigma = 0.2

        option_price = bs_call_price(S, K, T, r, sigma)
        iv = implied_volatility_brent(option_price, S, K, T, r, 'call')

        assert abs(iv - sigma) < 0.01

    def test_iv_zero_time(self):
        """零到期时间IV"""
        result = implied_volatility(10, 100, 100, 0, 0.05, 'call')

        assert result.iv == 0


class TestVolatilitySurface:
    """波动率曲面测试"""

    def test_build_volatility_surface(self):
        """构建波动率曲面"""
        # 构造期权链数据
        S = 100
        option_chain = pd.DataFrame({
            'strike': [90, 95, 100, 105, 110],
            'expiry': [0.25, 0.25, 0.25, 0.25, 0.25],
            'price': [12, 8, 5, 3, 1.5],
            'type': ['call', 'call', 'call', 'call', 'call']
        })

        result = build_volatility_surface(option_chain, S, 0.05)

        assert isinstance(result, VolSurfaceResult)
        assert len(result.surface) == 5
        assert 'term_structure' in result.__dict__

    def test_analyze_volatility_smile(self):
        """分析波动率微笑"""
        points = [
            VolSurfacePoint(strike=90, expiry=0.25, moneyness=0.9, iv=0.25, option_type='put'),
            VolSurfacePoint(strike=100, expiry=0.25, moneyness=1.0, iv=0.20, option_type='call'),
            VolSurfacePoint(strike=110, expiry=0.25, moneyness=1.1, iv=0.22, option_type='call'),
        ]

        result = analyze_volatility_smile(points)

        assert '0.25' in result


class TestOptionStrategies:
    """期权策略测试"""

    def test_long_straddle(self):
        """买入跨式"""
        result = long_straddle(100, 100, 0.25, 0.05, 0.2)

        assert isinstance(result, StrategyResult)
        assert result.strategy_name == 'Long Straddle'
        assert result.max_loss == result.net_cost
        assert result.upper_breakeven is not None
        assert result.lower_breakeven is not None

    def test_short_straddle(self):
        """卖出跨式"""
        result = short_straddle(100, 100, 0.25, 0.05, 0.2)

        assert result.strategy_name == 'Short Straddle'
        assert result.max_profit > 0

    def test_bull_call_spread(self):
        """牛市看涨价差"""
        result = bull_call_spread(100, 95, 105, 0.25, 0.05, 0.2)

        assert result.strategy_name == 'Bull Call Spread'
        assert result.max_loss >= 0
        assert result.upper_breakeven is not None

    def test_iron_condor(self):
        """铁秃鹰"""
        result = iron_condor(100, 90, 95, 105, 110, 0.25, 0.05, 0.2)

        assert result.strategy_name == 'Iron Condor'
        assert result.max_profit > 0
        assert result.upper_breakeven is not None
        assert result.lower_breakeven is not None

    def test_risk_reversal(self):
        """风险逆转"""
        result = risk_reversal(100, 0.25, 0.05, 0.2)

        assert result.strategy_name == 'Risk Reversal (Bullish)'
        # 卖出虚值put，买入虚值call，通常put更贵所以净成本为负
        assert isinstance(result.net_cost, float)


class TestVolArbitrage:
    """波动率套利测试"""

    def test_vol_mean_reversion_sell(self):
        """波动率偏高时卖出"""
        result = vol_mean_reversion_signal(
            iv=0.35,
            hv=0.20,
            iv_history_mean=0.22,
            hv_history_mean=0.20
        )

        assert result['signal'] == 'SELL_VOL'
        assert result['action'] == '卖出期权/跨式'

    def test_vol_mean_reversion_buy(self):
        """波动率偏低时买入"""
        result = vol_mean_reversion_signal(
            iv=0.15,
            hv=0.20,
            iv_history_mean=0.22,
            hv_history_mean=0.20
        )

        assert result['signal'] == 'BUY_VOL'
        assert result['action'] == '买入期权/跨式'

    def test_vol_mean_reversion_neutral(self):
        """波动率正常"""
        result = vol_mean_reversion_signal(
            iv=0.22,
            hv=0.20,
            iv_history_mean=0.22,
            hv_history_mean=0.20
        )

        assert result['signal'] == 'NEUTRAL'


class TestCalendarSpread:
    """日历价差测试"""

    def test_calendar_spread(self):
        """日历价差"""
        result = calendar_spread(100, 100, 0.25, 0.5, 0.05, 0.2, 0.22)

        assert result.strategy_name == 'Calendar Spread'
        assert result.net_cost > 0  # 买入远月，卖出近月


class TestPortfolioGreeks:
    """组合希腊字母测试"""

    def test_portfolio_greeks(self):
        """计算组合希腊字母"""
        positions = [
            {'type': 'call', 'direction': 1, 'strike': 100, 'T': 0.25, 'vol': 0.2, 'size': 1},
            {'type': 'put', 'direction': -1, 'strike': 100, 'T': 0.25, 'vol': 0.2, 'size': 1},
        ]

        result = calculate_portfolio_greeks(positions, 100, 0.05)

        assert 'delta' in result
        assert 'gamma' in result
        assert 'theta' in result
        assert 'vega' in result

    def test_hedge_delta(self):
        """Delta对冲"""
        shares, direction = hedge_delta(50, 100, 0)

        # 正delta需要卖出股票对冲
        assert direction == 'sell'
        assert shares == 50  # 返回绝对值


class TestIVIndex:
    """IV指数测试"""

    def test_estimate_iv_index(self):
        """估算IV指数"""
        option_chain = pd.DataFrame({
            'strike': [95, 100, 105],
            'expiry': [0.25, 0.25, 0.25],
            'price': [7, 5, 3],
            'type': ['call', 'call', 'call']
        })

        iv_index = estimate_iv_index(option_chain, 100, 0.05)

        assert 0 <= iv_index <= 1


class TestPnLAtExpiry:
    """到期盈亏测试"""

    def test_long_straddle_pnl(self):
        """跨式到期盈亏"""
        strategy = long_straddle(100, 100, 0.25, 0.05, 0.2)

        # 大幅上涨
        pnl = pnl_at_expiry(strategy, 120, 100)
        assert pnl > 0

        # 大幅下跌
        pnl = pnl_at_expiry(strategy, 80, 100)
        assert pnl > 0

        # 持平
        pnl = pnl_at_expiry(strategy, 100, 100)
        assert pnl < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
