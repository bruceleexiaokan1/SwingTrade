"""策略模块测试

测试：
- StrategyParams: 策略参数定义
- ParameterOptimizer: 参数优化器
- StrategyPortfolio: 多策略组合
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.backtest.strategy_params import StrategyParams
from src.backtest.optimizer import ParameterOptimizer, create_param_grid, OptimizationResult
from src.backtest.portfolio import (
    StrategyPortfolio, create_portfolio, AllocationMethod
)


class TestStrategyParams:
    """StrategyParams 测试"""

    def test_default_params(self):
        """默认参数"""
        params = StrategyParams()
        assert params.ma_short == 20
        assert params.ma_long == 60
        assert params.rsi_period == 14
        assert params.rsi_oversold == 35
        assert params.rsi_overbought == 80
        assert params.atr_stop_multiplier == 2.0
        assert params.max_open_positions == 5

    def test_custom_params(self):
        """自定义参数"""
        params = StrategyParams(
            ma_short=10,
            ma_long=30,
            rsi_oversold=30,
            atr_stop_multiplier=1.5,
            max_open_positions=3,
        )
        assert params.ma_short == 10
        assert params.ma_long == 30
        assert params.rsi_oversold == 30
        assert params.atr_stop_multiplier == 1.5
        assert params.max_open_positions == 3

    def test_aggressive_preset(self):
        """激进预设"""
        params = StrategyParams.aggressive()
        assert params.ma_short == 10
        assert params.ma_long == 30
        assert params.max_open_positions == 8
        assert params.atr_stop_multiplier == 1.5

    def test_conservative_preset(self):
        """保守预设"""
        params = StrategyParams.conservative()
        assert params.ma_short == 30
        assert params.ma_long == 120
        assert params.max_open_positions == 3
        assert params.atr_stop_multiplier == 2.5

    def test_to_dict(self):
        """转换为字典"""
        params = StrategyParams(ma_short=15, ma_long=45)
        d = params.to_dict()
        assert "indicator" in d
        assert "risk" in d
        assert d["indicator"]["ma_short"] == 15

    def test_all_params_have_defaults(self):
        """所有参数都有默认值"""
        params = StrategyParams()
        # 指标参数
        assert params.ma_short > 0
        assert params.ma_long > params.ma_short
        assert params.macd_fast > 0
        assert params.macd_slow > params.macd_fast
        assert params.rsi_period > 0
        assert 0 < params.rsi_oversold < params.rsi_overbought < 100
        # 风控参数
        assert 0 < params.trial_position_pct < 1
        assert 0 < params.max_single_loss_pct < 1
        assert params.max_open_positions > 0


class TestCreateParamGrid:
    """参数网格创建测试"""

    def test_full_grid(self):
        """完整网格"""
        grid = create_param_grid(
            ma_short=[10, 20],
            ma_long=[40, 60],
            rsi_oversold=[30, 35],
        )
        assert "ma_short" in grid
        assert "ma_long" in grid
        assert "rsi_oversold" in grid
        assert len(grid["ma_short"]) == 2
        assert len(grid["ma_long"]) == 2

    def test_partial_grid(self):
        """部分网格"""
        grid = create_param_grid(ma_short=[15, 25])
        assert "ma_short" in grid
        assert "ma_long" not in grid
        assert len(grid["ma_short"]) == 2

    def test_empty_grid(self):
        """空网格"""
        grid = create_param_grid()
        assert len(grid) == 0


class TestOptimizationResult:
    """优化结果测试"""

    def test_summary(self):
        """结果摘要"""
        result = OptimizationResult(
            best_params={"ma_short": 20, "ma_long": 60},
            best_metric=1.5,
            metric_name="sharpe_ratio",
            total_combinations=18,
            duration_seconds=10.5,
        )
        summary = result.summary()
        assert "1.5000" in summary
        assert "sharpe_ratio" in summary
        assert "18" in summary


class TestStrategyPortfolio:
    """策略组合测试"""

    def test_create_portfolio(self):
        """创建组合"""
        portfolio = StrategyPortfolio([
            StrategyParams(ma_short=10, ma_long=30),
            StrategyParams(ma_short=20, ma_long=60),
        ])
        assert portfolio.n_strategies == 2
        assert len(portfolio.strategies) == 2

    def test_equal_allocation(self):
        """等权分配"""
        portfolio = StrategyPortfolio([
            StrategyParams(),
            StrategyParams(),
            StrategyParams(),
        ])
        alloc = portfolio.allocate("equal")
        assert alloc["strategy_0"] == pytest.approx(1/3, rel=0.01)
        assert alloc["strategy_1"] == pytest.approx(1/3, rel=0.01)
        assert alloc["strategy_2"] == pytest.approx(1/3, rel=0.01)

    def test_two_strategy_allocation(self):
        """两个策略分配"""
        portfolio = StrategyPortfolio([
            StrategyParams(),
            StrategyParams(),
        ])
        alloc = portfolio.allocate("equal")
        assert alloc["strategy_0"] == 0.5
        assert alloc["strategy_1"] == 0.5

    def test_create_portfolio_from_config(self):
        """从配置创建组合"""
        portfolio = create_portfolio([
            {"ma_short": 10, "ma_long": 30},
            {"ma_short": 20, "ma_long": 60},
        ])
        assert portfolio.n_strategies == 2
        assert portfolio.strategies[0].ma_short == 10
        assert portfolio.strategies[1].ma_short == 20

    def test_invalid_empty_strategies(self):
        """空策略列表应该报错"""
        with pytest.raises(ValueError):
            StrategyPortfolio([])


class TestAllocationMethods:
    """分配方法测试"""

    def test_unknown_method_falls_back_to_equal(self):
        """未知方法回退到等权"""
        portfolio = StrategyPortfolio([
            StrategyParams(),
            StrategyParams(),
        ])
        alloc = portfolio.allocate("unknown_method")
        assert alloc["strategy_0"] == 0.5
        assert alloc["strategy_1"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
