"""多策略组合管理器

支持：
- 多策略同时运行
- 资金分配（等权/风险平价/动量）
- 组合回测分析
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class AllocationMethod(Enum):
    """资金分配方法"""
    EQUAL = "equal"              # 等权分配
    RISK_PARITY = "risk_parity"  # 风险平价
    MOMENTUM = "momentum"        # 动量加权


@dataclass
class PortfolioResult:
    """组合回测结果"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0

    # 各策略结果
    strategy_results: List = field(default_factory=list)

    # 分配权重
    allocations: Dict[str, float] = field(default_factory=dict)


class StrategyPortfolio:
    """
    多策略组合管理器

    功能：
    1. 管理多个策略实例
    2. 支持不同资金分配方法
    3. 组合回测分析
    """

    def __init__(
        self,
        strategies: List["StrategyParams"],
        initial_capital: float = 1_000_000.0,
    ):
        """
        初始化组合管理器

        Args:
            strategies: 策略参数列表
            initial_capital: 初始资金
        """
        if not strategies:
            raise ValueError("At least one strategy is required")

        self.strategies = strategies
        self.initial_capital = initial_capital
        self.n_strategies = len(strategies)

        # 默认等权分配
        self.allocations = {
            f"strategy_{i}": 1.0 / self.n_strategies
            for i in range(self.n_strategies)
        }

    def allocate(
        self,
        method: str = "equal",
        historical_results: Optional[List["BacktestResult"]] = None,
    ) -> Dict[str, float]:
        """
        资金分配

        Args:
            method: 分配方法 ("equal" / "risk_parity" / "momentum")
            historical_results: 各策略历史回测结果（用于风险平价/动量计算）

        Returns:
            分配权重字典
        """
        if method == "equal":
            return self._allocate_equal()
        elif method == "risk_parity":
            if historical_results is None:
                logger.warning("No historical results provided, falling back to equal allocation")
                return self._allocate_equal()
            return self._allocate_risk_parity(historical_results)
        elif method == "momentum":
            if historical_results is None:
                logger.warning("No historical results provided, falling back to equal allocation")
                return self._allocate_equal()
            return self._allocate_momentum(historical_results)
        else:
            logger.warning(f"Unknown allocation method: {method}, using equal")
            return self._allocate_equal()

    def _allocate_equal(self) -> Dict[str, float]:
        """等权分配"""
        return {
            f"strategy_{i}": 1.0 / self.n_strategies
            for i in range(self.n_strategies)
        }

    def _allocate_risk_parity(self, results: List["BacktestResult"]) -> Dict[str, float]:
        """
        风险平价分配

        原理：让各策略对组合总风险的贡献相等
        风险贡献 = 权重 × 策略波动率（用最大回撤近似）
        """
        allocations = {}

        # 计算各策略风险（用最大回撤作为风险指标）
        risks = []
        for result in results:
            # 用 max_drawdown 作为风险指标，回撤越大风险越高
            risk = abs(result.max_drawdown) if result.max_drawdown != 0 else 0.01
            risks.append(risk)

        # 风险平价权重 = 1 / 风险
        inv_risks = [1.0 / r for r in risks]
        total_inv_risk = sum(inv_risks)

        # 归一化
        for i, inv_risk in enumerate(inv_risks):
            allocations[f"strategy_{i}"] = inv_risk / total_inv_risk

        return allocations

    def _allocate_momentum(self, results: List["BacktestResult"]) -> Dict[str, float]:
        """
        动量加权分配

        原理：根据历史表现分配，表现好的分配更多
        使用夏普比率作为动量指标
        """
        allocations = {}

        # 计算各策略动量（用夏普比率）
        momenta = []
        for result in results:
            momentum = result.sharpe_ratio if result.sharpe_ratio > 0 else 0
            momenta.append(max(0, momentum))  # 负收益不给权重

        # 如果所有动量都为0，等权分配
        if sum(momenta) == 0:
            return self._allocate_equal()

        # 归一化
        total_momentum = sum(momenta)
        for i, momentum in enumerate(momenta):
            allocations[f"strategy_{i}"] = momentum / total_momentum

        return allocations

    def run_combined(
        self,
        backtest_fn,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
    ) -> List["BacktestResult"]:
        """
        运行组合回测

        Args:
            backtest_fn: 回测函数，接收 (stock_codes, start_date, end_date, params)
            stock_codes: 股票列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            各策略的回测结果列表
        """
        results = []

        for i, strategy in enumerate(self.strategies):
            logger.info(f"Running backtest for strategy {i+1}/{self.n_strategies}")

            result = backtest_fn(
                stock_codes=stock_codes,
                start_date=start_date,
                end_date=end_date,
                params=strategy
            )
            results.append(result)

        return results

    def analyze_portfolio(
        self,
        results: List["BacktestResult"],
        allocations: Optional[Dict[str, float]] = None,
    ) -> PortfolioResult:
        """
        分析组合表现

        Args:
            results: 各策略回测结果
            allocations: 分配权重

        Returns:
            PortfolioResult
        """
        if allocations is None:
            allocations = self.allocations

        # 计算加权组合指标
        total_return = 0.0
        total_sharpe = 0.0
        weighted_max_dd = 0.0
        total_trades = 0

        for i, result in enumerate(results):
            weight = allocations.get(f"strategy_{i}", 1.0 / self.n_strategies)
            total_return += result.total_return * weight
            total_sharpe += result.sharpe_ratio * weight
            weighted_max_dd += result.max_drawdown * weight
            total_trades += result.total_trades

        # 简化计算胜率（加权平均）
        win_rates = [r.win_rate for r in results]
        avg_win_rate = sum(win_rates) / len(win_rates)

        return PortfolioResult(
            total_return=total_return,
            sharpe_ratio=total_sharpe,
            max_drawdown=weighted_max_dd,
            win_rate=avg_win_rate,
            total_trades=total_trades,
            strategy_results=results,
            allocations=allocations,
        )


def create_portfolio(
    strategy_configs: List[Dict],
    initial_capital: float = 1_000_000.0,
) -> StrategyPortfolio:
    """
    从配置创建策略组合（便捷函数）

    Args:
        strategy_configs: 策略配置列表，每个配置是一个字典
        initial_capital: 初始资金

    Returns:
        StrategyPortfolio 实例

    Example:
        portfolio = create_portfolio([
            {"ma_short": 20, "ma_long": 60, "rsi_oversold": 35},
            {"ma_short": 10, "ma_long": 30, "rsi_oversold": 30},
        ])
    """
    from .strategy_params import StrategyParams

    strategies = []
    for config in strategy_configs:
        # 使用默认参数，然后用配置覆盖
        params = StrategyParams()
        for key, value in config.items():
            if hasattr(params, key):
                setattr(params, key, value)
        strategies.append(params)

    return StrategyPortfolio(strategies, initial_capital)
