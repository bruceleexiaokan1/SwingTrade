"""策略参数优化器

提供参数优化功能：
- 网格搜索（Grid Search）
- 并行回测加速
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Any]
    best_metric: float
    metric_name: str
    all_results: List[Dict] = field(default_factory=list)
    total_combinations: int = 0
    duration_seconds: float = 0.0

    def summary(self) -> str:
        """生成摘要"""
        return f"""=== Optimization Result ===
Best Metric ({self.metric_name}): {self.best_metric:.4f}
Best Params: {self.best_params}
Total Combinations: {self.total_combinations}
Duration: {self.duration_seconds:.1f}s
"""


class ParameterOptimizer:
    """
    策略参数优化器

    使用网格搜索找到最优参数组合：
    1. 定义参数网格
    2. 并行回测每个组合
    3. 返回最优参数
    """

    def __init__(
        self,
        backtest_fn: Callable,
        default_params: "StrategyParams" = None,
        n_workers: int = 4
    ):
        """
        初始化优化器

        Args:
            backtest_fn: 回测函数，接收 (stock_codes, start_date, end_date, params) 返回 BacktestResult
            default_params: 默认策略参数
            n_workers: 并行工作线程数
        """
        self.backtest_fn = backtest_fn
        self.default_params = default_params
        self.n_workers = n_workers

    def grid_search(
        self,
        param_grid: Dict[str, List],
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        metric: str = "sharpe_ratio",
        minimize: bool = False,
    ) -> OptimizationResult:
        """
        网格搜索最优参数

        Args:
            param_grid: 参数网格定义，如 {"ma_short": [10, 20], "ma_long": [30, 60]}
            stock_codes: 股票列表
            start_date: 开始日期
            end_date: 结束日期
            metric: 优化目标指标名（sharpe_ratio / total_return / max_drawdown / win_rate）
            minimize: 是否最小化目标（True=最小化，False=最大化）

        Returns:
            OptimizationResult: 最优参数和结果
        """
        import time
        from ..backtest.strategy_params import StrategyParams

        start_time = time.time()

        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        logger.info(f"Starting grid search: {len(combinations)} combinations")
        logger.info(f"Metric: {metric} ({'minimize' if minimize else 'maximize'})")

        # 使用默认参数作为基础
        if self.default_params is None:
            self.default_params = StrategyParams()

        # 存储所有结果
        all_results = []

        # 并行回测
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {}
            for combo in combinations:
                params = self._create_params_from_combination(
                    self.default_params, param_names, combo
                )
                future = executor.submit(
                    self.backtest_fn,
                    stock_codes,
                    start_date,
                    end_date,
                    params
                )
                futures[future] = dict(zip(param_names, combo))

            for future in as_completed(futures):
                combo_dict = futures[future]
                try:
                    result = future.result()
                    metric_value = self._extract_metric(result, metric)

                    result_entry = {
                        "params": combo_dict.copy(),
                        "metric": metric_value,
                        "sharpe_ratio": result.sharpe_ratio,
                        "total_return": result.total_return,
                        "max_drawdown": result.max_drawdown,
                        "win_rate": result.win_rate,
                        "total_trades": result.total_trades,
                    }
                    all_results.append(result_entry)

                    if minimize:
                        is_better = metric_value < (all_results[0]["metric"] if all_results else float('inf'))
                    else:
                        is_better = metric_value > (all_results[0]["metric"] if all_results else float('-inf'))

                    if is_better and metric_value is not None:
                        best_metric = metric_value
                        best_params = combo_dict.copy()

                except Exception as e:
                    logger.warning(f"Backtest failed for {combo_dict}: {e}")

        # 按指标排序
        all_results.sort(
            key=lambda x: x["metric"] if x["metric"] is not None else 0,
            reverse=not minimize
        )

        duration = time.time() - start_time

        # 返回最优结果
        if all_results:
            return OptimizationResult(
                best_params=all_results[0]["params"],
                best_metric=all_results[0]["metric"],
                metric_name=metric,
                all_results=all_results,
                total_combinations=len(combinations),
                duration_seconds=duration,
            )
        else:
            return OptimizationResult(
                best_params={},
                best_metric=0.0,
                metric_name=metric,
                all_results=[],
                total_combinations=len(combinations),
                duration_seconds=duration,
            )

    def _create_params_from_combination(
        self,
        base_params: "StrategyParams",
        param_names: List[str],
        param_values: tuple
    ) -> "StrategyParams":
        """从参数组合创建 StrategyParams"""
        import copy
        params = copy.deepcopy(base_params)

        for name, value in zip(param_names, param_values):
            if hasattr(params, name):
                setattr(params, name, value)

        return params

    def _extract_metric(self, result: "BacktestResult", metric: str) -> Optional[float]:
        """从回测结果提取指标"""
        if metric == "sharpe_ratio":
            return result.sharpe_ratio
        elif metric == "total_return":
            return result.total_return
        elif metric == "max_drawdown":
            return result.max_drawdown
        elif metric == "win_rate":
            return result.win_rate
        elif metric == "profit_factor":
            return result.profit_factor
        elif metric == "calmar_ratio":
            return result.calmar_ratio
        elif metric == "annualized_return":
            return result.annualized_return
        else:
            logger.warning(f"Unknown metric: {metric}")
            return None


def create_param_grid(
    ma_short: List[int] = None,
    ma_long: List[int] = None,
    rsi_period: List[int] = None,
    rsi_oversold: List[int] = None,
    rsi_overbought: List[int] = None,
    atr_stop_multiplier: List[float] = None,
    atr_trailing_multiplier: List[float] = None,
    entry_confidence_threshold: List[float] = None,
    min_profit_loss_ratio: List[float] = None,
) -> Dict[str, List]:
    """
    创建参数网格（便捷函数）

    Args:
        ma_short: 短期均线周期候选值
        ma_long: 长期均线周期候选值
        rsi_period: RSI周期候选值
        rsi_oversold: RSI超卖阈值候选值
        rsi_overbought: RSI超买阈值候选值
        atr_stop_multiplier: ATR止损倍数候选值
        atr_trailing_multiplier: ATR追踪止损倍数候选值
        entry_confidence_threshold: 入场置信度阈值候选值
        min_profit_loss_ratio: 最小盈亏比候选值

    Returns:
        参数网格字典
    """
    grid = {}

    if ma_short is not None:
        grid["ma_short"] = ma_short
    if ma_long is not None:
        grid["ma_long"] = ma_long
    if rsi_period is not None:
        grid["rsi_period"] = rsi_period
    if rsi_oversold is not None:
        grid["rsi_oversold"] = rsi_oversold
    if rsi_overbought is not None:
        grid["rsi_overbought"] = rsi_overbought
    if atr_stop_multiplier is not None:
        grid["atr_stop_multiplier"] = atr_stop_multiplier
    if atr_trailing_multiplier is not None:
        grid["atr_trailing_multiplier"] = atr_trailing_multiplier
    if entry_confidence_threshold is not None:
        grid["entry_confidence_threshold"] = entry_confidence_threshold
    if min_profit_loss_ratio is not None:
        grid["min_profit_loss_ratio"] = min_profit_loss_ratio

    return grid
