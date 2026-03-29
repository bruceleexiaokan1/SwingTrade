"""参数网格扫描回测器

支持多参数组合的并行回测，自动生成敏感性分析报告。
"""

import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import sys
sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

from src.backtest.engine import SwingBacktester
from src.backtest.strategy_params import StrategyParams
from src.backtest.performance import PerformanceAnalyzer
from src.data.loader import StockDataLoader

logger = logging.getLogger(__name__)


@dataclass
class GridResult:
    """单次参数组合的回测结果"""
    params: Dict[str, Any]
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    trades_per_year: float = 0.0
    avg_holding_days: float = 0.0

    # 结构止损统计
    structure_stop_1_count: int = 0
    structure_stop_2_count: int = 0
    atr_stop_count: int = 0
    trailing_stop_count: int = 0
    t1_count: int = 0
    t2_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": self.params,
            "metrics": {
                "total_return": f"{self.total_return:.2%}",
                "annualized_return": f"{self.annualized_return:.2%}",
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "calmar_ratio": self.calmar_ratio,
                "max_drawdown": f"{self.max_drawdown:.2%}",
                "win_rate": f"{self.win_rate:.2%}",
                "profit_factor": self.profit_factor,
                "total_trades": self.total_trades,
                "trades_per_year": self.trades_per_year,
                "avg_holding_days": self.avg_holding_days,
            },
            "exit_reasons": {
                "structure_stop_1": self.structure_stop_1_count,
                "structure_stop_2": self.structure_stop_2_count,
                "atr_stop": self.atr_stop_count,
                "trailing_stop": self.trailing_stop_count,
                "t1": self.t1_count,
                "t2": self.t2_count,
            }
        }


class ParameterGridRunner:
    """
    参数网格扫描回测器

    使用方式:
        runner = ParameterGridRunner(
            stockdata_root="/path/to/StockData",
            start_date="2023-01-01",
            end_date="2024-12-31",
        )

        # 定义参数网格
        grid = {
            "min_profit_loss_ratio": [1.5, 2.0, 2.5, 3.0, 3.5],
            "atr_stop_multiplier": [1.5, 2.0, 2.5],
        }

        results = runner.run_grid(grid, codes=["600000", "600519"])
        runner.generate_report(results, "reports/grid_scan.html")
    """

    def __init__(
        self,
        stockdata_root: str = "/Users/bruce/workspace/trade/StockData",
        initial_capital: float = 1_000_000.0,
        n_workers: int = None,
    ):
        self.stockdata_root = stockdata_root
        self.initial_capital = initial_capital
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.loader = StockDataLoader(stockdata_root=stockdata_root)

    def run_grid(
        self,
        param_grid: Dict[str, List[Any]],
        codes: List[str],
        start_date: str,
        end_date: str,
        other_params: Dict[str, Any] = None,
    ) -> List[GridResult]:
        """
        运行参数网格扫描

        Args:
            param_grid: 参数网格定义，如 {"min_profit_loss_ratio": [1.5, 2.0]}
            codes: 股票代码列表
            start_date: 回测开始日期
            end_date: 回测结束日期
            other_params: 其他固定参数

        Returns:
            GridResult 列表
        """
        other_params = other_params or {}

        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        logger.info(f"参数网格: {len(param_names)} 个参数, {len(combinations)} 种组合")

        results = []

        # 并行回测
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {}
            for combo in combinations:
                params = dict(zip(param_names, combo))
                params.update(other_params)

                future = executor.submit(
                    self._run_single,
                    params,
                    codes,
                    start_date,
                    end_date,
                )
                futures[future] = params

            for future in as_completed(futures):
                params = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"完成: {params}")
                except Exception as e:
                    logger.error(f"失败 {params}: {e}")

        return results

    def _run_single(
        self,
        params: Dict[str, Any],
        codes: List[str],
        start_date: str,
        end_date: str,
    ) -> GridResult:
        """运行单次回测"""
        # 分离引擎参数和策略参数
        # 引擎直接使用的参数
        engine_params = [
            "atr_stop_multiplier",
            "atr_trailing_multiplier",
            "min_profit_loss_ratio",
            "entry_confidence_threshold",
            "max_open_positions",
            "atr_circuit_breaker",
            "trial_position_pct",
            "max_single_loss_pct",
        ]

        engine_kwargs = {"initial_capital": self.initial_capital}
        strategy_kwargs = {}

        for k, v in params.items():
            if k in engine_params:
                engine_kwargs[k] = v
            else:
                strategy_kwargs[k] = v

        # 构建策略参数对象
        strategy_params = StrategyParams(**strategy_kwargs)
        engine_kwargs["strategy_params"] = strategy_params

        # 创建回测器
        backtester = SwingBacktester(**engine_kwargs)

        # 运行回测（内部加载数据）
        result = backtester.run(
            stock_codes=codes,
            start_date=start_date,
            end_date=end_date,
            data_loader=self.loader,
            stockdata_root=self.stockdata_root,
        )

        # 提取绩效指标
        grid_result = GridResult(params=params)

        grid_result.total_return = result.total_return
        grid_result.annualized_return = result.annualized_return
        grid_result.sharpe_ratio = result.sharpe_ratio
        grid_result.sortino_ratio = result.sortino_ratio
        grid_result.calmar_ratio = result.calmar_ratio
        grid_result.max_drawdown = result.max_drawdown
        grid_result.win_rate = result.win_rate
        grid_result.profit_factor = result.profit_factor
        grid_result.total_trades = result.total_trades
        grid_result.trades_per_year = result.trades_per_year
        grid_result.avg_holding_days = result.avg_holding_days

        # 统计出场原因（从 closed positions 获取 exit_reason）
        exit_counts = {
            "structure_stop_1": 0,
            "structure_stop_2": 0,
            "atr_stop": 0,
            "trailing_stop": 0,
            "take_profit_1": 0,
            "take_profit_2": 0,
        }

        for pos in result.positions:
            if pos.status == "closed" and pos.exit_reason:
                reason = pos.exit_reason
                if "structure_stop_1" in reason:
                    exit_counts["structure_stop_1"] += 1
                elif "structure_stop_2" in reason:
                    exit_counts["structure_stop_2"] += 1
                elif "atr_stop" in reason or "stop_loss" in reason:
                    exit_counts["atr_stop"] += 1
                elif "trailing" in reason:
                    exit_counts["trailing_stop"] += 1
                elif "take_profit_1" in reason:
                    exit_counts["take_profit_1"] += 1
                elif "take_profit_2" in reason:
                    exit_counts["take_profit_2"] += 1

        grid_result.structure_stop_1_count = exit_counts["structure_stop_1"]
        grid_result.structure_stop_2_count = exit_counts["structure_stop_2"]
        grid_result.atr_stop_count = exit_counts["atr_stop"]
        grid_result.trailing_stop_count = exit_counts["trailing_stop"]
        grid_result.t1_count = exit_counts["take_profit_1"]
        grid_result.t2_count = exit_counts["take_profit_2"]

        return grid_result

    def generate_report(
        self,
        results: List[GridResult],
        output_path: str,
    ) -> None:
        """生成 HTML 报告"""
        import webbrowser
        from pathlib import Path

        # 转换为 DataFrame 便于分析
        rows = []
        for r in results:
            row = {**r.params}
            row.update({
                "total_return": r.total_return,
                "annualized_return": r.annualized_return,
                "sharpe_ratio": r.sharpe_ratio,
                "sortino_ratio": r.sortino_ratio,
                "calmar_ratio": r.calmar_ratio,
                "max_drawdown": r.max_drawdown,
                "win_rate": r.win_rate,
                "profit_factor": r.profit_factor,
                "total_trades": r.total_trades,
                "structure_stop_1": r.structure_stop_1_count,
                "structure_stop_2": r.structure_stop_2_count,
                "atr_stop": r.atr_stop_count,
                "trailing_stop": r.trailing_stop_count,
            })
            rows.append(row)

        df = pd.DataFrame(rows)

        # 按夏普比率排序
        df_sorted = df.sort_values("sharpe_ratio", ascending=False)

        # 生成 HTML
        html = self._generate_html(df_sorted, results)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"报告已生成: {output_path}")

    def _generate_html(self, df: pd.DataFrame, results: List[GridResult]) -> str:
        """生成 HTML 报告内容"""
        # 找出最优参数组合
        best = df.iloc[0] if len(df) > 0 else None

        # 预计算最优值
        if best is not None:
            best_sharpe = f"{best['sharpe_ratio']:.2f}"
            best_annual = f"{best['annualized_return']:.2%}"
            best_dd = f"{best['max_drawdown']:.2%}"
        else:
            best_sharpe = best_annual = best_dd = "N/A"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_combinations = len(df)

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>参数网格扫描报告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f7fa; padding: 20px; }}
        .container {{ max-width: 1400px; margin: auto; }}
        h1 {{ color: #1a1a2e; margin-bottom: 20px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .metric-card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .metric-card h3 {{ color: #666; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
        .metric-card .value {{ font-size: 24px; font-weight: bold; color: #1a1a2e; }}
        .metric-card .value.positive {{ color: #2ecc71; }}
        .metric-card .value.negative {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 30px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; color: #666; font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 1px; }}
        tr:hover {{ background: #f8f9fa; }}
        .params {{ font-family: monospace; background: #f8f9fa; padding: 2px 6px; border-radius: 4px; }}
        .best-row {{ background: #e8f5e9 !important; }}
        .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .chart-card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .chart-card h3 {{ color: #1a1a2e; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>参数网格扫描报告</h1>
        <p style="color: #666; margin-bottom: 20px;">生成时间: {timestamp}</p>

        <div class="summary">
            <div class="metric-card">
                <h3>最优夏普比率</h3>
                <div class="value positive">{best_sharpe}</div>
            </div>
            <div class="metric-card">
                <h3>最优年化收益</h3>
                <div class="value positive">{best_annual}</div>
            </div>
            <div class="metric-card">
                <h3>最大回撤</h3>
                <div class="value negative">{best_dd}</div>
            </div>
            <div class="metric-card">
                <h3>总测试组合</h3>
                <div class="value">{total_combinations}</div>
            </div>
        </div>

        <div class="charts">
            <div class="chart-card">
                <h3>夏普比率 vs 参数组合</h3>
                <canvas id="sharpeChart"></canvas>
            </div>
            <div class="chart-card">
                <h3>收益与回撤分布</h3>
                <canvas id="returnChart"></canvas>
            </div>
        </div>

        <h2 style="margin-bottom: 15px;">详细参数组合结果</h2>
        <table>
            <thead>
                <tr>
                    <th>参数组合</th>
                    <th>总收益</th>
                    <th>年化收益</th>
                    <th>夏普比率</th>
                    <th>索提诺比率</th>
                    <th>卡玛比率</th>
                    <th>最大回撤</th>
                    <th>胜率</th>
                    <th>盈亏比</th>
                    <th>交易次数</th>
                    <th>结构止损1</th>
                    <th>结构止损2</th>
                    <th>ATR止损</th>
                    <th>追踪止损</th>
                </tr>
            </thead>
            <tbody>
"""

        for i, row in df.iterrows():
            is_best = (i == 0)
            best_class = "best-row" if is_best else ""

            params_str = ", ".join([f"{k}={v}" for k, v in row.items() if k in [
                "min_profit_loss_ratio", "atr_stop_multiplier", "atr_trailing_multiplier", "rsi_oversold"
            ]])

            html += f"""                <tr class="{best_class}">
                    <td><span class="params">{params_str}</span></td>
                    <td>{row["total_return"]:.2%}</td>
                    <td>{row["annualized_return"]:.2%}</td>
                    <td>{row["sharpe_ratio"]:.2f}</td>
                    <td>{row["sortino_ratio"]:.2f}</td>
                    <td>{row["calmar_ratio"]:.2f}</td>
                    <td>{row["max_drawdown"]:.2%}</td>
                    <td>{row["win_rate"]:.2%}</td>
                    <td>{row["profit_factor"]:.2f}</td>
                    <td>{row["total_trades"]}</td>
                    <td>{row["structure_stop_1"]}</td>
                    <td>{row["structure_stop_2"]}</td>
                    <td>{row["atr_stop"]}</td>
                    <td>{row["trailing_stop"]}</td>
                </tr>
"""

        html += """            </tbody>
        </table>
    </div>
</body>
</html>"""

        return html


def run_quick_scan(
    codes: List[str],
    start_date: str,
    end_date: str,
    output_dir: str = "reports",
) -> List[GridResult]:
    """
    快速参数扫描

    测试核心参数的敏感性：
    - min_profit_loss_ratio
    - atr_stop_multiplier
    - atr_trailing_multiplier
    - rsi_oversold
    """
    runner = ParameterGridRunner()

    # 简化网格（先跑小规模）
    grid = {
        "min_profit_loss_ratio": [1.5, 2.0, 2.5, 3.0],
        "atr_stop_multiplier": [1.5, 2.0, 2.5],
    }

    results = runner.run_grid(
        param_grid=grid,
        codes=codes,
        start_date=start_date,
        end_date=end_date,
        other_params={
            "atr_trailing_multiplier": 3.0,
            "rsi_oversold": 35,
        }
    )

    output_path = f"{output_dir}/parameter_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    runner.generate_report(results, output_path)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 测试数据
    test_codes = ["600000", "600519", "000858"]

    results = run_quick_scan(
        codes=test_codes,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    print(f"\\n扫描完成: {len(results)} 种参数组合")
