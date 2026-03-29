#!/usr/bin/env python3
"""多参数组合优化测试脚本

测试多参数组合的效果：
- min_profit_loss_ratio × atr_stop_multiplier
- min_profit_loss_ratio × atr_trailing_multiplier
- atr_stop_multiplier × atr_trailing_multiplier

生成热力图可视化。

使用方式:
    python3 scripts/backtest/run_multi_param_test.py --codes 600000,600519 --start 2024-01-01 --end 2024-12-31
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

import numpy as np

from scripts.backtest.parameter_grid_runner import ParameterGridRunner, GridResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_2d_grid_test(
    runner: ParameterGridRunner,
    param_x: str,
    param_y: str,
    values_x: List[float],
    values_y: List[float],
    codes: List[str],
    start_date: str,
    end_date: str,
    fixed_params: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    运行二维参数网格测试

    Returns:
        (sharpe_matrix, return_matrix, drawdown_matrix, x_labels, y_labels)
    """
    grid = {
        param_x: values_x,
        param_y: values_y,
    }

    results = runner.run_grid(
        param_grid=grid,
        codes=codes,
        start_date=start_date,
        end_date=end_date,
        other_params=fixed_params,
    )

    # 构建矩阵
    sharpe_matrix = np.full((len(values_y), len(values_x)), np.nan)
    return_matrix = np.full((len(values_y), len(values_x)), np.nan)
    drawdown_matrix = np.full((len(values_y), len(values_x)), np.nan)

    x_labels = [str(v) for v in values_x]
    y_labels = [str(v) for v in values_y]

    # 填充结果
    for r in results:
        try:
            ix = values_x.index(r.params[param_x])
            iy = values_y.index(r.params[param_y])
            sharpe_matrix[iy, ix] = r.sharpe_ratio
            return_matrix[iy, ix] = r.annualized_return
            drawdown_matrix[iy, ix] = r.max_drawdown
        except (ValueError, KeyError):
            continue

    return sharpe_matrix, return_matrix, drawdown_matrix, x_labels, y_labels


def generate_heatmap_html(
    param_x: str,
    param_y: str,
    sharpe_matrix: np.ndarray,
    return_matrix: np.ndarray,
    drawdown_matrix: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    output_path: str,
) -> None:
    """生成热力图 HTML 报告"""

    # 找到最优位置
    valid_sharpe = sharpe_matrix[~np.isnan(sharpe_matrix)]
    if len(valid_sharpe) > 0:
        best_idx = np.nanargmax(sharpe_matrix)
        best_iy, best_ix = np.unravel_index(best_idx, sharpe_matrix.shape)
        best_sharpe = sharpe_matrix[best_iy, best_ix]
        best_return = return_matrix[best_iy, best_ix]
        best_params = f"{param_x}={x_labels[best_ix]}, {param_y}={y_labels[best_iy]}"
    else:
        best_sharpe = best_return = 0
        best_params = "N/A"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 将矩阵转换为JSON
    def matrix_to_json(m):
        return np.nan_to_num(m).tolist()

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>多参数组合热力图</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #f5f7fa; padding: 20px; }}
        .container {{ max-width: 1400px; margin: auto; }}
        h1 {{ color: #1a1a2e; margin-bottom: 10px; }}
        .meta {{ color: #666; margin-bottom: 20px; }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .card h3 {{ color: #666; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
        .card .value {{ font-size: 24px; font-weight: bold; color: #1a1a2e; }}
        .card .value.positive {{ color: #2ecc71; }}
        .card .value.negative {{ color: #e74c3c; }}
        .heatmap-container {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .heatmap-container h2 {{ color: #1a1a2e; margin-bottom: 15px; }}
        .heatmap {{ width: 100%; height: 400px; }}
        .controls {{ margin-bottom: 20px; }}
        .controls button {{
            padding: 8px 16px;
            margin-right: 10px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            background: #3498db;
            color: white;
            font-size: 14px;
        }}
        .controls button:hover {{ background: #2980b9; }}
        .controls button.active {{ background: #1a1a2e; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-top: 20px; }}
        th, td {{ padding: 10px 12px; text-align: center; border: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .best {{ background: #e8f5e9; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>多参数组合热力图</h1>
        <p class="meta">
            {param_x} vs {param_y} | 生成时间: {timestamp}
        </p>

        <div class="summary">
            <div class="card">
                <h3>最优夏普比率</h3>
                <div class="value positive">{best_sharpe:.2f}</div>
            </div>
            <div class="card">
                <h3>对应年化收益</h3>
                <div class="value positive">{best_return:.2%}</div>
            </div>
            <div class="card">
                <h3>最优参数组合</h3>
                <div class="value" style="font-size: 16px;">{best_params}</div>
            </div>
        </div>

        <div class="controls">
            <button class="active" onclick="showChart('sharpe')">夏普比率</button>
            <button onclick="showChart('return')">年化收益</button>
            <button onclick="showChart('drawdown')">最大回撤</button>
        </div>

        <div class="heatmap-container">
            <h2 id="chartTitle">夏普比率热力图</h2>
            <canvas id="heatmapCanvas" class="heatmap"></canvas>
        </div>

        <h2 style="margin-bottom: 15px;">详细数据表</h2>
        <table id="dataTable">
            <thead>
                <tr>
                    <th>{param_y} \\ {param_x}</th>
                    {''.join([f'<th>{x}</th>' for x in x_labels])}
                </tr>
            </thead>
            <tbody>
"""

    # 生成表格
    for iy, y_label in enumerate(y_labels):
        html += f"""                <tr>
                    <th>{y_label}</th>
"""
        for ix, x_label in enumerate(x_labels):
            sharpe = sharpe_matrix[iy, ix]
            is_best = (iy == best_iy and ix == best_ix) if not np.isnan(sharpe) else False
            cell_class = "best" if is_best else ""
            value_str = f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A"
            html += f"""                    <td class="{cell_class}">{value_str}</td>
"""
        html += """                </tr>
"""

    html += """            </tbody>
        </table>
    </div>

    <script>
        const sharpeData = """ + matrix_to_json(sharpe_matrix) + """;
        const returnData = """ + matrix_to_json(return_matrix) + """;
        const drawdownData = """ + matrix_to_json(drawdown_matrix) + """;
        const xLabels = """ + str(x_labels) + """;
        const yLabels = """ + str(y_labels) + """;

        let currentChart = null;

        function showChart(type) {
            const canvas = document.getElementById('heatmapCanvas');
            const title = document.getElementById('chartTitle');
            const buttons = document.querySelectorAll('.controls button');

            buttons.forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');

            let data, label, color;

            if (type === 'sharpe') {
                data = sharpeData;
                label = '夏普比率';
                color = (v) => v > 1.5 ? '#2ecc71' : v > 1.0 ? '#f1c40f' : v > 0 ? '#e67e22' : '#e74c3c';
            } else if (type === 'return') {
                data = returnData;
                label = '年化收益率';
                color = (v) => v > 0.3 ? '#2ecc71' : v > 0.1 ? '#f1c40f' : v > 0 ? '#e67e22' : '#e74c3c';
            } else {
                data = drawdownData;
                label = '最大回撤';
                color = (v) => v < 0.1 ? '#2ecc71' : v < 0.2 ? '#f1c40f' : v < 0.3 ? '#e67e22' : '#e74c3c';
            }

            title.textContent = label + '热力图';

            if (currentChart) {
                currentChart.destroy();
            }

            // 创建伪热力图使用柱状图
            const datasets = [];
            for (let i = 0; i < data.length; i++) {
                for (let j = 0; j < data[i].length; j++) {
                    if (!isNaN(data[i][j])) {
                        datasets.push({
                            label: yLabels[i] + ' x ' + xLabels[j],
                            data: [data[i][j]],
                            backgroundColor: color(data[i][j]),
                        });
                    }
                }
            }

            currentChart = new Chart(canvas, {
                type: 'bar',
                data: {
                    labels: [label],
                    datasets: datasets.slice(0, 50)  // 限制显示数量
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: { title: { display: true, text: label } }
                    }
                }
            });
        }

        showChart('sharpe');
    </script>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"热力图报告已生成: {output_path}")


def run_multi_param_tests(
    codes: List[str],
    start_date: str,
    end_date: str,
    output_dir: str = "reports",
) -> None:
    """运行多参数组合测试"""

    runner = ParameterGridRunner()

    # 固定参数
    fixed = {
        "atr_trailing_multiplier": 3.0,
        "rsi_oversold": 35,
    }

    # 测试1: min_profit_loss_ratio × atr_stop_multiplier
    logger.info("\n" + "="*60)
    logger.info("测试1: min_profit_loss_ratio × atr_stop_multiplier")
    logger.info("="*60)

    sharpe, returns, dd, x_labels, y_labels = run_2d_grid_test(
        runner,
        param_x="min_profit_loss_ratio",
        param_y="atr_stop_multiplier",
        values_x=[1.5, 2.0, 2.5, 3.0, 3.5],
        values_y=[1.5, 2.0, 2.5, 3.0],
        codes=codes,
        start_date=start_date,
        end_date=end_date,
        fixed_params=fixed,
    )

    generate_heatmap_html(
        param_x="min_profit_loss_ratio",
        param_y="atr_stop_multiplier",
        sharpe_matrix=sharpe,
        return_matrix=returns,
        drawdown_matrix=dd,
        x_labels=x_labels,
        y_labels=y_labels,
        output_path=f"{output_dir}/heatmap_ratio_vs_atr_stop_{datetime.now().strftime('%Y%m%d')}.html",
    )

    # 测试2: min_profit_loss_ratio × atr_trailing_multiplier
    logger.info("\n" + "="*60)
    logger.info("测试2: min_profit_loss_ratio × atr_trailing_multiplier")
    logger.info("="*60)

    fixed2 = {
        "atr_stop_multiplier": 2.0,
        "rsi_oversold": 35,
    }

    sharpe2, returns2, dd2, x_labels2, y_labels2 = run_2d_grid_test(
        runner,
        param_x="min_profit_loss_ratio",
        param_y="atr_trailing_multiplier",
        values_x=[1.5, 2.0, 2.5, 3.0, 3.5],
        values_y=[2.0, 2.5, 3.0, 3.5, 4.0],
        codes=codes,
        start_date=start_date,
        end_date=end_date,
        fixed_params=fixed2,
    )

    generate_heatmap_html(
        param_x="min_profit_loss_ratio",
        param_y="atr_trailing_multiplier",
        sharpe_matrix=sharpe2,
        return_matrix=returns2,
        drawdown_matrix=dd2,
        x_labels=x_labels2,
        y_labels=y_labels2,
        output_path=f"{output_dir}/heatmap_ratio_vs_atr_trailing_{datetime.now().strftime('%Y%m%d')}.html",
    )

    # 测试3: atr_stop_multiplier × atr_trailing_multiplier
    logger.info("\n" + "="*60)
    logger.info("测试3: atr_stop_multiplier × atr_trailing_multiplier")
    logger.info("="*60)

    fixed3 = {
        "min_profit_loss_ratio": 3.0,
        "rsi_oversold": 35,
    }

    sharpe3, returns3, dd3, x_labels3, y_labels3 = run_2d_grid_test(
        runner,
        param_x="atr_stop_multiplier",
        param_y="atr_trailing_multiplier",
        values_x=[1.5, 2.0, 2.5, 3.0],
        values_y=[2.0, 2.5, 3.0, 3.5, 4.0],
        codes=codes,
        start_date=start_date,
        end_date=end_date,
        fixed_params=fixed3,
    )

    generate_heatmap_html(
        param_x="atr_stop_multiplier",
        param_y="atr_trailing_multiplier",
        sharpe_matrix=sharpe3,
        return_matrix=returns3,
        drawdown_matrix=dd3,
        x_labels=x_labels3,
        y_labels=y_labels3,
        output_path=f"{output_dir}/heatmap_atr_stop_vs_trailing_{datetime.now().strftime('%Y%m%d')}.html",
    )

    logger.info("\n多参数组合测试完成!")


def main():
    parser = argparse.ArgumentParser(description="多参数组合优化测试")
    parser.add_argument("--codes", type=str, default="600000,600519,000858",
                        help="股票代码列表，逗号分隔")
    parser.add_argument("--start", type=str, default="2024-01-01",
                        help="回测开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31",
                        help="回测结束日期 (YYYY-MM-DD)")
    parser.add_argument("--workers", type=int, default=4,
                        help="并行工作进程数")

    args = parser.parse_args()
    codes = args.codes.split(",")

    Path("reports").mkdir(exist_ok=True)

    run_multi_param_tests(codes, args.start, args.end)


if __name__ == "__main__":
    main()
