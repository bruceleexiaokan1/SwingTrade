#!/usr/bin/env python3
"""跨时间段稳健性验证脚本

在不同市场周期（牛市/熊市/震荡/反弹）中验证策略的稳健性。

使用方式:
    python3 scripts/backtest/run_cross_period_test.py --codes 600000,600519 --output reports/cross_period
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

from src.backtest.engine import SwingBacktester
from src.backtest.strategy_params import StrategyParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# 市场周期定义
MARKET_PERIODS = [
    {
        "name": "牛市",
        "start": "2019-01-01",
        "end": "2020-12-31",
        "description": "指数单边上涨，趋势明显",
    },
    {
        "name": "震荡市(2021)",
        "start": "2021-01-01",
        "end": "2021-12-31",
        "description": "区间波动，结构性行情",
    },
    {
        "name": "熊市",
        "start": "2022-01-01",
        "end": "2022-12-31",
        "description": "指数单边下跌",
    },
    {
        "name": "震荡市(2023)",
        "start": "2023-01-01",
        "end": "2023-12-31",
        "description": "区间波动，结构性行情",
    },
    {
        "name": "反弹市",
        "start": "2024-01-01",
        "end": "2024-06-30",
        "description": "下跌后反弹，趋势明显",
    },
    {
        "name": "震荡市(2024下)",
        "start": "2024-07-01",
        "end": "2024-12-31",
        "description": "区间波动",
    },
]


def run_period_test(
    period: Dict,
    codes: List[str],
) -> Dict:
    """运行单个时间段的回测"""

    logger.info(f"\n{'='*60}")
    logger.info(f"测试: {period['name']} ({period['start']} ~ {period['end']})")
    logger.info(f"特征: {period['description']}")
    logger.info("="*60)

    params = StrategyParams(
        min_profit_loss_ratio=3.0,
        atr_stop_multiplier=2.0,
        atr_trailing_multiplier=3.0,
        rsi_oversold=35,
    )

    backtester = SwingBacktester(
        initial_capital=1_000_000,
        strategy_params=params,
    )

    try:
        result = backtester.run(
            stock_codes=codes,
            start_date=period["start"],
            end_date=period["end"],
        )

        # 统计出场原因
        exit_counts = {
            "structure_stop_1": 0,
            "structure_stop_2": 0,
            "atr_stop": 0,
            "trailing_stop": 0,
            "t1": 0,
            "t2": 0,
            "other": 0,
        }

        for trade in result.trades:
            reason = trade.exit_reason or ""
            if "structure_stop_1" in reason:
                exit_counts["structure_stop_1"] += 1
            elif "structure_stop_2" in reason:
                exit_counts["structure_stop_2"] += 1
            elif "atr_stop" in reason or "stop_loss" in reason:
                exit_counts["atr_stop"] += 1
            elif "trailing" in reason:
                exit_counts["trailing_stop"] += 1
            elif "take_profit_1" in reason:
                exit_counts["t1"] += 1
            elif "take_profit_2" in reason:
                exit_counts["t2"] += 1
            else:
                exit_counts["other"] += 1

        period_result = {
            "name": period["name"],
            "start": period["start"],
            "end": period["end"],
            "description": period["description"],
            "total_trades": result.total_trades,
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "exit_counts": exit_counts,
            "success": True,
        }

        logger.info(f"结果: 总收益={result.total_return:.2%}, 夏普={result.sharpe_ratio:.2f}, 最大回撤={result.max_drawdown:.2%}")

    except Exception as e:
        logger.error(f"回测失败: {e}")
        period_result = {
            "name": period["name"],
            "start": period["start"],
            "end": period["end"],
            "description": period["description"],
            "success": False,
            "error": str(e),
        }

    return period_result


def generate_cross_period_report(
    results: List[Dict],
    output_path: str,
) -> None:
    """生成跨时间段验证报告"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 计算汇总统计
    successful = [r for r in results if r.get("success", False)]
    if successful:
        avg_return = sum(r["total_return"] for r in successful) / len(successful)
        avg_sharpe = sum(r["sharpe_ratio"] for r in successful) / len(successful)
        avg_dd = sum(r["max_drawdown"] for r in successful) / len(successful)
        avg_trades = sum(r["total_trades"] for r in successful) / len(successful)

        # 找出最好和最差的时期
        best = max(successful, key=lambda x: x["total_return"])
        worst = min(successful, key=lambda x: x["total_return"])
    else:
        avg_return = avg_sharpe = avg_dd = avg_trades = 0
        best = worst = None

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>跨时间段稳健性验证报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #f5f7fa; padding: 20px; }}
        .container {{ max-width: 1200px; margin: auto; }}
        h1 {{ color: #1a1a2e; margin-bottom: 10px; }}
        .meta {{ color: #666; margin-bottom: 20px; }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .card h3 {{ color: #666; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
        .card .value {{ font-size: 22px; font-weight: bold; color: #1a1a2e; }}
        .card .value.positive {{ color: #2ecc71; }}
        .card .value.negative {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 30px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; color: #666; font-weight: 600; text-transform: uppercase; font-size: 11px; }}
        tr:hover {{ background: #f8f9fa; }}
        .period-name {{ font-weight: bold; }}
        .period-desc {{ font-size: 12px; color: #999; }}
        .positive {{ color: #2ecc71; }}
        .negative {{ color: #e74c3c; }}
        .best-row {{ background: #e8f5e9; }}
        .worst-row {{ background: #ffe6e6; }}
        .chart-container {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>跨时间段稳健性验证报告</h1>
        <p class="meta">生成时间: {timestamp}</p>

        <div class="summary">
            <div class="card">
                <h3>平均年化收益</h3>
                <div class="value {'positive' if avg_return > 0 else 'negative'}">{avg_return:.2%}</div>
            </div>
            <div class="card">
                <h3>平均夏普比率</h3>
                <div class="value {'positive' if avg_sharpe > 1 else ''}">{avg_sharpe:.2f}</div>
            </div>
            <div class="card">
                <h3>平均最大回撤</h3>
                <div class="value negative">{avg_dd:.2%}</div>
            </div>
            <div class="card">
                <h3>平均交易次数</h3>
                <div class="value">{avg_trades:.0f}</div>
            </div>
            <div class="card">
                <h3>最差时期</h3>
                <div class="value" style="font-size: 16px;">{worst['name'] if worst else 'N/A'}</div>
            </div>
            <div class="card">
                <h3>最优时期</h3>
                <div class="value" style="font-size: 16px;">{best['name'] if best else 'N/A'}</div>
            </div>
        </div>

        <h2 style="margin-bottom: 15px;">各时期表现详情</h2>
        <table>
            <thead>
                <tr>
                    <th>时期</th>
                    <th>时间范围</th>
                    <th>总收益</th>
                    <th>年化收益</th>
                    <th>夏普比率</th>
                    <th>最大回撤</th>
                    <th>胜率</th>
                    <th>交易次数</th>
                    <th>结构止损1</th>
                    <th>结构止损2</th>
                    <th>ATR止损</th>
                    <th>追踪止损</th>
                </tr>
            </thead>
            <tbody>
"""

    for r in results:
        if not r.get("success", False):
            html += f"""                <tr>
                    <td class="period-name">{r['name']}</td>
                    <td>{r['start']} ~ {r['end']}</td>
                    <td colspan="9" style="color: #e74c3c;">失败: {r.get('error', 'Unknown')}</td>
                </tr>
"""
            continue

        is_best = (best and r["name"] == best["name"])
        is_worst = (worst and r["name"] == worst["name"])
        row_class = "best-row" if is_best else ("worst-row" if is_worst else "")

        exit_c = r.get("exit_counts", {})

        html += f"""                <tr class="{row_class}">
                    <td class="period-name">{r['name']}<br><span class="period-desc">{r['description']}</span></td>
                    <td>{r['start']} ~ {r['end']}</td>
                    <td class="{'positive' if r['total_return'] > 0 else 'negative'}">{r['total_return']:.2%}</td>
                    <td class="{'positive' if r['annualized_return'] > 0 else 'negative'}">{r['annualized_return']:.2%}</td>
                    <td>{r['sharpe_ratio']:.2f}</td>
                    <td class="negative">{r['max_drawdown']:.2%}</td>
                    <td>{r['win_rate']:.2%}</td>
                    <td>{r['total_trades']}</td>
                    <td>{exit_c.get('structure_stop_1', 0)}</td>
                    <td>{exit_c.get('structure_stop_2', 0)}</td>
                    <td>{exit_c.get('atr_stop', 0)}</td>
                    <td>{exit_c.get('trailing_stop', 0)}</td>
                </tr>
"""

    html += """            </tbody>
        </table>

        <h2 style="margin-bottom: 15px;">稳健性分析</h2>
        <div style="background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); line-height: 1.8;">
            <h3 style="margin-bottom: 10px;">评估标准</h3>
            <ul>
                <li><strong>夏普比率</strong>: > 1.5 优秀, 1.0~1.5 良好, < 1.0 一般</li>
                <li><strong>最大回撤</strong>: < 10% 优秀, 10%~20% 良好, > 20% 风险较高</li>
                <li><strong>胜率</strong>: > 50% 为正向策略</li>
            </ul>

            <h3 style="margin: 20px 0 10px;">跨周期稳定性评估</h3>
            <p>策略应在不同市场环境下保持稳定的夏普比率和可控的最大回撤。如果某类市场环境下表现明显较差，需要针对性优化。</p>
        </div>
    </div>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"跨时间段报告已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="跨时间段稳健性验证")
    parser.add_argument("--codes", type=str, default="600000,600519,000858,000001,600036",
                        help="股票代码列表，逗号分隔")
    parser.add_argument("--output", type=str, default="reports/cross_period",
                        help="输出目录")

    args = parser.parse_args()
    codes = args.codes.split(",")

    Path("reports").mkdir(exist_ok=True)

    results = []
    for period in MARKET_PERIODS:
        result = run_period_test(period, codes)
        results.append(result)

    # 生成报告
    output_path = f"{args.output}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    generate_cross_period_report(results, output_path)

    logger.info("\n跨时间段验证完成!")


if __name__ == "__main__":
    main()
