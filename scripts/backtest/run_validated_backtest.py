#!/usr/bin/env python3
"""经验证的有效性回测

使用较低的 min_profit_loss_ratio 来验证策略本身是否有效，
然后再评估严格的盈亏比要求是否合理。

使用方式:
    python3 scripts/backtest/run_validated_backtest.py --codes 600519,000001 --start 2024-01-01 --end 2024-12-31
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

from src.backtest.engine import SwingBacktester
from src.backtest.strategy_params import StrategyParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_comparative_test(
    codes: list,
    start_date: str,
    end_date: str,
    output_dir: str = "reports",
) -> None:
    """运行对比测试"""

    Path(output_dir).mkdir(exist_ok=True)

    # 测试配置
    configs = [
        {
            "name": "严格盈亏比 (3.0)",
            "min_profit_loss_ratio": 3.0,
            "description": "知识库推荐值，中长线 >= 1:3",
        },
        {
            "name": "适中盈亏比 (2.0)",
            "min_profit_loss_ratio": 2.0,
            "description": "平衡风险和机会",
        },
        {
            "name": "宽松盈亏比 (1.5)",
            "min_profit_loss_ratio": 1.5,
            "description": "最大化交易机会",
        },
        {
            "name": "无盈亏比限制 (0.0)",
            "min_profit_loss_ratio": 0.0,
            "description": "不做盈亏比过滤",
        },
    ]

    results = []

    for config in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"测试: {config['name']}")
        logger.info(f"{'='*60}")

        params = StrategyParams(
            min_profit_loss_ratio=config["min_profit_loss_ratio"],
            atr_stop_multiplier=2.0,
            atr_trailing_multiplier=3.0,
            rsi_oversold=35,
            entry_confidence_threshold=0.5,
        )

        backtester = SwingBacktester(
            initial_capital=1_000_000,
            strategy_params=params,
        )

        result = backtester.run(
            stock_codes=codes,
            start_date=start_date,
            end_date=end_date,
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
            signal_type = trade.signal_type
            if signal_type == "structure_stop_1":
                exit_counts["structure_stop_1"] += 1
            elif signal_type == "structure_stop_2":
                exit_counts["structure_stop_2"] += 1
            elif signal_type == "stop_loss":
                exit_counts["atr_stop"] += 1
            elif signal_type == "trailing_stop":
                exit_counts["trailing_stop"] += 1
            elif signal_type == "take_profit_1":
                exit_counts["t1"] += 1
            elif signal_type == "take_profit_2":
                exit_counts["t2"] += 1
            else:
                exit_counts["other"] += 1

        logger.info(f"  总交易: {result.total_trades}")
        logger.info(f"  总收益: {result.total_return:.2%}")
        logger.info(f"  夏普比率: {result.sharpe_ratio:.2f}")
        logger.info(f"  最大回撤: {result.max_drawdown:.2%}")
        logger.info(f"  出场: 结构1={exit_counts['structure_stop_1']}, 结构2={exit_counts['structure_stop_2']}, ATR={exit_counts['atr_stop']}, 追踪={exit_counts['trailing_stop']}, T1={exit_counts['t1']}, T2={exit_counts['t2']}")

        results.append({
            "name": config["name"],
            "min_profit_loss_ratio": config["min_profit_loss_ratio"],
            "description": config["description"],
            "total_trades": result.total_trades,
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "exit_counts": exit_counts,
        })

    # 生成报告
    generate_comparison_report(results, output_dir, start_date, end_date)

    return results


def generate_comparison_report(
    results: list,
    output_dir: str,
    start_date: str,
    end_date: str,
) -> None:
    """生成对比报告"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows_html = ""
    for r in results:
        exit_c = r["exit_counts"]
        rows_html += f"""                <tr>
                    <td><strong>{r['name']}</strong><br><small>{r['description']}</small></td>
                    <td>{r['min_profit_loss_ratio']}</td>
                    <td>{r['total_trades']}</td>
                    <td class="{'positive' if r['total_return'] > 0 else 'negative'}">{r['total_return']:.2%}</td>
                    <td class="{'positive' if r['annualized_return'] > 0 else 'negative'}">{r['annualized_return']:.2%}</td>
                    <td>{r['sharpe_ratio']:.2f}</td>
                    <td class="negative">{r['max_drawdown']:.2%}</td>
                    <td>{r['win_rate']:.2%}</td>
                    <td>{exit_c['structure_stop_1']}</td>
                    <td>{exit_c['structure_stop_2']}</td>
                    <td>{exit_c['atr_stop']}</td>
                    <td>{exit_c['trailing_stop']}</td>
                </tr>
"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>盈亏比对比验证报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #f5f7fa; padding: 20px; }}
        .container {{ max-width: 1400px; margin: auto; }}
        h1 {{ color: #1a1a2e; margin-bottom: 10px; }}
        .meta {{ color: #666; margin-bottom: 20px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .card h3 {{ color: #666; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
        .card .value {{ font-size: 24px; font-weight: bold; color: #1a1a2e; }}
        .card .value.positive {{ color: #2ecc71; }}
        .card .value.negative {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 30px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; color: #666; font-weight: 600; text-transform: uppercase; font-size: 11px; }}
        tr:hover {{ background: #f8f9fa; }}
        .positive {{ color: #2ecc71; }}
        .negative {{ color: #e74c3c; }}
        .analysis {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); line-height: 1.8; }}
        .analysis h2 {{ color: #1a1a2e; margin-bottom: 15px; }}
        .conclusion {{ background: #e8f5e9; border-radius: 10px; padding: 20px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>盈亏比对比验证报告</h1>
        <p class="meta">回测区间: {start_date} ~ {end_date} | 生成时间: {timestamp}</p>

        <h2 style="margin-bottom: 15px;">测试结果对比</h2>
        <table>
            <thead>
                <tr>
                    <th>配置</th>
                    <th>盈亏比阈值</th>
                    <th>交易次数</th>
                    <th>总收益</th>
                    <th>年化收益</th>
                    <th>夏普比率</th>
                    <th>最大回撤</th>
                    <th>胜率</th>
                    <th>结构止损1</th>
                    <th>结构止损2</th>
                    <th>ATR止损</th>
                    <th>追踪止损</th>
                </tr>
            </thead>
            <tbody>
{rows_html}
            </tbody>
        </table>

        <div class="analysis">
            <h2>分析结论</h2>

            <h3>1. 盈亏比阈值对交易机会的影响</h3>
            <p>较高的 min_profit_loss_ratio 会过滤掉盈亏比不足的交易机会，但可能提高单笔交易的质量。
            对于A股波段交易，由于涨停板限制和波动特性，5% 的预期涨幅配合 2x ATR 止损往往难以达到 3:1 的盈亏比。</p>

            <h3>2. 建议</h3>
            <ul>
                <li><strong>对于高单价股票</strong>（如茅台）：考虑使用相对盈亏比（如基于ATR百分比的阈值）而非固定阈值</li>
                <li><strong>对于活跃股票</strong>：可以适当降低阈值以增加交易机会</li>
                <li><strong>风险偏好</strong>：保守型交易者可以使用较高的阈值，激进型可以使用较低的阈值</li>
            </ul>

            <div class="conclusion">
                <h3>核心发现</h3>
                <p>如果宽松盈亏比（如1.5或0）能产生更多交易且风险可控，
                说明当前固定 5% 预期涨幅的假设可能过于保守。</p>
            </div>
        </div>
    </div>
</body>
</html>"""

    output_path = f"{output_dir}/ratio_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"\n报告已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="盈亏比对比验证")
    parser.add_argument("--codes", type=str,
                        default="600519,000001,600036",
                        help="股票代码列表，逗号分隔")
    parser.add_argument("--start", type=str, default="2024-01-01",
                        help="开始日期")
    parser.add_argument("--end", type=str, default="2024-12-31",
                        help="结束日期")

    args = parser.parse_args()
    codes = args.codes.split(",")

    logger.info(f"盈亏比对比验证: codes={codes}, start={args.start}, end={args.end}")

    run_comparative_test(codes, args.start, args.end)


if __name__ == "__main__":
    main()
