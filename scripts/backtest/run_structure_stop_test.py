#!/usr/bin/env python3
"""结构止损专项验证脚本

验证新实现的关键结构破坏止损是否正确工作：
1. structure_stop_1: 跌破入场后前一根K线最低点
2. structure_stop_2: 跌破前3日最低点

使用方式:
    python3 scripts/backtest/run_structure_stop_test.py --codes 600000 --start 2024-01-01 --end 2024-12-31
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

import pandas as pd
from src.backtest.engine import SwingBacktester
from src.backtest.strategy_params import StrategyParams
from src.backtest.models import Position

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def count_exit_reasons(trades: List) -> Dict[str, int]:
    """统计出场原因"""
    counts = {
        "structure_stop_1": 0,
        "structure_stop_2": 0,
        "atr_stop": 0,
        "trailing_stop": 0,
        "t1": 0,
        "t2": 0,
        "rsi_overbought": 0,
        "other": 0,
    }

    for trade in trades:
        reason = trade.exit_reason or ""
        if "structure_stop_1" in reason:
            counts["structure_stop_1"] += 1
        elif "structure_stop_2" in reason:
            counts["structure_stop_2"] += 1
        elif "atr_stop" in reason or "stop_loss" in reason:
            counts["atr_stop"] += 1
        elif "trailing" in reason:
            counts["trailing_stop"] += 1
        elif "take_profit_1" in reason:
            counts["t1"] += 1
        elif "take_profit_2" in reason:
            counts["t2"] += 1
        elif "rsi" in reason.lower():
            counts["rsi_overbought"] += 1
        else:
            counts["other"] += 1

    return counts


def test_structure_stop_scenario(
    runner: SwingBacktester,
    structure_config: Dict,
    description: str,
) -> Dict:
    """
    测试特定结构止损配置

    Args:
        runner: 回测器
        structure_config: 结构止损配置
            - use_entry_prev_low: 是否使用 entry_prev_low
            - use_lowest_3d_low: 是否使用 lowest_3d_low
        description: 测试描述
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"测试: {description}")
    logger.info(f"{'='*60}")

    # 运行回测
    result = runner.run(
        stock_codes=structure_config.get("codes", ["600000"]),
        start_date=structure_config.get("start", "2024-01-01"),
        end_date=structure_config.get("end", "2024-12-31"),
    )

    # 统计出场原因
    exit_counts = count_exit_reasons(result.trades)

    logger.info(f"总交易次数: {result.total_trades}")
    logger.info(f"总收益: {result.total_return:.2%}")
    logger.info(f"夏普比率: {result.sharpe_ratio:.2f}")
    logger.info(f"最大回撤: {result.max_drawdown:.2%}")
    logger.info(f"\n出场原因统计:")
    for reason, count in exit_counts.items():
        if count > 0:
            logger.info(f"  {reason}: {count}")

    return {
        "description": description,
        "total_trades": result.total_trades,
        "total_return": result.total_return,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "exit_counts": exit_counts,
    }


def run_structure_stop_tests(
    codes: List[str],
    start_date: str,
    end_date: str,
) -> List[Dict]:
    """运行结构止损测试"""
    results = []

    # 测试1: 标准配置（启用全部结构止损）
    logger.info("\n" + "="*60)
    logger.info("测试1: 标准配置（启用全部结构止损）")
    logger.info("="*60)

    params = StrategyParams(
        min_profit_loss_ratio=3.0,
        atr_stop_multiplier=2.0,
        atr_trailing_multiplier=3.0,
    )

    backtester = SwingBacktester(
        initial_capital=1_000_000,
        strategy_params=params,
    )

    result1 = test_structure_stop_scenario(
        backtester,
        {"codes": codes, "start": start_date, "end": end_date},
        "标准配置（全部启用）",
    )
    results.append(result1)

    # 测试2: 禁用结构止损1
    logger.info("\n" + "="*60)
    logger.info("测试2: 禁用结构止损1（仅 entry_prev_low）")
    logger.info("="*60)

    # 通过修改持仓的 entry_prev_low 为 0 来禁用
    backtester2 = SwingBacktester(
        initial_capital=1_000_000,
        strategy_params=params,
    )
    # 注意：这需要在持仓创建时设置，暂时无法通过回测器级别禁用

    # 测试3: 不同市场的结构止损表现
    market_periods = [
        ("2024-01-01", "2024-06-30", "2024上半年"),
        ("2024-07-01", "2024-12-31", "2024下半年"),
    ]

    for period_start, period_end, period_name in market_periods:
        logger.info(f"\n{'='*60}")
        logger.info(f"测试: {period_name} ({period_start} ~ {period_end})")
        logger.info("="*60)

        backtester_period = SwingBacktester(
            initial_capital=1_000_000,
            strategy_params=params,
        )

        result_period = test_structure_stop_scenario(
            backtester_period,
            {"codes": codes, "start": period_start, "end": period_end},
            f"{period_name}结构止损表现",
        )
        results.append(result_period)

    return results


def generate_comparison_report(results: List[Dict], output_path: str) -> None:
    """生成对比报告"""
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>结构止损验证报告</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .positive {{ color: #2ecc71; }}
        .negative {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <h1>结构止损验证报告</h1>
    <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <h2>测试结果汇总</h2>
    <table>
        <tr>
            <th>测试场景</th>
            <th>总收益</th>
            <th>夏普比率</th>
            <th>最大回撤</th>
            <th>交易次数</th>
            <th>结构止损1</th>
            <th>结构止损2</th>
            <th>ATR止损</th>
            <th>追踪止损</th>
        </tr>
"""

    for r in results:
        exit_c = r["exit_counts"]
        html += f"""        <tr>
            <td>{r['description']}</td>
            <td class="{'positive' if r['total_return'] > 0 else 'negative'}">{r['total_return']:.2%}</td>
            <td>{r['sharpe_ratio']:.2f}</td>
            <td class="negative">{r['max_drawdown']:.2%}</td>
            <td>{r['total_trades']}</td>
            <td>{exit_c.get('structure_stop_1', 0)}</td>
            <td>{exit_c.get('structure_stop_2', 0)}</td>
            <td>{exit_c.get('atr_stop', 0)}</td>
            <td>{exit_c.get('trailing_stop', 0)}</td>
        </tr>
"""

    html += """    </table>

    <h2>分析</h2>
    <ul>
        <li>结构止损1 (entry_prev_low): 跌破入场后前一根K线最低点</li>
        <li>结构止损2 (lowest_3d_low): 跌破前3日最低点</li>
        <li>ATR止损: 跌破入场价 - N*ATR</li>
        <li>追踪止损: 持仓最高价 - N*ATR</li>
    </ul>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"报告已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="结构止损专项验证")
    parser.add_argument("--codes", type=str, default="600000,600519",
                        help="股票代码列表，逗号分隔")
    parser.add_argument("--start", type=str, default="2024-01-01",
                        help="回测开始日期")
    parser.add_argument("--end", type=str, default="2024-12-31",
                        help="回测结束日期")

    args = parser.parse_args()
    codes = args.codes.split(",")

    logger.info(f"结构止损验证: codes={codes}, start={args.start}, end={args.end}")

    results = run_structure_stop_tests(codes, args.start, args.end)

    # 生成报告
    output_path = f"reports/structure_stop_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    generate_comparison_report(results, output_path)

    logger.info("\n结构止损验证完成!")


if __name__ == "__main__":
    main()
