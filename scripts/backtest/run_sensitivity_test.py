#!/usr/bin/env python3
"""核心参数敏感性测试脚本

测试新实现的参数对回测绩效的影响：
1. min_profit_loss_ratio: [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
2. atr_stop_multiplier: [1.5, 2.0, 2.5, 3.0]
3. atr_trailing_multiplier: [2.0, 2.5, 3.0, 3.5, 4.0]
4. rsi_oversold: [25, 30, 35, 40]

使用方式:
    python3 scripts/backtest/run_sensitivity_test.py --codes 600000,600519 --start 2024-01-01 --end 2024-12-31
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

from scripts.backtest.parameter_grid_runner import ParameterGridRunner, run_quick_scan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ]
)
logger = logging.getLogger(__name__)


def run_min_profit_loss_ratio_test(
    runner: ParameterGridRunner,
    codes: list,
    start_date: str,
    end_date: str,
) -> None:
    """测试 min_profit_loss_ratio 敏感性"""
    logger.info("=" * 60)
    logger.info("测试 min_profit_loss_ratio 敏感性")
    logger.info("=" * 60)

    grid = {
        "min_profit_loss_ratio": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    }

    # 固定其他参数
    other_params = {
        "atr_stop_multiplier": 2.0,
        "atr_trailing_multiplier": 3.0,
        "rsi_oversold": 35,
    }

    results = runner.run_grid(
        param_grid=grid,
        codes=codes,
        start_date=start_date,
        end_date=end_date,
        other_params=other_params,
    )

    # 输出结果
    print("\nmin_profit_loss_ratio 敏感性测试结果:")
    print("-" * 80)
    print(f"{'ratio':>8} {'总收益':>10} {'年化收益':>10} {'夏普':>8} {'最大回撤':>10} {'交易次数':>8}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x.params["min_profit_loss_ratio"]):
        p = r.params["min_profit_loss_ratio"]
        print(f"{p:>8.1f} {r.total_return:>10.2%} {r.annualized_return:>10.2%} {r.sharpe_ratio:>8.2f} {r.max_drawdown:>10.2%} {r.total_trades:>8}")

    # 生成报告
    output_path = f"reports/sensitivity_min_profit_loss_ratio_{datetime.now().strftime('%Y%m%d')}.html"
    runner.generate_report(results, output_path)
    logger.info(f"报告已生成: {output_path}")


def run_atr_stop_test(
    runner: ParameterGridRunner,
    codes: list,
    start_date: str,
    end_date: str,
) -> None:
    """测试 ATR 止损倍数敏感性"""
    logger.info("=" * 60)
    logger.info("测试 atr_stop_multiplier 敏感性")
    logger.info("=" * 60)

    grid = {
        "atr_stop_multiplier": [1.5, 2.0, 2.5, 3.0],
    }

    other_params = {
        "min_profit_loss_ratio": 3.0,  # 使用知识库推荐值
        "atr_trailing_multiplier": 3.0,
        "rsi_oversold": 35,
    }

    results = runner.run_grid(
        param_grid=grid,
        codes=codes,
        start_date=start_date,
        end_date=end_date,
        other_params=other_params,
    )

    print("\nATR 止损倍数敏感性测试结果:")
    print("-" * 80)
    print(f"{'倍数':>8} {'总收益':>10} {'年化收益':>10} {'夏普':>8} {'最大回撤':>10} {'交易次数':>8}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x.params["atr_stop_multiplier"]):
        p = r.params["atr_stop_multiplier"]
        print(f"{p:>8.1f} {r.total_return:>10.2%} {r.annualized_return:>10.2%} {r.sharpe_ratio:>8.2f} {r.max_drawdown:>10.2%} {r.total_trades:>8}")

    output_path = f"reports/sensitivity_atr_stop_{datetime.now().strftime('%Y%m%d')}.html"
    runner.generate_report(results, output_path)
    logger.info(f"报告已生成: {output_path}")


def run_atr_trailing_test(
    runner: ParameterGridRunner,
    codes: list,
    start_date: str,
    end_date: str,
) -> None:
    """测试 ATR 追踪止损倍数敏感性"""
    logger.info("=" * 60)
    logger.info("测试 atr_trailing_multiplier 敏感性")
    logger.info("=" * 60)

    grid = {
        "atr_trailing_multiplier": [2.0, 2.5, 3.0, 3.5, 4.0],
    }

    other_params = {
        "min_profit_loss_ratio": 3.0,
        "atr_stop_multiplier": 2.0,
        "rsi_oversold": 35,
    }

    results = runner.run_grid(
        param_grid=grid,
        codes=codes,
        start_date=start_date,
        end_date=end_date,
        other_params=other_params,
    )

    print("\nATR 追踪止损倍数敏感性测试结果:")
    print("-" * 80)
    print(f"{'倍数':>8} {'总收益':>10} {'年化收益':>10} {'夏普':>8} {'最大回撤':>10} {'交易次数':>8}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x.params["atr_trailing_multiplier"]):
        p = r.params["atr_trailing_multiplier"]
        print(f"{p:>8.1f} {r.total_return:>10.2%} {r.annualized_return:>10.2%} {r.sharpe_ratio:>8.2f} {r.max_drawdown:>10.2%} {r.total_trades:>8}")

    output_path = f"reports/sensitivity_atr_trailing_{datetime.now().strftime('%Y%m%d')}.html"
    runner.generate_report(results, output_path)
    logger.info(f"报告已生成: {output_path}")


def run_rsi_threshold_test(
    runner: ParameterGridRunner,
    codes: list,
    start_date: str,
    end_date: str,
) -> None:
    """测试 RSI 超卖阈值敏感性"""
    logger.info("=" * 60)
    logger.info("测试 rsi_oversold 敏感性")
    logger.info("=" * 60)

    grid = {
        "rsi_oversold": [25, 30, 35, 40],
    }

    other_params = {
        "min_profit_loss_ratio": 3.0,
        "atr_stop_multiplier": 2.0,
        "atr_trailing_multiplier": 3.0,
    }

    results = runner.run_grid(
        param_grid=grid,
        codes=codes,
        start_date=start_date,
        end_date=end_date,
        other_params=other_params,
    )

    print("\nRSI 超卖阈值敏感性测试结果:")
    print("-" * 80)
    print(f"{'RSI':>8} {'总收益':>10} {'年化收益':>10} {'夏普':>8} {'最大回撤':>10} {'交易次数':>8}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x.params["rsi_oversold"]):
        p = r.params["rsi_oversold"]
        print(f"{p:>8.0f} {r.total_return:>10.2%} {r.annualized_return:>10.2%} {r.sharpe_ratio:>8.2f} {r.max_drawdown:>10.2%} {r.total_trades:>8}")

    output_path = f"reports/sensitivity_rsi_{datetime.now().strftime('%Y%m%d')}.html"
    runner.generate_report(results, output_path)
    logger.info(f"报告已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="核心参数敏感性测试")
    parser.add_argument("--codes", type=str, default="600000,600519,000858,000001,600036",
                        help="股票代码列表，逗号分隔")
    parser.add_argument("--start", type=str, default="2024-01-01",
                        help="回测开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31",
                        help="回测结束日期 (YYYY-MM-DD)")
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "ratio", "atr_stop", "atr_trailing", "rsi"],
                        help="选择要运行的测试")
    parser.add_argument("--workers", type=int, default=4,
                        help="并行工作进程数")

    args = parser.parse_args()

    codes = args.codes.split(",")
    logger.info(f"测试参数: codes={codes}, start={args.start}, end={args.end}")

    # 创建报告目录
    Path("reports").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    runner = ParameterGridRunner(n_workers=args.workers)

    if args.test == "all":
        run_min_profit_loss_ratio_test(runner, codes, args.start, args.end)
        run_atr_stop_test(runner, codes, args.start, args.end)
        run_atr_trailing_test(runner, codes, args.start, args.end)
        run_rsi_threshold_test(runner, codes, args.start, args.end)
    elif args.test == "ratio":
        run_min_profit_loss_ratio_test(runner, codes, args.start, args.end)
    elif args.test == "atr_stop":
        run_atr_stop_test(runner, codes, args.start, args.end)
    elif args.test == "atr_trailing":
        run_atr_trailing_test(runner, codes, args.start, args.end)
    elif args.test == "rsi":
        run_rsi_threshold_test(runner, codes, args.start, args.end)

    logger.info("敏感性测试完成!")


if __name__ == "__main__":
    main()
