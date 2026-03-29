#!/usr/bin/env python3
"""完整回测套件执行器

执行所有回测测试并生成汇总报告。

使用方式:
    python3 scripts/backtest/run_full_backtest_suite.py --codes 600519,000001,600036 --start 2024-01-01 --end 2024-12-31
"""

import argparse
import logging
import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

from scripts.backtest.parameter_grid_runner import ParameterGridRunner, GridResult
from scripts.backtest.run_sensitivity_test import (
    run_min_profit_loss_ratio_test,
    run_atr_stop_test,
    run_atr_trailing_test,
    run_rsi_threshold_test,
)
from scripts.backtest.run_structure_stop_test import run_structure_stop_tests, generate_comparison_report as generate_structure_report
from scripts.backtest.run_multi_param_test import run_multi_param_tests
from scripts.backtest.run_structure_ab_test import run_ab_test, generate_ab_report
from scripts.backtest.run_cross_period_test import MARKET_PERIODS, run_period_test, generate_cross_period_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_full_suite(
    codes: list,
    start_date: str,
    end_date: str,
    output_dir: str = "reports",
) -> dict:
    """运行完整回测套件"""

    Path(output_dir).mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_summary = {
        "timestamp": timestamp,
        "codes": codes,
        "start_date": start_date,
        "end_date": end_date,
        "tests": {},
    }

    # 1. 参数敏感性测试
    logger.info("\n" + "="*70)
    logger.info("Step 1: 参数敏感性测试")
    logger.info("="*70)

    try:
        runner = ParameterGridRunner()

        logger.info("  - min_profit_loss_ratio 敏感性...")
        run_min_profit_loss_ratio_test(runner, codes, start_date, end_date)

        logger.info("  - ATR止损倍数敏感性...")
        run_atr_stop_test(runner, codes, start_date, end_date)

        logger.info("  - ATR追踪止损敏感性...")
        run_atr_trailing_test(runner, codes, start_date, end_date)

        logger.info("  - RSI阈值敏感性...")
        run_rsi_threshold_test(runner, codes, start_date, end_date)

        results_summary["tests"]["sensitivity"] = "completed"

    except Exception as e:
        logger.error(f"  参数敏感性测试失败: {e}")
        results_summary["tests"]["sensitivity"] = f"failed: {e}"

    # 2. 结构止损验证
    logger.info("\n" + "="*70)
    logger.info("Step 2: 结构止损验证")
    logger.info("="*70)

    try:
        structure_results = run_structure_stop_tests(codes, start_date, end_date)
        structure_report_path = f"{output_dir}/structure_stop_{timestamp}.html"
        generate_structure_report(structure_results, structure_report_path)
        results_summary["tests"]["structure_stop"] = "completed"
        results_summary["structure_report"] = structure_report_path

    except Exception as e:
        logger.error(f"  结构止损验证失败: {e}")
        results_summary["tests"]["structure_stop"] = f"failed: {e}"

    # 3. 多参数组合热力图
    logger.info("\n" + "="*70)
    logger.info("Step 3: 多参数组合热力图")
    logger.info("="*70)

    try:
        run_multi_param_tests(codes, start_date, end_date, output_dir)
        results_summary["tests"]["multi_param"] = "completed"

    except Exception as e:
        logger.error(f"  多参数组合测试失败: {e}")
        results_summary["tests"]["multi_param"] = f"failed: {e}"

    # 4. 结构止损A/B对比
    logger.info("\n" + "="*70)
    logger.info("Step 4: 结构止损A/B对比")
    logger.info("="*70)

    try:
        ab_results = run_ab_test(codes, start_date, end_date)
        ab_report_path = f"{output_dir}/structure_ab_test_{timestamp}.html"
        generate_ab_report(ab_results, ab_report_path)
        results_summary["tests"]["ab_test"] = "completed"
        results_summary["ab_report"] = ab_report_path

    except Exception as e:
        logger.error(f"  A/B对比测试失败: {e}")
        results_summary["tests"]["ab_test"] = f"failed: {e}"

    # 5. 跨时间段验证
    logger.info("\n" + "="*70)
    logger.info("Step 5: 跨时间段稳健性验证")
    logger.info("="*70)

    try:
        cross_period_results = []
        for period in MARKET_PERIODS:
            result = run_period_test(period, codes)
            cross_period_results.append(result)

        cross_report_path = f"{output_dir}/cross_period_{timestamp}.html"
        generate_cross_period_report(cross_period_results, cross_report_path)
        results_summary["tests"]["cross_period"] = "completed"
        results_summary["cross_period_report"] = cross_report_path

    except Exception as e:
        logger.error(f"  跨时间段验证失败: {e}")
        results_summary["tests"]["cross_period"] = f"failed: {e}"

    # 保存汇总结果
    summary_path = f"{output_dir}/backtest_summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\n汇总结果已保存: {summary_path}")

    return results_summary


def generate_master_report(summary: dict, output_path: str) -> None:
    """生成汇总报告主页"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 收集所有报告链接
    reports = []
    if "structure_report" in summary:
        reports.append({
            "name": "结构止损验证报告",
            "path": summary["structure_report"],
            "icon": "🛡️",
        })
    if "ab_report" in summary:
        reports.append({
            "name": "结构止损A/B对比报告",
            "path": summary["ab_report"],
            "icon": "📊",
        })
    if "cross_period_report" in summary:
        reports.append({
            "name": "跨时间段稳健性报告",
            "path": summary["cross_period_report"],
            "icon": "📈",
        })

    # 添加参数扫描报告
    scan_reports = list(Path("reports").glob("sensitivity_*.html"))
    for r in scan_reports:
        reports.append({
            "name": f"敏感性测试: {r.stem}",
            "path": str(r),
            "icon": "🔬",
        })

    heatmap_reports = list(Path("reports").glob("heatmap_*.html"))
    for r in heatmap_reports:
        reports.append({
            "name": f"热力图: {r.stem}",
            "path": str(r),
            "icon": "🔥",
        })

    reports_html = ""
    for r in reports:
        reports_html += f"""                <li>
                    <a href="{r['path']}">{r['icon']} {r['name']}</a>
                </li>
"""

    tests_status = "<br>".join([
        f"  {'✅' if v == 'completed' else '❌'} {k}: {v}"
        for k, v in summary.get("tests", {}).items()
    ])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>回测报告汇总</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 40px 20px; }}
        .container {{ max-width: 1000px; margin: auto; }}
        .card {{ background: white; border-radius: 16px; padding: 40px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); margin-bottom: 30px; }}
        h1 {{ color: #1a1a2e; margin-bottom: 10px; font-size: 32px; }}
        .subtitle {{ color: #666; margin-bottom: 30px; }}
        .meta {{ background: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 30px; }}
        .meta-item {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; }}
        .meta-item:last-child {{ border-bottom: none; }}
        .meta-label {{ color: #666; }}
        .meta-value {{ font-weight: 600; }}
        .status {{ margin-bottom: 30px; }}
        .status-item {{ padding: 10px; border-radius: 8px; margin-bottom: 8px; background: #f8f9fa; }}
        h2 {{ color: #1a1a2e; margin-bottom: 20px; font-size: 24px; }}
        ul {{ list-style: none; }}
        li {{ padding: 12px 0; border-bottom: 1px solid #eee; }}
        li:last-child {{ border-bottom: none; }}
        a {{ color: #3498db; text-decoration: none; font-size: 16px; }}
        a:hover {{ color: #2980b9; text-decoration: underline; }}
        .footer {{ text-align: center; color: #999; margin-top: 30px; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>波段交易回测报告汇总</h1>
            <p class="subtitle">大规模回测验证 - 参数敏感性分析与稳健性检验</p>

            <div class="meta">
                <div class="meta-item">
                    <span class="meta-label">执行时间</span>
                    <span class="meta-value">{timestamp}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">测试股票</span>
                    <span class="meta-value">{', '.join(summary.get('codes', []))}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">回测区间</span>
                    <span class="meta-value">{summary.get('start_date', 'N/A')} ~ {summary.get('end_date', 'N/A')}</span>
                </div>
            </div>

            <div class="status">
                <h2>测试执行状态</h2>
                <div class="status-item">✅ 参数敏感性测试: {summary.get('tests', {}).get('sensitivity', 'N/A')}</div>
                <div class="status-item">{'✅' if summary.get('tests', {}).get('structure_stop') == 'completed' else '❌'} 结构止损验证: {summary.get('tests', {}).get('structure_stop', 'N/A')}</div>
                <div class="status-item">{'✅' if summary.get('tests', {}).get('multi_param') == 'completed' else '❌'} 多参数组合热力图: {summary.get('tests', {}).get('multi_param', 'N/A')}</div>
                <div class="status-item">{'✅' if summary.get('tests', {}).get('ab_test') == 'completed' else '❌'} A/B对比测试: {summary.get('tests', {}).get('ab_test', 'N/A')}</div>
                <div class="status-item">{'✅' if summary.get('tests', {}).get('cross_period') == 'completed' else '❌'} 跨时间段验证: {summary.get('tests', {}).get('cross_period', 'N/A')}</div>
            </div>

            <h2>测试报告链接</h2>
            <ul>
{reports_html}
            </ul>
        </div>

        <div class="footer">
            波段交易量化系统 - 回测验证报告<br>
            生成时间: {timestamp}
        </div>
    </div>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"汇总报告已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="完整回测套件执行器")
    parser.add_argument("--codes", type=str,
                        default="600519,000001,600036,601318,600886,600900",
                        help="股票代码列表，逗号分隔")
    parser.add_argument("--start", type=str, default="2024-01-01",
                        help="回测开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31",
                        help="回测结束日期 (YYYY-MM-DD)")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式（仅运行关键测试）")
    parser.add_argument("--output", type=str, default="reports",
                        help="输出目录")

    args = parser.parse_args()
    codes = args.codes.split(",")

    logger.info("="*70)
    logger.info("波段交易回测套件")
    logger.info("="*70)
    logger.info(f"股票: {codes}")
    logger.info(f"区间: {args.start} ~ {args.end}")
    logger.info(f"模式: {'快速' if args.quick else '完整'}")
    logger.info("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.quick:
        # 快速模式：仅运行关键测试
        runner = ParameterGridRunner()

        # min_profit_loss_ratio 测试
        run_min_profit_loss_ratio_test(runner, codes, args.start, args.end)

        summary = {
            "timestamp": timestamp,
            "codes": codes,
            "start_date": args.start,
            "end_date": args.end,
            "tests": {"sensitivity": "completed"},
        }

    else:
        # 完整模式
        summary = run_full_suite(codes, args.start, args.end, args.output)

    # 生成汇总报告
    master_path = f"{args.output}/index_{timestamp}.html"
    generate_master_report(summary, master_path)

    logger.info("\n" + "="*70)
    logger.info("回测套件执行完成!")
    logger.info(f"汇总报告: {master_path}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
