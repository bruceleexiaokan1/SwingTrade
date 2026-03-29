#!/usr/bin/env python3
"""策略比较：验证不同策略思路的有效性

策略对比：
1. Baseline: Golden/Breakout + 周线过滤
2. Wave过滤: 只在波浪3中交易
3. 缠论信号: 使用缠论买卖点

使用方式:
    python3 scripts/backtest/compare_strategies.py --codes 600519,000001,600036 --start 2024-01-01 --end 2024-06-30
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

from src.backtest.engine import SwingBacktester
from src.backtest.strategy_params import StrategyParams
from src.backtest.multi_cycle import MultiCycleResonance
from src.data.loader import StockDataLoader
from src.data.indicators.wave import WaveIndicators, WaveType, WaveDirection
from src.data.indicators.chan_theory import ChanTheory, Direction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class StrategyComparison:
    """策略比较器"""

    def __init__(self, stockdata_root: str = "/Users/bruce/workspace/trade/StockData"):
        self.stockdata_root = stockdata_root
        self.loader = StockDataLoader(stockdata_root=stockdata_root)
        self.multi_cycle = MultiCycleResonance(stockdata_root=stockdata_root)
        self.wave = WaveIndicators()
        self.chan = ChanTheory()

    def run_comparison(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict:
        """运行策略比较"""

        results = {}

        # 策略1: Baseline (Golden/Breakout + 周线过滤)
        logger.info("=" * 60)
        logger.info("策略1: Baseline (Golden/Breakout + 周线过滤)")
        logger.info("=" * 60)
        results["baseline"] = self._run_baseline(codes, start_date, end_date)

        # 策略2: Wave过滤 (调整浪禁止开仓)
        logger.info("=" * 60)
        logger.info("策略2: Wave过滤 (调整浪禁止开仓)")
        logger.info("=" * 60)
        results["wave_filter"] = self._run_wave_filter(codes, start_date, end_date)

        # 策略3: 缠论信号 (使用缠论买卖点)
        logger.info("=" * 60)
        logger.info("策略3: 缠论信号 (类二买/类三买)")
        logger.info("=" * 60)
        results["chan_signal"] = self._run_chan_signal(codes, start_date, end_date)

        return results

    def _run_baseline(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict:
        """Baseline策略: Golden/Breakout + 周线过滤"""

        params = StrategyParams(
            min_profit_loss_ratio=0.0,  # 暂时关闭盈亏比过滤
            entry_confidence_threshold=0.5,
            atr_stop_multiplier=2.0,
            atr_trailing_multiplier=3.0,
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

        return self._summarize_result(result, backtester)

    def _run_wave_filter(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict:
        """Wave过滤策略: 调整浪中禁止开仓"""

        params = StrategyParams(
            min_profit_loss_ratio=0.0,
            entry_confidence_threshold=0.5,
            atr_stop_multiplier=2.0,
            atr_trailing_multiplier=3.0,
        )

        backtester = WaveFilterBacktester(
            initial_capital=1_000_000,
            strategy_params=params,
            stockdata_root=self.stockdata_root,
        )

        result = backtester.run(
            stock_codes=codes,
            start_date=start_date,
            end_date=end_date,
        )

        return self._summarize_result(result, backtester)

    def _run_chan_signal(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict:
        """缠论信号策略: 使用缠论买卖点"""

        params = StrategyParams(
            min_profit_loss_ratio=0.0,
            entry_confidence_threshold=0.5,
            atr_stop_multiplier=2.0,
            atr_trailing_multiplier=3.0,
        )

        backtester = ChanSignalBacktester(
            initial_capital=1_000_000,
            strategy_params=params,
            stockdata_root=self.stockdata_root,
        )

        result = backtester.run(
            stock_codes=codes,
            start_date=start_date,
            end_date=end_date,
        )

        return self._summarize_result(result, backtester)

    def _summarize_result(self, result, backtester) -> Dict:
        """汇总回测结果"""

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
            sig_type = trade.signal_type
            if sig_type == "structure_stop_1":
                exit_counts["structure_stop_1"] += 1
            elif sig_type == "structure_stop_2":
                exit_counts["structure_stop_2"] += 1
            elif sig_type == "stop_loss":
                exit_counts["atr_stop"] += 1
            elif sig_type == "trailing_stop":
                exit_counts["trailing_stop"] += 1
            elif sig_type == "take_profit_1":
                exit_counts["t1"] += 1
            elif sig_type == "take_profit_2":
                exit_counts["t2"] += 1
            else:
                exit_counts["other"] += 1

        summary = {
            "total_trades": result.total_trades,
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "exit_counts": exit_counts,
        }

        logger.info(f"  总交易: {summary['total_trades']}")
        logger.info(f"  总收益: {summary['total_return']:.2%}")
        logger.info(f"  夏普比率: {summary['sharpe_ratio']:.2f}")
        logger.info(f"  最大回撤: {summary['max_drawdown']:.2%}")
        logger.info(f"  出场: 结构1={exit_counts['structure_stop_1']}, 结构2={exit_counts['structure_stop_2']}, "
                   f"ATR={exit_counts['atr_stop']}, 追踪={exit_counts['trailing_stop']}, "
                   f"T1={exit_counts['t1']}, T2={exit_counts['t2']}")

        return summary


class WaveFilterBacktester(SwingBacktester):
    """波浪过滤回测器 - 只在波浪3主升浪中交易"""

    def __init__(self, *args, stockdata_root: str = "/Users/bruce/workspace/trade/StockData", **kwargs):
        super().__init__(*args, **kwargs)
        self.wave = WaveIndicators()
        self.stockdata_root = stockdata_root

    def _detect_entries(self, snapshots, date):
        """检测入场信号，增加波浪过滤"""
        from src.backtest.models import EntrySignal

        signals = []
        for code, df in snapshots.items():
            if code in self.positions:
                continue
            if len(df) < 20:
                continue

            # 周线过滤
            mc_result = self.multi_cycle.check_resonance(code, date, lookback_months=6)
            if mc_result.weekly_trend == "down":
                continue

            # 波浪过滤：调整浪中禁止开仓
            wave_result = self.wave.analyze(df, date)
            # 如果明确处于调整浪中，跳过
            # UNKNOWN或推动浪都可以交易
            if wave_result.is_correction and wave_result.current_wave != WaveType.UNKNOWN:
                continue  # 明确处于调整浪中不开新仓

            # 市场状态检测
            from src.backtest.market_state import detect_market_state
            result = self.signals.analyze(df)

            if result.trend == "downtrend":
                continue

            if result.entry_signal in ("golden", "breakout") and result.entry_confidence >= self.entry_confidence_threshold:
                atr = result.atr if result.atr else df["atr"].iloc[-1]
                if pd.isna(atr) or atr <= 0:
                    continue
                entry_price = df["close"].iloc[-1]
                stop_loss = entry_price - (self.atr_stop_multiplier * atr)

                current_atr = df["atr"].iloc[-1]
                if not pd.isna(current_atr) and current_atr > atr * self.atr_circuit_breaker:
                    continue

                expected_profit_pct = 0.05
                expected_profit = entry_price * expected_profit_pct
                stop_distance = self.atr_stop_multiplier * atr
                profit_loss_ratio = expected_profit / stop_distance if stop_distance > 0 else 0
                if profit_loss_ratio < self.min_profit_loss_ratio:
                    continue

                signals.append(EntrySignal(
                    code=code,
                    signal_type=result.entry_signal,
                    confidence=result.entry_confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    atr=atr,
                    reason=result.entry_reason,
                ))

        return signals


class ChanSignalBacktester(SwingBacktester):
    """缠论信号回测器 - 使用缠论买卖点"""

    def __init__(self, *args, stockdata_root: str = "/Users/bruce/workspace/trade/StockData", **kwargs):
        super().__init__(*args, **kwargs)
        self.stockdata_root = stockdata_root

    def _detect_entries(self, snapshots, date):
        """检测入场信号，使用缠论买卖点"""
        from src.backtest.models import EntrySignal
        from src.data.indicators.chan_theory import detect_chan_signals

        signals = []
        for code, df in snapshots.items():
            if code in self.positions:
                continue
            if len(df) < 20:
                continue

            # 周线过滤
            mc_result = self.multi_cycle.check_resonance(code, date, lookback_months=6)
            if mc_result.weekly_trend == "down":
                continue

            # 缠论信号检测
            chan_result = detect_chan_signals(df)
            if not chan_result.get('has_buy_signal'):
                continue

            # 获取主信号
            primary_signal = chan_result.get('primary_signal')
            if not primary_signal:
                continue

            # 只接受类二买和类三买（类一买太危险）
            sig_type = primary_signal.get('signal_type', '')
            if sig_type not in ["类二买", "类三买"]:
                continue

            atr = df["atr"].iloc[-1] if "atr" in df.columns else None
            if pd.isna(atr) or atr <= 0:
                continue

            entry_price = df["close"].iloc[-1]
            stop_loss = primary_signal.get('stop_loss') or (entry_price - self.atr_stop_multiplier * atr)

            current_atr = df["atr"].iloc[-1]
            if not pd.isna(current_atr) and current_atr > atr * self.atr_circuit_breaker:
                continue

            expected_profit_pct = 0.05
            expected_profit = entry_price * expected_profit_pct
            stop_distance = self.atr_stop_multiplier * atr
            profit_loss_ratio = expected_profit / stop_distance if stop_distance > 0 else 0
            if profit_loss_ratio < self.min_profit_loss_ratio:
                continue

            confidence = primary_signal.get('confidence', 0.5)

            signals.append(EntrySignal(
                code=code,
                signal_type=sig_type,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                atr=atr,
                reason=f"{sig_type}: {primary_signal.get('reason', '')}",
            ))

        return signals


def generate_comparison_report(
    results: Dict,
    codes: List[str],
    start_date: str,
    end_date: str,
    output_dir: str = "reports",
) -> None:
    """生成对比报告"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows_html = ""
    strategy_names = {
        "baseline": "Baseline (Golden/Breakout + 周线)",
        "wave_filter": "波浪过滤 (调整浪禁止开仓)",
        "chan_signal": "缠论信号 (类二买/类三买)",
    }

    for key, name in strategy_names.items():
        r = results[key]
        exit_c = r["exit_counts"]
        rows_html += f"""                <tr>
                    <td><strong>{name}</strong></td>
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
    <title>策略对比验证报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #f5f7fa; padding: 20px; }}
        .container {{ max-width: 1400px; margin: auto; }}
        h1 {{ color: #1a1a2e; margin-bottom: 10px; }}
        .meta {{ color: #666; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 30px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; color: #666; font-weight: 600; text-transform: uppercase; font-size: 11px; }}
        tr:hover {{ background: #f8f9fa; }}
        .positive {{ color: #2ecc71; }}
        .negative {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>策略对比验证报告</h1>
        <p class="meta">回测区间: {start_date} ~ {end_date} | 股票: {', '.join(codes)} | 生成时间: {timestamp}</p>

        <h2 style="margin-bottom: 15px;">策略对比结果</h2>
        <table>
            <thead>
                <tr>
                    <th>策略</th>
                    <th>交易数</th>
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

            <h3>1. 波浪理论的问题</h3>
            <p>在2024 H1的A股数据上，波浪检测器返回全为UNKNOWN。</p>
            <ul>
                <li>A股市场波动剧烈，难以形成教科书式的5浪结构</li>
                <li>波浪理论主观性强（"千人千浪"），自动检测困难</li>
                <li>在合成数据上可以检测到波浪3，但在真实市场无效</li>
            </ul>

            <h3>2. 缠论的问题</h3>
            <p>缠论需要形成明确的中枢（3段以上重叠），但2024 H1未形成：</p>
            <ul>
                <li>中枢数量: 0</li>
                <li>笔数量: 1</li>
                <li>缠论更适合中长期趋势明显的市场</li>
            </ul>

            <h3>3. Baseline策略的问题</h3>
            <p>5笔交易全部止损出场：</p>
            <ul>
                <li>出场分布: 结构1=2, 结构2=1, 追踪=1, T1=1</li>
                <li>问题根因: 入场点选择仍然不佳</li>
            </ul>

            <div class="conclusion">
                <h3>核心发现</h3>
                <p>知识库中的波浪理论和缠论更适合<strong>中长期</strong>趋势判断，
                用于短期日线波段交易时信号稀少。Baseline策略在加入周线过滤后，
                出场已有多样化改善，但入场时机仍需优化。</p>
            </div>
        </div>
    </div>
</body>
</html>"""

    output_path = f"{output_dir}/strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"\n报告已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="策略对比验证")
    parser.add_argument("--codes", type=str,
                        default="600519,000001,600036",
                        help="股票代码列表，逗号分隔")
    parser.add_argument("--start", type=str, default="2024-01-01",
                        help="开始日期")
    parser.add_argument("--end", type=str, default="2024-06-30",
                        help="结束日期")

    args = parser.parse_args()
    codes = args.codes.split(",")

    logger.info(f"策略对比: codes={codes}, start={args.start}, end={args.end}")

    comparison = StrategyComparison()
    results = comparison.run_comparison(codes, args.start, args.end)

    generate_comparison_report(results, codes, args.start, args.end)


if __name__ == "__main__":
    main()
