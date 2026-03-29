#!/usr/bin/env python3
"""入场信号诊断脚本

分析为何入场信号未触发。

使用方式:
    python3 scripts/backtest/debug_entry_signals.py --code 600519 --start 2024-01-01 --end 2024-06-30
"""

import argparse
import sys
from datetime import datetime

sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

import pandas as pd
from src.data.loader import StockDataLoader
from src.data.indicators.signals import SwingSignals
from src.data.fetcher.price_converter import convert_to_forward_adj
from src.backtest.strategy_params import StrategyParams


def diagnose_entry_signals(code: str, start_date: str, end_date: str):
    """诊断入场信号"""

    print(f"\n{'='*70}")
    print(f"入场信号诊断: {code} ({start_date} ~ {end_date})")
    print(f"{'='*70}")

    # 加载数据
    loader = StockDataLoader(stockdata_root='/Users/bruce/workspace/trade/StockData')
    df = loader.load_daily(code, start_date, end_date)

    if df is None or df.empty:
        print(f"❌ 无法加载数据")
        return

    print(f"✓ 加载数据: {len(df)} 条")

    # 计算指标
    params = StrategyParams()
    signals = SwingSignals(params)
    df = convert_to_forward_adj(df)
    df = signals.calculate_all(df)

    print(f"\n数据概览:")
    print(f"  日期范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    print(f"  价格范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")

    # 逐日分析信号
    entry_signals_found = []
    atr_multiplier = 2.0

    for i in range(60, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]

        # 计算信号（使用完整历史数据）
        result = signals.detect_entry(df.iloc[:i+1])

        if result and result[0] in ("golden", "breakout"):
            entry_signal, confidence, reason = result
            entry_price = row["close"]
            atr = row.get("atr", 0)

            if pd.isna(atr) or atr <= 0:
                continue

            stop_loss = entry_price - (atr_multiplier * atr)

            # 计算盈亏比
            expected_profit_pct = 0.05
            expected_profit = entry_price * expected_profit_pct
            stop_distance = atr_multiplier * atr
            profit_loss_ratio = expected_profit / stop_distance if stop_distance > 0 else 0

            # ATR熔断检查
            entry_atr = atr
            current_atr = atr
            circuit_breaker_ok = True
            if not pd.isna(current_atr) and current_atr > entry_atr * 3.0:
                circuit_breaker_ok = False

            # 最小盈亏比检查
            ratio_ok = profit_loss_ratio >= params.min_profit_loss_ratio

            signal_info = {
                "date": row["date"],
                "close": entry_price,
                "atr": atr,
                "stop_loss": stop_loss,
                "profit_loss_ratio": profit_loss_ratio,
                "entry_signal": entry_signal,
                "confidence": confidence,
                "reason": reason,
                "ratio_ok": ratio_ok,
                "circuit_breaker_ok": circuit_breaker_ok,
                "rsi14": row.get("rsi14", 0),
                "ma20": row.get("ma20", 0),
                "ma60": row.get("ma60", 0),
            }
            entry_signals_found.append(signal_info)

    # 汇总统计
    print(f"\n信号统计:")
    print(f"  总信号数: {len(entry_signals_found)}")

    if entry_signals_found:
        print(f"\n信号详情:")
        for sig in entry_signals_found[:10]:
            ratio_status = "✓" if sig["ratio_ok"] else "✗"
            circuit_status = "✓" if sig["circuit_breaker_ok"] else "✗"
            print(f"\n  日期: {sig['date']}")
            print(f"    信号类型: {sig['entry_signal']}")
            print(f"    置信度: {sig['confidence']:.2f}")
            print(f"    入场价: {sig['close']:.2f}, ATR: {sig['atr']:.2f}")
            print(f"    止损价: {sig['stop_loss']:.2f}")
            print(f"    盈亏比: {sig['profit_loss_ratio']:.2f} (要求{ratio_status}: {sig['ratio_ok']})")
            print(f"    ATR熔断: {circuit_status} ({sig['circuit_breaker_ok']})")
            print(f"    原因: {sig['reason'][:80]}...")

    else:
        print(f"\n未找到入场信号，分析原因:")

        # 分析最近的数据
        recent = df.tail(30)
        print(f"\n  最近30日统计:")
        print(f"    RSI14: {recent['rsi14'].mean():.1f} (范围: {recent['rsi14'].min():.1f} ~ {recent['rsi14'].max():.1f})")
        print(f"    MA20: {recent['close'].iloc[-1]:.2f} vs MA20: {recent['ma20'].iloc[-1]:.2f}")
        print(f"    MA60: {recent['close'].iloc[-1]:.2f} vs MA60: {recent['ma60'].iloc[-1]:.2f}")

        # 检查MA趋势
        ma20_trend = "上涨" if recent['ma20'].iloc[-1] > recent['ma20'].iloc[0] else "下跌"
        ma60_trend = "上涨" if recent['ma60'].iloc[-1] > recent['ma60'].iloc[0] else "下跌"
        print(f"    MA20趋势: {ma20_trend}")
        print(f"    MA60趋势: {ma60_trend}")

        # 检查ATR是否过大（导致熔断）
        atr_values = recent['atr'].dropna()
        if len(atr_values) > 0:
            avg_atr = atr_values.mean()
            print(f"    平均ATR: {avg_atr:.2f}")
            print(f"    ATR/价格比: {avg_atr/recent['close'].iloc[-1]*100:.2f}%")

        # 检查RSI是否在合理范围
        rsi_values = recent['rsi14'].dropna()
        if len(rsi_values) > 0:
            if rsi_values.iloc[-1] > 60:
                print(f"    ⚠️ RSI({rsi_values.iloc[-1]:.1f}) > 60，可能已超买")
            elif rsi_values.iloc[-1] < 40:
                print(f"    ✓ RSI({rsi_values.iloc[-1]:.1f}) < 40，可能有入场机会")

    return entry_signals_found


def main():
    parser = argparse.ArgumentParser(description="入场信号诊断")
    parser.add_argument("--code", type=str, default="600519",
                        help="股票代码")
    parser.add_argument("--start", type=str, default="2024-01-01",
                        help="开始日期")
    parser.add_argument("--end", type=str, default="2024-06-30",
                        help="结束日期")

    args = parser.parse_args()

    diagnose_entry_signals(args.code, args.start, args.end)


if __name__ == "__main__":
    main()
