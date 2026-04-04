#!/usr/bin/env python3
"""
PATTERN_1 v1.1 扩展WFA验证 + RSI参数对比
验证RSI=55是否比RSI=58更稳健
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_data(max_stocks=800):
    data_dir = Path("/Users/bruce/workspace/trade/StockData/raw/daily")
    all_files = list(data_dir.glob("*.parquet"))
    if len(all_files) > max_stocks:
        np.random.seed(42)
        all_files = list(np.random.choice(all_files, max_stocks, replace=False))

    dfs = []
    for f in all_files:
        try:
            df = pd.read_parquet(f)
            df['code'] = f.stem
            dfs.append(df)
        except:
            continue

    data = pd.concat(dfs, ignore_index=True)
    data['date'] = pd.to_datetime(data['date'])
    return data.sort_values(['code', 'date']).reset_index(drop=True)

def compute_indicators(df):
    result = []
    for code, group in df.groupby('code'):
        g = group.sort_values('date').copy()
        if len(g) < 60:
            continue

        for w in [5, 10, 20, 60]:
            g[f'ma{w}'] = g['close'].rolling(w).mean()

        delta = g['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        g['rsi'] = 100 - (100 / (1 + rs))

        high_low = g['high'] - g['low']
        high_close = np.abs(g['high'] - g['close'].shift())
        low_close = np.abs(g['low'] - g['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        g['atr'] = tr.rolling(14).mean()
        g['atr_pct'] = g['atr'] / g['close'] * 100

        g['vol_ma5'] = g['volume'].rolling(5).mean()
        g['vol_ratio'] = g['volume'] / g['vol_ma5'].replace(0, np.nan)

        g['high_20d'] = g['high'].rolling(20).max().shift(1)
        g['trend_up'] = (g['ma5'] > g['ma20']) & (g['close'] > g['ma20'])
        g['above_ma20'] = (g['close'] > g['ma20']).astype(int)
        g['price_above_ma20'] = (g['close'] - g['ma20']) / g['ma20'] * 100

        result.append(g)

    return pd.concat(result, ignore_index=True)

def compute_market_indicators(df):
    daily = df.groupby('date').agg({'above_ma20': 'mean'}).reset_index()
    daily.columns = ['date', 'breadth']
    daily['breadth_ma3'] = daily['breadth'].rolling(3).mean()

    stock_ret = df.groupby('date').apply(
        lambda x: x['close'].pct_change().median(), include_groups=False
    ).reset_index()
    stock_ret.columns = ['date', 'market_ret']
    stock_ret['market_ret_ma5'] = stock_ret['market_ret'].rolling(5).mean()
    stock_ret['mom5'] = stock_ret['market_ret_ma5'] - stock_ret['market_ret_ma5'].shift(5)

    daily = daily.merge(stock_ret, on='date', how='left')
    return daily

def dynamic_rsi_upper(atr_pct):
    """原始v1.1动态RSI"""
    if atr_pct < 2.0:
        return 58
    elif atr_pct <= 3.0:
        return 57
    else:
        return 55

def backtest(signals_df, full_df, initial_capital=100000, max_positions=5,
             stop_loss=0.08, max_hold=20):
    date_to_data = {}
    for date, group in full_df.groupby('date'):
        date_to_data[date] = group.set_index('code').to_dict('index')

    date_to_signals = {}
    for date, group in signals_df.groupby('date'):
        date_to_signals[date] = group

    signal_dates = sorted(date_to_signals.keys())
    if not signal_dates:
        return [], [initial_capital], []

    capital = float(initial_capital)
    positions = {}
    trades = []
    equity = [initial_capital]
    equity_dates = [signal_dates[0]]

    for date in signal_dates:
        day_data = date_to_data.get(date, {})
        day_signals = date_to_signals.get(date, pd.DataFrame())

        for code in list(positions.keys()):
            pos = positions[code]
            if code not in day_data:
                continue

            current = day_data[code]
            hold_days = (date - pos['entry_date']).days
            pnl_pct = (current['close'] - pos['entry_price']) / pos['entry_price']

            should_exit = False
            exit_reason = None
            if pnl_pct < -stop_loss:
                should_exit = True
                exit_reason = 'stop_loss'
            elif hold_days >= max_hold:
                should_exit = True
                exit_reason = 'time_exit'

            if should_exit:
                sell_price = current['close'] * 0.999
                capital = capital + pos['shares'] * sell_price
                trades.append({
                    'code': code,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'hold_days': hold_days,
                    'pnl_pct': (sell_price - pos['entry_price']) / pos['entry_price'],
                    'exit_reason': exit_reason,
                })
                del positions[code]

        if len(positions) < max_positions:
            for _, row in day_signals.iterrows():
                code = row['code']
                if code in positions:
                    continue

                breadth = row.get('breadth_ma3', 0.5)
                if breadth > 0.50:
                    base_pos = 0.50
                elif breadth >= 0.35:
                    base_pos = 0.33
                else:
                    base_pos = 0.20

                rsi = row.get('rsi', 55)
                if rsi < 52:
                    adj = 1.2
                elif rsi < 55:
                    adj = 1.1
                elif rsi <= 58:
                    adj = 1.0
                else:
                    adj = 0.7

                entry_price = row['close'] * 1.001
                position_value = capital * base_pos * adj

                if position_value > 1000:
                    shares = int(position_value / entry_price)
                    if shares > 0:
                        positions[code] = {
                            'entry_date': date,
                            'entry_price': entry_price,
                            'shares': shares,
                        }
                        capital -= shares * entry_price

        pos_value = sum(pos['shares'] * day_data[code]['close']
                       for code, pos in positions.items() if code in day_data)
        equity.append(capital + pos_value)
        equity_dates.append(date)

    if positions and signal_dates:
        last_date = signal_dates[-1]
        last_data = date_to_data.get(last_date, {})
        for code, pos in positions.items():
            if code in last_data:
                sell_price = last_data[code]['close'] * 0.999
                capital = capital + pos['shares'] * sell_price
                trades.append({
                    'code': code,
                    'entry_date': pos['entry_date'],
                    'exit_date': last_date,
                    'hold_days': (last_date - pos['entry_date']).days,
                    'pnl_pct': (sell_price - pos['entry_price']) / pos['entry_price'],
                    'exit_reason': 'final_liquidation',
                })

    return trades, equity, equity_dates

def analyze_trades(trades, equity, equity_dates, initial_capital=100000):
    if not trades:
        return {}

    df = pd.DataFrame(trades)
    equity_df = pd.DataFrame({'date': equity_dates, 'equity': equity})
    equity_df['year'] = pd.to_datetime(equity_df['date']).dt.year

    years = sorted(equity_df['year'].unique())
    yearly = {}

    for i, year in enumerate(years):
        yeq = equity_df[equity_df['year'] == year]['equity']
        if len(yeq) > 1:
            start_equity = initial_capital if i == 0 else equity_df[equity_df['year'] == years[i-1]]['equity'].iloc[-1]
            end_equity = yeq.iloc[-1]
            y_return = (end_equity - start_equity) / start_equity
            y_rolling = yeq.expanding().max()
            y_dd = (yeq - y_rolling) / y_rolling
            y_max_dd = abs(y_dd.min()) if len(y_dd) > 0 else 0
        else:
            y_return = 0
            y_max_dd = 0

        ydf = df[pd.to_datetime(df['entry_date']).dt.year == year]
        yearly[year] = {
            'n_trades': len(ydf),
            'return': y_return,
            'max_drawdown': y_max_dd,
        }

    eq_series = pd.Series(equity)
    total_return = (eq_series.iloc[-1] - initial_capital) / initial_capital
    rolling_max = eq_series.expanding().max()
    drawdown = (eq_series - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min())

    return {
        'n_trades': len(df),
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'yearly': yearly,
    }

def run_wfa(signals_df, full_df, train_months=6, test_months=3, min_train_signals=5, min_test_signals=3):
    """运行扩展WFA"""
    wfa_results = []

    start_date = pd.Timestamp('2021-07-01')
    end_date = pd.Timestamp('2026-01-01')

    current_train_end = start_date

    while True:
        train_start = current_train_end - pd.DateOffset(months=train_months)
        test_start = current_train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end_date:
            break

        train_signals = signals_df[
            (signals_df['date'] >= train_start) &
            (signals_df['date'] < test_start)
        ]

        test_signals = signals_df[
            (signals_df['date'] >= test_start) &
            (signals_df['date'] < test_end)
        ]

        if len(train_signals) < min_train_signals or len(test_signals) < min_test_signals:
            current_train_end = test_end
            continue

        train_trades, train_equity, train_dates = backtest(train_signals, full_df)
        train_result = analyze_trades(train_trades, train_equity, train_dates)

        test_trades, test_equity, test_dates = backtest(test_signals, full_df)
        test_result = analyze_trades(test_trades, test_equity, test_dates)

        if test_result.get('n_trades', 0) == 0:
            current_train_end = test_end
            continue

        wfa_results.append({
            'train_period': f"{train_start.strftime('%Y-%m')} ~ {(test_start-pd.DateOffset(days=1)).strftime('%Y-%m')}",
            'test_period': f"{test_start.strftime('%Y-%m')} ~ {(test_end-pd.DateOffset(days=1)).strftime('%Y-%m')}",
            'train_trades': train_result.get('n_trades', 0),
            'train_return': train_result.get('total_return', 0) * 100,
            'train_dd': train_result.get('max_drawdown', 0) * 100,
            'test_trades': test_result.get('n_trades', 0),
            'test_return': test_result.get('total_return', 0) * 100,
            'test_dd': test_result.get('max_drawdown', 0) * 100,
            'yearly': {str(k): v for k, v in test_result.get('yearly', {}).items()},
        })

        current_train_end = test_end

    return wfa_results

# ==================== 主程序 ====================

print("=" * 80)
print("PATTERN_1 扩展WFA验证 + RSI参数对比")
print("=" * 80)

# 加载数据（扩大样本）
print("\n[1] 加载数据...")
df = load_data(800)
df = compute_indicators(df)
market = compute_market_indicators(df)
df = df.merge(market[['date', 'breadth', 'breadth_ma3', 'market_ret', 'market_ret_ma5', 'mom5']], on='date', how='left')
df = df[df['date'] >= '2021-01-01'].copy()

print(f"股票数: {df['code'].nunique()}")
print(f"数据天数: {df['date'].nunique()}")

# ==================== v1.1 信号（RSI=58动态上限）====================
print("\n[2] 构建v1.1信号 (RSI动态上限)...")

df['rsi_upper'] = df.apply(lambda r: dynamic_rsi_upper(r['atr_pct']), axis=1)

mask_v11 = (
    (df['close'] > df['high_20d']) &
    (df['trend_up']) &
    (df['rsi'] >= 50) & (df['rsi'] <= df['rsi_upper']) &
    (df['vol_ratio'] >= 0.8) & (df['vol_ratio'] <= 2.0) &
    (df['atr_pct'] < 3) &
    (df['breadth_ma3'] > 0.50) &
    (df['price_above_ma20'] >= 2)
)

signals_v11 = df[mask_v11].copy()
print(f"v1.1信号数: {len(signals_v11)}")

# ==================== 固定RSI=58信号 ====================
print("\n[3] 构建固定RSI=58信号...")

mask_rsi58 = (
    (df['close'] > df['high_20d']) &
    (df['trend_up']) &
    (df['rsi'] >= 50) & (df['rsi'] <= 58) &
    (df['vol_ratio'] >= 0.8) & (df['vol_ratio'] <= 2.0) &
    (df['atr_pct'] < 3) &
    (df['breadth_ma3'] > 0.50) &
    (df['price_above_ma20'] >= 2)
)

signals_rsi58 = df[mask_rsi58].copy()
print(f"RSI=58信号数: {len(signals_rsi58)}")

# ==================== 固定RSI=55信号 ====================
print("\n[4] 构建固定RSI=55信号...")

mask_rsi55 = (
    (df['close'] > df['high_20d']) &
    (df['trend_up']) &
    (df['rsi'] >= 50) & (df['rsi'] <= 55) &
    (df['vol_ratio'] >= 0.8) & (df['vol_ratio'] <= 2.0) &
    (df['atr_pct'] < 3) &
    (df['breadth_ma3'] > 0.50) &
    (df['price_above_ma20'] >= 2)
)

signals_rsi55 = df[mask_rsi55].copy()
print(f"RSI=55信号数: {len(signals_rsi55)}")

# ==================== 运行WFA ====================
print("\n[5] 运行扩展WFA...")
print("训练窗口: 6个月 | 测试窗口: 3个月 | 步长: 1个月")
print("最低信号要求: 训练5/测试3")

# v1.1 WFA
wfa_v11 = run_wfa(signals_v11, df, min_train_signals=5, min_test_signals=3)

# RSI=58 WFA
wfa_rsi58 = run_wfa(signals_rsi58, df, min_train_signals=5, min_test_signals=3)

# RSI=55 WFA
wfa_rsi55 = run_wfa(signals_rsi55, df, min_train_signals=5, min_test_signals=3)

# ==================== 输出结果 ====================

def print_wfa_summary(name, wfa_results):
    if not wfa_results:
        print(f"\n{name}: 无有效WFA窗口")
        return None

    test_returns = [r['test_return'] for r in wfa_results]
    train_returns = [r['train_return'] for r in wfa_results]
    positive_count = sum(1 for r in wfa_results if r['test_return'] > 0)

    print(f"\n{'='*80}")
    print(f"{name} WFA结果")
    print(f"{'='*80}")

    print(f"\n{'训练期':<18} {'测试期':<18} {'训练交易':>8} {'训练收益':>10} {'测试交易':>8} {'测试收益':>10}")
    print("-" * 80)

    for r in wfa_results:
        marker = "✓" if r['test_return'] > 0 else "✗"
        print(f"{r['train_period']:<18} {r['test_period']:<18} {r['train_trades']:>8} {r['train_return']:>+9.1f}% {r['test_trades']:>8} {r['test_return']:>+9.1f}% {marker}")

    print(f"\n统计:")
    print(f"  窗口数: {len(wfa_results)}")
    print(f"  训练平均收益: {np.mean(train_returns):+.1f}%")
    print(f"  测试平均收益: {np.mean(test_returns):+.1f}%")
    print(f"  正收益周期: {positive_count}/{len(wfa_results)} ({positive_count/len(wfa_results)*100:.0f}%)")

    decay = np.mean(test_returns) / np.mean(train_returns) * 100 if np.mean(train_returns) != 0 else float('inf')
    print(f"  衰减率: {decay:.0f}%")

    return {
        'n_windows': len(wfa_results),
        'train_avg': np.mean(train_returns),
        'test_avg': np.mean(test_returns),
        'positive_ratio': positive_count / len(wfa_results),
        'decay_rate': decay,
    }

v11_summary = print_wfa_summary("v1.1 (动态RSI)", wfa_v11)
rsi58_summary = print_wfa_summary("RSI=58 固定", wfa_rsi58)
rsi55_summary = print_wfa_summary("RSI=55 固定", wfa_rsi55)

# ==================== 对比 ====================

print(f"\n{'='*80}")
print("RSI参数对比")
print(f"{'='*80}")

print(f"\n{'参数':<20} {'WFA窗口':>8} {'测试平均收益':>12} {'正收益比例':>12} {'vs v1.1':>10}")
print("-" * 70)

if v11_summary:
    base = v11_summary['test_avg']
else:
    base = 0

for name, summary in [("v1.1 (动态RSI)", v11_summary), ("RSI=58 固定", rsi58_summary), ("RSI=55 固定", rsi55_summary)]:
    if summary:
        vs_base = summary['test_avg'] - base if name != "v1.1 (动态RSI)" else 0
        print(f"{name:<20} {summary['n_windows']:>8} {summary['test_avg']:>+11.1f}% {summary['positive_ratio']*100:>11.0f}% {vs_base:>+9.1f}%")

# ==================== 结论 ====================

print(f"\n{'='*80}")
print("WFA验证结论")
print(f"{'='*80}")

best = None
best_return = -float('inf')

for name, summary in [("v1.1 (动态RSI)", v11_summary), ("RSI=58 固定", rsi58_summary), ("RSI=55 固定", rsi55_summary)]:
    if summary and summary['test_avg'] > best_return:
        best_return = summary['test_avg']
        best = name

if best:
    print(f"\n最佳参数: {best} (测试期平均收益 {best_return:+.1f}%)")

# 判断RSI=55是否显著优于v1.1
if rsi55_summary and v11_summary:
    diff = rsi55_summary['test_avg'] - v11_summary['test_avg']
    if diff > 5:
        print(f"\nRSI=55比v1.1好 {diff:+.1f}%，建议更新参数")
    elif diff < -5:
        print(f"\nRSI=55比v1.1差 {diff:.1f}%，保持v1.1")
    else:
        print(f"\n差异不显著({diff:+.1f}%)，保持v1.1")

# 保存结果
report = {
    'date': '2026-04-04',
    'strategy': 'PATTERN_1 参数对比',
    'v11_wfa': {
        'windows': wfa_v11,
        'summary': v11_summary,
    },
    'rsi58_wfa': {
        'windows': wfa_rsi58,
        'summary': rsi58_summary,
    },
    'rsi55_wfa': {
        'windows': wfa_rsi55,
        'summary': rsi55_summary,
    },
}

output_path = Path('/Users/bruce/workspace/trade/SwingTrade/reports/t6_wfa_extended_results.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f"\n报告已保存: {output_path}")