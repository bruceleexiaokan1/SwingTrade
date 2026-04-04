#!/usr/bin/env python3
"""
PATTERN_1 v1.1 优化验证
测试候选优化方向：
1. E7: 2% → 3% (Sharpe可能+39%)
2. RSI: 动态 → 固定55 (收益可能更高)
3. 组合: RSI=55 + E7=3%
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

def compute_metrics(trades, equity, equity_dates, initial_capital=100000):
    if not trades or len(equity) < 2:
        return None

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

    equity_series = pd.Series(equity)
    total_return = (equity_series.iloc[-1] - initial_capital) / initial_capital

    returns = equity_series.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0

    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min())

    return {
        'n_trades': len(df),
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'yearly': yearly,
    }

def run_wfa(signals_df, full_df, train_months=6, test_months=3, min_train_signals=5, min_test_signals=3):
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
        test_trades, test_equity, test_dates = backtest(test_signals, full_df)

        train_result = compute_metrics(train_trades, train_equity, train_dates)
        test_result = compute_metrics(test_trades, test_equity, test_dates)

        if not test_result or test_result['n_trades'] == 0:
            current_train_end = test_end
            continue

        wfa_results.append({
            'train_period': f"{train_start.strftime('%Y-%m')} ~ {(test_start-pd.DateOffset(days=1)).strftime('%Y-%m')}",
            'test_period': f"{test_start.strftime('%Y-%m')} ~ {(test_end-pd.DateOffset(days=1)).strftime('%Y-%m')}",
            'train_trades': train_result.get('n_trades', 0) if train_result else 0,
            'train_return': train_result.get('total_return', 0) * 100 if train_result else 0,
            'train_sharpe': train_result.get('sharpe', 0) if train_result else 0,
            'test_trades': test_result.get('n_trades', 0),
            'test_return': test_result.get('total_return', 0) * 100,
            'test_sharpe': test_result.get('sharpe', 0),
            'test_dd': test_result.get('max_drawdown', 0) * 100,
            'yearly': {str(k): v for k, v in test_result.get('yearly', {}).items()},
        })

        current_train_end = test_end

    return wfa_results

# ==================== 主程序 ====================

print("=" * 80)
print("PATTERN_1 v1.1 优化方向验证")
print("=" * 80)

# 加载数据
print("\n[1] 加载数据...")
df = load_data(800)
df = compute_indicators(df)
market = compute_market_indicators(df)
df = df.merge(market[['date', 'breadth', 'breadth_ma3', 'market_ret', 'market_ret_ma5', 'mom5']], on='date', how='left')
df = df[df['date'] >= '2021-01-01'].copy()

print(f"股票数: {df['code'].nunique()}")

# 构建不同版本的信号
print("\n[2] 构建信号...")

# v1.1: 动态RSI + E7=2%
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

# Option A: E7=3%
mask_e7_3 = (
    (df['close'] > df['high_20d']) &
    (df['trend_up']) &
    (df['rsi'] >= 50) & (df['rsi'] <= df['rsi_upper']) &
    (df['vol_ratio'] >= 0.8) & (df['vol_ratio'] <= 2.0) &
    (df['atr_pct'] < 3) &
    (df['breadth_ma3'] > 0.50) &
    (df['price_above_ma20'] >= 3)
)
signals_e7_3 = df[mask_e7_3].copy()

# Option B: RSI固定55
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

# Option C: RSI固定55 + E7=3%
mask_combo = (
    (df['close'] > df['high_20d']) &
    (df['trend_up']) &
    (df['rsi'] >= 50) & (df['rsi'] <= 55) &
    (df['vol_ratio'] >= 0.8) & (df['vol_ratio'] <= 2.0) &
    (df['atr_pct'] < 3) &
    (df['breadth_ma3'] > 0.50) &
    (df['price_above_ma20'] >= 3)
)
signals_combo = df[mask_combo].copy()

print(f"v1.1 (动态RSI + E7=2%): {len(signals_v11)} 信号")
print(f"Option A (动态RSI + E7=3%): {len(signals_e7_3)} 信号")
print(f"Option B (RSI=55 + E7=2%): {len(signals_rsi55)} 信号")
print(f"Option C (RSI=55 + E7=3%): {len(signals_combo)} 信号")

# ==================== 全量回测对比 ====================

print("\n[3] 全量回测对比...")

def full_backtest(name, signals):
    trades, equity, equity_dates = backtest(signals, df)
    result = compute_metrics(trades, equity, equity_dates)
    return result, trades, equity

v11_result, v11_trades, v11_equity = full_backtest("v1.1", signals_v11)
e7_3_result, e7_3_trades, e7_3_equity = full_backtest("E7=3%", signals_e7_3)
rsi55_result, rsi55_trades, rsi55_equity = full_backtest("RSI=55", signals_rsi55)
combo_result, combo_trades, combo_equity = full_backtest("Combo", signals_combo)

print("\n全量回测统计:")
print(f"{'版本':<25} {'信号数':>8} {'交易数':>8} {'总收益':>12} {'Sharpe':>8} {'最大DD':>10}")
print("-" * 75)

for name, signals, result in [
    ("v1.1 (动态RSI + E7=2%)", signals_v11, v11_result),
    ("Option A (动态RSI + E7=3%)", signals_e7_3, e7_3_result),
    ("Option B (RSI=55 + E7=2%)", signals_rsi55, rsi55_result),
    ("Option C (RSI=55 + E7=3%)", signals_combo, combo_result),
]:
    if result:
        print(f"{name:<25} {len(signals):>8} {result['n_trades']:>8} {result['total_return']*100:>+11.1f}% {result['sharpe']:>8.2f} {result['max_drawdown']*100:>9.1f}%")

# ==================== WFA验证 ====================

print("\n[4] WFA验证 (6个月训练 + 3个月测试)...")

wfa_v11 = run_wfa(signals_v11, df)
wfa_e7_3 = run_wfa(signals_e7_3, df)
wfa_rsi55 = run_wfa(signals_rsi55, df)
wfa_combo = run_wfa(signals_combo, df)

def wfa_summary(name, wfa_results):
    if not wfa_results:
        return None
    test_returns = [r['test_return'] for r in wfa_results]
    test_sharpes = [r['test_sharpe'] for r in wfa_results]
    positive = sum(1 for r in wfa_results if r['test_return'] > 0)
    return {
        'n_windows': len(wfa_results),
        'test_avg_return': np.mean(test_returns),
        'test_avg_sharpe': np.mean(test_sharpes),
        'positive_ratio': positive / len(wfa_results),
        'windows': wfa_results,
    }

v11_wfa = wfa_summary("v1.1", wfa_v11)
e7_3_wfa = wfa_summary("E7=3%", wfa_e7_3)
rsi55_wfa = wfa_summary("RSI=55", wfa_rsi55)
combo_wfa = wfa_summary("Combo", wfa_combo)

print("\nWFA统计:")
print(f"{'版本':<25} {'窗口数':>8} {'测试收益':>12} {'测试Sharpe':>12} {'正收益':>10}")
print("-" * 70)

for name, wfa in [
    ("v1.1 (动态RSI + E7=2%)", v11_wfa),
    ("Option A (动态RSI + E7=3%)", e7_3_wfa),
    ("Option B (RSI=55 + E7=2%)", rsi55_wfa),
    ("Option C (RSI=55 + E7=3%)", combo_wfa),
]:
    if wfa:
        print(f"{name:<25} {wfa['n_windows']:>8} {wfa['test_avg_return']:>+11.1f}% {wfa['test_avg_sharpe']:>11.2f} {wfa['positive_ratio']*100:>9.0f}%")

# ==================== 详细WFA对比 ====================

print("\n" + "=" * 80)
print("详细WFA对比 (按版本)")
print("=" * 80)

def print_wfa_detail(name, wfa_results):
    if not wfa_results:
        print(f"\n{name}: 无有效窗口")
        return
    print(f"\n{name}:")
    print(f"{'测试期':<18} {'收益':>10} {'Sharpe':>8} {'DD':>8}")
    print("-" * 50)
    for r in wfa_results:
        marker = "✓" if r['test_return'] > 0 else "✗"
        print(f"{r['test_period']:<18} {r['test_return']:>+9.1f}% {r['test_sharpe']:>8.2f} {r['test_dd']:>7.1f}% {marker}")

print_wfa_detail("v1.1 (动态RSI + E7=2%)", wfa_v11)
print_wfa_detail("Option A (动态RSI + E7=3%)", wfa_e7_3)
print_wfa_detail("Option B (RSI=55 + E7=2%)", wfa_rsi55)
print_wfa_detail("Option C (RSI=55 + E7=3%)", wfa_combo)

# ==================== 综合判定 ====================

print("\n" + "=" * 80)
print("综合判定")
print("=" * 80)

print("\n候选优化方向评估:")

candidates = []

# Option A vs v1.1
if e7_3_wfa and v11_wfa:
    ret_diff = e7_3_wfa['test_avg_return'] - v11_wfa['test_avg_return']
    sharpe_diff = e7_3_wfa['test_avg_sharpe'] - v11_wfa['test_avg_sharpe']
    pos_diff = e7_3_wfa['positive_ratio'] - v11_wfa['positive_ratio']
    candidates.append({
        'name': 'Option A: E7=3%',
        'ret_diff': ret_diff,
        'sharpe_diff': sharpe_diff,
        'pos_diff': pos_diff,
        'e7_3_wfa': e7_3_wfa,
        'v11_wfa': v11_wfa,
    })

# Option B vs v1.1
if rsi55_wfa and v11_wfa:
    ret_diff = rsi55_wfa['test_avg_return'] - v11_wfa['test_avg_return']
    sharpe_diff = rsi55_wfa['test_avg_sharpe'] - v11_wfa['test_avg_sharpe']
    pos_diff = rsi55_wfa['positive_ratio'] - v11_wfa['positive_ratio']
    candidates.append({
        'name': 'Option B: RSI=55',
        'ret_diff': ret_diff,
        'sharpe_diff': sharpe_diff,
        'pos_diff': pos_diff,
        'rsi55_wfa': rsi55_wfa,
        'v11_wfa': v11_wfa,
    })

# Option C vs v1.1
if combo_wfa and v11_wfa:
    ret_diff = combo_wfa['test_avg_return'] - v11_wfa['test_avg_return']
    sharpe_diff = combo_wfa['test_avg_sharpe'] - v11_wfa['test_avg_sharpe']
    pos_diff = combo_wfa['positive_ratio'] - v11_wfa['positive_ratio']
    candidates.append({
        'name': 'Option C: RSI=55 + E7=3%',
        'ret_diff': ret_diff,
        'sharpe_diff': sharpe_diff,
        'pos_diff': pos_diff,
        'combo_wfa': combo_wfa,
        'v11_wfa': v11_wfa,
    })

for c in candidates:
    print(f"\n{c['name']} vs v1.1:")
    print(f"  收益差异: {c['ret_diff']:+.1f}%")
    print(f"  Sharpe差异: {c['sharpe_diff']:+.2f}")
    print(f"  正收益比例差异: {c['pos_diff']*100:+.0f}%")
    print(f"  v1.1 窗口: {c['v11_wfa']['n_windows']}, {c['name'].split(':')[0]} 窗口: {c.get('e7_3_wfa', c.get('rsi55_wfa', c.get('combo_wfa')))['n_windows']}")

# 选择最佳候选
best = None
best_score = -float('inf')

for c in candidates:
    # 综合评分：收益差异 * 0.4 + Sharpe差异 * 0.4 + 正收益比例 * 0.2
    score = c['ret_diff'] * 0.4 + c['sharpe_diff'] * 10 * 0.4 + c['pos_diff'] * 100 * 0.2
    c['score'] = score
    if score > best_score:
        best_score = score
        best = c

print("\n" + "=" * 80)
print("最终推荐")
print("=" * 80)

if best and best['score'] > 0:
    print(f"\n✅ 推荐优化: {best['name']}")
    print(f"   综合评分: {best['score']:+.2f}")
    print(f"   收益提升: {best['ret_diff']:+.1f}%")
    print(f"   Sharpe提升: {best['sharpe_diff']:+.2f}")
    print(f"   正收益比例: {best['pos_diff']*100:+.0f}%")
else:
    print("\n❌ 保持v1.1，无显著优化方向")

# 保存结果
report = {
    'date': '2026-04-04',
    'strategy': 'PATTERN_1 优化验证',
    'v11': {
        'signals': len(signals_v11),
        'n_trades': v11_result['n_trades'] if v11_result else 0,
        'total_return': v11_result['total_return'] if v11_result else 0,
        'sharpe': v11_result['sharpe'] if v11_result else 0,
        'wfa': v11_wfa,
    },
    'option_a_e7_3': {
        'signals': len(signals_e7_3),
        'n_trades': e7_3_result['n_trades'] if e7_3_result else 0,
        'total_return': e7_3_result['total_return'] if e7_3_result else 0,
        'sharpe': e7_3_result['sharpe'] if e7_3_result else 0,
        'wfa': e7_3_wfa,
    },
    'option_b_rsi55': {
        'signals': len(signals_rsi55),
        'n_trades': rsi55_result['n_trades'] if rsi55_result else 0,
        'total_return': rsi55_result['total_return'] if rsi55_result else 0,
        'sharpe': rsi55_result['sharpe'] if rsi55_result else 0,
        'wfa': rsi55_wfa,
    },
    'option_c_combo': {
        'signals': len(signals_combo),
        'n_trades': combo_result['n_trades'] if combo_result else 0,
        'total_return': combo_result['total_return'] if combo_result else 0,
        'sharpe': combo_result['sharpe'] if combo_result else 0,
        'wfa': combo_wfa,
    },
    'candidates': [
        {
            'name': c['name'],
            'score': c.get('score', 0),
            'ret_diff': c['ret_diff'],
            'sharpe_diff': c['sharpe_diff'],
            'pos_diff': c['pos_diff'],
        }
        for c in candidates
    ],
    'recommendation': best['name'] if best and best['score'] > 0 else 'keep_v1.1',
}

output_path = Path('/Users/bruce/workspace/trade/SwingTrade/reports/t6_optimization_validation.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f"\n报告已保存: {output_path}")