#!/usr/bin/env python3
"""
PATTERN_1 v1.1 E7优化深度验证
Option A: E7=3% 是否显著优于E7=2%
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_data(max_stocks=1000):
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

def statistical_test(wfa_v11, wfa_e7_3):
    """
    配对统计检验：v1.1 vs E7=3%
    只比较两者都有数据的窗口
    """
    # 匹配同期窗口
    v11_by_period = {r['test_period']: r for r in wfa_v11}
    e7_by_period = {r['test_period']: r for r in wfa_e7_3}

    common_periods = set(v11_by_period.keys()) & set(e7_by_period.keys())

    if len(common_periods) < 3:
        return None, None, None

    v11_returns = [v11_by_period[p]['test_return'] for p in common_periods]
    e7_returns = [e7_by_period[p]['test_return'] for p in common_periods]

    # 配对差值
    diffs = [e - v for v, e in zip(v11_returns, e7_returns)]

    # t检验
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    n = len(diffs)
    t_stat = mean_diff / (std_diff / np.sqrt(n)) if std_diff > 0 else 0

    # p值（双尾）
    from scipy import stats
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-1))

    return mean_diff, t_stat, p_value

# ==================== 主程序 ====================

print("=" * 80)
print("PATTERN_1 v1.1 E7优化深度验证")
print("E7=2% vs E7=3% 统计显著性检验")
print("=" * 80)

# 加载数据（扩大到1000股票）
print("\n[1] 加载数据...")
df = load_data(1000)
df = compute_indicators(df)
market = compute_market_indicators(df)
df = df.merge(market[['date', 'breadth', 'breadth_ma3', 'market_ret', 'market_ret_ma5', 'mom5']], on='date', how='left')
df = df[df['date'] >= '2021-01-01'].copy()

print(f"股票数: {df['code'].nunique()}")

# 构建信号
df['rsi_upper'] = df.apply(lambda r: dynamic_rsi_upper(r['atr_pct']), axis=1)

# v1.1: E7=2%
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

print(f"v1.1 (E7=2%): {len(signals_v11)} 信号")
print(f"Option A (E7=3%): {len(signals_e7_3)} 信号")

# WFA
print("\n[2] WFA验证...")
wfa_v11 = run_wfa(signals_v11, df)
wfa_e7_3 = run_wfa(signals_e7_3, df)

print(f"v1.1 WFA窗口: {len(wfa_v11)}")
print(f"E7=3% WFA窗口: {len(wfa_e7_3)}")

# ==================== 配对统计检验 ====================

print("\n[3] 配对统计检验 (仅比较同期窗口)...")

mean_diff, t_stat, p_value = statistical_test(wfa_v11, wfa_e7_3)

if mean_diff is not None:
    print(f"\n配对检验结果:")
    print(f"  平均收益差异: {mean_diff:+.2f}%")
    print(f"  t统计量: {t_stat:.2f}")
    print(f"  p值: {p_value:.4f}")

    if p_value < 0.05:
        print(f"  结论: 差异显著 (p<0.05)，E7=3% 显著优于 E7=2%")
    elif p_value < 0.10:
        print(f"  结论: 边缘显著 (p<0.10)，E7=3% 可能优于 E7=2%")
    else:
        print(f"  结论: 差异不显著 (p>=0.10)，无法确定E7=3%更优")
else:
    print("  窗口数不足，无法进行统计检验")

# ==================== 年度收益对比 ====================

print("\n" + "=" * 80)
print("年度收益对比")
print("=" * 80)

# v1.1年度
v11_yearly = {}
for r in wfa_v11:
    for year, stats in r['yearly'].items():
        if year not in v11_yearly:
            v11_yearly[year] = []
        v11_yearly[year].append(stats['return'] * 100)

# E7=3%年度
e7_yearly = {}
for r in wfa_e7_3:
    for year, stats in r['yearly'].items():
        if year not in e7_yearly:
            e7_yearly[year] = []
        e7_yearly[year].append(stats['return'] * 100)

all_years = sorted(set(v11_yearly.keys()) | set(e7_yearly.keys()))

print(f"\n{'年份':<6} {'v1.1平均':>12} {'E7=3%平均':>12} {'差异':>10} {'v1.1样本':>10} {'E7样本':>10}")
print("-" * 65)

for year in all_years:
    v11_avg = np.mean(v11_yearly.get(year, [0]))
    e7_avg = np.mean(e7_yearly.get(year, [0]))
    diff = e7_avg - v11_avg
    v11_n = len(v11_yearly.get(year, []))
    e7_n = len(e7_yearly.get(year, []))
    print(f"{year:<6} {v11_avg:>+11.1f}% {e7_avg:>+11.1f}% {diff:>+9.1f}% {v11_n:>10} {e7_n:>10}")

# ==================== 综合评估 ====================

print("\n" + "=" * 80)
print("综合评估")
print("=" * 80)

# 计算整体统计
v11_all_returns = [r['test_return'] for r in wfa_v11]
e7_all_returns = [r['test_return'] for r in wfa_e7_3]

v11_positive = sum(1 for r in v11_all_returns if r > 0) / len(v11_all_returns) * 100 if v11_all_returns else 0
e7_positive = sum(1 for r in e7_all_returns if r > 0) / len(e7_all_returns) * 100 if e7_all_returns else 0

print(f"\n{'指标':<25} {'v1.1 (E7=2%)':>15} {'E7=3%':>15} {'差异':>10}")
print("-" * 65)
print(f"{'WFA窗口数':<25} {len(wfa_v11):>15} {len(wfa_e7_3):>15}")
print(f"{'平均测试收益':<25} {np.mean(v11_all_returns):>+14.1f}% {np.mean(e7_all_returns):>+14.1f}% {(np.mean(e7_all_returns)-np.mean(v11_all_returns)):>+9.1f}%")
print(f"{'正收益周期比例':<25} {v11_positive:>14.1f}% {e7_positive:>14.1f}% {(e7_positive-v11_positive):>+9.0f}%")
print(f"{'最大回撤平均':<25} {np.mean([r['test_dd'] for r in wfa_v11]):>+14.1f}% {np.mean([r['test_dd'] for r in wfa_e7_3]):>+14.1f}%")

# 最终判定
print("\n" + "=" * 80)
print("最终判定")
print("=" * 80)

# 评分标准
score_e7_3 = 0
score_v11 = 0

# 1. 收益
if np.mean(e7_all_returns) > np.mean(v11_all_returns):
    score_e7_3 += 1
    print("✅ E7=3% 平均收益更高")
else:
    score_v11 += 1
    print("❌ E7=3% 平均收益不更高")

# 2. 正收益比例
if e7_positive > v11_positive:
    score_e7_3 += 1
    print("✅ E7=3% 正收益比例更高")
elif e7_positive == v11_positive:
    score_e7_3 += 0.5
    print("⚠️ E7=3% 正收益比例相当")
else:
    score_v11 += 1
    print("❌ E7=3% 正收益比例不更高")

# 3. 窗口数（统计可靠性）
if len(wfa_e7_3) >= 10:
    score_e7_3 += 1
    print("✅ E7=3% 窗口数足够 (>=10)")
else:
    score_e7_3 -= 0.5
    print(f"⚠️ E7=3% 窗口数偏少 ({len(wfa_e7_3)})")

# 4. 统计显著性
if p_value and p_value < 0.10:
    score_e7_3 += 1
    print(f"✅ E7=3% 差异边缘显著 (p={p_value:.3f})")
else:
    print(f"⚠️ E7=3% 差异不显著 (p={p_value:.4f if p_value else 'N/A'})")

# 5. 最大回撤
e7_avg_dd = np.mean([r['test_dd'] for r in wfa_e7_3])
v11_avg_dd = np.mean([r['test_dd'] for r in wfa_v11])
if e7_avg_dd < v11_avg_dd:
    score_e7_3 += 1
    print("✅ E7=3% 平均回撤更小")
else:
    score_v11 += 1
    print("❌ E7=3% 平均回撤不更小")

print(f"\n总分: v1.1 = {score_v11:.1f}, E7=3% = {score_e7_3:.1f}")

if score_e7_3 > score_v11 + 0.5:
    recommendation = "E7=3%"
elif score_v11 > score_e7_3 + 0.5:
    recommendation = "v1.1"
else:
    recommendation = "v1.1 (差异不显著)"

print(f"\n{'='*80}")
print(f"建议: {recommendation}")
print(f"{'='*80}")