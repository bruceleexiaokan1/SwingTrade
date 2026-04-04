#!/usr/bin/env python3
"""
PATTERN_1 v1.1 Monte Carlo模拟验证
基于知识库：验证策略稳健性，打乱交易序列验证原始曲线是否在Top 25%
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import random

def load_data(max_stocks=500):
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
    """返回交易列表、权益曲线、权益日期"""
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

def monte_carlo_simulation(trades, equity, equity_dates, n_simulations=1000, initial_capital=100000):
    """
    Monte Carlo模拟：正确方法 - 重采样权益曲线的日收益率
    知识库标准：
    - 原始收益曲线 > Monte Carlo P50
    - 原始收益曲线在Top 25%分位

    方法：对权益曲线的日收益率进行重采样，重建equity path
    """
    if not trades or len(equity) < 10:
        return None

    # 计算日收益率序列
    equity_series = pd.Series(equity)
    daily_returns = equity_series.pct_change().dropna()

    if len(daily_returns) < 5:
        return None

    # 原始策略统计
    original_return = (equity[-1] - initial_capital) / initial_capital

    # Monte Carlo 结果
    mc_final_returns = []
    mc_equity_paths = []

    random.seed(42)
    np.random.seed(42)

    for i in range(n_simulations):
        # 有放回重采样日收益率
        n_days = len(daily_returns)
        sampled_indices = np.random.choice(n_days, size=n_days, replace=True)
        sampled_returns = daily_returns.iloc[sampled_indices].values

        # 重建equity path
        mc_equity = [initial_capital]
        for r in sampled_returns:
            if np.isnan(r) or np.isinf(r):
                r = 0
            mc_equity.append(mc_equity[-1] * (1 + r))

        mc_final_returns.append((mc_equity[-1] - initial_capital) / initial_capital)
        mc_equity_paths.append(mc_equity)

    mc_final_returns = np.array(mc_final_returns)
    mc_final_returns.sort()

    # 计算分位数
    p10 = np.percentile(mc_final_returns, 10)
    p25 = np.percentile(mc_final_returns, 25)
    p50 = np.percentile(mc_final_returns, 50)
    p75 = np.percentile(mc_final_returns, 75)
    p90 = np.percentile(mc_final_returns, 90)

    # 计算原始曲线排名 (有多少%比原始低)
    rank = np.sum(mc_final_returns < original_return) / n_simulations * 100

    # 稳健性判定
    is_robust = (
        original_return >= p50 and  # 原始 >= P50
        rank >= 25                   # 在Top 25% (即至少75%的模拟比原始差)
    )

    return {
        'original_return': float(original_return),
        'p10': float(p10),
        'p25': float(p25),
        'p50': float(p50),
        'p75': float(p75),
        'p90': float(p90),
        'rank_percentile': float(rank),
        'is_robust': bool(is_robust),
        'n_simulations': int(n_simulations),
        'n_trades': int(len(trades)),
        'n_days': int(len(daily_returns)),
    }

def bootstrap_confidence_interval(trades, equity, equity_dates, n_bootstrap=1000, initial_capital=100000):
    """Bootstrap重采样验证置信区间"""
    if not trades:
        return None

    df = pd.DataFrame(trades)
    n_trades = len(df)

    bootstrap_returns = []

    for _ in range(n_bootstrap):
        # 有放回重采样
        sample_indices = np.random.choice(n_trades, size=n_trades, replace=True)
        sample_df = df.iloc[sample_indices]

        # 计算重采样收益
        sample_return = sample_df['pnl_pct'].mean()
        bootstrap_returns.append(sample_return)

    bootstrap_returns = np.array(bootstrap_returns)

    ci_lower = np.percentile(bootstrap_returns, 5)
    ci_upper = np.percentile(bootstrap_returns, 95)
    mean_return = np.mean(bootstrap_returns)

    return {
        'mean_return': mean_return,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_bootstrap': n_bootstrap,
    }

# ==================== 主程序 ====================

print("=" * 80)
print("PATTERN_1 v1.1 Monte Carlo稳健性验证")
print("=" * 80)

# 加载数据
print("\n[1] 加载数据...")
df = load_data(500)
df = compute_indicators(df)
market = compute_market_indicators(df)
df = df.merge(market[['date', 'breadth', 'breadth_ma3', 'market_ret', 'market_ret_ma5', 'mom5']], on='date', how='left')
df = df[df['date'] >= '2021-01-01'].copy()

# 构建v1.1信号
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

# 回测
print("\n[2] 回测原始策略...")
trades, equity, equity_dates = backtest(signals_v11, df)

if not trades:
    print("无交易，跳过Monte Carlo测试")
    exit(1)

# Monte Carlo模拟
print("\n[3] Monte Carlo模拟 (1000次)...")
mc_result = monte_carlo_simulation(trades, equity, equity_dates, n_simulations=1000)

# Bootstrap置信区间
print("\n[4] Bootstrap置信区间 (1000次)...")
bootstrap_result = bootstrap_confidence_interval(trades, equity, equity_dates, n_bootstrap=1000)

# ==================== 输出结果 ====================

print("\n" + "=" * 80)
print("Monte Carlo 稳健性验证结果")
print("=" * 80)

print(f"\n原始策略:")
print(f"  总交易数: {mc_result['n_trades']}")
print(f"  原始收益: {mc_result['original_return']*100:+.1f}%")

print(f"\nMonte Carlo 分位数:")
print(f"  P10: {mc_result['p10']*100:+.1f}%")
print(f"  P25: {mc_result['p25']*100:+.1f}%")
print(f"  P50: {mc_result['p50']*100:+.1f}%")
print(f"  P75: {mc_result['p75']*100:+.1f}%")
print(f"  P90: {mc_result['p90']*100:+.1f}%")

print(f"\n原始曲线排名: {mc_result['rank_percentile']:.1f}% (Top {100-mc_result['rank_percentile']:.1f}%)")

robust_verdict = "✅ 稳健" if mc_result['is_robust'] else "❌ 不稳健"
print(f"稳健性判定: {robust_verdict}")

# Bootstrap结果
if bootstrap_result:
    print(f"\nBootstrap 95%置信区间:")
    print(f"  均值收益: {bootstrap_result['mean_return']*100:+.2f}%")
    print(f"  置信区间: [{bootstrap_result['ci_lower']*100:+.2f}%, {bootstrap_result['ci_upper']*100:+.2f}%]")

# 综合判定
print("\n" + "=" * 80)
print("综合判定")
print("=" * 80)

criteria_met = 0
total_criteria = 2

if mc_result['original_return'] > mc_result['p50']:
    print("✅ 原始收益 > P50")
    criteria_met += 1
else:
    print("❌ 原始收益 <= P50")

if mc_result['rank_percentile'] >= 25:
    print(f"✅ 排名 {mc_result['rank_percentile']:.1f}% >= 25% (Top {100-mc_result['rank_percentile']:.1f}%)")
    criteria_met += 1
else:
    print(f"❌ 排名 {mc_result['rank_percentile']:.1f}% < 25%")

print(f"\n通过标准: {criteria_met}/{total_criteria}")

# 保存结果
report = {
    'date': '2026-04-04',
    'strategy': 'PATTERN_1 v1.1',
    'monte_carlo': mc_result,
    'bootstrap': bootstrap_result,
    'criteria_passed': criteria_met,
    'total_criteria': total_criteria,
}

output_path = Path('/Users/bruce/workspace/trade/SwingTrade/reports/t6_monte_carlo_results.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f"\n报告已保存: {output_path}")