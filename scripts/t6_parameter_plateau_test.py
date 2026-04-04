#!/usr/bin/env python3
"""
PATTERN_1 v1.1 参数Plateau检测
验证关键参数是否在稳定 plateau 上而非过拟合岛屿
知识库方法：生成热力图，确认参数有宽稳定区
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

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

def dynamic_rsi_upper_test(atr_pct, base_upper):
    """测试用的动态RSI上限"""
    if atr_pct < 2.0:
        return base_upper
    elif atr_pct <= 3.0:
        return base_upper - 1
    else:
        return base_upper - 3

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

def grid_search(df, rsi_upper_range, e7_range):
    """
    参数网格搜索
    生成热力图数据：RSI上限 vs E7过滤
    """
    results = []

    for rsi_upper in rsi_upper_range:
        for e7_threshold in e7_range:
            # 构建信号
            mask = (
                (df['close'] > df['high_20d']) &
                (df['trend_up']) &
                (df['rsi'] >= 50) & (df['rsi'] <= rsi_upper) &
                (df['vol_ratio'] >= 0.8) & (df['vol_ratio'] <= 2.0) &
                (df['atr_pct'] < 3) &
                (df['breadth_ma3'] > 0.50) &
                (df['price_above_ma20'] >= e7_threshold)
            )

            signals = df[mask].copy()

            if len(signals) < 10:
                results.append({
                    'rsi_upper': int(rsi_upper),
                    'e7_threshold': float(e7_threshold),
                    'n_signals': 0,
                    'return': np.nan,
                    'sharpe': np.nan,
                    'max_drawdown': np.nan,
                })
                continue

            trades, equity, equity_dates = backtest(signals, df)

            if not trades or len(equity) < 2:
                results.append({
                    'rsi_upper': int(rsi_upper),
                    'e7_threshold': float(e7_threshold),
                    'n_signals': len(signals),
                    'n_trades': 0,
                    'return': np.nan,
                    'sharpe': np.nan,
                    'max_drawdown': np.nan,
                })
                continue

            # 计算收益
            total_return = (equity[-1] - 100000) / 100000

            # 计算Sharpe（简化）
            equity_series = pd.Series(equity)
            returns = equity_series.pct_change().dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

            # 计算最大回撤
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_dd = abs(drawdown.min())

            results.append({
                'rsi_upper': int(rsi_upper),
                'e7_threshold': float(e7_threshold),
                'n_signals': len(signals),
                'n_trades': len(trades),
                'return': float(total_return),
                'sharpe': float(sharpe),
                'max_drawdown': float(max_dd),
            })

    return results

def analyze_plateau(results):
    """
    分析参数plateau
    判定标准：
    - 最优参数周围有宽稳定区（相邻参数表现接近）
    - 不是孤立的峰值（岛屿）
    """
    # 转为DataFrame
    df = pd.DataFrame(results)
    df = df.dropna(subset=['return'])

    if len(df) == 0:
        return {'is_plateau': False, 'reason': 'No valid results'}

    # 找到最优参数
    best_idx = df['return'].idxmax()
    best = df.loc[best_idx]

    # 找相邻参数
    neighbors = df[
        (abs(df['rsi_upper'] - best['rsi_upper']) <= 1) &
        (abs(df['e7_threshold'] - best['e7_threshold']) <= 0.5) &
        (df['rsi_upper'] != best['rsi_upper']) | (df['e7_threshold'] != best['e7_threshold'])
    ]

    if len(neighbors) == 0:
        # 没有相邻参数（孤立峰值）
        return {
            'is_plateau': False,
            'reason': 'Isolated peak - potential overfitting',
            'best_params': {
                'rsi_upper': int(best['rsi_upper']),
                'e7_threshold': float(best['e7_threshold']),
                'return': float(best['return']),
            }
        }

    # 计算相邻参数的平均表现
    neighbor_avg_return = neighbors['return'].mean()
    neighbor_min_return = neighbors['return'].min()

    # plateau判定：相邻参数表现不低于最优的70%
    plateau_threshold = best['return'] * 0.7

    is_plateau = neighbor_min_return >= plateau_threshold

    return {
        'is_plateau': is_plateau,
        'reason': 'Wide stable plateau' if is_plateau else 'Narrow peak',
        'best_params': {
            'rsi_upper': int(best['rsi_upper']),
            'e7_threshold': float(best['e7_threshold']),
            'return': float(best['return']),
            'sharpe': float(best['sharpe']),
            'max_drawdown': float(best['max_drawdown']),
        },
        'neighbor_avg_return': float(neighbor_avg_return),
        'neighbor_min_return': float(neighbor_min_return),
        'neighbor_count': len(neighbors),
        'decay_rate': float((best['return'] - neighbor_avg_return) / best['return']) if best['return'] != 0 else 0,
    }

# ==================== 主程序 ====================

print("=" * 80)
print("PATTERN_1 v1.1 参数Plateau检测")
print("=" * 80)

# 加载数据
print("\n[1] 加载数据...")
df = load_data(500)
df = compute_indicators(df)
market = compute_market_indicators(df)
df = df.merge(market[['date', 'breadth', 'breadth_ma3', 'market_ret', 'market_ret_ma5', 'mom5']], on='date', how='left')
df = df[df['date'] >= '2021-01-01'].copy()

# 网格搜索
print("\n[2] 参数网格搜索...")
print("RSI上限范围: 53-62")
print("E7过滤范围: 0.5%-5%")

rsi_upper_range = range(53, 63)  # 53 to 62
e7_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

results = grid_search(df, rsi_upper_range, e7_range)

# 转为DataFrame便于展示
results_df = pd.DataFrame(results)
valid_results = results_df.dropna(subset=['return'])

# ==================== 输出热力图 ====================

print("\n" + "=" * 80)
print("热力图：RSI上限 vs E7过滤 (收益率)")
print("=" * 80)

# Pivot table
pivot_return = valid_results.pivot_table(
    values='return',
    index='e7_threshold',
    columns='rsi_upper',
    aggfunc='first'
) * 100

print("\n" + pivot_return.round(1).to_string())

# ==================== Plateau分析 ====================

print("\n" + "=" * 80)
print("Plateau 分析结果")
print("=" * 80)

plateau_result = analyze_plateau(results)

if plateau_result['is_plateau']:
    print("✅ 参数在稳定Plateau上 - 非过拟合")
else:
    print("⚠️ 参数可能是孤立峰值 - 需谨慎")

print(f"\n原因: {plateau_result['reason']}")

if 'best_params' in plateau_result:
    bp = plateau_result['best_params']
    print(f"\n最优参数:")
    print(f"  RSI上限: {bp['rsi_upper']}")
    print(f"  E7过滤: {bp['e7_threshold']}%")
    print(f"  收益率: {bp['return']*100:+.1f}%")
    if 'sharpe' in bp:
        print(f"  Sharpe: {bp['sharpe']:.2f}")
        print(f"  最大回撤: {bp['max_drawdown']*100:.1f}%")

if 'neighbor_avg_return' in plateau_result:
    print(f"\n相邻参数表现:")
    print(f"  相邻参数数: {plateau_result['neighbor_count']}")
    print(f"  平均收益: {plateau_result['neighbor_avg_return']*100:+.1f}%")
    print(f"  最低收益: {plateau_result['neighbor_min_return']*100:+.1f}%")
    print(f"  衰减率: {plateau_result['decay_rate']*100:.1f}%")

# ==================== 最优参数 vs 当前参数对比 ====================

print("\n" + "=" * 80)
print("当前参数 vs 最优参数")
print("=" * 80)

# 当前参数（v1.1标准）
current_return = valid_results[
    (valid_results['rsi_upper'] == 58) &
    (valid_results['e7_threshold'] == 2.0)
]['return'].values

best_return = plateau_result['best_params']['return']

if len(current_return) > 0:
    print(f"\n当前参数 (RSI上限=58, E7=2%):")
    print(f"  收益率: {current_return[0]*100:+.1f}%")
    print(f"  vs 最优: {(current_return[0]/best_return)*100:.1f}%")

# ==================== 保存结果 ====================

report = {
    'date': '2026-04-04',
    'strategy': 'PATTERN_1 v1.1',
    'plateau_analysis': plateau_result,
    'grid_results': [
        {
            'rsi_upper': int(r['rsi_upper']),
            'e7_threshold': float(r['e7_threshold']),
            'n_signals': int(r['n_signals']) if not np.isnan(r.get('n_signals', np.nan)) else 0,
            'n_trades': int(r['n_trades']) if not np.isnan(r.get('n_trades', np.nan)) else 0,
            'return': float(r['return']) if not np.isnan(r.get('return', np.nan)) else None,
            'sharpe': float(r['sharpe']) if not np.isnan(r.get('sharpe', np.nan)) else None,
            'max_drawdown': float(r['max_drawdown']) if not np.isnan(r.get('max_drawdown', np.nan)) else None,
        }
        for r in results
    ],
}

output_path = Path('/Users/bruce/workspace/trade/SwingTrade/reports/t6_parameter_plateau_results.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f"\n报告已保存: {output_path}")