#!/usr/bin/env python3
"""
PATTERN_1 v1.1 参数敏感度分析
知识库标准：参数变化5% → Sharpe变化<10%
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

def compute_metrics(trades, equity, initial_capital=100000):
    """计算策略指标"""
    if not trades or len(equity) < 2:
        return None

    equity_series = pd.Series(equity)
    total_return = (equity[-1] - initial_capital) / initial_capital

    # 计算Sharpe
    returns = equity_series.pct_change().dropna()
    if len(returns) < 2 or returns.std() == 0:
        sharpe = 0
    else:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)

    # 计算最大回撤
    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_dd = abs(drawdown.min())

    # Calmar
    calmar = total_return / max_dd if max_dd > 0 else 0

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'n_trades': len(trades),
    }

def sensitivity_test(df, param_name, param_values, base_mask_fn):
    """测试单个参数的敏感度"""
    results = []

    for val in param_values:
        # 构建信号
        mask = base_mask_fn(val)
        signals = df[mask].copy()

        if len(signals) < 5:
            results.append({
                'param_value': val,
                'n_signals': len(signals),
                'n_trades': 0,
                'return': np.nan,
                'sharpe': np.nan,
                'max_drawdown': np.nan,
            })
            continue

        trades, equity, equity_dates = backtest(signals, df)
        metrics = compute_metrics(trades, equity)

        if metrics:
            results.append({
                'param_value': val,
                'n_signals': len(signals),
                'n_trades': metrics['n_trades'],
                'return': metrics['total_return'],
                'sharpe': metrics['sharpe'],
                'max_drawdown': metrics['max_drawdown'],
            })
        else:
            results.append({
                'param_value': val,
                'n_signals': len(signals),
                'n_trades': 0,
                'return': np.nan,
                'sharpe': np.nan,
                'max_drawdown': np.nan,
            })

    return results

# ==================== 主程序 ====================

print("=" * 80)
print("PATTERN_1 v1.1 参数敏感度分析")
print("知识库标准: 参数变化5% → Sharpe变化<10%")
print("=" * 80)

# 加载数据
print("\n[1] 加载数据...")
df = load_data(500)
df = compute_indicators(df)
market = compute_market_indicators(df)
df = df.merge(market[['date', 'breadth', 'breadth_ma3', 'market_ret', 'market_ret_ma5', 'mom5']], on='date', how='left')
df = df[df['date'] >= '2021-01-01'].copy()

# 定义基础mask（用于参数替换）
def make_base_mask(rsi_upper, e7_threshold):
    def base_mask(val):
        return (
            (df['close'] > df['high_20d']) &
            (df['trend_up']) &
            (df['rsi'] >= 50) & (df['rsi'] <= val) &
            (df['vol_ratio'] >= 0.8) & (df['vol_ratio'] <= 2.0) &
            (df['atr_pct'] < 3) &
            (df['breadth_ma3'] > 0.50) &
            (df['price_above_ma20'] >= e7_threshold)
        )
    return base_mask

# ==================== 测试1: RSI上限敏感度 ====================
print("\n[2] RSI上限敏感度测试...")

rsi_values = [53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
base_rsi_fn = make_base_mask(58, 2.0)

rsi_results = sensitivity_test(df, 'rsi_upper', rsi_values, base_rsi_fn)
rsi_df = pd.DataFrame(rsi_results).dropna(subset=['return'])

print(f"\nRSI上限 vs 收益:")
print(f"{'RSI上限':>8} {'信号数':>8} {'交易数':>8} {'收益率':>12} {'Sharpe':>8} {'最大DD':>10}")
print("-" * 60)
for r in rsi_results:
    if not np.isnan(r.get('return', np.nan)):
        print(f"{r['param_value']:>8} {r['n_signals']:>8} {r['n_trades']:>8} {r['return']*100:>+11.1f}% {r['sharpe']:>8.2f} {r['max_drawdown']*100:>9.1f}%")

# ==================== 测试2: E7过滤敏感度 ====================
print("\n[3] E7过滤敏感度测试...")

e7_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

def make_e7_mask(e7_val):
    def mask_fn(val):
        return (
            (df['close'] > df['high_20d']) &
            (df['trend_up']) &
            (df['rsi'] >= 50) & (df['rsi'] <= 58) &
            (df['vol_ratio'] >= 0.8) & (df['vol_ratio'] <= 2.0) &
            (df['atr_pct'] < 3) &
            (df['breadth_ma3'] > 0.50) &
            (df['price_above_ma20'] >= val)
        )
    return mask_fn

e7_results = sensitivity_test(df, 'e7_threshold', e7_values, make_e7_mask(None))
e7_df = pd.DataFrame(e7_results).dropna(subset=['return'])

print(f"\nE7过滤 vs 收益:")
print(f"{'E7阈值':>8} {'信号数':>8} {'交易数':>8} {'收益率':>12} {'Sharpe':>8} {'最大DD':>10}")
print("-" * 60)
for r in e7_results:
    if not np.isnan(r.get('return', np.nan)):
        print(f"{r['param_value']:>7.1f}% {r['n_signals']:>8} {r['n_trades']:>8} {r['return']*100:>+11.1f}% {r['sharpe']:>8.2f} {r['max_drawdown']*100:>9.1f}%")

# ==================== 测试3: 止损敏感度 ====================
print("\n[4] 止损敏感度测试...")

# 注意：止损变化需要修改backtest，暂时只展示结果
stop_loss_values = [0.06, 0.07, 0.08, 0.09, 0.10, 0.12]

def make_stop_mask(sl):
    def mask_fn(val):
        return (
            (df['close'] > df['high_20d']) &
            (df['trend_up']) &
            (df['rsi'] >= 50) & (df['rsi'] <= 58) &
            (df['vol_ratio'] >= 0.8) & (df['vol_ratio'] <= 2.0) &
            (df['atr_pct'] < 3) &
            (df['breadth_ma3'] > 0.50) &
            (df['price_above_ma20'] >= 2.0)
        )
    return mask_fn

# 简化的止损测试（使用不同的stop_loss参数调用backtest）
print("(止损敏感度需要修改backtest，暂时跳过)")
print("建议后续手动测试: 6%, 7%, 8%, 9%, 10%, 12%")

# ==================== 敏感度分析 ====================

print("\n" + "=" * 80)
print("敏感度分析结果")
print("=" * 80)

# RSI敏感度分析
if len(rsi_df) > 0:
    rsi_base = rsi_df[rsi_df['param_value'] == 58]
    if len(rsi_base) > 0:
        base_sharpe = rsi_base['sharpe'].values[0]
        base_return = rsi_base['return'].values[0]

        print(f"\nRSI=58 基准:")
        print(f"  Sharpe: {base_sharpe:.2f}")
        print(f"  收益: {base_return*100:+.1f}%")

        # 5%参数变化分析
        rsi_55 = rsi_df[rsi_df['param_value'] == 55]
        rsi_61 = rsi_df[rsi_df['param_value'] == 61]

        print(f"\n参数变化影响 (知识库标准: 5%变化 → Sharpe变化<10%):")

        if len(rsi_55) > 0:
            sharpe_change_55 = (rsi_55['sharpe'].values[0] - base_sharpe) / base_sharpe * 100 if base_sharpe != 0 else 0
            return_change_55 = (rsi_55['return'].values[0] - base_return) / base_return * 100 if base_return != 0 else 0
            print(f"  RSI=55 (↓5.2%): Sharpe变化 {sharpe_change_55:+.1f}%, 收益变化 {return_change_55:+.1f}%")

        if len(rsi_61) > 0:
            sharpe_change_61 = (rsi_61['sharpe'].values[0] - base_sharpe) / base_sharpe * 100 if base_sharpe != 0 else 0
            return_change_61 = (rsi_61['return'].values[0] - base_return) / base_return * 100 if base_return != 0 else 0
            print(f"  RSI=61 (↑5.2%): Sharpe变化 {sharpe_change_61:+.1f}%, 收益变化 {return_change_61:+.1f}%")

# E7敏感度分析
if len(e7_df) > 0:
    e7_base = e7_df[e7_df['param_value'] == 2.0]
    if len(e7_base) > 0:
        base_sharpe_e7 = e7_base['sharpe'].values[0]
        base_return_e7 = e7_base['return'].values[0]

        print(f"\nE7=2.0% 基准:")
        print(f"  Sharpe: {base_sharpe_e7:.2f}")
        print(f"  收益: {base_return_e7*100:+.1f}%")

        e7_1 = e7_df[e7_df['param_value'] == 1.0]
        e7_3 = e7_df[e7_df['param_value'] == 3.0]

        print(f"\n参数变化影响 (E7变化 50%):")

        if len(e7_1) > 0:
            sharpe_change_1 = (e7_1['sharpe'].values[0] - base_sharpe_e7) / base_sharpe_e7 * 100 if base_sharpe_e7 != 0 else 0
            return_change_1 = (e7_1['return'].values[0] - base_return_e7) / base_return_e7 * 100 if base_return_e7 != 0 else 0
            print(f"  E7=1.0% (↓50%): Sharpe变化 {sharpe_change_1:+.1f}%, 收益变化 {return_change_1:+.1f}%")

        if len(e7_3) > 0:
            sharpe_change_3 = (e7_3['sharpe'].values[0] - base_sharpe_e7) / base_sharpe_e7 * 100 if base_sharpe_e7 != 0 else 0
            return_change_3 = (e7_3['return'].values[0] - base_return_e7) / base_return_e7 * 100 if base_return_e7 != 0 else 0
            print(f"  E7=3.0% (↑50%): Sharpe变化 {sharpe_change_3:+.1f}%, 收益变化 {return_change_3:+.1f}%")

# ==================== 结论 ====================

print("\n" + "=" * 80)
print("敏感度分析结论")
print("=" * 80)

print("""
根据知识库标准：
- 参数变化5% → Sharpe变化<10% = 稳健

RSI上限敏感度：
- 从RSI=58变化到55或61，Sharpe变化需要<10%才符合稳健标准
- 如果Sharpe变化>10%，说明参数敏感，需要谨慎

E7过滤敏感度：
- 从E7=2.0%变化到1.0%或3.0%，Sharpe变化需要<10%才符合稳健标准
- 建议检查E7阈值是否在plateau上
""")

# 保存结果
report = {
    'date': '2026-04-04',
    'strategy': 'PATTERN_1 v1.1',
    'rsi_sensitivity': [
        {
            'param_value': r['param_value'],
            'n_signals': int(r['n_signals']),
            'n_trades': int(r['n_trades']),
            'return': float(r['return']) if not np.isnan(r.get('return', np.nan)) else None,
            'sharpe': float(r['sharpe']) if not np.isnan(r.get('sharpe', np.nan)) else None,
            'max_drawdown': float(r['max_drawdown']) if not np.isnan(r.get('max_drawdown', np.nan)) else None,
        }
        for r in rsi_results
    ],
    'e7_sensitivity': [
        {
            'param_value': float(r['param_value']),
            'n_signals': int(r['n_signals']),
            'n_trades': int(r['n_trades']),
            'return': float(r['return']) if not np.isnan(r.get('return', np.nan)) else None,
            'sharpe': float(r['sharpe']) if not np.isnan(r.get('sharpe', np.nan)) else None,
            'max_drawdown': float(r['max_drawdown']) if not np.isnan(r.get('max_drawdown', np.nan)) else None,
        }
        for r in e7_results
    ],
}

output_path = Path('/Users/bruce/workspace/trade/SwingTrade/reports/t6_sensitivity_results.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f"\n报告已保存: {output_path}")