#!/usr/bin/env python3
"""
PATTERN_1 v1.1 Walk-Forward Analysis (WFA)
验证策略稳健性，避免过拟合
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

# ==================== 主程序 ====================

print("=" * 100)
print("PATTERN_1 v1.1 Walk-Forward Analysis (WFA)")
print("=" * 100)

# 加载数据
print("\n[1] 加载数据...")
df = load_data(500)
df = compute_indicators(df)
market = compute_market_indicators(df)
df = df.merge(market[['date', 'breadth', 'breadth_ma3', 'market_ret', 'market_ret_ma5', 'mom5']], on='date', how='left')
df = df[df['date'] >= '2021-01-01'].copy()

# 计算RSI上限
df['rsi_upper'] = df.apply(lambda r: dynamic_rsi_upper(r['atr_pct']), axis=1)

# 构建v1.1信号
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
print(f"总信号数: {len(signals_v11)}")

# ==================== WFA窗口设计 ====================
# 使用6个月训练，3个月测试的滚动窗口
# 2021-07开始训练，2022-01开始测试

print("\n[2] WFA窗口设计")
print("训练窗口: 6个月 | 测试窗口: 3个月 | 滚动步长: 1个月")

wfa_results = []

# WFA周期
# 训练期: 6个月
# 测试期: 3个月
# 起始: 2021-07 (需要6个月数据预热)

train_months = 6
test_months = 3

# 生成WFA窗口
start_date = pd.Timestamp('2021-07-01')
end_date = pd.Timestamp('2026-01-01')  # 确保有足够测试数据

current_train_end = start_date

while True:
    train_start = current_train_end - pd.DateOffset(months=train_months)
    test_start = current_train_end
    test_end = test_start + pd.DateOffset(months=test_months)

    if test_end > end_date:
        break

    # 训练期数据
    train_signals = signals_v11[
        (signals_v11['date'] >= train_start) &
        (signals_v11['date'] < test_start)
    ]

    # 测试期数据
    test_signals = signals_v11[
        (signals_v11['date'] >= test_start) &
        (signals_v11['date'] < test_end)
    ]

    if len(train_signals) < 10 or len(test_signals) < 5:
        current_train_end = test_end
        continue

    # 训练期分析
    train_trades, train_equity, train_dates = backtest(train_signals, df)
    train_result = analyze_trades(train_trades, train_equity, train_dates)

    # 测试期分析
    test_trades, test_equity, test_dates = backtest(test_signals, df)
    test_result = analyze_trades(test_trades, test_equity, test_dates)

    if test_result['n_trades'] == 0:
        current_train_end = test_end
        continue

    wfa_results.append({
        'train_period': f"{train_start.strftime('%Y-%m')} ~ {(test_start-pd.DateOffset(days=1)).strftime('%Y-%m')}",
        'test_period': f"{test_start.strftime('%Y-%m')} ~ {(test_end-pd.DateOffset(days=1)).strftime('%Y-%m')}",
        'train_trades': train_result['n_trades'],
        'train_return': train_result['total_return'] * 100,
        'train_dd': train_result['max_drawdown'] * 100,
        'test_trades': test_result['n_trades'],
        'test_return': test_result['total_return'] * 100,
        'test_dd': test_result['max_drawdown'] * 100,
        'yearly': test_result['yearly'],
    })

    current_train_end = test_end

# ==================== 结果汇总 ====================

print("\n" + "=" * 100)
print("Walk-Forward Analysis 结果")
print("=" * 100)

# 打印WFA表格
print(f"\n{'训练期':<18} {'测试期':<18} {'训练交易':>8} {'训练收益':>10} {'测试交易':>8} {'测试收益':>10} {'vs训练':>10}")
print("-" * 100)

positive_count = 0
for r in wfa_results:
    diff = r['test_return'] - r['train_return']
    marker = "✓" if r['test_return'] > 0 else "✗"
    if r['test_return'] > 0:
        positive_count += 1
    print(f"{r['train_period']:<18} {r['test_period']:<18} {r['train_trades']:>8} {r['train_return']:>+9.1f}% {r['test_trades']:>8} {r['test_return']:>+9.1f}% {diff:>+9.1f}% {marker}")

# 汇总统计
print("\n" + "=" * 100)
print("WFA 统计汇总")
print("=" * 100)

test_returns = [r['test_return'] for r in wfa_results]
train_returns = [r['train_return'] for r in wfa_results]

print(f"\n训练期:")
print(f"  平均收益: {np.mean(train_returns):+.1f}%")
print(f"  收益标准差: {np.std(train_returns):.1f}%")

print(f"\n测试期:")
print(f"  平均收益: {np.mean(test_returns):+.1f}%")
print(f"  收益标准差: {np.std(test_returns):.1f}%")
print(f"  正收益周期数: {positive_count}/{len(wfa_results)} ({positive_count/len(wfa_results)*100:.0f}%)")
print(f"  最大回撤: {max([r['test_dd'] for r in wfa_results]):.1f}%")
print(f"  平均回撤: {np.mean([r['test_dd'] for r in wfa_results]):.1f}%")

# In-Sample vs Out-of-Sample对比
print(f"\n过拟合检测:")
print(f"  训练期平均收益: {np.mean(train_returns):+.1f}%")
print(f"  测试期平均收益: {np.mean(test_returns):+.1f}%")
print(f"  衰减率: {(np.mean(test_returns)/np.mean(train_returns))*100:.0f}%")
print(f"  结论: {'稳健' if np.mean(test_returns)/np.mean(train_returns) > 0.5 else '可能过拟'}")

# ==================== 年化收益分析 ====================

print("\n" + "=" * 100)
print("测试期年度收益分析")
print("=" * 100)

# 汇总各年度在测试期的表现
yearly_returns = {2021: [], 2022: [], 2023: [], 2024: [], 2025: [], 2026: []}

for r in wfa_results:
    for year, stats in r['yearly'].items():
        yearly_returns[year].append(stats['return'] * 100)

print(f"\n{'年份':<6} {'WFA次数':>8} {'平均收益':>10} {'胜率':>10} {'最小收益':>10} {'最大收益':>10}")
print("-" * 60)

for year in sorted(yearly_returns.keys()):
    returns = yearly_returns[year]
    if len(returns) > 0:
        positive = sum(1 for r in returns if r > 0)
        print(f"{year:<6} {len(returns):>8} {np.mean(returns):>+9.1f}% {positive/len(returns)*100:>9.0f}% {min(returns):>+9.1f}% {max(returns):>+9.1f}%")

# ==================== 最终结论 ====================

print("\n" + "=" * 100)
print("WFA 验证结论")
print("=" * 100)

avg_test = np.mean(test_returns)
pos_ratio = positive_count / len(wfa_results)

if avg_test > 10 and pos_ratio > 0.6:
    verdict = "✅ WFA验证通过 - 策略稳健"
elif avg_test > 0 and pos_ratio > 0.4:
    verdict = "⚠️ WFA验证基本通过 - 策略可用但需监控"
else:
    verdict = "❌ WFA验证失败 - 策略可能过拟"

print(f"""
评估结果: {verdict}

详细指标:
  - 测试期平均年化收益: {avg_test:+.1f}%
  - 正收益周期比例: {pos_ratio*100:.0f}%
  - 训练/测试收益衰减率: {(np.mean(test_returns)/np.mean(train_returns))*100:.0f}%
  - WFA窗口数: {len(wfa_results)}

解读:
  - WFA通过 = 测试期表现与训练期接近，策略非过拟
  - WFA失败 = 测试期远差于训练期，策略可能过拟
""")

# ==================== 保存结果 ====================

wfa_report = {
    'date': '2026-04-04',
    'strategy': 'PATTERN_1 v1.1',
    'train_months': train_months,
    'test_months': test_months,
    'windows': [
        {
            'train_period': r['train_period'],
            'test_period': r['test_period'],
            'train_trades': r['train_trades'],
            'train_return': r['train_return'],
            'train_dd': r['train_dd'],
            'test_trades': r['test_trades'],
            'test_return': r['test_return'],
            'test_dd': r['test_dd'],
            'yearly': {str(k): v for k, v in r['yearly'].items()},
        }
        for r in wfa_results
    ],
    'summary': {
        'train_avg_return': float(np.mean(train_returns)),
        'test_avg_return': float(np.mean(test_returns)),
        'positive_ratio': pos_ratio,
        'decay_rate': float(np.mean(test_returns)/np.mean(train_returns)) if np.mean(train_returns) != 0 else 0,
        'verdict': verdict,
    }
}

# 保存JSON
output_path = Path('/Users/bruce/workspace/trade/SwingTrade/reports/PATTERN_1_WFA_REPORT.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(wfa_report, f, indent=2, default=str)

print(f"\nWFA报告已保存: {output_path}")
