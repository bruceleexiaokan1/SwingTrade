#!/usr/bin/env python3
"""动量因子单因子回测

使用滚动窗口计算因子值，进行分组回测
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))


def load_daily_data(data_dir: Path, min_days: int = 200) -> pd.DataFrame:
    """加载日线数据"""
    files = list(data_dir.glob('*.parquet'))

    all_data = []
    for f in files:
        df = pd.read_parquet(f)
        if len(df) >= min_days:
            all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])
    combined = combined.sort_values(['code', 'date'])

    return combined


def compute_momentum_factor(daily: pd.DataFrame, lookback: int = 120) -> pd.DataFrame:
    """
    计算动量因子: 过去N日累计收益

    每个月末计算一次因子值
    """
    daily = daily.sort_values(['code', 'date']).reset_index(drop=True)

    # 计算日收益率
    daily['ret'] = daily.groupby('code')['close'].pct_change()

    # 计算累计收益 (使用滚动窗口, min_periods确保有效性)
    daily['cum_ret'] = daily.groupby('code')['ret'].transform(
        lambda x: x.rolling(lookback, min_periods=lookback).sum()
    )

    # 取每月最后一个交易日
    daily['year_month'] = daily['date'].dt.to_period('M')

    # 按月取最后一条记录 (仅保留有因子值的)
    factor_df = daily.dropna(subset=['cum_ret']).groupby(['code', 'year_month']).last().reset_index()
    factor_df = factor_df[['date', 'code', 'cum_ret']].dropna()
    factor_df = factor_df.rename(columns={'cum_ret': 'factor_value'})

    return factor_df


def compute_forward_returns(daily: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    """
    计算未来N日收益

    对于每只股票，计算每个日期之后的forward_days收益
    """
    daily = daily.sort_values(['code', 'date']).reset_index(drop=True)

    # 日收益率
    daily['ret'] = daily.groupby('code')['close'].pct_change()

    # 未来N日累计收益 = shift(-N) 之后 N天的收益之和
    daily['fwd_ret'] = daily.groupby('code')['ret'].transform(
        lambda x: x.shift(-forward_days + 1).rolling(window=forward_days, min_periods=forward_days).sum()
    )

    # 只保留有效的fwd_ret
    result = daily[['date', 'code', 'fwd_ret']].dropna()

    return result


def run_backtest(
    daily: pd.DataFrame,
    factor_name: str,
    lookback: int,
    forward_days: int = 5,
    n_groups: int = 10,
    commission_rate: float = 0.0003,  # 佣金0.03%
    stamp_tax: float = 0.001,          # 印花税0.1% (卖出时)
) -> dict:
    """运行单因子回测

    Args:
        commission_rate: 佣金率 (双向)
        stamp_tax: 印花税率 (卖出时)
    """

    print(f"\n{'='*60}")
    print(f"因子回测: {factor_name} (lookback={lookback})")
    print(f"{'='*60}")

    # 1. 计算因子值
    print("1. 计算因子值...")
    factor_df = compute_momentum_factor(daily, lookback=lookback)
    factor_df = factor_df.rename(columns={'factor_value': factor_name})
    print(f"   因子值数量: {len(factor_df)}")

    if len(factor_df) == 0:
        print("   无因子数据")
        return {}

    # 检查因子值是否有效
    if factor_df[factor_name].abs().max() > 10:  # 收益不可能大于1000%
        print(f"   警告: 因子值异常 (max={factor_df[factor_name].abs().max()})")
        # 过滤异常值
        factor_df = factor_df[factor_df[factor_name].abs() < 10]
        print(f"   过滤后: {len(factor_df)}")

    print(f"   因子均值: {factor_df[factor_name].mean():.4f}")
    print(f"   因子标准差: {factor_df[factor_name].std():.4f}")
    print(f"   因子范围: [{factor_df[factor_name].min():.4f}, {factor_df[factor_name].max():.4f}]")

    # 2. 计算未来收益
    print(f"2. 计算未来{forward_days}日收益...")
    forward_df = compute_forward_returns(daily, forward_days)
    print(f"   收益数据: {len(forward_df)}")

    # 3. 合并
    print("3. 合并数据...")
    merged = factor_df.merge(forward_df, on=['date', 'code'], how='inner')
    merged = merged.dropna()
    print(f"   合并后: {len(merged)}")

    if len(merged) < 50:
        print("   数据不足")
        return {}

    # 4. 分组回测
    print(f"4. 分组回测 (n_groups={n_groups})...")

    try:
        merged['group'] = pd.qcut(
            merged[factor_name],
            q=n_groups,
            labels=range(1, n_groups + 1),
            duplicates='drop'
        )
    except Exception as e:
        print(f"   分组失败: {e}")
        return {}

    # 计算每组统计 (原始收益)
    group_results = []
    for grp, data in merged.groupby('group', observed=True):
        group_results.append({
            'group': grp,
            'avg_return': data['fwd_ret'].mean(),
            'std_return': data['fwd_ret'].std(),
            'count': len(data),
            'median_return': data['fwd_ret'].median()
        })

    group_df = pd.DataFrame(group_results)
    print(group_df.to_string(index=False))

    # 5. 多空收益 (原始)
    print("5. 多空组合...")
    long_group = group_df.loc[group_df['avg_return'].idxmax(), 'group']
    short_group = group_df.loc[group_df['avg_return'].idxmin(), 'group']
    long_return = group_df.loc[group_df['avg_return'].idxmax(), 'avg_return']
    short_return = group_df.loc[group_df['avg_return'].idxmin(), 'avg_return']
    long_short = long_return - short_return

    # 扣除交易成本后的收益
    # 多空组合: 买入做多组, 卖出做空组
    # 买入时: 佣金
    # 卖出时: 佣金 + 印花税
    total_cost = commission_rate * 2 + stamp_tax  # 买卖一次完整交易的成本
    long_short_net = long_short - total_cost

    print(f"   做多组: G{long_group} (收益: {long_return*100:.2f}%)")
    print(f"   做空组: G{short_group} (收益: {short_return*100:.2f}%)")
    print(f"   多空收益差(毛): {long_short*100:.2f}%")
    print(f"   交易成本: {total_cost*100:.2f}% (佣金{commission_rate*100:.2f}%×2 + 印花税{stamp_tax*100:.2f}%)")
    print(f"   多空收益差(净): {long_short_net*100:.2f}%")

    # 6. 单调性
    print("6. 单调性检验...")
    correlation = group_df['group'].corr(group_df['avg_return'])

    is_monotonic = True
    for i in range(len(group_df) - 1):
        if group_df['avg_return'].iloc[i] > group_df['avg_return'].iloc[i + 1]:
            is_monotonic = False
            break

    diffs = []
    for i in range(len(group_df) - 1):
        diff = group_df['avg_return'].iloc[i + 1] - group_df['avg_return'].iloc[i]
        diffs.append(diff)
    positive_ratio = sum(1 for d in diffs if d > 0) / len(diffs) if diffs else 0
    monotonic_score = positive_ratio if not is_monotonic else 1.0

    print(f"   相关性: {correlation:.4f}")
    print(f"   单调性: {'是' if is_monotonic else '否'} (评分: {monotonic_score:.2f})")

    # 7. IC
    print("7. IC分析...")
    ic, pval = spearmanr(merged[factor_name], merged['fwd_ret'])
    print(f"   IC: {ic:.4f}, p-value: {pval:.4f}")
    print(f"   样本: {len(merged)}")

    return {
        'factor': factor_name,
        'group_results': group_df,
        'long_short': {
            'long_group': int(long_group),
            'short_group': int(short_group),
            'long_return': float(long_return),
            'short_return': float(short_return),
            'long_short_return': float(long_short),
            'long_short_net': float(long_short_net),
            'total_cost': float(total_cost)
        },
        'monotonicity': {
            'correlation': float(correlation),
            'is_monotonic': is_monotonic,
            'monotonic_score': float(monotonic_score)
        },
        'ic': float(ic),
        'p_value': float(pval),
        'n_samples': len(merged)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='动量因子单因子回测')
    parser.add_argument('--holding', type=int, default=10, help='持仓天数 (默认10)')
    args = parser.parse_args()

    forward_days = args.holding

    print("=" * 60)
    print(f"动量因子单因子回测 (持仓{forward_days}日)")
    print("=" * 60)

    data_dir = PROJECT_ROOT / "StockData" / "raw" / "daily"
    print(f"\n加载日线数据...")
    daily = load_daily_data(data_dir)
    print(f"日线数据: {len(daily)} 行, {daily['code'].nunique()} 只股票")
    print(f"日期范围: {daily['date'].min().date()} ~ {daily['date'].max().date()}")

    n_groups = 10

    results = {}

    # 回测不同周期的动量因子
    for periods, name in [(60, 'ret_3m'), (120, 'ret_6m'), (240, 'ret_12m')]:
        try:
            results[name] = run_backtest(
                daily, name, lookback=periods,
                forward_days=forward_days, n_groups=n_groups
            )
        except Exception as e:
            print(f"回测失败: {e}")

    # 汇总
    print("\n" + "=" * 60)
    print("回测结果汇总")
    print("=" * 60)

    summary = []
    for name, result in results.items():
        if result:
            ls = result['long_short']
            mono = result['monotonicity']
            summary.append({
                '因子': name,
                'IC': f"{result['ic']:.4f}",
                'p-value': f"{result['p_value']:.4f}",
                '多空(毛)': f"{ls['long_short_return']*100:.2f}%",
                '多空(净)': f"{ls['long_short_net']*100:.2f}%",
                '单调性': f"{mono['monotonic_score']:.2f}",
            })

    if summary:
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))

    # 成本分析
    print("\n交易成本敏感性分析:")
    print("-" * 60)
    print(f"单次交易成本: 佣金0.03%×2 + 印花税0.10% = 0.16%")
    print()

    valid = []
    for name, result in results.items():
        if not result:
            continue
        ic = result['ic']
        ls_gross = result['long_short']['long_short_return']
        ls_net = result['long_short']['long_short_net']
        mono = result['monotonicity']['monotonic_score']

        effective = abs(ic) > 0.02 and ls_net > 0 and mono > 0.5
        status = "✓ 有效" if effective else "○ 待优化"
        valid.append(name) if effective else None

        cost_ratio = ls_gross / result['long_short']['total_cost'] if result['long_short']['total_cost'] > 0 else 0
        print(f"  {name}: {status}")
        print(f"    IC={ic:.4f}, 多空(毛)={ls_gross*100:.2f}%, 多空(净)={ls_net*100:.2f}%, 单调={mono:.2f}")
        print(f"    成本覆盖率: {cost_ratio:.1f}x (毛收益/成本)")

    print()
    if valid:
        print(f"有效因子(扣除成本后): {', '.join(valid)}")
    else:
        print("提示: 扣除交易成本后，IC/多空收益/单调性需进一步优化")

    return results


if __name__ == '__main__':
    results = main()
