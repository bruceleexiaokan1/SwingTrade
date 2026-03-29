"""分组回测框架

提供因子分组回测功能:
- 按因子值分N组
- 计算各组未来收益
- 检验单调性
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

from ..exceptions import EvaluationError


def group_backtest(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_groups: int = 10,
    factor_col: str = None
) -> pd.DataFrame:
    """
    分组回测

    按因子值分n_groups组，计算每组的平均未来收益

    Args:
        factor_values: 因子值DataFrame (date, code, factor_value)
        forward_returns: 未来收益DataFrame (date, code, return)
        n_groups: 分组数，默认10
        factor_col: 因子列名，默认使用factor_value

    Returns:
        DataFrame: group, avg_return, count, std_return
    """
    if factor_col is None:
        factor_col = "factor_value"

    if factor_values.empty or forward_returns.empty:
        raise EvaluationError("Empty DataFrame provided")

    # 合并
    merged = factor_values.merge(
        forward_returns,
        on=["date", "code"],
        how="inner"
    )

    if merged.empty:
        raise EvaluationError("No overlapping data")

    # 获取收益列
    return_cols = [c for c in forward_returns.columns if c not in ["date", "code"]]
    if not return_cols:
        raise EvaluationError("No return column found")
    return_col = return_cols[0]

    # 去除NaN
    merged = merged.dropna(subset=[factor_col, return_col])

    if len(merged) == 0:
        raise EvaluationError("No valid data after dropping NaN")

    # 按因子值分位分组
    merged["group"] = pd.qcut(
        merged[factor_col],
        q=n_groups,
        labels=range(1, n_groups + 1),
        duplicates="drop"
    )

    # 计算每组统计
    results = []
    for grp, group_data in merged.groupby("group"):
        results.append({
            "group": grp,
            "avg_return": group_data[return_col].mean(),
            "std_return": group_data[return_col].std(),
            "count": len(group_data),
            "median_return": group_data[return_col].median()
        })

    return pd.DataFrame(results)


def calculate_long_short_return(
    group_results: pd.DataFrame,
    long_group: int = None,
    short_group: int = None
) -> dict:
    """
    计算多空组合收益

    Args:
        group_results: 分组回测结果
        long_group: 做多组号，默认最高组
        short_group: 做空组号，默认最低组

    Returns:
        dict: {long_short_return, long_return, short_return, spread}
    """
    if group_results.empty:
        return {
            "long_short_return": np.nan,
            "long_return": np.nan,
            "short_return": np.nan,
            "spread": np.nan
        }

    # 找到收益最高和最低的组
    if long_group is None:
        long_group = group_results["avg_return"].idxmax()
    if short_group is None:
        short_group = group_results["avg_return"].idxmin()

    long_row = group_results[group_results["group"] == long_group]
    short_row = group_results[group_results["group"] == short_group]

    if long_row.empty or short_row.empty:
        return {
            "long_short_return": np.nan,
            "long_return": np.nan,
            "short_return": np.nan,
            "spread": np.nan
        }

    long_return = long_row["avg_return"].values[0]
    short_return = short_row["avg_return"].values[0]
    long_short = long_return - short_return

    return {
        "long_short_return": long_short,
        "long_return": long_return,
        "short_return": short_return,
        "long_group": int(long_group),
        "short_group": int(short_group),
        "spread": abs(long_group - short_group)
    }


def check_monotonicity(
    group_results: pd.DataFrame,
) -> dict:
    """
    检验分组收益的单调性

    Args:
        group_results: 分组回测结果

    Returns:
        dict: {is_monotonic, correlation, monotonic_score}
    """
    if group_results.empty or len(group_results) < 2:
        return {
            "is_monotonic": False,
            "correlation": np.nan,
            "monotonic_score": 0.0
        }

    # 计算组号和收益的相关性
    correlation = group_results["group"].corr(group_results["avg_return"])

    # 检查单调性：组号增加，收益也应该增加
    is_monotonic = True
    for i in range(len(group_results) - 1):
        if group_results["avg_return"].iloc[i] > group_results["avg_return"].iloc[i + 1]:
            is_monotonic = False
            break

    # 单调性评分：0-1之间
    if is_monotonic:
        monotonic_score = 1.0
    else:
        # 计算偏离程度
        diffs = []
        for i in range(len(group_results) - 1):
            diff = (group_results["avg_return"].iloc[i + 1] -
                    group_results["avg_return"].iloc[i])
            diffs.append(diff)

        # 正相关越多越接近1
        positive_ratio = sum(1 for d in diffs if d > 0) / len(diffs)
        monotonic_score = max(0, positive_ratio)

    return {
        "is_monotonic": is_monotonic,
        "correlation": correlation,
        "monotonic_score": monotonic_score
    }


__all__ = [
    "group_backtest",
    "calculate_long_short_return",
    "check_monotonicity",
]
