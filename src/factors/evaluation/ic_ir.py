"""IC/IR评估框架

提供因子有效性评估:
- IC (Information Coefficient): 因子与未来收益的秩相关性
- IR (Information Ratio): IC均值 / IC标准差
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from typing import Optional, Tuple

from ..exceptions import EvaluationError


def calculate_ic(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "spearman"
) -> pd.DataFrame:
    """
    计算IC (Information Coefficient)

    IC = factor_value 与 forward_return 的相关系数

    Args:
        factor_values: 因子值DataFrame (date, code, factor_name)
        forward_returns: 未来收益DataFrame (date, code, return)
        method: 相关系数方法 ("spearman" 或 "pearson")

    Returns:
        DataFrame: date, ic_value, p_value

    Raises:
        EvaluationError: 数据无效时
    """
    if factor_values.empty or forward_returns.empty:
        raise EvaluationError("Empty DataFrame provided")

    # 合并因子值和未来收益
    merged = factor_values.merge(
        forward_returns,
        on=["date", "code"],
        how="inner"
    )

    if merged.empty:
        raise EvaluationError("No overlapping data between factor and returns")

    # 获取因子列名
    factor_cols = [c for c in factor_values.columns if c not in ["date", "code"]]
    if len(factor_cols) == 0:
        raise EvaluationError("No factor column found")

    factor_col = factor_cols[0]
    return_col = [c for c in forward_returns.columns if c not in ["date", "code"]][0]

    # 计算每日IC
    results = []
    for date, group in merged.groupby("date"):
        if len(group) < 10:  # 样本量太少不计算
            continue

        valid = group.dropna(subset=[factor_col, return_col])
        if len(valid) < 10:
            continue

        try:
            if method == "spearman":
                ic, p_value = spearmanr(valid[factor_col], valid[return_col])
            else:
                ic, p_value = pearsonr(valid[factor_col], valid[return_col])
        except Exception:
            ic = np.nan
            p_value = np.nan

        results.append({
            "date": date,
            "ic_value": ic,
            "p_value": p_value
        })

    return pd.DataFrame(results)


def calculate_ir(
    ic_series: pd.Series,
    lookback: Optional[int] = None
) -> dict:
    """
    计算IR (Information Ratio)

    IR = IC均值 / IC标准差

    Args:
        ic_series: IC时间序列
        lookback: 回溯期，None表示全量

    Returns:
        dict: {ic_mean, ic_std, ir, ic_count, valid_ratio}
    """
    if lookback is not None:
        ic = ic_series.iloc[-lookback:]
    else:
        ic = ic_series

    # 去除NaN
    ic = ic.dropna()

    if len(ic) == 0:
        return {
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ir": np.nan,
            "ic_count": 0,
            "valid_ratio": 0.0
        }

    ic_mean = ic.mean()
    ic_std = ic.std()

    if ic_std == 0 or np.isnan(ic_std):
        ir = np.nan
    else:
        ir = ic_mean / ic_std

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ir": ir,
        "ic_count": len(ic),
        "valid_ratio": len(ic) / len(ic_series)
    }


def batch_calculate_ic(
    factor_wide: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: list,
    method: str = "spearman"
) -> dict:
    """
    批量计算多个因子的IC

    Args:
        factor_wide: 因子宽表
        forward_returns: 未来收益
        factor_names: 因子名列表
        method: 相关系数方法

    Returns:
        dict: {factor_name: {ic_stats, ic_series}}
    """
    results = {}

    for factor_name in factor_names:
        if factor_name not in factor_wide.columns:
            continue

        try:
            factor_df = factor_wide[["date", "code", factor_name]].copy()
            ic_df = calculate_ic(factor_df, forward_returns, method)

            if not ic_df.empty:
                ir_stats = calculate_ir(ic_df["ic_value"])

                results[factor_name] = {
                    "ic_mean": ir_stats["ic_mean"],
                    "ic_std": ir_stats["ic_std"],
                    "ir": ir_stats["ir"],
                    "ic_count": ir_stats["ic_count"],
                    "valid_ratio": ir_stats["valid_ratio"],
                    "ic_series": ic_df
                }
        except Exception as e:
            results[factor_name] = {
                "error": str(e)
            }

    return results


__all__ = [
    "calculate_ic",
    "calculate_ir",
    "batch_calculate_ic",
]
