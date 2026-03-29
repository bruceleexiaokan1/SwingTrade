"""价格转换工具

支持后复权和前复权之间的转换
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def convert_to_forward_adj(df: pd.DataFrame) -> pd.DataFrame:
    """
    将后复权价格转换为前复权价格

    前复权价格 = 后复权价格 * (最新adj_factor / 当时的adj_factor)

    这样转换后，所有历史价格都按照最新的复权因子调整，
    使得价格可以直接比较（反映真实成本）。

    Args:
        df: 包含 adj_factor 和 close_adj/open_adj/high_adj/low_adj 的 DataFrame

    Returns:
        添加了 forward_close/forward_open/forward_high/forward_low 列的 DataFrame
    """
    if df is None or len(df) == 0:
        return df

    if "adj_factor" not in df.columns or "close_adj" not in df.columns:
        logger.warning("Missing required columns for forward adjustment")
        return df

    df = df.copy()

    # 获取最新的复权因子（通常是最近一个交易日的因子）
    latest_adj_factor = df["adj_factor"].iloc[-1]

    if latest_adj_factor <= 0 or pd.isna(latest_adj_factor):
        logger.warning(f"Invalid latest adj_factor: {latest_adj_factor}, using 1.0")
        latest_adj_factor = 1.0

    # 安全处理：避免除以零
    adj_factor_safe = df["adj_factor"].replace(0.0, 1.0)

    # 计算前复权价格
    # 前复权价 = 后复权价 * (最新因子 / 历史因子)
    # 这样可以归一化所有价格到最新因子水平
    adj_ratio = latest_adj_factor / adj_factor_safe

    df["forward_close"] = df["close_adj"] * adj_ratio
    df["forward_open"] = df["open_adj"] * adj_ratio
    df["forward_high"] = df["high_adj"] * adj_ratio
    df["forward_low"] = df["low_adj"] * adj_ratio

    logger.debug(
        f"Forward adjustment: latest_adj_factor={latest_adj_factor}, "
        f"date_range={df['date'].iloc[0]}~{df['date'].iloc[-1]}"
    )

    return df


def convert_to_post_adj(df: pd.DataFrame, base_adj_factor: float = 1.0) -> pd.DataFrame:
    """
    将前复权价格转换为后复权价格

    后复权价格 = 前复权价格 / (最新adj_factor / 基准因子)

    Args:
        df: 包含前复权价格列的 DataFrame
        base_adj_factor: 基准复权因子，默认为 1.0

    Returns:
        转换后的 DataFrame
    """
    if df is None or len(df) == 0:
        return df

    df = df.copy()

    # 获取最新的前复权因子（用于归一化）
    if "adj_factor" in df.columns:
        latest_adj_factor = df["adj_factor"].iloc[-1]
    else:
        latest_adj_factor = 1.0

    if latest_adj_factor <= 0 or pd.isna(latest_adj_factor):
        latest_adj_factor = 1.0

    # 计算转换比率
    adj_ratio = latest_adj_factor / base_adj_factor

    # 如果有前复权列，进行转换
    if "forward_close" in df.columns:
        df["close_adj"] = df["forward_close"] / adj_ratio
        df["open_adj"] = df["forward_open"] / adj_ratio
        df["high_adj"] = df["forward_high"] / adj_ratio
        df["low_adj"] = df["forward_low"] / adj_ratio

    return df


def get_current_adj_factor(df: pd.DataFrame) -> float:
    """
    获取最新的复权因子

    Args:
        df: 包含 adj_factor 列的 DataFrame

    Returns:
        最新的复权因子值
    """
    if df is None or len(df) == 0:
        return 1.0

    if "adj_factor" not in df.columns:
        return 1.0

    return df["adj_factor"].iloc[-1]
