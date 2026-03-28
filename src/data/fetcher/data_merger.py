"""数据合并工具

处理日线数据和复权因子的合并
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def merge_daily_with_adj_factor(
    daily_df: pd.DataFrame,
    adj_df: pd.DataFrame,
    code: str
) -> pd.DataFrame:
    """
    合并日线数据和复权因子，计算后复权价格

    Args:
        daily_df: 日线数据（包含 open, high, low, close, volume 等）
        adj_df: 复权因子数据（包含 date, adj_factor）
        code: 股票代码（用于日志）

    Returns:
        合并后的 DataFrame，添加 adj_factor 和后复权价格列
    """
    if daily_df is None or len(daily_df) == 0:
        return daily_df

    # 如果没有复权因子数据
    if adj_df is None or len(adj_df) == 0:
        logger.warning(f"No adj_factor data for {code}, using 1.0 as default")
        daily_df = daily_df.copy()
        daily_df["adj_factor"] = 1.0
        return daily_df

    # 合并复权因子
    df = daily_df.merge(adj_df, on="date", how="left")

    # 检查复权因子缺失情况
    missing_count = df["adj_factor"].isna().sum()
    total_count = len(df)

    if missing_count > 0:
        if missing_count == total_count:
            # 全部缺失，无法用前向填充
            logger.warning(f"All adj_factor missing for {code}, using 1.0 as default")
            df["adj_factor"] = 1.0
        else:
            # 部分缺失，用前向填充
            logger.warning(
                f"Partial adj_factor missing for {code}: {missing_count}/{total_count}, "
                f"using forward fill"
            )
            df["adj_factor"] = df["adj_factor"].ffill()

            # 如果第一个值就是 NaN，用 1.0 填充
            if pd.isna(df["adj_factor"].iloc[0]):
                df["adj_factor"].iloc[0] = 1.0

    # 计算后复权价格
    adj_factor = df["adj_factor"].values
    df["close_adj"] = df["close"] * adj_factor
    df["open_adj"] = df["open"] * adj_factor
    df["high_adj"] = df["high"] * adj_factor
    df["low_adj"] = df["low"] * adj_factor

    return df


def validate_date_freshness(df: pd.DataFrame, expected_date: str, code: str) -> bool:
    """
    验证数据新鲜度

    Args:
        df: 日线数据
        expected_date: 期望的日期 (YYYY-MM-DD)
        code: 股票代码（用于日志）

    Returns:
        True if date matches expected_date
    """
    if df is None or len(df) == 0:
        return False

    actual_dates = df["date"].unique()
    if len(actual_dates) == 1:
        actual_date = actual_dates[0]
        if actual_date != expected_date:
            logger.error(
                f"Date mismatch for {code}: expected {expected_date}, got {actual_date}"
            )
            return False
        logger.debug(f"Date validated for {code}: {actual_date}")
        return True
    else:
        logger.warning(f"Multiple dates in result for {code}: {actual_dates}")
        return True  # 允许范围查询返回多日数据
