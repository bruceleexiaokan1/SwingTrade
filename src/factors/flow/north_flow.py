"""北向资金因子

提供北向资金相关因子:
- north_hold_chg: 北向持股变化
- north_hold_ratio: 北向持股占比
"""

import numpy as np
import pandas as pd

from ..factor_base import FactorBase


class NorthHoldChange(FactorBase):
    """
    北向持股变化因子

    北向持股变化 = 当日持股 - 前日持股
    """

    name = "north_hold_chg"
    category = "flow"
    description = "北向持股变化量"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算北向持股变化

        数据要求:
            - north_hold: 北向持股量 (可选)

        Returns:
            DataFrame: date, code, factor_value
        """
        required_cols = ["date", "code"]
        self.validate_data(data, required_cols)

        df = data.copy()
        df = df.sort_values(["code", "date"])

        if "north_hold" in df.columns:
            # 计算持股变化
            df["factor_value"] = df.groupby("code")["north_hold"].diff()
        else:
            df["factor_value"] = np.nan

        # 取最后一天的值
        result = []
        for code, group in df.groupby("code"):
            if len(group) > 0:
                val = group["factor_value"].iloc[-1]
                result.append({
                    "date": group["date"].iloc[-1],
                    "code": code,
                    "factor_value": val if not pd.isna(val) else 0.0
                })

        return pd.DataFrame(result)


class NorthHoldRatio(FactorBase):
    """
    北向持股占比因子

    北向持股占比 = 北向持股 / 总股本 * 100
    """

    name = "north_hold_ratio"
    category = "flow"
    description = "北向持股占比 (%)"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算北向持股占比
        """
        required_cols = ["date", "code"]
        self.validate_data(data, required_cols)

        df = data.copy()

        if "north_hold" in df.columns and "outstanding_share" in df.columns:
            df["factor_value"] = df["north_hold"] / df["outstanding_share"] * 100
        else:
            df["factor_value"] = np.nan

        result = []
        for code, group in df.groupby("code"):
            if len(group) > 0:
                val = group["factor_value"].iloc[-1]
                result.append({
                    "date": group["date"].iloc[-1],
                    "code": code,
                    "factor_value": val if not pd.isna(val) else 0.0
                })

        return pd.DataFrame(result)


__all__ = [
    "NorthHoldChange",
    "NorthHoldRatio",
]
