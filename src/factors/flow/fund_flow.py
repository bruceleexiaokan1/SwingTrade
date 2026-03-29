"""资金流因子

提供资金流相关因子:
- fund_flow_main: 主力资金净流入占比
- fund_flow_big: 大单资金净流入占比
"""

import numpy as np
import pandas as pd

from ..factor_base import FactorBase


class FundFlowMain(FactorBase):
    """
    主力资金净流入占比因子

    主力资金占比 = 主力净流入 / 成交额 * 100
    """

    name = "fund_flow_main"
    category = "flow"
    description = "主力资金净流入占比 (%)"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算主力资金占比

        数据要求:
            - main_net_inflow: 主力净流入 (可选)
            - amount: 成交额

        Returns:
            DataFrame: date, code, factor_value
        """
        required_cols = ["date", "code"]
        self.validate_data(data, required_cols)

        df = data.copy()

        # 检查是否有main_net_inflow列
        if "main_net_inflow" in df.columns and "amount" in df.columns:
            # 计算主力资金占比
            df["factor_value"] = df["main_net_inflow"] / df["amount"] * 100
        else:
            # 如果没有主力资金数据，返回NaN
            df["factor_value"] = np.nan

        # 取最后一天的值
        result = []
        for code, group in df.groupby("code"):
            if len(group) > 0:
                result.append({
                    "date": group["date"].iloc[-1],
                    "code": code,
                    "factor_value": group["factor_value"].iloc[-1]
                    if not pd.isna(group["factor_value"].iloc[-1])
                    else 0.0
                })

        return pd.DataFrame(result)


class FundFlowBig(FactorBase):
    """
    大单资金净流入占比因子
    """

    name = "fund_flow_big"
    category = "flow"
    description = "大单资金净流入占比 (%)"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算大单资金占比
        """
        required_cols = ["date", "code"]
        self.validate_data(data, required_cols)

        df = data.copy()

        if "big_net_inflow" in df.columns and "amount" in df.columns:
            df["factor_value"] = df["big_net_inflow"] / df["amount"] * 100
        else:
            df["factor_value"] = np.nan

        result = []
        for code, group in df.groupby("code"):
            if len(group) > 0:
                result.append({
                    "date": group["date"].iloc[-1],
                    "code": code,
                    "factor_value": group["factor_value"].iloc[-1]
                    if not pd.isna(group["factor_value"].iloc[-1])
                    else 0.0
                })

        return pd.DataFrame(result)


__all__ = [
    "FundFlowMain",
    "FundFlowBig",
]
