"""换手率因子

提供换手率相关因子:
- turnover: 日换手率
- turnover_ma20: 20日平均换手率
- turnover_std20: 20日换手率波动
"""

import numpy as np
import pandas as pd

from ..factor_base import FactorBase


class TurnoverRate(FactorBase):
    """
    日换手率因子

    换手率 = 成交量 / 流通股本 * 100%
    """

    name = "turnover"
    category = "turnover"
    description = "日换手率 (成交量/流通股本*100%)"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算日换手率

        Returns:
            DataFrame: date, code, factor_value
        """
        required_cols = ["date", "code", "volume", "outstanding_share"]
        self.validate_data(data, required_cols)

        df = data.copy()
        df = df.sort_values(["code", "date"])

        # 计算换手率
        df["factor_value"] = df["volume"] / df["outstanding_share"] * 100

        # 取最后一天的值
        result = []
        for code, group in df.groupby("code"):
            if len(group) > 0:
                result.append({
                    "date": group["date"].iloc[-1],
                    "code": code,
                    "factor_value": group["factor_value"].iloc[-1]
                })

        return pd.DataFrame(result)


class TurnoverMA20(FactorBase):
    """
    20日平均换手率因子
    """

    name = "turnover_ma20"
    category = "turnover"
    description = "20日平均换手率"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算20日平均换手率

        Returns:
            DataFrame: date, code, factor_value
        """
        required_cols = ["date", "code", "volume", "outstanding_share"]
        self.validate_data(data, required_cols)

        df = data.copy()
        df = df.sort_values(["code", "date"])

        # 计算日换手率
        df["daily_turnover"] = df["volume"] / df["outstanding_share"] * 100

        # 计算20日移动平均
        df["factor_value"] = df.groupby("code")["daily_turnover"].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean()
        )

        # 取最后一天的值
        result = []
        for code, group in df.groupby("code"):
            if len(group) > 0:
                result.append({
                    "date": group["date"].iloc[-1],
                    "code": code,
                    "factor_value": group["factor_value"].iloc[-1]
                })

        return pd.DataFrame(result)


class TurnoverStd20(FactorBase):
    """
    20日换手率波动因子
    """

    name = "turnover_std20"
    category = "turnover"
    description = "20日换手率标准差"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算20日换手率标准差

        Returns:
            DataFrame: date, code, factor_value
        """
        required_cols = ["date", "code", "volume", "outstanding_share"]
        self.validate_data(data, required_cols)

        df = data.copy()
        df = df.sort_values(["code", "date"])

        # 计算日换手率
        df["daily_turnover"] = df["volume"] / df["outstanding_share"] * 100

        # 计算20日标准差
        df["factor_value"] = df.groupby("code")["daily_turnover"].transform(
            lambda x: x.rolling(window=20, min_periods=1).std()
        )

        # 取最后一天的值
        result = []
        for code, group in df.groupby("code"):
            if len(group) > 0:
                result.append({
                    "date": group["date"].iloc[-1],
                    "code": code,
                    "factor_value": group["factor_value"].iloc[-1]
                })

        return pd.DataFrame(result)


class AmountDaily(FactorBase):
    """
    日成交额因子
    """

    name = "amount"
    category = "turnover"
    description = "日成交额 (万元)"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算日成交额

        Returns:
            DataFrame: date, code, factor_value
        """
        required_cols = ["date", "code", "amount"]
        self.validate_data(data, required_cols)

        df = data.copy()
        df = df.sort_values(["code", "date"])

        # 转换为万元
        df["factor_value"] = df["amount"] / 10000

        # 取最后一天的值
        result = []
        for code, group in df.groupby("code"):
            if len(group) > 0:
                result.append({
                    "date": group["date"].iloc[-1],
                    "code": code,
                    "factor_value": group["factor_value"].iloc[-1]
                })

        return pd.DataFrame(result)


__all__ = [
    "TurnoverRate",
    "TurnoverMA20",
    "TurnoverStd20",
    "AmountDaily",
]
