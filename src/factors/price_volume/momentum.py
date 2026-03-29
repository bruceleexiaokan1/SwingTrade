"""动量因子

提供价格动量相关因子:
- ret_3m: 3个月收益率
- ret_6m: 6个月收益率
- ret_12m: 12个月收益率
- rs_120: 120日相对强度
"""

import numpy as np
import pandas as pd
from typing import Optional

from ..factor_base import FactorBase
from ..exceptions import FactorCalculationError


class MomentumRet3M(FactorBase):
    """
    3个月动量因子

    计算窗口内的收益率
    """

    name = "ret_3m"
    category = "momentum"
    description = "3个月收益率动量"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算3个月动量

        Returns:
            DataFrame: date, code, factor_value
        """
        self.validate_data(data)

        df = data.copy()
        df = df.sort_values(["code", "date"])

        # 计算3个月前收盘价 (约60个交易日)
        periods = 60

        result = []
        for code, group in df.groupby("code"):
            if len(group) >= periods + 1:
                price_now = group["close"].iloc[-1]
                price_then = group["close"].iloc[-(periods + 1)]
                ret = (price_now / price_then) - 1
                result.append({
                    "date": group["date"].iloc[-1],
                    "code": code,
                    "factor_value": ret
                })

        return pd.DataFrame(result)


class MomentumRet6M(FactorBase):
    """
    6个月动量因子
    """

    name = "ret_6m"
    category = "momentum"
    description = "6个月收益率动量"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算6个月动量"""
        self.validate_data(data)

        df = data.copy()
        df = df.sort_values(["code", "date"])

        periods = 120

        result = []
        for code, group in df.groupby("code"):
            if len(group) >= periods + 1:
                price_now = group["close"].iloc[-1]
                price_then = group["close"].iloc[-(periods + 1)]
                ret = (price_now / price_then) - 1
                result.append({
                    "date": group["date"].iloc[-1],
                    "code": code,
                    "factor_value": ret
                })

        return pd.DataFrame(result)


class MomentumRet12M(FactorBase):
    """
    12个月动量因子
    """

    name = "ret_12m"
    category = "momentum"
    description = "12个月收益率动量"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算12个月动量"""
        self.validate_data(data)

        df = data.copy()
        df = df.sort_values(["code", "date"])

        periods = 240

        result = []
        for code, group in df.groupby("code"):
            if len(group) >= periods + 1:
                price_now = group["close"].iloc[-1]
                price_then = group["close"].iloc[-(periods + 1)]
                ret = (price_now / price_then) - 1
                result.append({
                    "date": group["date"].iloc[-1],
                    "code": code,
                    "factor_value": ret
                })

        return pd.DataFrame(result)


class MomentumRS120(FactorBase):
    """
    120日相对强度因子

    RS = 个股收益率 - 基准指数收益率
    """

    name = "rs_120"
    category = "momentum"
    description = "120日相对强度 (相对于基准指数)"

    def __init__(self, index_code: str = "000300.SH"):
        """
        初始化RS因子

        Args:
            index_code: 基准指数代码，默认沪深300
        """
        super().__init__()
        self.index_code = index_code

    def calculate(self, data: pd.DataFrame, index_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        计算相对强度

        Args:
            data: 个股数据
            index_data: 指数数据，如果为None则用data自己计算

        Returns:
            DataFrame: date, code, factor_value
        """
        self.validate_data(data)

        df = data.copy()
        df = df.sort_values(["code", "date"])

        periods = 120

        # 准备指数收益率
        idx_ret = 0
        if index_data is not None:
            idx_df = index_data[index_data["code"] == self.index_code].copy()
            if len(idx_df) >= periods + 1:
                idx_df = idx_df.sort_values("date")
                idx_ret = (idx_df["close"].iloc[-1] / idx_df["close"].iloc[-(periods + 1)]) - 1

        result = []
        for code, group in df.groupby("code"):
            if len(group) >= periods + 1:
                stock_ret = (group["close"].iloc[-1] / group["close"].iloc[-(periods + 1)]) - 1
                rs = stock_ret - idx_ret
                result.append({
                    "date": group["date"].iloc[-1],
                    "code": code,
                    "factor_value": rs
                })

        return pd.DataFrame(result)


__all__ = [
    "MomentumRet3M",
    "MomentumRet6M",
    "MomentumRet12M",
    "MomentumRS120",
]
