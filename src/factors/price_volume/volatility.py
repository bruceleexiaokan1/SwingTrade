"""波动率因子

提供波动率相关因子:
- vol_20: 20日历史波动率
- atr_14_pct: ATR百分比
- beta_60: 60日Beta
"""

import numpy as np
import pandas as pd
from typing import Optional

from ..factor_base import FactorBase


class VolatilityVol20(FactorBase):
    """
    20日历史波动率因子

    使用日收益率标准差年化
    """

    name = "vol_20"
    category = "volatility"
    description = "20日历史波动率 (年化)"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算20日波动率

        Returns:
            DataFrame: date, code, factor_value
        """
        self.validate_data(data)

        df = data.copy()
        df = df.sort_values(["code", "date"])

        # 计算日收益率
        df["return"] = df.groupby("code")["close"].pct_change()

        # 计算20日滚动波动率 (年化因子: sqrt(252))
        def calc_vol(group):
            if len(group) < 20:
                return pd.Series({"factor_value": np.nan})

            vol = group["return"].iloc[-20:].std() * np.sqrt(252)
            return pd.Series({"factor_value": vol})

        # 对每只股票取最后一天的波动率
        result = []
        for code, group in df.groupby("code"):
            if len(group) >= 20:
                vol = group["return"].iloc[-20:].std() * np.sqrt(252)
                result.append({
                    "date": group["date"].iloc[-1],
                    "code": code,
                    "factor_value": vol
                })

        return pd.DataFrame(result)


class VolatilityATR14Pct(FactorBase):
    """
    14日ATR百分比因子

    ATR = Average True Range
    ATR百分比 = ATR / 收盘价 * 100
    """

    name = "atr_14_pct"
    category = "volatility"
    description = "14日ATR百分比 (ATR/收盘价*100)"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算ATR百分比

        Returns:
            DataFrame: date, code, factor_value
        """
        required_cols = ["date", "code", "close", "high", "low"]
        self.validate_data(data, required_cols)

        df = data.copy()
        df = df.sort_values(["code", "date"])

        # 计算True Range
        df["prev_close"] = df.groupby("code")["close"].shift(1)
        df["tr1"] = df["high"] - df["low"]
        df["tr2"] = (df["high"] - df["prev_close"]).abs()
        df["tr3"] = (df["low"] - df["prev_close"]).abs()
        df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)

        # 计算14日ATR
        df["atr"] = df.groupby("code")["tr"].transform(
            lambda x: x.rolling(window=14, min_periods=1).mean()
        )

        # 计算ATR百分比
        df["atr_pct"] = df["atr"] / df["close"] * 100

        # 取最后一天的值
        result = []
        for code, group in df.groupby("code"):
            if len(group) > 0:
                result.append({
                    "date": group["date"].iloc[-1],
                    "code": code,
                    "factor_value": group["atr_pct"].iloc[-1]
                })

        return pd.DataFrame(result)


class RiskBeta60(FactorBase):
    """
    60日Beta因子

    Beta = Cov(个股收益, 指数收益) / Var(指数收益)
    """

    name = "beta_60"
    category = "volatility"
    description = "60日Beta (相对于沪深300)"

    def __init__(self, index_code: str = "000300.SH"):
        super().__init__()
        self.index_code = index_code

    def calculate(self, data: pd.DataFrame, index_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        计算Beta

        Args:
            data: 个股数据
            index_data: 指数数据

        Returns:
            DataFrame: date, code, factor_value
        """
        required_cols = ["date", "code", "close"]
        self.validate_data(data, required_cols)

        df = data.copy()
        df = df.sort_values(["code", "date"])

        # 计算个股日收益率
        df["return"] = df.groupby("code")["close"].pct_change()

        # 获取指数收益率
        if index_data is not None:
            idx_df = index_data[index_data["code"] == self.index_code].copy()
            idx_df = idx_df.sort_values("date")
            idx_df["idx_return"] = idx_df["close"].pct_change()
            idx_df = idx_df[["date", "idx_return"]]
        else:
            # 如果没有指数数据，用市场平均收益代替
            market_return = df.groupby("date")["return"].mean().reset_index()
            market_return.columns = ["date", "idx_return"]
            idx_df = market_return

        # 合并
        df = df.merge(idx_df, on="date", how="left")

        # 计算60日Beta
        def calc_beta(group):
            if len(group) < 20:
                return np.nan

            valid = group.dropna(subset=["return", "idx_return"])
            if len(valid) < 20:
                return np.nan

            cov = valid["return"].cov(valid["idx_return"])
            var = valid["idx_return"].var()

            if var == 0 or pd.isna(var):
                return np.nan

            return cov / var

        # 取最后一天的值
        result = []
        for code, group in df.groupby("code"):
            if len(group) >= 20:
                valid = group.dropna(subset=["return", "idx_return"])
                if len(valid) >= 20:
                    cov = valid["return"].cov(valid["idx_return"])
                    var = valid["idx_return"].var()
                    if var > 0:
                        beta = cov / var
                    else:
                        beta = np.nan
                else:
                    beta = np.nan

                result.append({
                    "date": group["date"].iloc[-1],
                    "code": code,
                    "factor_value": beta
                })

        return pd.DataFrame(result)


__all__ = [
    "VolatilityVol20",
    "VolatilityATR14Pct",
    "RiskBeta60",
]
