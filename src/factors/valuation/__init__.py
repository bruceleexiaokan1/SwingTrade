"""估值因子模块

提供估值相关因子:
- pe: 市盈率
- pb: 市净率
- ep: 市盈率倒数

支持从日线数据+财务数据计算
包含穿越时间风险防护: 财务数据仅在披露日后才可用
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from ..factor_base import FactorBase


# 财务数据缓存目录
FINANCIAL_CACHE_DIR = Path(__file__).parent.parent.parent.parent / "StockData" / "financial"


class PETTM(FactorBase):
    """
    TTM市盈率因子

    PE = 股价 / EPS_TTM

    数据依赖:
        - close: 股价 (来自日线)
        - eps_ttm: TTm每股收益 (来自财务数据)

    穿越风险防护:
        - 财报数据仅在披露截止日后才可用
    """

    name = "pe_ttm"
    category = "valuation"
    description = "TTM市盈率"

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__()
        self.cache_dir = Path(cache_dir) if cache_dir else FINANCIAL_CACHE_DIR

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算TTM市盈率

        Args:
            data: 日线数据 (需要包含 date, code, close)

        Returns:
            DataFrame: date, code, factor_value
        """
        required_cols = ["date", "code", "close"]
        self.validate_data(data, required_cols)

        df = data.copy()
        df = df.sort_values(["code", "date"])
        df["date"] = pd.to_datetime(df["date"])
        df["code_raw"] = df["code"].str.replace(r'\.(SH|SZ)', '', regex=True)

        as_of_date = df["date"].max()
        eps_data = self._load_financial_data(as_of_date)

        if eps_data.empty:
            df["factor_value"] = np.nan
        else:
            df = df.merge(eps_data, left_on="code_raw", right_on="code", how="left", suffixes=('', '_fin'))
            df["factor_value"] = df["close"] / df["eps_value"]

        result = []
        for code, group in df.groupby("code"):
            if len(group) > 0:
                last_row = group.iloc[-1]
                result.append({
                    "date": last_row["date"],
                    "code": code,
                    "factor_value": last_row["factor_value"]
                })

        return pd.DataFrame(result)

    def _load_financial_data(self, as_of_date: pd.Timestamp) -> pd.DataFrame:
        """
        加载财务EPS数据 (穿越防护)

        只返回在as_of_date之前已披露的财务数据
        """
        all_data = []

        if not self.cache_dir.exists():
            return pd.DataFrame()

        for cache_file in self.cache_dir.glob("*.parquet"):
            code = cache_file.stem
            try:
                fin_df = pd.read_parquet(cache_file)
                if "报告期" in fin_df.columns and "eps_basic" in fin_df.columns:
                    # 穿越防护: 检查available_date
                    if "available_date" in fin_df.columns:
                        fin_df = fin_df[fin_df["available_date"] <= as_of_date]
                        if fin_df.empty:
                            continue

                    fin_df = fin_df.sort_values("报告期", ascending=False)
                    latest = fin_df.iloc[0]
                    all_data.append({
                        "code": code,
                        "eps_value": latest["eps_basic"]
                    })
            except Exception:
                pass

        if not all_data:
            return pd.DataFrame()

        return pd.DataFrame(all_data)


class PBTMR(FactorBase):
    """
    市净率因子

    PB = 股价 / 每股净资产
    穿越风险防护
    """

    name = "pb_tmr"
    category = "valuation"
    description = "市净率"

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__()
        self.cache_dir = Path(cache_dir) if cache_dir else FINANCIAL_CACHE_DIR

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算市净率"""
        required_cols = ["date", "code", "close"]
        self.validate_data(data, required_cols)

        df = data.copy()
        df = df.sort_values(["code", "date"])
        df["date"] = pd.to_datetime(df["date"])
        df["code_raw"] = df["code"].str.replace(r'\.(SH|SZ)', '', regex=True)

        as_of_date = df["date"].max()
        bps_data = self._load_financial_data(as_of_date)

        if bps_data.empty:
            df["factor_value"] = np.nan
        else:
            df = df.merge(bps_data, left_on="code_raw", right_on="code", how="left", suffixes=('', '_fin'))
            df["factor_value"] = df["close"] / df["bps_value"]

        result = []
        for code, group in df.groupby("code"):
            if len(group) > 0:
                last_row = group.iloc[-1]
                result.append({
                    "date": last_row["date"],
                    "code": code,
                    "factor_value": last_row["factor_value"]
                })

        return pd.DataFrame(result)

    def _load_financial_data(self, as_of_date: pd.Timestamp) -> pd.DataFrame:
        """加载财务BPS数据 (穿越防护)"""
        all_data = []

        if not self.cache_dir.exists():
            return pd.DataFrame()

        for cache_file in self.cache_dir.glob("*.parquet"):
            code = cache_file.stem
            try:
                fin_df = pd.read_parquet(cache_file)
                if "报告期" in fin_df.columns and "bps" in fin_df.columns:
                    if "available_date" in fin_df.columns:
                        fin_df = fin_df[fin_df["available_date"] <= as_of_date]
                        if fin_df.empty:
                            continue

                    fin_df = fin_df.sort_values("报告期", ascending=False)
                    latest = fin_df.iloc[0]
                    all_data.append({
                        "code": code,
                        "bps_value": latest["bps"]
                    })
            except Exception:
                pass

        if not all_data:
            return pd.DataFrame()

        return pd.DataFrame(all_data)


class EPInverse(FactorBase):
    """
    市盈率倒数因子 (EP)

    EP = 1 / PE = EPS / Price
    亏损股票的EP为负值，保留在序列中
    穿越风险防护
    """

    name = "ep"
    category = "valuation"
    description = "市盈率倒数 (EP)"

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__()
        self.cache_dir = Path(cache_dir) if cache_dir else FINANCIAL_CACHE_DIR

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算EP"""
        required_cols = ["date", "code", "close"]
        self.validate_data(data, required_cols)

        df = data.copy()
        df = df.sort_values(["code", "date"])
        df["date"] = pd.to_datetime(df["date"])
        df["code_raw"] = df["code"].str.replace(r'\.(SH|SZ)', '', regex=True)

        as_of_date = df["date"].max()
        eps_data = self._load_financial_data(as_of_date)

        if eps_data.empty:
            df["factor_value"] = np.nan
        else:
            df = df.merge(eps_data, left_on="code_raw", right_on="code", how="left", suffixes=('', '_fin'))
            df["factor_value"] = df["eps_value"] / df["close"]

        result = []
        for code, group in df.groupby("code"):
            if len(group) > 0:
                last_row = group.iloc[-1]
                result.append({
                    "date": last_row["date"],
                    "code": code,
                    "factor_value": last_row["factor_value"]
                })

        return pd.DataFrame(result)

    def _load_financial_data(self, as_of_date: pd.Timestamp) -> pd.DataFrame:
        """加载财务EPS数据 (穿越防护)"""
        all_data = []

        if not self.cache_dir.exists():
            return pd.DataFrame()

        for cache_file in self.cache_dir.glob("*.parquet"):
            code = cache_file.stem
            try:
                fin_df = pd.read_parquet(cache_file)
                if "报告期" in fin_df.columns and "eps_basic" in fin_df.columns:
                    if "available_date" in fin_df.columns:
                        fin_df = fin_df[fin_df["available_date"] <= as_of_date]
                        if fin_df.empty:
                            continue

                    fin_df = fin_df.sort_values("报告期", ascending=False)
                    latest = fin_df.iloc[0]
                    all_data.append({
                        "code": code,
                        "eps_value": latest["eps_basic"]
                    })
            except Exception:
                pass

        if not all_data:
            return pd.DataFrame()

        return pd.DataFrame(all_data)


__all__ = [
    "PETTM",
    "PBTMR",
    "EPInverse",
]
