"""质量因子模块

提供质量相关因子:
- roe: 净资产收益率
- roa: 资产收益率
- gross_margin: 毛利率

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


class ROETTM(FactorBase):
    """
    TTM净资产收益率因子

    ROE = 净利润 / 平均净资产 * 100%

    穿越风险防护:
    - 财报数据仅在披露截止日后才可用
    - Q1(3月报): 4月30日后
    - Q2(6月报): 8月31日后
    - Q3(9月报): 10月31日后
    - Q4(年报): 次年4月30日后
    """

    name = "roe_ttm"
    category = "quality"
    description = "TTM净资产收益率 (%)"

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__()
        self.cache_dir = Path(cache_dir) if cache_dir else FINANCIAL_CACHE_DIR

    def calculate(self, data: pd.DataFrame, as_of_date: Optional[str] = None) -> pd.DataFrame:
        """
        计算TTM ROE

        Args:
            data: 日线数据 (需要包含 date, code)
            as_of_date: 计算因子值的截止日期，用于穿越风险防护
                       如果不提供，使用data中最晚日期

        Returns:
            DataFrame: date, code, factor_value
        """
        required_cols = ["date", "code"]
        self.validate_data(data, required_cols)

        df = data.copy()
        df = df.sort_values(["code", "date"])
        df["date"] = pd.to_datetime(df["date"])
        df["code_raw"] = df["code"].str.replace(r'\.(SH|SZ)', '', regex=True)

        # 确定计算日期
        if as_of_date is None:
            as_of_date = df["date"].max()
        else:
            as_of_date = pd.to_datetime(as_of_date)

        roe_data = self._load_financial_data(as_of_date)

        if roe_data.empty:
            df["factor_value"] = np.nan
        else:
            df = df.merge(roe_data, left_on="code_raw", right_on="code", how="left", suffixes=('', '_fin'))
            df["factor_value"] = df["roe_value"]

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
        加载财务ROE数据 (仅加载在as_of_date之前已披露的数据)

        Args:
            as_of_date: 截止日期，只返回 available_date <= as_of_date 的数据

        Returns:
            DataFrame: code, roe_value (最新的已披露ROE)
        """
        all_data = []

        if not self.cache_dir.exists():
            return pd.DataFrame()

        for cache_file in self.cache_dir.glob("*.parquet"):
            code = cache_file.stem
            try:
                fin_df = pd.read_parquet(cache_file)
                if "报告期" in fin_df.columns and "roe" in fin_df.columns:
                    # 检查是否有 available_date 字段 (穿越防护)
                    if "available_date" in fin_df.columns:
                        # 过滤: 只使用在as_of_date之前已披露的数据
                        fin_df = fin_df[fin_df["available_date"] <= as_of_date]
                        if fin_df.empty:
                            continue

                    # 取最新的可用数据
                    fin_df = fin_df.sort_values("报告期", ascending=False)
                    latest = fin_df.iloc[0]
                    all_data.append({
                        "code": code,
                        "roe_value": latest["roe"]
                    })
            except Exception:
                pass

        if not all_data:
            return pd.DataFrame()

        return pd.DataFrame(all_data)


class ROATTM(FactorBase):
    """
    TTM资产收益率因子

    ROA = 净利润 / 总资产 * 100%

    注意: 当前实现使用 ROE * 0.5 作为简化
    完整实现需要总资产数据
    包含穿越时间风险防护
    """

    name = "roa_ttm"
    category = "quality"
    description = "TTM资产收益率 (%)"

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__()
        self.cache_dir = Path(cache_dir) if cache_dir else FINANCIAL_CACHE_DIR

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算TTM ROA"""
        required_cols = ["date", "code"]
        self.validate_data(data, required_cols)

        df = data.copy()
        df = df.sort_values(["code", "date"])
        df["date"] = pd.to_datetime(df["date"])
        df["code_raw"] = df["code"].str.replace(r'\.(SH|SZ)', '', regex=True)

        as_of_date = df["date"].max()
        roa_data = self._load_financial_data(as_of_date)

        if roa_data.empty:
            df["factor_value"] = np.nan
        else:
            df = df.merge(roa_data, left_on="code_raw", right_on="code", how="left", suffixes=('', '_fin'))
            df["factor_value"] = df["roe_value"] * 0.5  # 简化估计

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
        """加载财务数据 (穿越防护)"""
        all_data = []

        if not self.cache_dir.exists():
            return pd.DataFrame()

        for cache_file in self.cache_dir.glob("*.parquet"):
            code = cache_file.stem
            try:
                fin_df = pd.read_parquet(cache_file)
                if "报告期" in fin_df.columns and "roe" in fin_df.columns:
                    if "available_date" in fin_df.columns:
                        fin_df = fin_df[fin_df["available_date"] <= as_of_date]
                        if fin_df.empty:
                            continue

                    fin_df = fin_df.sort_values("报告期", ascending=False)
                    latest = fin_df.iloc[0]
                    all_data.append({
                        "code": code,
                        "roe_value": latest["roe"]
                    })
            except Exception:
                pass

        if not all_data:
            return pd.DataFrame()

        return pd.DataFrame(all_data)


class GrossMargin(FactorBase):
    """
    毛利率因子

    毛利率 = (营业收入 - 营业成本) / 营业收入 * 100%
    包含穿越时间风险防护
    """

    name = "gross_margin"
    category = "quality"
    description = "毛利率 (%)"

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__()
        self.cache_dir = Path(cache_dir) if cache_dir else FINANCIAL_CACHE_DIR

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算毛利率"""
        required_cols = ["date", "code"]
        self.validate_data(data, required_cols)

        df = data.copy()
        df = df.sort_values(["code", "date"])
        df["date"] = pd.to_datetime(df["date"])
        df["code_raw"] = df["code"].str.replace(r'\.(SH|SZ)', '', regex=True)

        as_of_date = df["date"].max()
        margin_data = self._load_financial_data(as_of_date)

        if margin_data.empty:
            df["factor_value"] = np.nan
        else:
            df = df.merge(margin_data, left_on="code_raw", right_on="code", how="left", suffixes=('', '_fin'))
            df["factor_value"] = df["gross_margin_value"]

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
        """加载财务毛利率数据 (穿越防护)"""
        all_data = []

        if not self.cache_dir.exists():
            return pd.DataFrame()

        for cache_file in self.cache_dir.glob("*.parquet"):
            code = cache_file.stem
            try:
                fin_df = pd.read_parquet(cache_file)
                if "报告期" in fin_df.columns and "gross_margin" in fin_df.columns:
                    if "available_date" in fin_df.columns:
                        fin_df = fin_df[fin_df["available_date"] <= as_of_date]
                        if fin_df.empty:
                            continue

                    fin_df = fin_df.sort_values("报告期", ascending=False)
                    latest = fin_df.iloc[0]
                    all_data.append({
                        "code": code,
                        "gross_margin_value": latest["gross_margin"]
                    })
            except Exception:
                pass

        if not all_data:
            return pd.DataFrame()

        return pd.DataFrame(all_data)


class DebtRatio(FactorBase):
    """
    资产负债率因子

    资产负债率 = 负债合计 / 资产合计 * 100%
    包含穿越时间风险防护
    """

    name = "debt_ratio"
    category = "quality"
    description = "资产负债率 (%)"

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__()
        self.cache_dir = Path(cache_dir) if cache_dir else FINANCIAL_CACHE_DIR

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算资产负债率"""
        required_cols = ["date", "code"]
        self.validate_data(data, required_cols)

        df = data.copy()
        df = df.sort_values(["code", "date"])
        df["date"] = pd.to_datetime(df["date"])
        df["code_raw"] = df["code"].str.replace(r'\.(SH|SZ)', '', regex=True)

        as_of_date = df["date"].max()
        debt_data = self._load_financial_data(as_of_date)

        if debt_data.empty:
            df["factor_value"] = np.nan
        else:
            df = df.merge(debt_data, left_on="code_raw", right_on="code", how="left", suffixes=('', '_fin'))
            df["factor_value"] = df["debt_ratio_value"]

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
        """加载资产负债率数据 (穿越防护)"""
        all_data = []

        if not self.cache_dir.exists():
            return pd.DataFrame()

        for cache_file in self.cache_dir.glob("*.parquet"):
            code = cache_file.stem
            try:
                fin_df = pd.read_parquet(cache_file)
                if "报告期" in fin_df.columns and "debt_ratio" in fin_df.columns:
                    if "available_date" in fin_df.columns:
                        fin_df = fin_df[fin_df["available_date"] <= as_of_date]
                        if fin_df.empty:
                            continue

                    fin_df = fin_df.sort_values("报告期", ascending=False)
                    latest = fin_df.iloc[0]
                    all_data.append({
                        "code": code,
                        "debt_ratio_value": latest["debt_ratio"]
                    })
            except Exception:
                pass

        if not all_data:
            return pd.DataFrame()

        return pd.DataFrame(all_data)


__all__ = [
    "ROETTM",
    "ROATTM",
    "GrossMargin",
    "DebtRatio",
]
