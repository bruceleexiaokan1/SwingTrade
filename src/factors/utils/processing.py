"""因子清洗处理工具

提供标准化的因子数据清洗流程:
1. fillna - 缺失值处理
2. winsorize - 去极值 (MAD法)
3. standardize - Z-score标准化
4. neutralize - 中性化 (市值+行业回归残差)
"""

import numpy as np
import pandas as pd
from typing import Optional, Union

from ..exceptions import ProcessingError


class FactorProcessor:
    """
    因子处理器

    提供标准化的清洗流程

    示例:
        processor = FactorProcessor()

        # 完整流程
        result = processor.process(
            factor_df,
            market_cap=market_cap,
            industry=industry
        )

        # 或单独使用
        result = processor.fillna(factor_df, method="industry_median")
        result = processor.winsorize(factor_df, n_std=3)
        result = processor.standardize(factor_df)
        result = processor.neutralize(factor_df, market_cap, industry)
    """

    def __init__(self):
        """初始化处理器"""
        pass

    def fillna(
        self,
        df: pd.DataFrame,
        method: str = "industry_median",
        factor_col: str = "factor_value",
        industry_col: str = "industry"
    ) -> pd.DataFrame:
        """
        缺失值处理

        Args:
            df: 因子DataFrame
            method: 填充方法
                - "zero": 用0填充
                - "median": 用全局中位数填充
                - "industry_median": 按行业中位数填充
            factor_col: 因子列名
            industry_col: 行业列名

        Returns:
            pd.DataFrame: 处理后的DataFrame
        """
        result = df.copy()

        if factor_col not in result.columns:
            raise ProcessingError(f"Column {factor_col} not found")

        if result[factor_col].isna().sum() == 0:
            return result

        if method == "zero":
            result[factor_col] = result[factor_col].fillna(0)

        elif method == "median":
            median_val = result[factor_col].median()
            result[factor_col] = result[factor_col].fillna(median_val)

        elif method == "industry_median":
            if industry_col in result.columns:
                # 按行业中位数填充
                industry_medians = result.groupby(industry_col)[factor_col].transform(
                    "median"
                )
                result[factor_col] = result[factor_col].fillna(industry_medians)
                # 剩余NaN用全局中位数填充
                remaining_na = result[factor_col].isna()
                if remaining_na.sum() > 0:
                    global_median = result.loc[~remaining_na, factor_col].median()
                    result.loc[remaining_na, factor_col] = global_median
            else:
                # 没有行业信息，用中位数
                median_val = result[factor_col].median()
                result[factor_col] = result[factor_col].fillna(median_val)

        else:
            raise ProcessingError(f"Unknown fill method: {method}")

        return result

    def winsorize(
        self,
        df: pd.DataFrame,
        n_std: float = 3,
        factor_col: str = "factor_value"
    ) -> pd.DataFrame:
        """
        MAD去极值

        使用修正的Z-score方法:
        Z = (X - M) / (1.4826 * MAD)
        超出 ±n_std 的值被截断到边界

        Args:
            df: 因子DataFrame
            n_std: 截断阈值 (默认3倍)
            factor_col: 因子列名

        Returns:
            pd.DataFrame: 处理后的DataFrame
        """
        result = df.copy()

        if factor_col not in result.columns:
            raise ProcessingError(f"Column {factor_col} not found")

        # 计算中位数
        median = result[factor_col].median()

        # 计算MAD (Median Absolute Deviation)
        mad = (result[factor_col] - median).abs().median()

        if mad == 0 or pd.isna(mad):
            # MAD为0时，不做处理
            return result

        # 计算修正Z-score
        z_score = (result[factor_col] - median) / (1.4826 * mad)

        # 截断
        upper_bound = median + n_std * 1.4826 * mad
        lower_bound = median - n_std * 1.4826 * mad

        result[factor_col] = result[factor_col].clip(lower_bound, upper_bound)

        return result

    def standardize(
        self,
        df: pd.DataFrame,
        factor_col: str = "factor_value",
        groupby: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Z-score标准化

        对每个截面计算: z = (x - mean) / std

        Args:
            df: 因子DataFrame
            factor_col: 因子列名
            groupby: 分组列名 (如 "date")，None表示全局标准化

        Returns:
            pd.DataFrame: 标准化后的DataFrame
        """
        result = df.copy()

        if factor_col not in result.columns:
            raise ProcessingError(f"Column {factor_col} not found")

        if groupby is None:
            # 全局标准化
            mean = result[factor_col].mean()
            std = result[factor_col].std()
            if std > 0:
                result[factor_col] = (result[factor_col] - mean) / std
        else:
            # 按截面标准化
            def zscore(x):
                mean = x.mean()
                std = x.std()
                if std > 0:
                    return (x - mean) / std
                return x - mean

            result[factor_col] = result.groupby(groupby)[factor_col].transform(zscore)

        return result

    def neutralize(
        self,
        df: pd.DataFrame,
        market_cap: Optional[pd.Series] = None,
        industry: Optional[pd.Series] = None,
        factor_col: str = "factor_value"
    ) -> pd.DataFrame:
        """
        中性化

        对市值和行业做回归，取残差

        Args:
            df: 因子DataFrame
            market_cap: 市值序列 (index为code)
            industry: 行业序列 (index为code)
            factor_col: 因子列名

        Returns:
            pd.DataFrame: 中性化后的DataFrame
        """
        result = df.copy()

        if factor_col not in result.columns:
            raise ProcessingError(f"Column {factor_col} not found")

        if market_cap is None and industry is None:
            return result

        # 检查数据对齐
        if market_cap is not None:
            if not isinstance(market_cap, pd.Series):
                raise ProcessingError("market_cap must be a pandas Series")
            result = result.set_index("code")
            result["ln_market_cap"] = np.log(market_cap.reindex(result.index))
            result["ln_market_cap"] = result["ln_market_cap"].fillna(
                result["ln_market_cap"].median()
            )
        else:
            result["ln_market_cap"] = 0

        if industry is not None:
            if not isinstance(industry, pd.Series):
                raise ProcessingError("industry must be a pandas Series")
            # 创建行业哑变量
            industry_dummies = pd.get_dummies(industry.reindex(result.index), prefix="ind")
            industry_cols = industry_dummies.columns.tolist()
            result = pd.concat([result, industry_dummies], axis=1)
        else:
            industry_cols = []

        # 回归取残差
        from sklearn.linear_model import LinearRegression

        y = result[factor_col].values

        if industry_cols:
            X = result[["ln_market_cap"] + industry_cols].values
        else:
            X = result[["ln_market_cap"]].values

        # 去除NaN
        valid_mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))
        if valid_mask.sum() > 0:
            model = LinearRegression()
            model.fit(X[valid_mask], y[valid_mask])
            y_pred = model.predict(X)
            result[factor_col] = y - y_pred
        else:
            result[factor_col] = 0

        result = result.reset_index()

        # 清理临时列
        if "ln_market_cap" in result.columns:
            result = result.drop(columns=["ln_market_cap"])

        # 清理行业哑变量
        cols_to_drop = [c for c in result.columns if c.startswith("ind_")]
        if cols_to_drop:
            result = result.drop(columns=cols_to_drop)

        return result

    def process(
        self,
        df: pd.DataFrame,
        market_cap: Optional[pd.Series] = None,
        industry: Optional[pd.Series] = None,
        factor_col: str = "factor_value",
        industry_col: str = "industry",
        fillna_method: str = "industry_median",
        winsorize_std: float = 3,
        neutralize: bool = True
    ) -> pd.DataFrame:
        """
        完整处理流水线

        顺序执行:
        1. fillna (行业均值填充)
        2. winsorize (MAD去极值)
        3. standardize (Z-score标准化)
        4. neutralize (可选, 市值+行业中性化)

        Args:
            df: 因子DataFrame
            market_cap: 市值序列
            industry: 行业序列
            factor_col: 因子列名
            industry_col: 行业列名 (用于fillna)
            fillna_method: 缺失值填充方法
            winsorize_std: 去极值阈值
            neutralize: 是否做中性化

        Returns:
            pd.DataFrame: 处理后的DataFrame
        """
        # 1. 缺失值处理
        result = self.fillna(df, method=fillna_method, industry_col=industry_col)

        # 2. 去极值
        result = self.winsorize(result, n_std=winsorize_std)

        # 3. 标准化
        result = self.standardize(result)

        # 4. 中性化
        if neutralize and (market_cap is not None or industry is not None):
            result = self.neutralize(result, market_cap, industry)

        return result


# 便捷函数
def fillna(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """fillna便捷函数"""
    return FactorProcessor().fillna(df, **kwargs)


def winsorize(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """winsorize便捷函数"""
    return FactorProcessor().winsorize(df, **kwargs)


def standardize(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """standardize便捷函数"""
    return FactorProcessor().standardize(df, **kwargs)


def neutralize(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """neutralize便捷函数"""
    return FactorProcessor().neutralize(df, **kwargs)


def process(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """process便捷函数"""
    return FactorProcessor().process(df, **kwargs)


__all__ = [
    "FactorProcessor",
    "fillna",
    "winsorize",
    "standardize",
    "neutralize",
    "process",
]
