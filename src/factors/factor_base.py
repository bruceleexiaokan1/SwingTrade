"""因子基类定义

所有因子必须继承FactorBase并实现calculate()方法
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import pandas as pd


@dataclass
class FactorMetadata:
    """因子元数据"""

    name: str                    # 因子名称 (如 "ret_3m")
    category: str               # 因子类别 (如 "momentum")
    description: str            # 因子描述
    version: str = "1.0"         # 因子版本
    author: str = ""             # 作者
    created_date: str = ""      # 创建日期
    dependencies: List[str] = field(default_factory=list)  # 依赖的因子

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "created_date": self.created_date,
            "dependencies": self.dependencies,
        }


class FactorBase(ABC):
    """
    因子基类

    所有因子必须继承此类并实现:
    1. name: 因子名称
    2. category: 因子类别
    3. description: 因子描述
    4. calculate(): 计算逻辑

    示例:
        class MomentumRet3M(FactorBase):
            name = "ret_3m"
            category = "momentum"
            description = "3个月收益率动量"

            def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
                # 计算逻辑
                result = ...
                return result
    """

    # 类属性 (子类必须定义)
    name: str = ""
    category: str = ""
    description: str = ""

    def __init__(self):
        """初始化因子"""
        if not self.name:
            raise ValueError("Factor name must be defined")
        if not self.category:
            raise ValueError("Factor category must be defined")

        self._metadata = FactorMetadata(
            name=self.name,
            category=self.category,
            description=self.description,
        )

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算因子

        Args:
            data: 输入数据，必须包含以下列:
                - date: 日期 (str)
                - code: 股票代码 (str)
                - close: 收盘价 (float)
                - open/high/low: 价格 (float, 可选)
                - volume: 成交量 (int, 可选)
                - amount: 成交额 (float, 可选)

        Returns:
            DataFrame，必须包含以下列:
                - date: 日期 (str)
                - code: 股票代码 (str)
                - factor_value: 因子值 (float)

        Raises:
            FactorCalculationError: 计算失败时
        """
        pass

    def get_metadata(self) -> FactorMetadata:
        """获取因子元数据"""
        return self._metadata

    def validate_data(self, data: pd.DataFrame, required_cols: List[str] = None) -> bool:
        """
        验证输入数据

        Args:
            data: 待验证的DataFrame
            required_cols: 必需的列，默认为 ['date', 'code', 'close']

        Returns:
            bool: 是否有效

        Raises:
            DataValidationError: 数据无效时
        """
        from .exceptions import DataValidationError

        if required_cols is None:
            required_cols = ["date", "code", "close"]

        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise DataValidationError(
                f"Missing required columns: {missing}",
                missing_columns=missing,
            )

        return True

    def get_required_data(self) -> List[str]:
        """
        返回计算所需的最小数据列

        子类可重写此方法指定必需列

        Returns:
            List[str]: 必需的数据列
        """
        return ["date", "code", "close"]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, category={self.category})>"


class CompositeFactor(FactorBase):
    """
    复合因子基类

    用于组合多个因子的场景
    """

    def __init__(self, sub_factors: List[FactorBase], weights: List[float] = None):
        """
        初始化复合因子

        Args:
            sub_factors: 子因子列表
            weights: 权重列表，默认等权
        """
        super().__init__()

        if len(sub_factors) == 0:
            raise ValueError("At least one sub-factor is required")

        self.sub_factors = sub_factors
        self.weights = weights or [1.0 / len(sub_factors)] * len(sub_factors)

        if len(self.weights) != len(self.sub_factors):
            raise ValueError("Number of weights must match number of sub-factors")

        # 复合因子的元数据
        self._metadata.dependencies = [f.name for f in sub_factors]

    @abstractmethod
    def combine(self, factor_values: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        组合子因子

        Args:
            factor_values: {因子名: DataFrame} 的字典

        Returns:
            DataFrame: 组合后的因子值
        """
        pass


__all__ = [
    "FactorBase",
    "CompositeFactor",
    "FactorMetadata",
]
