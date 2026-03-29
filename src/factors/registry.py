"""因子注册表

提供因子的注册、发现、批量计算功能
"""

import logging
from typing import Dict, List, Optional, Set
from threading import RLock

import pandas as pd

from .factor_base import FactorBase
from .exceptions import FactorNotFoundError, FactorValidationError

logger = logging.getLogger(__name__)


class FactorRegistry:
    """
    因子注册表

    线程安全的因子注册和管理，支持:
    - 因子注册/注销
    - 按类别查询因子
    - 批量计算因子
    - 因子依赖管理

    示例:
        registry = FactorRegistry()

        # 注册因子
        registry.register(MomentumRet3M())

        # 获取因子
        factor = registry.get_factor("ret_3m")

        # 按类别列出
        momentum_factors = registry.list_factors(category="momentum")

        # 批量计算
        result = registry.calculate_all(data, ["ret_3m", "vol_20"])
    """

    _instance: Optional["FactorRegistry"] = None
    _lock = RLock()

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化注册表"""
        if self._initialized:
            return

        self._factors: Dict[str, FactorBase] = {}
        self._category_index: Dict[str, Set[str]] = {}  # category -> factor_names
        self._initialized = True
        logger.info("FactorRegistry initialized")

    def register(self, factor: FactorBase, allow_overwrite: bool = False) -> None:
        """
        注册因子

        Args:
            factor: 因子实例
            allow_overwrite: 是否允许覆盖已有因子

        Raises:
            FactorValidationError: 因子已存在且不允许覆盖
        """
        name = factor.name

        if name in self._factors and not allow_overwrite:
            raise FactorValidationError(
                f"Factor {name} already registered. Use allow_overwrite=True to replace."
            )

        self._factors[name] = factor

        # 更新类别索引
        category = factor.category
        if category not in self._category_index:
            self._category_index[category] = set()
        self._category_index[category].add(name)

        logger.debug(f"Registered factor: {name} (category: {category})")

    def unregister(self, name: str) -> bool:
        """
        注销因子

        Args:
            name: 因子名称

        Returns:
            bool: 是否成功注销
        """
        if name not in self._factors:
            return False

        factor = self._factors[name]
        category = factor.category

        # 从类别索引移除
        if category in self._category_index:
            self._category_index[category].discard(name)

        # 从注册表移除
        del self._factors[name]

        logger.debug(f"Unregistered factor: {name}")
        return True

    def get_factor(self, name: str) -> FactorBase:
        """
        获取因子

        Args:
            name: 因子名称

        Returns:
            FactorBase: 因子实例

        Raises:
            FactorNotFoundError: 因子不存在
        """
        if name not in self._factors:
            raise FactorNotFoundError(name)
        return self._factors[name]

    def list_factors(self, category: Optional[str] = None) -> List[str]:
        """
        列出因子

        Args:
            category: 筛选类别，None表示所有

        Returns:
            List[str]: 因子名称列表
        """
        if category is None:
            return list(self._factors.keys())

        if category not in self._category_index:
            return []

        return list(self._category_index[category])

    def get_categories(self) -> List[str]:
        """
        获取所有类别

        Returns:
            List[str]: 类别列表
        """
        return list(self._category_index.keys())

    def calculate_single(
        self,
        name: str,
        data: pd.DataFrame,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        计算单个因子

        Args:
            name: 因子名称
            data: 输入数据
            validate: 是否验证数据

        Returns:
            DataFrame: date, code, factor_value
        """
        factor = self.get_factor(name)

        if validate:
            required_cols = factor.get_required_data()
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise FactorValidationError(
                    f"Missing columns for factor {name}: {missing}",
                    {"factor": name, "missing": missing}
                )

        result = factor.calculate(data)

        # 验证输出格式
        if not isinstance(result, pd.DataFrame):
            raise FactorValidationError(
                f"Factor {name} calculate() must return DataFrame"
            )

        required_output = {"date", "code", "factor_value"}
        if not required_output.issubset(result.columns):
            raise FactorValidationError(
                f"Factor {name} output must contain: {required_output}"
            )

        return result

    def calculate_all(
        self,
        data: pd.DataFrame,
        factor_names: List[str],
        validate: bool = True,
        drop_na: bool = True
    ) -> pd.DataFrame:
        """
        批量计算因子

        Args:
            data: 输入数据
            factor_names: 要计算的因子列表
            validate: 是否验证数据
            drop_na: 是否丢弃NaN值

        Returns:
            DataFrame: date, code, factor1, factor2, ...
        """
        if len(factor_names) == 0:
            return pd.DataFrame()

        results = []

        for name in factor_names:
            try:
                result = self.calculate_single(name, data, validate=validate)
                result = result.rename(columns={"factor_value": name})
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to calculate factor {name}: {e}")
                raise

        # 合并所有因子
        if not results:
            return pd.DataFrame()

        # 使用 date + code 作为key进行合并
        merged = results[0]
        for result in results[1:]:
            merged = merged.merge(result, on=["date", "code"], how="outer")

        if drop_na:
            # 保留至少有一个因子值的行
            factor_cols = factor_names
            mask = merged[factor_cols].notna().any(axis=1)
            merged = merged[mask]

        return merged

    def get_factor_info(self, name: str) -> dict:
        """
        获取因子信息

        Args:
            name: 因子名称

        Returns:
            dict: 因子元数据
        """
        factor = self.get_factor(name)
        return factor.get_metadata().to_dict()

    def clear(self) -> int:
        """
        清空注册表

        Returns:
            int: 清空的因子数量
        """
        count = len(self._factors)
        self._factors.clear()
        self._category_index.clear()
        logger.info(f"Cleared {count} factors from registry")
        return count

    def __len__(self) -> int:
        """返回注册因子数量"""
        return len(self._factors)

    def __contains__(self, name: str) -> bool:
        """检查因子是否已注册"""
        return name in self._factors


# 全局注册表实例
_default_registry: Optional[FactorRegistry] = None


def get_registry() -> FactorRegistry:
    """获取全局注册表实例"""
    global _default_registry
    if _default_registry is None:
        _default_registry = FactorRegistry()
    return _default_registry


def register_factor(factor: FactorBase, allow_overwrite: bool = False) -> None:
    """快捷注册函数"""
    get_registry().register(factor, allow_overwrite)


def list_registered_factors(category: Optional[str] = None) -> List[str]:
    """快捷列出函数"""
    return get_registry().list_factors(category)


__all__ = [
    "FactorRegistry",
    "get_registry",
    "register_factor",
    "list_registered_factors",
]
