"""因子库模块

提供标准化的因子计算、存储和评估框架

主要组件:
    - FactorBase: 因子基类
    - FactorRegistry: 因子注册表
    - 清洗处理: fillna, winsorize, standardize, neutralize
    - 评估框架: IC/IR计算, 分组回测

示例:
    from factors import FactorRegistry, register_factor
    from factors.price_volume import MomentumRet3M

    # 注册因子
    registry = get_registry()
    registry.register(MomentumRet3M())

    # 计算因子
    result = registry.calculate_all(data, ["ret_3m"])
"""

from .factor_base import FactorBase, CompositeFactor, FactorMetadata
from .registry import (
    FactorRegistry,
    get_registry,
    register_factor,
    list_registered_factors,
)
from .exceptions import (
    FactorError,
    FactorNotFoundError,
    FactorValidationError,
    FactorCalculationError,
    DataValidationError,
    ProcessingError,
    EvaluationError,
)

__all__ = [
    # 基类和注册表
    "FactorBase",
    "CompositeFactor",
    "FactorMetadata",
    "FactorRegistry",
    "get_registry",
    "register_factor",
    "list_registered_factors",
    # 异常
    "FactorError",
    "FactorNotFoundError",
    "FactorValidationError",
    "FactorCalculationError",
    "DataValidationError",
    "ProcessingError",
    "EvaluationError",
]
