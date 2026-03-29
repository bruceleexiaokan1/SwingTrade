"""因子库自定义异常"""


class FactorError(Exception):
    """因子库基础异常"""
    pass


class FactorNotFoundError(FactorError):
    """因子不存在"""

    def __init__(self, factor_name: str):
        self.factor_name = factor_name
        super().__init__(f"Factor not found: {factor_name}")


class FactorValidationError(FactorError):
    """因子数据验证失败"""

    def __init__(self, message: str, details: dict = None):
        self.details = details or {}
        super().__init__(message)


class FactorCalculationError(FactorError):
    """因子计算失败"""

    def __init__(self, factor_name: str, message: str):
        self.factor_name = factor_name
        super().__init__(f"Factor calculation failed for {factor_name}: {message}")


class DataValidationError(FactorError):
    """输入数据验证失败"""

    def __init__(self, message: str, missing_columns: list = None):
        self.missing_columns = missing_columns or []
        super().__init__(message)


class ProcessingError(FactorError):
    """因子处理失败"""
    pass


class EvaluationError(FactorError):
    """因子评估失败"""
    pass


__all__ = [
    "FactorError",
    "FactorNotFoundError",
    "FactorValidationError",
    "FactorCalculationError",
    "DataValidationError",
    "ProcessingError",
    "EvaluationError",
]
