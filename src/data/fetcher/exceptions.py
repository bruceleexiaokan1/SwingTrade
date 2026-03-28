"""StockData Fetcher - 数据采集模块

异常定义
"""

from typing import Optional


class FetcherError(Exception):
    """采集器基础异常"""

    def __init__(self, message: str, code: Optional[str] = None, date: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.date = date


class NetworkError(FetcherError):
    """网络错误（可重试）"""
    pass


class SourceError(FetcherError):
    """数据源错误（可能需要切换备源）"""

    def __init__(self, message: str, source: str, code: Optional[str] = None, date: Optional[str] = None):
        super().__init__(message, code, date)
        self.source = source


class ValidationError(FetcherError):
    """数据格式/字段验证错误（不重试）"""
    pass


class QualityError(FetcherError):
    """数据质量不达标（拒绝写入）"""

    def __init__(self, message: str, score: float, code: Optional[str] = None, date: Optional[str] = None):
        super().__init__(message, code, date)
        self.score = score


class WriteError(FetcherError):
    """写入错误"""
    pass


class ConfigurationError(FetcherError):
    """配置错误"""
    pass
