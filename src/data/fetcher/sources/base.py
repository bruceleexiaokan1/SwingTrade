"""数据源基类

所有数据源适配器必须实现此接口
"""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class DataSource(ABC):
    """数据源抽象基类"""

    name: str = "base"

    @abstractmethod
    def fetch_daily(
        self,
        code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        拉取日线数据

        Args:
            code: 股票代码，格式如 "600519" 或 "000001"
            start_date: 开始日期，格式 "YYYY-MM-DD"
            end_date: 结束日期，格式 "YYYY-MM-DD"

        Returns:
            DataFrame，包含以下列：
            - date: 交易日期
            - code: 股票代码
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - close: 收盘价
            - volume: 成交量
            - amount: 成交额
            - pct_chg: 涨跌幅
            - adj_factor: 复权因子

        Raises:
            NetworkError: 网络错误
            SourceError: 数据源返回错误
        """
        pass

    @abstractmethod
    def fetch_stock_list(self) -> pd.DataFrame:
        """
        拉取股票列表

        Returns:
            DataFrame，包含以下列：
            - code: 股票代码
            - name: 股票名称
            - market: 市场标识 (sh/sz/bj)
            - list_date: 上市日期
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        检查数据源是否可用

        Returns:
            True if available, False otherwise
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} ({self.name})>"
