"""Tushare 数据源适配器"""

import os
import time
import logging
from typing import Optional
from datetime import datetime
import pandas as pd

from .base import DataSource
from ..exceptions import NetworkError, SourceError, ConfigurationError

logger = logging.getLogger(__name__)


class TushareSource(DataSource):
    """Tushare 数据源（主力日线数据源）

    限流保护：
    - 200积分限制：50次/分钟
    - 实现 90% 阈值保护（45次/分钟）
    - 每分钟自动重置计数器
    """

    name = "tushare"

    # 限流配置
    RATE_LIMIT_PER_MINUTE = 50      # 200积分限制
    RATE_THRESHOLD = 0.9            # 90% 阈值
    SAFETY_LIMIT = int(RATE_LIMIT_PER_MINUTE * RATE_THRESHOLD)  # 45次

    def __init__(self, token: Optional[str] = None):
        """
        初始化 Tushare 数据源

        Args:
            token: Tushare API Token，若不提供则从环境变量 TUSHARE_TOKEN 读取
        """
        self.token = token or os.getenv("TUSHARE_TOKEN")
        if not self.token:
            raise ConfigurationError(
                "TUSHARE_TOKEN environment variable is not set",
                date=None
            )

        try:
            import tushare as ts
            self.pro = ts.pro_api(self.token)
        except ImportError:
            raise ConfigurationError(
                "tushare package is not installed",
                date=None
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize Tushare: {e}",
                date=None
            )

        # 限流状态
        self._call_count = 0
        self._last_reset_time = time.time()
        self._rate_limit_warnings = 0

    def fetch_daily(
        self,
        code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        从 Tushare 拉取日线数据

        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame
        """
        # 限流检查
        self._check_rate_limit()

        # 转换日期格式：YYYY-MM-DD -> YYYYMMDD
        start_str = start_date.replace("-", "")
        end_str = end_date.replace("-", "")

        # Tushare 代码格式：600519.SH 或 000001.SZ
        ts_code = self._to_ts_code(code)

        try:
            df = self.pro.daily(
                ts_code=ts_code,
                start_date=start_str,
                end_date=end_str
            )
        except Exception as e:
            raise NetworkError(
                f"Failed to fetch daily data: {e}",
                code=code,
                date=end_date
            )

        if df is None or len(df) == 0:
            # 返回空 DataFrame 但不抛异常，这是正常情况（如停牌）
            return pd.DataFrame()

        # 字段映射（Tushare原始列名 → 标准列名）
        df = df.rename(columns={
            "ts_code": "code",
            "trade_date": "date",
            "vol": "volume"
        })

        # 验证数据格式（映射后的列名）
        df = self._validate_daily_dataframe(df, code)

        # 转换日期格式
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        # 确保字段类型正确
        df["volume"] = df["volume"].astype("int64")

        return df

    def fetch_stock_list(self) -> pd.DataFrame:
        """
        从 Tushare 拉取股票列表

        Returns:
            DataFrame，包含全市场活跃股票
        """
        try:
            # 获取 A 股全量股票列表
            df = self.pro.stock_basic(
                exchange="",
                list_status="L",  # L=上市, D=退市, P=暂停上市
                fields="ts_code,symbol,name,area,industry,list_date,market"
            )
        except Exception as e:
            raise SourceError(
                f"Failed to fetch stock list: {e}",
                source=self.name,
                date=None
            )

        if df is None or len(df) == 0:
            return pd.DataFrame()

        # 字段映射和转换
        df = df.rename(columns={
            "ts_code": "code",
            "symbol": "symbol",
            "name": "name",
            "list_date": "list_date",
            "market": "market_ext"
        })

        # 从 ts_code 提取市场标识
        # 600519.SH -> sh, 000001.SZ -> sz
        df["market"] = df["code"].apply(self._extract_market)

        # 只保留必要字段
        df = df[["code", "name", "market", "list_date"]]

        return df

    def is_available(self) -> bool:
        """检查 Tushare 是否可用"""
        try:
            # 尝试获取一只股票的基本信息来测试连接
            self.pro.stock_basic(ts_code="600519.SH", fields="ts_code")
            return True
        except Exception as e:
            logger.warning(f"Tushare health check failed: {e}")
            return False

    def _to_ts_code(self, code: str) -> str:
        """将股票代码转换为 Tushare 格式"""
        if code.startswith(("6", "5")):
            return f"{code}.SH"
        elif code.startswith(("0", "1", "2", "3")):
            return f"{code}.SZ"
        elif code.startswith(("4", "8", "9")):
            # 北交所或特殊板块
            return f"{code}.BJ"
        else:
            # 默认上海
            return f"{code}.SH"

    def _check_rate_limit(self):
        """
        检查并更新限流状态

        - 每分钟重置计数器
        - 达到 90% 阈值时等待
        """
        now = time.time()

        # 每分钟重置
        if now - self._last_reset_time >= 60:
            self._call_count = 0
            self._last_reset_time = now
            self._rate_limit_warnings = 0

        # 检查是否接近阈值
        if self._call_count >= self.SAFETY_LIMIT:
            # 需要等待到下一分钟
            sleep_time = 60 - (now - self._last_reset_time)
            if sleep_time > 0:
                logger.warning(
                    f"Rate limit threshold reached ({self._call_count}/{self.SAFETY_LIMIT}), "
                    f"sleeping {sleep_time:.1f}s"
                )
                time.sleep(sleep_time)
                self._call_count = 0
                self._last_reset_time = time.time()

        self._call_count += 1

    def _validate_daily_dataframe(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        """
        验证日线数据格式

        Args:
            df: 待验证的 DataFrame
            code: 股票代码（用于错误信息）

        Returns:
            验证通过的 DataFrame

        Raises:
            SourceError: 数据格式异常
        """
        required_cols = {"date", "open", "high", "low", "close", "volume"}

        if df is None or len(df) == 0:
            return df

        # 检查列是否存在
        missing = required_cols - set(df.columns)
        if missing:
            raise SourceError(
                f"Missing columns: {missing}",
                source=self.name,
                code=code
            )

        # 检查数据类型
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise SourceError(
                        f"Column {col} is not numeric type",
                        source=self.name,
                        code=code
                    )

        return df

    def fetch_adj_factor(
        self,
        code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取复权因子

        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame，包含 date, adj_factor 列
        """
        # 限流检查
        self._check_rate_limit()

        ts_code = self._to_ts_code(code)
        start_str = start_date.replace("-", "")
        end_str = end_date.replace("-", "")

        try:
            df = self.pro.adj_factor(
                ts_code=ts_code,
                start_date=start_str,
                end_date=end_str
            )
        except Exception as e:
            raise NetworkError(
                f"Failed to fetch adj_factor: {e}",
                code=code,
                date=end_date
            )

        if df is None or len(df) == 0:
            return pd.DataFrame()

        df = df.rename(columns={"trade_date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        return df[["date", "adj_factor"]]

    def _extract_market(self, ts_code: str) -> str:
        """从 Tushare 代码提取市场标识"""
        if ts_code.endswith(".SH"):
            return "sh"
        elif ts_code.endswith(".SZ"):
            return "sz"
        elif ts_code.endswith(".BJ"):
            return "bj"
        else:
            return "unknown"
