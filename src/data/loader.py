"""
StockData 数据加载器

提供统一的数据加载接口，支持热/温/冷分层访问
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd


class StockDataLoader:
    """
    StockData 数据加载器

    数据分层：
    - Hot (热数据): 当日实时行情，来自 SQLite latest_quote
    - Warm (温数据): 近 60 天数据，来自 warm/daily_summary
    - Cold (冷数据): 历史数据，来自 raw/daily/{code}.parquet
    """

    WARM_RETENTION_DAYS = 60

    def __init__(self, stockdata_root: str):
        """
        初始化加载器

        Args:
            stockdata_root: StockData 根目录
        """
        self.root = Path(stockdata_root)
        self.daily_dir = self.root / "raw" / "daily"
        self.warm_dir = self.root / "warm" / "daily_summary"
        self.db_path = self.root / "sqlite" / "market.db"

    def load_daily(
        self,
        code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        加载日线数据（自动热/温/冷分层）

        Args:
            code: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)，None 表示从头
            end_date: 结束日期 (YYYY-MM-DD)，None 表示到最新

        Returns:
            pd.DataFrame: 日线数据（按 date 排序），无数据时返回空 DataFrame
        """
        parquet_path = self.daily_dir / f"{code}.parquet"

        if not parquet_path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_parquet(str(parquet_path))
        except Exception:
            return pd.DataFrame()

        # 过滤日期范围
        if start_date is not None:
            df = df[df['date'] >= start_date]
        if end_date is not None:
            df = df[df['date'] <= end_date]

        # 按日期排序
        df = df.sort_values('date')

        return df

    def load_warm_summary(self, date: str) -> pd.DataFrame:
        """
        加载指定日期的全市场汇总

        Args:
            date: 日期 (YYYY-MM-DD)

        Returns:
            pd.DataFrame: 全市场当日汇总，无数据时返回空 DataFrame
        """
        date_str = date.replace('-', '')
        summary_path = self.warm_dir / f"{date_str}.parquet"

        if not summary_path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_parquet(str(summary_path))
            return df
        except Exception:
            return pd.DataFrame()

    def load_realtime(self, code: str) -> Dict[str, Any]:
        """
        加载实时行情（热数据）

        Args:
            code: 股票代码

        Returns:
            dict: 实时行情数据，无数据时返回空字典
        """
        if not self.db_path.exists():
            return {}

        conn = sqlite3.connect(str(self.db_path))
        try:
            row = conn.execute(
                "SELECT * FROM latest_quote WHERE code = ?",
                [code]
            ).fetchone()

            if row is None:
                return {}

            columns = [desc[0] for desc in conn.execute(
                "SELECT * FROM latest_quote LIMIT 0"
            ).description]

            return dict(zip(columns, row))
        finally:
            conn.close()

    def search_stocks(
        self,
        filters: Optional[Dict[str, Any]] = None,
        date: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        条件选股

        Args:
            filters: 筛选条件，如 {'pct_chg': ('>', 0.05), 'volume': ('>', 1000000)}
            date: 日期 (YYYY-MM-DD)，None 表示使用最新汇总
            limit: 返回最大数量

        Returns:
            pd.DataFrame: 符合条件的结果，无结果时返回空 DataFrame
        """
        # 加载指定日期的汇总数据
        if date is not None:
            df = self.load_warm_summary(date)
        else:
            # 使用最新的汇总文件
            df = self._get_latest_warm_summary()

        if df.empty:
            return pd.DataFrame()

        # 应用筛选条件
        if filters:
            for field, (op, value) in filters.items():
                if field not in df.columns:
                    continue
                if op == '>':
                    df = df[df[field] > value]
                elif op == '<':
                    df = df[df[field] < value]
                elif op == '>=':
                    df = df[df[field] >= value]
                elif op == '<=':
                    df = df[df[field] <= value]
                elif op == '==':
                    df = df[df[field] == value]
                elif op == '!=':
                    df = df[df[field] != value]

        return df.head(limit)

    def _get_latest_warm_summary(self) -> pd.DataFrame:
        """获取最新的温数据汇总"""
        if not self.warm_dir.exists():
            return pd.DataFrame()

        parquet_files = list(self.warm_dir.glob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()

        # 按文件名排序，取最新的
        latest = sorted(parquet_files)[-1]
        try:
            return pd.read_parquet(str(latest))
        except Exception:
            return pd.DataFrame()

    def get_stock_info(self, code: str) -> Dict[str, Any]:
        """
        获取股票基本信息

        Args:
            code: 股票代码

        Returns:
            dict: 股票信息，无数据时返回空字典
        """
        if not self.db_path.exists():
            return {}

        conn = sqlite3.connect(str(self.db_path))
        try:
            row = conn.execute(
                "SELECT * FROM stocks WHERE code = ?",
                [code]
            ).fetchone()

            if row is None:
                return {}

            columns = [desc[0] for desc in conn.execute(
                "SELECT * FROM stocks LIMIT 0"
            ).description]

            return dict(zip(columns, row))
        finally:
            conn.close()

    def get_trading_dates(
        self,
        code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[str]:
        """
        获取交易日期列表

        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            List[str]: 交易日期列表
        """
        df = self.load_daily(code, start_date, end_date)
        if df.empty:
            return []

        df['date'] = pd.to_datetime(df['date'])
        return sorted(df['date'].dt.strftime('%Y-%m-%d').tolist())

    def is_trading_day(self, date: str) -> bool:
        """
        检查是否为交易日

        Args:
            date: 日期 (YYYY-MM-DD)

        Returns:
            bool: 是否为交易日
        """
        # 简单判断：非周末
        dt = pd.to_datetime(date)
        return dt.weekday() < 5
