"""
StockData 数据加载器测试
"""

import pytest
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.insert(0, 'src')

from data.loader import StockDataLoader


class TestStockDataLoader:
    """数据加载器测试"""

    def test_init(self, temp_stockdata):
        """初始化加载器"""
        loader = StockDataLoader(stockdata_root=str(temp_stockdata))
        assert loader.root == temp_stockdata
        assert loader.daily_dir == temp_stockdata / "raw" / "daily"

    def test_load_daily_no_data(self, temp_stockdata):
        """无数据时返回空 DataFrame"""
        loader = StockDataLoader(stockdata_root=str(temp_stockdata))
        df = loader.load_daily("000001")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_load_daily_with_data(self, temp_stockdata):
        """加载有数据的股票"""
        loader = StockDataLoader(stockdata_root=str(temp_stockdata))

        # 创建测试数据
        daily_dir = temp_stockdata / "raw" / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-03-01', '2026-03-02', '2026-03-03']),
            'code': ['000001'] * 3,
            'open': [10.0, 10.2, 10.4],
            'high': [10.5, 10.7, 10.8],
            'low': [9.8, 10.0, 10.2],
            'close': [10.2, 10.5, 10.6],
            'volume': [1000000, 1100000, 1200000],
            'amount': [10200000, 11550000, 12600000],
            'adj_factor': [1.0, 1.0, 1.0],
            'turnover': [0.05, 0.055, 0.06],
            'is_halt': [False, False, False],
            'pct_chg': [0.02, 0.029, 0.01],
        })

        path = daily_dir / "000001.parquet"
        df.to_parquet(str(path), engine='pyarrow')

        # 测试加载
        result = loader.load_daily("000001")
        assert len(result) == 3
        assert 'close' in result.columns

    def test_load_daily_date_filter(self, temp_stockdata):
        """日期过滤"""
        loader = StockDataLoader(stockdata_root=str(temp_stockdata))

        daily_dir = temp_stockdata / "raw" / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-03-01', '2026-03-02', '2026-03-03', '2026-03-04']),
            'code': ['000001'] * 4,
            'open': [10.0, 10.1, 10.2, 10.3],
            'high': [10.5, 10.6, 10.7, 10.8],
            'low': [9.8, 9.9, 10.0, 10.1],
            'close': [10.2, 10.3, 10.4, 10.5],
            'volume': [1000000, 1100000, 1200000, 1300000],
            'amount': [10200000, 11200000, 12200000, 13200000],
            'adj_factor': [1.0] * 4,
            'turnover': [0.05] * 4,
            'is_halt': [False] * 4,
            'pct_chg': [0.02] * 4,
        })

        path = daily_dir / "000001.parquet"
        df.to_parquet(str(path), engine='pyarrow')

        # 只加载 2026-03-02 到 2026-03-03
        result = loader.load_daily("000001", "2026-03-02", "2026-03-03")
        assert len(result) == 2
        assert result['date'].iloc[0] >= pd.Timestamp('2026-03-02')
        assert result['date'].iloc[-1] <= pd.Timestamp('2026-03-03')

    def test_load_realtime_no_data(self, temp_stockdata):
        """无实时数据时返回空字典"""
        loader = StockDataLoader(stockdata_root=str(temp_stockdata))
        result = loader.load_realtime("000001")
        assert result == {}

    def test_load_realtime_with_data(self, temp_stockdata):
        """加载实时行情"""
        loader = StockDataLoader(stockdata_root=str(temp_stockdata))

        # 创建 SQLite
        db_dir = temp_stockdata / "sqlite"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / "market.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS latest_quote (
                code TEXT PRIMARY KEY,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                amount REAL,
                pct_chg REAL,
                update_time TEXT
            )
        """)
        conn.execute("""
            INSERT OR REPLACE INTO latest_quote
            (code, date, open, high, low, close, volume, amount, pct_chg, update_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ["000001", "2026-03-28", 10.0, 10.5, 9.8, 10.2, 1000000, 10200000, 0.02, "2026-03-28 15:00:00"])
        conn.commit()
        conn.close()

        result = loader.load_realtime("000001")
        assert result['code'] == "000001"
        assert result['close'] == 10.2

    def test_search_stocks_empty(self, temp_stockdata):
        """无汇总数据时返回空 DataFrame"""
        loader = StockDataLoader(stockdata_root=str(temp_stockdata))
        result = loader.search_stocks()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_search_stocks_with_filter(self, temp_stockdata):
        """条件选股"""
        loader = StockDataLoader(stockdata_root=str(temp_stockdata))

        # 创建温数据汇总
        warm_dir = temp_stockdata / "warm" / "daily_summary"
        warm_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            'code': ['000001', '000002', '000003'],
            'name': ['股票A', '股票B', '股票C'],
            'close': [10.0, 20.0, 30.0],
            'pct_chg': [0.05, 0.02, -0.03],
            'volume': [1000000, 2000000, 3000000],
        })

        path = warm_dir / "20260328.parquet"
        df.to_parquet(str(path), engine='pyarrow')

        # 筛选涨幅 > 0
        result = loader.search_stocks({'pct_chg': ('>', 0)})
        assert len(result) == 2
        assert all(result['pct_chg'] > 0)

    def test_get_stock_info(self, temp_stockdata):
        """获取股票信息"""
        loader = StockDataLoader(stockdata_root=str(temp_stockdata))

        # 创建 SQLite
        db_dir = temp_stockdata / "sqlite"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / "market.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stocks (
                code TEXT PRIMARY KEY,
                name TEXT,
                market TEXT,
                list_date TEXT
            )
        """)
        conn.execute("""
            INSERT OR REPLACE INTO stocks (code, name, market, list_date)
            VALUES (?, ?, ?, ?)
        """, ["000001", "平安银行", "sz", "1991-04-03"])
        conn.commit()
        conn.close()

        result = loader.get_stock_info("000001")
        assert result['name'] == "平安银行"
        assert result['market'] == "sz"

    def test_get_trading_dates(self, temp_stockdata):
        """获取交易日期列表"""
        loader = StockDataLoader(stockdata_root=str(temp_stockdata))

        daily_dir = temp_stockdata / "raw" / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-03-01', '2026-03-02', '2026-03-03']),
            'code': ['000001'] * 3,
            'open': [10.0, 10.1, 10.2],
            'high': [10.5, 10.6, 10.7],
            'low': [9.8, 9.9, 10.0],
            'close': [10.2, 10.3, 10.4],
            'volume': [1000000, 1100000, 1200000],
            'amount': [10200000, 11200000, 12200000],
            'adj_factor': [1.0] * 3,
            'turnover': [0.05] * 3,
            'is_halt': [False] * 3,
            'pct_chg': [0.02] * 3,
        })

        path = daily_dir / "000001.parquet"
        df.to_parquet(str(path), engine='pyarrow')

        dates = loader.get_trading_dates("000001")
        assert len(dates) == 3
        assert dates[0] == "2026-03-01"

    def test_is_trading_day(self, temp_stockdata):
        """判断交易日"""
        loader = StockDataLoader(stockdata_root=str(temp_stockdata))
        assert loader.is_trading_day("2026-03-30") is True  # Monday
        assert loader.is_trading_day("2026-03-29") is False  # Sunday
