"""
StockData 测试配置
提供测试隔离环境
"""

import pytest
import tempfile
import shutil
import os
import sqlite3
from pathlib import Path
from contextlib import contextmanager


@pytest.fixture
def temp_stockdata(tmp_path):
    """
    创建临时 StockData 环境
    测试结束后自动清理
    """
    stockdata_root = tmp_path / "stockdata"
    stockdata_root.mkdir(parents=True)

    # 创建必要目录
    dirs = [
        "raw/daily",
        "processed/adj",
        "warm/daily_summary",
        "sqlite",
        "cache/realtime",
        "status",
        "logs",
    ]
    for d in dirs:
        (stockdata_root / d).mkdir(parents=True)

    # 保存原环境变量
    original_root = os.environ.get('STOCKDATA_ROOT')

    # 设置测试环境变量
    os.environ['STOCKDATA_ROOT'] = str(stockdata_root)

    yield stockdata_root

    # 恢复原环境变量
    if original_root:
        os.environ['STOCKDATA_ROOT'] = original_root
    else:
        os.environ.pop('STOCKDATA_ROOT', None)


@pytest.fixture
def temp_sqlite(temp_stockdata):
    """创建临时 SQLite 数据库"""
    db_path = temp_stockdata / "sqlite" / "market.db"

    conn = sqlite3.connect(str(db_path))
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')

    # 创建表结构
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            code TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            market TEXT NOT NULL,
            list_date TEXT,
            delist_date TEXT,
            is_active INTEGER DEFAULT 1
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_index (
            code TEXT PRIMARY KEY,
            latest_date TEXT NOT NULL,
            file_path TEXT NOT NULL,
            row_count INTEGER,
            start_date TEXT,
            end_date TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()

    yield db_path


@pytest.fixture
def sample_stockdata(temp_stockdata):
    """创建示例股票数据"""
    import pandas as pd

    stockdata_root = temp_stockdata

    # 创建示例日线数据
    data = {
        'date': ['2026-03-01', '2026-03-02', '2026-03-03'],
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
    }

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])

    # 写入 Parquet
    parquet_path = stockdata_root / "raw/daily/000001.parquet"
    df.to_parquet(str(parquet_path), engine='pyarrow', compression='snappy')

    return {
        'root': stockdata_root,
        'df': df,
        'parquet_path': parquet_path,
        'code': '000001'
    }


@contextmanager
def isolated_environment():
    """上下文管理器：隔离环境"""
    original_env = os.environ.copy()
    temp_dir = tempfile.mkdtemp()

    try:
        os.environ['STOCKDATA_ROOT'] = temp_dir
        yield temp_dir
    finally:
        os.environ.clear()
        os.environ.update(original_env)
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_env(monkeypatch):
    """Mock 环境变量"""
    def set_env(key, value):
        monkeypatch.setenv(key, value)
    return set_env
