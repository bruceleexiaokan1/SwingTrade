"""
StockData 数据库初始化脚本
创建 market.db 和所有必要的表
"""

import sqlite3
import os
from pathlib import Path


def get_db_path() -> str:
    """获取数据库路径"""
    return os.path.join(
        os.environ.get('STOCKDATA_ROOT', '/Users/bruce/workspace/trade/StockData'),
        'sqlite',
        'market.db'
    )


def init_database(db_path: str = None):
    """初始化 SQLite 数据库"""

    if db_path is None:
        db_path = get_db_path()

    # 确保目录存在
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # 连接数据库
    conn = sqlite3.connect(db_path)

    # 启用 WAL 模式
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('PRAGMA busy_timeout=30000')

    # 创建股票代码表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            code TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            market TEXT NOT NULL,
            list_date TEXT,
            delist_date TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # 创建日线数据索引表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_index (
            code TEXT PRIMARY KEY,
            latest_date TEXT NOT NULL,
            file_path TEXT NOT NULL,
            row_count INTEGER,
            start_date TEXT,
            end_date TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # 创建最新行情缓存表
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

    # 创建数据更新记录表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS update_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_type TEXT NOT NULL,
            code TEXT,
            update_date TEXT NOT NULL,
            status TEXT NOT NULL,
            row_count INTEGER,
            error_msg TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # 创建检查点记录表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # 创建索引
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_daily_code_date
        ON daily_index(code, latest_date)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_update_log_date
        ON update_log(data_type, update_date)
    """)

    # 提交并关闭
    conn.commit()
    conn.close()

    print(f"数据库初始化完成: {db_path}")


def reset_database(db_path: str = None):
    """重置数据库（删除所有表，重新创建）"""

    if db_path is None:
        db_path = get_db_path()

    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"已删除旧数据库: {db_path}")

    init_database(db_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='初始化 StockData 数据库')
    parser.add_argument('--reset', action='store_true', help='重置数据库')
    args = parser.parse_args()

    if args.reset:
        reset_database()
    else:
        init_database()
