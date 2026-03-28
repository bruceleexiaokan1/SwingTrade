"""
StockData 备份脚本测试
"""

import pytest
import os
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, 'scripts')

from maintenance.backup import (
    create_backup_dirs,
    backup_raw_daily,
    backup_sqlite,
    cleanup_old_backups,
    run_backup,
    get_stockdata_root,
    get_backup_root,
)


class TestBackup:
    """备份功能测试"""

    def setup_method(self):
        """使用临时目录"""
        self.temp_stockdata = tempfile.mkdtemp(prefix="stockdata_backup_test_")
        self.temp_backup = tempfile.mkdtemp(prefix="backup_root_test_")

        # 创建测试数据
        raw_dir = Path(self.temp_stockdata) / "raw" / "daily"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # 创建测试 parquet 文件
        import pandas as pd
        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-03-29']),
            'code': ['000001'],
            'close': [10.0],
            'volume': [1000000],
        })
        df.to_parquet(str(raw_dir / "000001.parquet"), engine='pyarrow')

        # 创建 SQLite
        sqlite_dir = Path(self.temp_stockdata) / "sqlite"
        sqlite_dir.mkdir(parents=True, exist_ok=True)
        import sqlite3
        conn = sqlite3.connect(str(sqlite_dir / "market.db"))
        conn.execute("CREATE TABLE stocks (code TEXT PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO stocks VALUES ('000001', '测试股票')")
        conn.commit()
        conn.close()

    def teardown_method(self):
        shutil.rmtree(self.temp_stockdata, ignore_errors=True)
        shutil.rmtree(self.temp_backup, ignore_errors=True)

    def test_create_backup_dirs(self):
        """创建备份目录"""
        os.environ['BACKUP_ROOT'] = self.temp_backup

        create_backup_dirs()

        today = datetime.now().strftime('%Y%m%d')
        assert (Path(self.temp_backup) / "daily" / today).exists()
        assert (Path(self.temp_backup) / "weekly").exists()
        assert (Path(self.temp_backup) / "monthly").exists()

    def test_backup_raw_daily(self):
        """备份 raw/daily"""
        os.environ['STOCKDATA_ROOT'] = self.temp_stockdata
        os.environ['BACKUP_ROOT'] = self.temp_backup

        result = backup_raw_daily()

        assert result is True

        today = datetime.now().strftime('%Y%m%d')
        backup_path = Path(self.temp_backup) / "daily" / today / "raw_daily" / "000001.parquet"
        assert backup_path.exists()

    def test_backup_sqlite(self):
        """备份 SQLite"""
        os.environ['STOCKDATA_ROOT'] = self.temp_stockdata
        os.environ['BACKUP_ROOT'] = self.temp_backup

        result = backup_sqlite()

        assert result is True

        today = datetime.now().strftime('%Y%m%d')
        backup_path = Path(self.temp_backup) / "daily" / today / "sqlite" / "market.db"
        assert backup_path.exists()

    def test_run_backup(self):
        """执行完整备份"""
        os.environ['STOCKDATA_ROOT'] = self.temp_stockdata
        os.environ['BACKUP_ROOT'] = self.temp_backup

        results = run_backup()

        assert results['success'] is True
        assert results['raw_daily'] is True
        assert results['sqlite'] is True
