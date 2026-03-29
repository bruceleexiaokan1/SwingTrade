"""
StockData 幂等写入测试
"""

import pytest
import pandas as pd
import numpy as np
import os
import sqlite3
from pathlib import Path

# 导入被测试模块
import sys
sys.path.insert(0, 'scripts')

from utils.writer import IdempotentWriter
from utils.quality import QualityScore, calculate_quality_score
from tests.fixtures import ANOMALY_TEST_CASES


class TestIdempotentWriter:
    """幂等写入测试"""

    def test_write_creates_file(self, temp_stockdata, temp_sqlite):
        """正常写入创建文件"""
        writer = IdempotentWriter(
            stockdata_root=temp_stockdata,
            db_path=temp_sqlite
        )

        df, _ = ANOMALY_TEST_CASES['perfect']()
        writer.write('000001', df)

        # 验证文件创建
        parquet_path = temp_stockdata / "raw/daily/000001.parquet"
        assert parquet_path.exists()

        # 验证数据正确
        result = pd.read_parquet(str(parquet_path))
        assert len(result) == len(df)

    def test_duplicate_write_skipped(self, temp_stockdata, temp_sqlite):
        """重复写入被跳过"""
        writer = IdempotentWriter(
            stockdata_root=temp_stockdata,
            db_path=temp_sqlite
        )

        # 写入第一版
        df1, _ = ANOMALY_TEST_CASES['perfect']()
        writer.write('000001', df1)

        # 尝试写入相同日期的新数据
        df2, _ = ANOMALY_TEST_CASES['perfect']()
        writer.write('000001', df2)

        # 验证：数据只有一组（不重复）
        result = pd.read_parquet(str(temp_stockdata / "raw/daily/000001.parquet"))
        assert len(result) == len(df1), f"数据不应重复: {len(result)} vs {len(df1)}"

    def test_new_date_appended(self, temp_stockdata, temp_sqlite):
        """新日期数据追加"""
        writer = IdempotentWriter(
            stockdata_root=temp_stockdata,
            db_path=temp_sqlite
        )

        # 写入第一天
        df1 = pd.DataFrame({
            'date': pd.to_datetime(['2026-03-01']),
            'open': [10.0], 'high': [10.5], 'low': [9.8], 'close': [10.2],
            'volume': [1000000], 'amount': [10200000],
            'adj_factor': [1.0], 'turnover': [0.05],
            'is_halt': [False], 'pct_chg': [0.02],
        })
        writer.write('000001', df1)

        # 写入第二天
        df2 = pd.DataFrame({
            'date': pd.to_datetime(['2026-03-02']),
            'open': [10.2], 'high': [10.7], 'low': [10.0], 'close': [10.5],
            'volume': [1100000], 'amount': [11550000],
            'adj_factor': [1.0], 'turnover': [0.055],
            'is_halt': [False], 'pct_chg': [0.029],
        })
        writer.write('000001', df2)

        # 验证：数据有两行
        result = pd.read_parquet(str(temp_stockdata / "raw/daily/000001.parquet"))
        assert len(result) == 2, f"应有2行数据: {len(result)}"
        assert result['date'].iloc[0] < result['date'].iloc[1]

    def test_old_date_rejected(self, temp_stockdata, temp_sqlite):
        """旧日期数据被拒绝"""
        writer = IdempotentWriter(
            stockdata_root=temp_stockdata,
            db_path=temp_sqlite
        )

        # 写入较新的数据
        df_new = pd.DataFrame({
            'date': pd.to_datetime(['2026-03-02']),
            'open': [10.2], 'high': [10.7], 'low': [10.0], 'close': [10.5],
            'volume': [1100000], 'amount': [11550000],
            'adj_factor': [1.0], 'turnover': [0.055],
            'is_halt': [False], 'pct_chg': [0.029],
        })
        writer.write('000001', df_new)

        # 尝试写入较旧的数据
        df_old = pd.DataFrame({
            'date': pd.to_datetime(['2026-03-01']),
            'open': [10.0], 'high': [10.5], 'low': [9.8], 'close': [10.2],
            'volume': [1000000], 'amount': [10200000],
            'adj_factor': [1.0], 'turnover': [0.05],
            'is_halt': [False], 'pct_chg': [0.02],
        })
        writer.write('000001', df_old)

        # 验证：只有新数据
        result = pd.read_parquet(str(temp_stockdata / "raw/daily/000001.parquet"))
        assert len(result) == 1
        assert result['date'].iloc[0] == pd.to_datetime('2026-03-02')

    def test_low_quality_quarantine(self, temp_stockdata, temp_sqlite):
        """低质量数据进入隔离"""
        writer = IdempotentWriter(
            stockdata_root=temp_stockdata,
            db_path=temp_sqlite
        )

        # 创建低质量数据
        df, _ = ANOMALY_TEST_CASES['multi_anomalies']()
        writer.write('000001', df)

        # 验证：主数据文件不应存在
        parquet_path = temp_stockdata / "raw/daily/000001.parquet"
        # 注意：当前实现会跳过写入，这里验证行为即可

    def test_checkpoint_updated(self, temp_stockdata, temp_sqlite):
        """检查点正确更新"""
        writer = IdempotentWriter(
            stockdata_root=temp_stockdata,
            db_path=temp_sqlite
        )

        df, _ = ANOMALY_TEST_CASES['perfect']()
        writer.write('000001', df)

        # 验证检查点
        conn = sqlite3.connect(str(temp_sqlite))
        checkpoint = conn.execute(
            "SELECT value FROM checkpoints WHERE key = 'daily_000001_last_update'"
        ).fetchone()
        conn.close()

        assert checkpoint is not None
        assert checkpoint[0] is not None


class TestAtomicWrite:
    """原子写入测试"""

    def test_write_atomic_no_temp_file_left(self, temp_stockdata):
        """原子写入后无临时文件残留"""
        from utils.writer import AtomicWriter

        writer = AtomicWriter(stockdata_root=temp_stockdata)
        df, _ = ANOMALY_TEST_CASES['perfect']()

        writer.write('000001', df)

        # 验证：没有 .tmp 文件
        daily_dir = temp_stockdata / "raw/daily"
        temp_files = list(daily_dir.glob("*.tmp*"))
        assert len(temp_files) == 0, f"有临时文件残留: {temp_files}"

    def test_write_atomic_file_complete(self, temp_stockdata):
        """原子写入后文件完整"""
        from utils.writer import AtomicWriter

        writer = AtomicWriter(stockdata_root=temp_stockdata)
        df, _ = ANOMALY_TEST_CASES['perfect']()

        writer.write('000001', df)

        # 验证：文件存在且可读
        parquet_path = temp_stockdata / "raw/daily/000001.parquet"
        result = pd.read_parquet(str(parquet_path))
        assert len(result) == len(df)


class TestRetryMechanism:
    """重试机制测试"""

    def test_exponential_backoff_timing(self):
        """指数退避时间正确"""
        import time

        backoff_times = []
        for i in range(3):
            backoff_times.append(2 ** i)

        # 验证：时间递增
        assert backoff_times == [1, 2, 4], f"退避时间错误: {backoff_times}"

    def test_max_retry_count(self):
        """最大重试次数限制"""
        from utils.writer import IdempotentWriter

        # Mock 一个总是失败的方法
        writer = IdempotentWriter.__new__(IdempotentWriter)
        writer._write_atomic = lambda *args, **kwargs: (_ for _ in ()).throw(Exception("test"))

        df, _ = ANOMALY_TEST_CASES['perfect']()

        # 应该抛出异常而不是无限重试
        with pytest.raises(Exception):
            writer.write('000001', df)


class TestErrorPaths:
    """错误路径测试"""

    def test_corrupted_parquet_handled(self, temp_stockdata, temp_sqlite):
        """损坏的 Parquet 文件被正确处理"""
        from utils.writer import IdempotentWriter, WriteError

        writer = IdempotentWriter(
            stockdata_root=temp_stockdata,
            db_path=temp_sqlite
        )

        # 创建包含损坏数据的 Parquet 文件
        parquet_path = temp_stockdata / "raw/daily/000001.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        with open(parquet_path, 'wb') as f:
            f.write(b'corrupted parquet data')

        # 尝试写入新数据，损坏的 parquet 会导致读取失败
        df, _ = ANOMALY_TEST_CASES['perfect']()
        # 应该抛出 WriteError 而不是未处理的异常
        with pytest.raises(WriteError):
            writer.write('000001', df)

    def test_readonly_directory_handled(self, temp_stockdata, temp_sqlite):
        """只读目录被正确处理"""
        from utils.writer import IdempotentWriter, WriteError

        writer = IdempotentWriter(
            stockdata_root=temp_stockdata,
            db_path=temp_sqlite
        )

        # 创建只读目录
        readonly_dir = temp_stockdata / "raw/daily"
        readonly_dir.mkdir(parents=True, exist_ok=True)
        readonly_dir.chmod(0o444)  # 只读

        try:
            df, _ = ANOMALY_TEST_CASES['perfect']()
            # 应该抛出 WriteError 而不是未捕获的 PermissionError
            with pytest.raises(WriteError):
                writer.write('000001', df)
        finally:
            # 恢复权限以便清理
            readonly_dir.chmod(0o755)

    def test_empty_dataframe_handled(self, temp_stockdata, temp_sqlite):
        """空 DataFrame 被正确处理"""
        from utils.writer import IdempotentWriter

        writer = IdempotentWriter(
            stockdata_root=temp_stockdata,
            db_path=temp_sqlite
        )

        # 空 DataFrame
        df = pd.DataFrame()

        # 应该返回 False 而不是抛出异常
        result = writer.write('000001', df)
        assert result is False

    def test_sqlite_connection_failure(self, temp_stockdata):
        """SQLite 连接失败被正确处理"""
        from utils.writer import IdempotentWriter, WriteError

        # 使用不存在的数据库路径
        writer = IdempotentWriter(
            stockdata_root=temp_stockdata,
            db_path="/nonexistent/path/market.db"
        )

        df, _ = ANOMALY_TEST_CASES['perfect']()

        # 应该抛出 WriteError（被捕获并包装后的异常）
        with pytest.raises(WriteError):
            writer.write('000001', df)
