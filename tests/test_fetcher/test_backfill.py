"""回填模块测试"""

import pytest
import sys
import os
import tempfile
import shutil
import sqlite3
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.fetcher.backfill import (
    BackfillFetcher,
    BackfillReport,
    BackfillResult,
    run_backfill
)


class TestBackfillFetcher:
    """BackfillFetcher 测试"""

    def setup_method(self):
        """使用临时目录"""
        self.temp_dir = tempfile.mkdtemp(prefix="backfill_test_")

        # 创建必要目录
        os.makedirs(os.path.join(self.temp_dir, "raw", "daily"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "sqlite"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "status"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "quarantine"), exist_ok=True)

        # 初始化 SQLite
        self._init_sqlite()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _init_sqlite(self):
        """初始化 SQLite"""
        db_path = os.path.join(self.temp_dir, "sqlite", "market.db")
        conn = sqlite3.connect(db_path)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute("""
            CREATE TABLE stocks (
                code TEXT PRIMARY KEY,
                name TEXT,
                is_active INTEGER DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE TABLE daily_index (
                code TEXT PRIMARY KEY,
                latest_date TEXT
            )
        """)
        # 插入测试股票
        conn.execute("INSERT INTO stocks (code, name) VALUES ('600519', '贵州茅台')")
        conn.execute("INSERT INTO stocks (code, name) VALUES ('000001', '平安银行')")
        conn.commit()
        conn.close()

    def _create_mock_daily(self, code="600519"):
        """创建模拟日线数据"""
        return pd.DataFrame({
            "date": ["2021-03-29", "2021-03-30", "2021-03-31"],
            "code": [code] * 3,
            "open": [1800.0, 1810.0, 1820.0],
            "high": [1850.0, 1860.0, 1870.0],
            "low": [1790.0, 1800.0, 1810.0],
            "close": [1820.0, 1830.0, 1840.0],
            "volume": [1000000, 1100000, 1200000],
            "pct_chg": [0.011, 0.005, 0.005]
        })

    def _create_mock_adj(self):
        """创建模拟复权因子"""
        return pd.DataFrame({
            "date": ["2021-03-29", "2021-03-30", "2021-03-31"],
            "adj_factor": [1.0, 1.0, 1.0]
        })

    def test_initialization(self):
        """初始化测试"""
        # 使用 mock 避免真实 API 调用
        mock_source = Mock()
        mock_scorer = Mock()

        fetcher = BackfillFetcher(
            stockdata_root=self.temp_dir,
            start_date="2021-03-01",
            end_date="2021-03-31",
            codes=["600519"],
            source=mock_source,
            scorer=mock_scorer
        )

        assert fetcher.stockdata_root == self.temp_dir
        assert fetcher.start_date == "2021-03-01"
        assert fetcher.end_date == "2021-03-31"
        # 36 * 0.8 = 28.8 → 28
        assert fetcher.rate_limit == 28
        assert fetcher._source == mock_source

    def test_fetch_stock_success(self):
        """成功回填单只股票"""
        mock_source = Mock()
        mock_scorer = Mock()

        # 模拟质量评分返回高分
        mock_score = Mock()
        mock_score.overall = 95.0
        mock_scorer.score.return_value = mock_score

        fetcher = BackfillFetcher(
            stockdata_root=self.temp_dir,
            start_date="2021-03-01",
            end_date="2021-03-31",
            codes=["600519"],
            source=mock_source,
            scorer=mock_scorer
        )

        mock_daily = self._create_mock_daily()
        mock_adj = self._create_mock_adj()
        mock_source.fetch_daily.return_value = mock_daily
        mock_source.fetch_adj_factor.return_value = mock_adj

        result = fetcher._fetch_stock("600519")

        assert result.status == "success"
        assert result.records_count == 3
        assert result.quality_score is not None

    def test_fetch_stock_no_data(self):
        """无数据返回"""
        mock_source = Mock()
        mock_scorer = Mock()

        fetcher = BackfillFetcher(
            stockdata_root=self.temp_dir,
            start_date="2021-03-01",
            end_date="2021-03-31",
            codes=["600519"],
            source=mock_source,
            scorer=mock_scorer
        )

        mock_source.fetch_daily.return_value = pd.DataFrame()
        mock_source.fetch_adj_factor.return_value = pd.DataFrame()

        result = fetcher._fetch_stock("600519")

        assert result.status == "failed"
        assert result.error == "no_data"

    def test_quarantine_low_quality(self):
        """低质量数据隔离"""
        mock_source = Mock()
        mock_scorer = Mock()

        # 模拟质量评分为0
        mock_score = Mock()
        mock_score.overall = 0
        mock_scorer.score.return_value = mock_score

        fetcher = BackfillFetcher(
            stockdata_root=self.temp_dir,
            start_date="2021-03-01",
            end_date="2021-03-31",
            codes=["600519"],
            source=mock_source,
            scorer=mock_scorer
        )

        mock_daily = pd.DataFrame({
            "date": ["2021-03-29"],
            "code": ["600519"],
            "open": [0],
            "high": [0],
            "low": [0],
            "close": [0],
            "volume": [0],
            "pct_chg": [0]
        })
        mock_adj = self._create_mock_adj()
        mock_source.fetch_daily.return_value = mock_daily
        mock_source.fetch_adj_factor.return_value = mock_adj

        result = fetcher._fetch_stock("600519")

        assert result.status == "quarantined"
        assert "records < 60" in result.quarantined_reason

        # 验证隔离文件存在
        quarantine_dir = os.path.join(self.temp_dir, "quarantine", "backfill")
        assert os.path.exists(quarantine_dir)

    def test_progress_persistence(self):
        """进度持久化"""
        mock_source = Mock()
        mock_scorer = Mock()

        fetcher = BackfillFetcher(
            stockdata_root=self.temp_dir,
            start_date="2021-03-01",
            end_date="2021-03-31",
            codes=["600519", "000001"],
            source=mock_source,
            scorer=mock_scorer
        )

        # 直接测试 _save_progress
        fetcher._save_progress("600519")

        # 验证进度文件
        progress_file = os.path.join(self.temp_dir, "status", "backfill_2021-03-01_2021-03-31.json")
        assert os.path.exists(progress_file)

        import json
        with open(progress_file) as f:
            data = json.load(f)
        assert "600519" in data["completed_codes"]

    def test_resume_from_checkpoint(self):
        """从断点恢复"""
        mock_source = Mock()
        mock_scorer = Mock()

        # 预先写入进度
        progress_file = os.path.join(self.temp_dir, "status", "backfill_2021-03-01_2021-03-31.json")
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)

        import json
        with open(progress_file, "w") as f:
            json.dump({
                "start_date": "2021-03-01",
                "end_date": "2021-03-31",
                "completed_codes": ["600519"]
            }, f)

        fetcher = BackfillFetcher(
            stockdata_root=self.temp_dir,
            start_date="2021-03-01",
            end_date="2021-03-31",
            codes=["600519", "000001"],
            source=mock_source,
            scorer=mock_scorer
        )

        completed = fetcher._load_progress()
        assert "600519" in completed
        assert "000001" not in completed

    def test_rate_limit_calculation(self):
        """速率限制计算"""
        mock_source = Mock()
        mock_scorer = Mock()

        fetcher = BackfillFetcher(
            stockdata_root=self.temp_dir,
            start_date="2021-03-01",
            end_date="2021-03-31",
            rate_limit_buffer=0.8,
            source=mock_source,
            scorer=mock_scorer
        )

        # 36 * 0.8 = 28.8 → 28
        assert fetcher.rate_limit == 28
        # 60 / 28 = 2.14
        assert abs(fetcher.min_interval - 2.14) < 0.01


class TestBackfillReport:
    """BackfillReport 测试"""

    def test_report_metrics(self):
        """报告指标计算"""
        report = BackfillReport(
            start_date="2021-03-01",
            end_date="2021-03-31",
            total_stocks=100
        )

        report.success_count = 90
        report.failed_count = 5
        report.quarantined_count = 5
        report.end_time = datetime.now()

        assert report.success_count == 90
        assert report.failed_count == 5
        assert report.quarantined_count == 5
        assert report.duration_minutes >= 0

    def test_report_to_dict(self):
        """报告转字典"""
        report = BackfillReport(
            start_date="2021-03-01",
            end_date="2021-03-31",
            total_stocks=1
        )

        result = BackfillResult(
            code="600519",
            start_date="2021-03-01",
            end_date="2021-03-31",
            status="success",
            records_count=100,
            quality_score=85.0
        )
        report.results.append(result)

        data = report.to_dict()

        assert data["start_date"] == "2021-03-01"
        assert data["total_stocks"] == 1
        assert len(data["results"]) == 1


class TestRateLimit:
    """速率限制测试"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp(prefix="ratelimit_test_")

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_rate_limit_sleep(self):
        """速率限制休眠"""
        mock_source = Mock()
        mock_scorer = Mock()

        fetcher = BackfillFetcher(
            stockdata_root=self.temp_dir,
            start_date="2021-03-01",
            end_date="2021-03-31",
            codes=["600519"],
            source=mock_source,
            scorer=mock_scorer
        )

        # 设置最小间隔
        fetcher.min_interval = 0.1
        # 模拟刚完成一次API调用，使elapsed < min_interval
        import time
        fetcher._last_call_time = time.time()

        start = time.time()
        fetcher._rate_limit_sleep()
        elapsed = time.time() - start

        # 至少休眠 0.1 秒
        assert elapsed >= 0.09


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
