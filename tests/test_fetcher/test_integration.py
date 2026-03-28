"""数据采集集成测试

使用临时目录和 Mock 数据源，不依赖外部 API。
测试完整流程：采集 -> 质量评分 -> 写入
"""

import pytest
import sys
import os
import tempfile
import shutil
import sqlite3
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.fetcher.fetch_daily import DailyFetcher
from src.data.fetcher.quality_scorer import QualityScorer, QualityScore
from src.data.fetcher.sources.tushare_source import TushareSource
from src.data.fetcher.sources.akshare_source import AkShareSource
from src.data.fetcher.retry_handler import RetryHandler, FetchResult


class MockTushareSource:
    """Mock Tushare 数据源"""

    name = "tushare"

    def __init__(self, return_data=None, error=None):
        self.return_data = return_data
        self.error = error
        self.fetch_count = 0

    def fetch_daily(self, code, start_date, end_date):
        self.fetch_count += 1
        if self.error:
            raise self.error
        if self.return_data is not None:
            return self.return_data.copy()
        return pd.DataFrame()

    def fetch_stock_list(self):
        return pd.DataFrame({
            "code": ["600519", "000001"],
            "name": ["贵州茅台", "平安银行"],
            "market": ["sh", "sz"]
        })


class MockAkShareSource:
    """Mock AkShare 数据源"""

    name = "akshare"

    def __init__(self, return_data=None):
        self.return_data = return_data

    def fetch_daily(self, code, start_date, end_date):
        if self.return_data is not None:
            return self.return_data.copy()
        return pd.DataFrame()


class TestDailyFetcherIntegration:
    """DailyFetcher 集成测试"""

    def setup_method(self):
        """使用临时目录作为 StockData root"""
        self.temp_dir = tempfile.mkdtemp(prefix="stockdata_test_")

        # 创建必要的目录结构
        os.makedirs(os.path.join(self.temp_dir, "raw", "daily"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "sqlite"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "processed"), exist_ok=True)

        # 初始化 SQLite 数据库
        self._init_sqlite_db()

        self.fetcher = DailyFetcher(
            stockdata_root=self.temp_dir,
            target_date="2026-03-27",
            use_akshare_verify=False  # 禁用外部验证
        )

    def teardown_method(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _init_sqlite_db(self):
        """初始化 SQLite 数据库"""
        db_path = os.path.join(self.temp_dir, "sqlite", "market.db")
        conn = sqlite3.connect(db_path)
        conn.execute('PRAGMA journal_mode=WAL')

        # 创建 stock_list 表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_list (
                code TEXT PRIMARY KEY,
                name TEXT,
                market TEXT,
                is_active INTEGER DEFAULT 1,
                list_date TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建 daily_index 表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_index (
                code TEXT PRIMARY KEY,
                latest_date TEXT,
                file_path TEXT,
                row_count INTEGER,
                start_date TEXT,
                end_date TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建 update_log 表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS update_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_type TEXT NOT NULL,
                code TEXT,
                update_date TEXT NOT NULL,
                status TEXT NOT NULL,
                row_count INTEGER,
                error_msg TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def _create_mock_data(self, code="600519", close=1820.0):
        """创建 Mock 日线数据"""
        return pd.DataFrame({
            "date": ["2026-03-27"],
            "code": [code],
            "open": [1800.0],
            "high": [1850.0],
            "low": [1790.0],
            "close": [close],
            "volume": [1000000],
            "pct_chg": [0.011]
        })

    def test_successful_fetch_and_write(self):
        """成功采集并写入"""
        mock_data = self._create_mock_data()

        with patch.object(TushareSource, '__init__', lambda x: None):
            with patch.object(TushareSource, 'fetch_daily', lambda x, c, s, e: mock_data):
                fetcher = DailyFetcher(
                    stockdata_root=self.temp_dir,
                    target_date="2026-03-27",
                    use_akshare_verify=False
                )
                fetcher.scraper = Mock()
                fetcher.scraper.name = "tushare"
                fetcher.scraper.fetch_daily = lambda c, s, e: mock_data

                result = fetcher._fetch_single("600519", "贵州茅台")

        assert result.fetch_status == "success"
        assert result.write_status == "success"
        assert result.quality_score >= 80

        # 验证文件已写入
        parquet_file = os.path.join(self.temp_dir, "raw", "daily", "600519.parquet")
        assert os.path.exists(parquet_file)

        # 验证 Parquet 内容
        df = pd.read_parquet(parquet_file)
        assert len(df) == 1
        assert df["close"].iloc[0] == 1820.0

    def test_quality_rejected_data(self):
        """质量不达标被拒绝"""
        # 价格超出正常范围（会被 base_result 检测为严重问题）
        mock_data = pd.DataFrame({
            "date": ["2026-03-27"],
            "code": ["600519"],
            "open": [0.001],
            "high": [0.005],
            "low": [0.001],
            "close": [0.002],
            "volume": [1000000]
        })

        with patch.object(TushareSource, '__init__', lambda x: None):
            fetcher = DailyFetcher(
                stockdata_root=self.temp_dir,
                target_date="2026-03-27",
                use_akshare_verify=False
            )
            fetcher.scraper = Mock()
            fetcher.scraper.name = "tushare"
            fetcher.scraper.fetch_daily = lambda c, s, e: mock_data

            result = fetcher._fetch_single("600519", "贵州茅台")

        assert result.fetch_status == "success"
        assert result.write_status == "rejected"
        assert result.fail_type == "quality"

    def test_idempotent_write(self):
        """幂等写入：同一数据多次写入不重复"""
        mock_data = self._create_mock_data()

        with patch.object(TushareSource, '__init__', lambda x: None):
            fetcher = DailyFetcher(
                stockdata_root=self.temp_dir,
                target_date="2026-03-27",
                use_akshare_verify=False
            )
            fetcher.scraper = Mock()
            fetcher.scraper.name = "tushare"
            fetcher.scraper.fetch_daily = lambda c, s, e: mock_data

            # 第一次写入
            result1 = fetcher._fetch_single("600519", "贵州茅台")
            # 第二次写入（同一数据）
            result2 = fetcher._fetch_single("600519", "贵州茅台")

        assert result1.write_status == "success"
        assert result2.write_status == "success"

        # 验证只有一条记录
        parquet_file = os.path.join(self.temp_dir, "raw", "daily", "600519.parquet")
        df = pd.read_parquet(parquet_file)
        assert len(df) == 1

    def test_multiple_days_accumulation(self):
        """多日数据累积"""
        # 第一天数据
        day1_data = pd.DataFrame({
            "date": ["2026-03-25"],
            "code": ["600519"],
            "open": [1780.0],
            "high": [1820.0],
            "low": [1770.0],
            "close": [1800.0],
            "volume": [900000]
        })

        # 第二天数据
        day2_data = pd.DataFrame({
            "date": ["2026-03-26"],
            "code": ["600519"],
            "open": [1800.0],
            "high": [1850.0],
            "low": [1790.0],
            "close": [1820.0],
            "volume": [1000000]
        })

        with patch.object(TushareSource, '__init__', lambda x: None):
            fetcher = DailyFetcher(
                stockdata_root=self.temp_dir,
                target_date="2026-03-26",
                use_akshare_verify=False
            )
            fetcher.scraper = Mock()
            fetcher.scraper.name = "tushare"

            # 写入第一天
            fetcher.scraper.fetch_daily = lambda c, s, e: day1_data
            fetcher._fetch_single("600519", "贵州茅台")

            # 写入第二天
            fetcher.scraper.fetch_daily = lambda c, s, e: day2_data
            fetcher._fetch_single("600519", "贵州茅台")

        # 验证有两条记录
        parquet_file = os.path.join(self.temp_dir, "raw", "daily", "600519.parquet")
        df = pd.read_parquet(parquet_file)
        assert len(df) == 2
        assert sorted(df["date"].tolist()) == ["2026-03-25", "2026-03-26"]

    def test_akshare_verification_triggered(self):
        """AkShare 验证被触发（质量分在 60-80 之间）"""
        # 创建刚好触发验证分数的数据（需要验证）
        mock_data = pd.DataFrame({
            "date": ["2026-03-27"],
            "code": ["600519"],
            "open": [1800.0],
            "high": [1850.0],
            "low": [1790.0],
            "close": [1820.0],
            "volume": [1000000]
            # 故意不提供 pct_chg，导致 field_completeness < 100
        })

        verify_data = pd.DataFrame({
            "date": ["2026-03-27"],
            "code": ["600519"],
            "close": [1820.0]
        })

        with patch.object(TushareSource, '__init__', lambda x: None):
            with patch.object(AkShareSource, '__init__', lambda x: None):
                fetcher = DailyFetcher(
                    stockdata_root=self.temp_dir,
                    target_date="2026-03-27",
                    use_akshare_verify=True
                )

                fetcher.scraper = Mock()
                fetcher.scraper.name = "tushare"
                fetcher.scraper.fetch_daily = lambda c, s, e: mock_data

                fetcher.akshare = Mock()
                fetcher.akshare.name = "akshare"
                fetcher.akshare.fetch_daily = lambda c, s, e: verify_data

                result = fetcher._fetch_single("600519", "贵州茅台")

        # 应该使用 AkShare 数据重新评分
        assert result.fetch_status == "success"
        assert result.quality_score is not None

    def test_fetch_with_network_error(self):
        """网络错误处理"""
        from src.data.fetcher.exceptions import NetworkError

        mock_error = NetworkError("Connection timeout")

        with patch.object(TushareSource, '__init__', lambda x: None):
            fetcher = DailyFetcher(
                stockdata_root=self.temp_dir,
                target_date="2026-03-27",
                use_akshare_verify=False
            )
            fetcher.scraper = Mock()
            fetcher.scraper.name = "tushare"
            fetcher.scraper.fetch_daily = lambda c, s, e: (_ for _ in ()).throw(mock_error)

            result = fetcher._fetch_single("600519", "贵州茅台")

        assert result.fetch_status == "failed"
        assert result.fail_type == "network"
        assert result.attempts == 2  # 重试了2次

    def test_sqlite_index_updated(self):
        """SQLite 索引正确更新"""
        mock_data = self._create_mock_data()

        with patch.object(TushareSource, '__init__', lambda x: None):
            fetcher = DailyFetcher(
                stockdata_root=self.temp_dir,
                target_date="2026-03-27",
                use_akshare_verify=False
            )
            fetcher.scraper = Mock()
            fetcher.scraper.name = "tushare"
            fetcher.scraper.fetch_daily = lambda c, s, e: mock_data

            fetcher._fetch_single("600519", "贵州茅台")

        # 验证 daily_index 已更新
        db_path = os.path.join(self.temp_dir, "sqlite", "market.db")
        conn = sqlite3.connect(db_path)
        index = conn.execute(
            "SELECT code, latest_date, start_date, end_date, row_count FROM daily_index"
        ).fetchone()
        conn.close()

        assert index is not None
        assert index[0] == "600519"
        assert index[1] == "2026-03-27"
        assert index[4] == 1  # row_count


class TestQualityScorerWithRealData:
    """使用真实数据结构的质量评分测试"""

    def setup_method(self):
        self.scorer = QualityScorer()

    def test_score_matches_expected_weights(self):
        """验证评分权重计算"""
        record = {
            "date": "2026-03-28",
            "code": "600519",
            "open": 1800.0,
            "high": 1850.0,
            "low": 1790.0,
            "close": 1820.0,
            "volume": 1000000,
            "pct_chg": 0.011
        }

        score = self.scorer.score(record)

        # 单源: source_consistency = 95
        # 全部字段: field_completeness = 100
        # 范围正常: range_validity = 100
        # 无历史: historical_anomaly = 100
        expected_overall = 95 * 0.30 + 100 * 0.20 + 100 * 0.30 + 100 * 0.20
        assert score.overall == expected_overall

    def test_verify_threshold_boundary(self):
        """验证阈值边界"""
        # 刚好 60 分
        score_60 = QualityScore(95, 100, 100, 100, 60)
        assert self.scorer.should_write(score_60) is True
        assert self.scorer.should_verify(score_60) is True
        assert self.scorer.should_reject(score_60) is False

        # 刚好 80 分
        score_80 = QualityScore(95, 100, 100, 100, 80)
        assert self.scorer.should_write(score_80) is True
        assert self.scorer.should_verify(score_80) is False
        assert self.scorer.should_reject(score_80) is False

        # 刚好 59 分
        score_59 = QualityScore(95, 100, 100, 100, 59)
        assert self.scorer.should_write(score_59) is False
        assert self.scorer.should_verify(score_59) is False
        assert self.scorer.should_reject(score_59) is True


class TestRetryHandlerUnit:
    """RetryHandler 单元测试"""

    def test_successful_fetch(self):
        """一次成功"""
        mock_source = Mock()
        mock_source.name = "test"
        mock_source.fetch_daily = Mock(return_value=pd.DataFrame({"close": [100]}))

        handler = RetryHandler(max_attempts=2)
        result = handler.fetch_with_retry(mock_source, "600519", "2026-03-27", "2026-03-27")

        assert result.fetch_status == "success"
        assert result.attempts == 1

    def test_immediate_retry_on_failure(self):
        """失败后立即重试"""
        call_count = 0

        def fail_twice(*args):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary error")
            return pd.DataFrame({"close": [100]})

        mock_source = Mock()
        mock_source.name = "test"
        mock_source.fetch_daily = fail_twice

        handler = RetryHandler(max_attempts=2)
        result = handler.fetch_with_retry(mock_source, "600519", "2026-03-27", "2026-03-27")

        assert result.fetch_status == "success"
        assert result.attempts == 2

    def test_no_dataReturned(self):
        """无数据返回"""
        mock_source = Mock()
        mock_source.name = "test"
        mock_source.fetch_daily = Mock(return_value=pd.DataFrame())

        handler = RetryHandler(max_attempts=2)
        result = handler.fetch_with_retry(mock_source, "600519", "2026-03-27", "2026-03-27")

        assert result.fetch_status == "failed"
        assert result.fail_reason == "no_data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
