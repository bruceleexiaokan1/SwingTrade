"""IndexFetcher 测试"""

import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from src.data.fetcher.index_fetcher import IndexFetcher, INDEX_POOL


class TestIndexFetcher:
    """IndexFetcher 测试"""

    def setup_method(self):
        """使用临时目录"""
        self.temp_dir = tempfile.mkdtemp(prefix="index_test_")
        os.makedirs(os.path.join(self.temp_dir, "raw", "index"), exist_ok=True)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """初始化测试"""
        with patch('src.data.fetcher.index_fetcher.os.getenv') as mock_getenv:
            mock_getenv.return_value = "fake_token"

            with patch('tushare.pro_api') as mock_pro:
                mock_pro.return_value = Mock()

                fetcher = IndexFetcher(
                    stockdata_root=self.temp_dir,
                    start_date="2021-03-29",
                    end_date="2026-03-28"
                )

                assert fetcher.stockdata_root == self.temp_dir
                assert fetcher.start_date == "2021-03-29"
                assert fetcher.end_date == "2026-03-28"

    def test_index_pool(self):
        """指数池测试"""
        assert len(INDEX_POOL) == 6

        codes = [idx["code"] for idx in INDEX_POOL]
        assert "000001.SH" in codes
        assert "000300.SH" in codes
        assert "399006.SZ" in codes
        assert "399001.SZ" in codes

    def test_rate_limit_calculation(self):
        """速率限制计算"""
        with patch('src.data.fetcher.index_fetcher.os.getenv') as mock_getenv:
            mock_getenv.return_value = "fake_token"

            with patch('tushare.pro_api') as mock_pro:
                mock_pro.return_value = Mock()

                fetcher = IndexFetcher(
                    stockdata_root=self.temp_dir,
                    start_date="2021-03-29",
                    rate_limit_buffer=0.8
                )

                # 36 * 0.8 = 28.8 → 28
                assert fetcher.rate_limit == 28
                # 60 / 28 = 2.14
                assert abs(fetcher.min_interval - 2.14) < 0.01

    def test_write_index(self):
        """写入指数数据"""
        with patch('src.data.fetcher.index_fetcher.os.getenv') as mock_getenv:
            mock_getenv.return_value = "fake_token"

            with patch('tushare.pro_api') as mock_pro:
                mock_pro.return_value = Mock()

                fetcher = IndexFetcher(
                    stockdata_root=self.temp_dir,
                    start_date="2021-03-29"
                )

                df = pd.DataFrame({
                    "date": ["2026-03-27", "2026-03-28"],
                    "code": ["000300.SH"] * 2,
                    "close": [4500.0, 4550.0],
                    "open": [4480.0, 4500.0],
                    "high": [4520.0, 4560.0],
                    "low": [4470.0, 4490.0],
                    "volume": [1000000, 1100000],
                    "pct_chg": [0.5, 1.0]
                })

                fetcher._write_index("000300.SH", df)

                # 验证文件存在
                assert os.path.exists(os.path.join(self.temp_dir, "raw", "index", "000300.SH.parquet"))

                # 验证内容
                read_df = pd.read_parquet(os.path.join(self.temp_dir, "raw", "index", "000300.SH.parquet"))
                assert len(read_df) == 2


class TestIndexFetcherIntegration:
    """IndexFetcher 集成测试（需要真实 API）"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp(prefix="index_integration_")
        os.makedirs(os.path.join(self.temp_dir, "raw", "index"), exist_ok=True)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fetch_single_index(self):
        """获取单个指数（真实验证）"""
        token = os.getenv("TUSHARE_TOKEN")
        if not token:
            pytest.skip("TUSHARE_TOKEN not set")

        from src.data.fetcher.index_fetcher import IndexFetcher

        fetcher = IndexFetcher(
            stockdata_root=self.temp_dir,
            start_date="2026-03-20",
            end_date="2026-03-28"
        )

        df = fetcher.fetch_index("000300.SH")

        assert len(df) > 0
        assert "close" in df.columns
        assert "volume" in df.columns
        assert df["close"].notna().all()
        assert (df["close"] > 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
