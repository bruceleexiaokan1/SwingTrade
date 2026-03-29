"""东方财富数据源测试"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.fetcher.sources.eastmoney_source import EastMoneySource
from src.data.fetcher.exceptions import NetworkError


class TestEastMoneySource:
    """EastMoneySource 测试"""

    def setup_method(self):
        """初始化"""
        # Mock akshare
        self.mock_ak = MagicMock()
        self.patcher = patch.dict('sys.modules', {'akshare': self.mock_ak})
        self.patcher.start()

    def teardown_method(self):
        """清理"""
        self.patcher.stop()

    def test_initialization(self):
        """初始化测试"""
        source = EastMoneySource()
        assert source.name == "eastmoney"

    def test_fetch_individual_fund_flow(self):
        """获取个股资金流"""
        source = EastMoneySource()

        # Mock 返回数据
        mock_df = pd.DataFrame({
            "日期": ["2026-03-27", "2026-03-28"],
            "收盘价": [1420.0, 1430.0],
            "涨跌幅": [0.5, 0.7],
            "主力净流入-净额": [-1000000, 2000000],
            "主力净流入-净占比": [-0.5, 1.0],
            "超大单净流入-净额": [-500000, 1000000],
            "超大单净流入-净占比": [-0.25, 0.5],
            "大单净流入-净额": [-300000, 500000],
            "大单净流入-净占比": [-0.15, 0.25],
            "中单净流入-净额": [200000, 300000],
            "中单净流入-净占比": [0.1, 0.15],
            "小单净流入-净额": [100000, -500000],
            "小单净流入-净占比": [0.05, -0.25]
        })
        source.ak.stock_individual_fund_flow.return_value = mock_df

        result = source.fetch_individual_fund_flow("600519", "sh")

        assert len(result) == 2
        assert "main_net_inflow" in result.columns
        assert "code" in result.columns
        assert result["code"].iloc[0] == "600519"

    def test_fetch_industry_fund_flow(self):
        """获取行业资金流"""
        source = EastMoneySource()

        mock_df = pd.DataFrame({
            "序号": [1, 2],
            "名称": ["有色金属", "医药生物"],
            "行业指数": [3500.0, 6500.0],
            "行业-涨跌幅": [2.5, 1.8],
            "流入资金": [100e8, 80e8],
            "流出资金": [80e8, 70e8],
            "净额": [20e8, 10e8],
            "公司家数": [50, 100],
            "领涨股": ["赣锋锂业", "恒瑞医药"],
            "领涨股-涨跌幅": [5.0, 3.0]
        })
        source.ak.stock_sector_fund_flow_rank.return_value = mock_df

        result = source.fetch_industry_fund_flow("今日")

        assert len(result) == 2
        assert "sector" in result.columns
        assert "net_inflow" in result.columns

    def test_fetch_hsgt_north_flow(self):
        """获取北向资金"""
        source = EastMoneySource()

        mock_df = pd.DataFrame({
            "交易日": ["2026-03-27", "2026-03-27"],
            "类型": ["沪港通", "深港通"],
            "板块": ["沪股通", "深股通"],
            "资金方向": ["北向", "北向"],
            "交易状态": [3, 3],
            "成交净买额": [1000000, 2000000],
            "资金净流入": [1000000, 2000000],
            "当日资金余额": [10000000, 20000000],
            "上涨数": [100, 200],
            "持平数": [30, 40],
            "下跌数": [70, 80],
            "相关指数": ["上证指数", "深证成指"],
            "指数涨跌幅": [0.5, 1.0]
        })
        source.ak.stock_hsgt_fund_flow_summary_em.return_value = mock_df

        result = source.fetch_hsgt_north_flow()

        assert len(result) == 2
        assert "net_inflow" in result.columns
        assert "direction" in result.columns

    def test_empty_dataframe(self):
        """空数据返回"""
        source = EastMoneySource()
        source.ak.stock_individual_fund_flow.return_value = pd.DataFrame()

        result = source.fetch_individual_fund_flow("600519", "sh")

        assert len(result) == 0

    def test_network_error(self):
        """网络错误处理"""
        source = EastMoneySource()
        source.ak.stock_individual_fund_flow.side_effect = Exception("Network error")

        with pytest.raises(NetworkError):
            source.fetch_individual_fund_flow("600519", "sh")

    def test_rate_limit(self):
        """速率限制"""
        source = EastMoneySource()

        # Mock 返回空数据以快速完成
        source.ak.stock_individual_fund_flow.return_value = pd.DataFrame()

        import time
        start = time.time()

        # 连续调用两次
        source.fetch_individual_fund_flow("600519", "sh")
        source.fetch_individual_fund_flow("000001", "sz")

        elapsed = time.time() - start

        # 两次调用之间应该有至少 0.2 秒间隔 (1/5 = 0.2s)
        assert elapsed >= 0.15, f"Rate limit not working, elapsed: {elapsed}"

    def test_to_em_market(self):
        """市场代码转换"""
        source = EastMoneySource()

        assert source._to_em_market("600519") == "sh"
        assert source._to_em_market("000001") == "sz"
        assert source._to_em_market("300001") == "sz"
        assert source._to_em_market("400001") == "bj"
        assert source._to_em_market("800001") == "bj"
        assert source._to_em_market("900001") == "bj"

    def test_fetch_daily_returns_empty(self):
        """fetch_daily 应返回空 DataFrame"""
        source = EastMoneySource()

        result = source.fetch_daily("600519", "2026-03-01", "2026-03-31")

        assert len(result) == 0

    def test_fetch_stock_list_returns_empty(self):
        """fetch_stock_list 应返回空 DataFrame"""
        source = EastMoneySource()

        result = source.fetch_stock_list()

        assert len(result) == 0

    def test_is_available_success(self):
        """可用性检查成功"""
        source = EastMoneySource()
        source.ak.stock_sector_fund_flow_rank.return_value = pd.DataFrame({"a": [1]})

        assert source.is_available() == True

    def test_is_available_failure(self):
        """可用性检查失败"""
        source = EastMoneySource()
        source.ak.stock_sector_fund_flow_rank.side_effect = Exception("Connection failed")

        assert source.is_available() == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
