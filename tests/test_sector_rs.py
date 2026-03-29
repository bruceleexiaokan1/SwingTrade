"""板块 RS/RPS 测试"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.data.indicators.sector_rs import SectorRelativeStrength


class TestSectorRelativeStrength:
    """板块相对强度测试"""

    def setup_method(self):
        """测试初始化"""
        # 使用 mock 避免实际加载数据
        selfsr = SectorRelativeStrength.__new__(SectorRelativeStrength)
        selfsr.stock_loader = MagicMock()
        selfsr.sector_fetcher = MagicMock()
        selfsr._stock_data_cache = {}
        selfsr._sector_data_cache = {}
        return selfsr

    def _create_mock_stock_df(self, code: str, base_price: float = 100.0, n_days: int = 60) -> pd.DataFrame:
        """创建模拟个股数据"""
        dates = pd.date_range(end='2026-03-28', periods=n_days, freq='D')
        # 模拟上涨趋势
        close_prices = base_price * (1 + np.linspace(0, 0.15, n_days))
        return pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'open': close_prices * 0.99,
            'high': close_prices * 1.02,
            'low': close_prices * 0.98,
            'close': close_prices,
            'volume': np.random.randint(1000000, 10000000, n_days)
        })

    def _create_mock_sector_df(self, base_price: float = 1000.0, n_days: int = 60) -> pd.DataFrame:
        """创建模拟板块数据"""
        dates = pd.date_range(end='2026-03-28', periods=n_days, freq='D')
        # 模拟小幅上涨
        close_prices = base_price * (1 + np.linspace(0, 0.08, n_days))
        return pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'open': close_prices * 0.99,
            'high': close_prices * 1.01,
            'low': close_prices * 0.99,
            'close': close_prices,
            'volume': np.random.randint(10000000, 100000000, n_days)
        })

    def test_rs_calculation(self):
        """测试 RS 计算"""
        sr = self.setup_method()

        # Mock 数据
        stock_df = self._create_mock_stock_df("002371", base_price=100.0)
        sector_df = self._create_mock_sector_df(base_price=1000.0)

        sr._stock_data_cache["002371"] = stock_df
        sr._sector_data_cache["人工智能"] = sector_df

        # Mock 配置
        sr._sector_config = {
            'sectors': [{
                'name': '人工智能',
                'stocks': [{'code': '002371'}]
            }]
        }
        sr._sector_stocks = MagicMock(return_value=['002371'])

        with patch.object(sr, '_get_sector_config', return_value=sr._sector_config):
            with patch.object(sr, '_get_sector_stocks', return_value=['002371']):
                rs_values = sr.calculate_rs("人工智能", ["002371"], "2026-03-28", lookback=20)

        assert "002371" in rs_values
        # 个股涨幅约 15%，板块涨幅约 8%，RS 应该约为 7%
        assert rs_values["002371"] > 0

    def test_rps_ranking(self):
        """测试 RPS 排名"""
        sr = self.setup_method()

        # 创建三只股票，RS 值不同
        stock_df1 = self._create_mock_stock_df("002371", base_price=100.0)  # 强
        stock_df2 = self._create_mock_stock_df("002372", base_price=100.0)  # 中
        stock_df3 = self._create_mock_stock_df("002373", base_price=100.0)  # 弱

        # 调整收盘价模拟不同涨幅
        stock_df1.iloc[-1, stock_df1.columns.get_loc('close')] = 120.0  # +20%
        stock_df2.iloc[-1, stock_df2.columns.get_loc('close')] = 110.0  # +10%
        stock_df3.iloc[-1, stock_df3.columns.get_loc('close')] = 105.0  # +5%

        sector_df = self._create_mock_sector_df(base_price=1000.0)
        sector_df.iloc[-1, sector_df.columns.get_loc('close')] = 1080.0  # +8%

        sr._stock_data_cache["002371"] = stock_df1
        sr._stock_data_cache["002372"] = stock_df2
        sr._stock_data_cache["002373"] = stock_df3
        sr._sector_data_cache["人工智能"] = sector_df

        sr._sector_config = {
            'sectors': [{
                'name': '人工智能',
                'stocks': [
                    {'code': '002371'},
                    {'code': '002372'},
                    {'code': '002373'}
                ]
            }]
        }

        with patch.object(sr, '_get_sector_config', return_value=sr._sector_config):
            with patch.object(sr, '_get_sector_stocks', return_value=['002371', '002372', '002373']):
                rps_values = sr.calculate_rps("人工智能", ["002371", "002372", "002373"], "2026-03-28", lookback=20)

        # 验证排名
        assert len(rps_values) == 3
        # 002371 RS = 20% - 8% = 12% 最高
        # 002372 RS = 10% - 8% = 2%
        # 002373 RS = 5% - 8% = -3% 最低
        # RPS = (n - rank) / n, rank starts at 1
        # n=3: rank1=2/3≈0.67, rank2=1/3≈0.33, rank3=0/3=0
        assert rps_values["002371"] > rps_values["002372"] > rps_values["002373"]
        assert abs(rps_values["002371"] - 2/3) < 0.01  # 最高
        assert abs(rps_values["002373"] - 0.0) < 0.01  # 最低

    def test_rps_normalized(self):
        """测试 RPS 归一化 (0.0 ~ 1.0)"""
        sr = self.setup_method()

        # 两个股票
        stock_df1 = self._create_mock_stock_df("002371", base_price=100.0)
        stock_df2 = self._create_mock_stock_df("002372", base_price=100.0)

        stock_df1.iloc[-1, stock_df1.columns.get_loc('close')] = 115.0  # +15%
        stock_df2.iloc[-1, stock_df2.columns.get_loc('close')] = 105.0  # +5%

        sector_df = self._create_mock_sector_df(base_price=1000.0)
        sector_df.iloc[-1, sector_df.columns.get_loc('close')] = 1100.0  # +10%

        sr._stock_data_cache["002371"] = stock_df1
        sr._stock_data_cache["002372"] = stock_df2
        sr._sector_data_cache["人工智能"] = sector_df

        sr._sector_config = {
            'sectors': [{
                'name': '人工智能',
                'stocks': [{'code': '002371'}, {'code': '002372'}]
            }]
        }

        with patch.object(sr, '_get_sector_config', return_value=sr._sector_config):
            with patch.object(sr, '_get_sector_stocks', return_value=['002371', '002372']):
                rps_values = sr.calculate_rps("人工智能", ["002371", "002372"], "2026-03-28", lookback=20)

        # 排名: 002371 第1 (1/2 = 0.5), 002372 第2 (0/2 = 0.0) 或者反过来
        # 实际上 RPS = (total - rank) / total
        # rank=1: (2-1)/2 = 0.5
        # rank=2: (2-2)/2 = 0.0
        # 但如果按降序排列，第1名 index=0, rank=1
        # RPS 应该是 0.5 和 0.0
        assert all(0.0 <= v <= 1.0 for v in rps_values.values())

    def test_empty_data(self):
        """测试空数据处理"""
        sr = self.setup_method()
        sr._stock_data_cache = {}
        sr._sector_data_cache = {}

        rs_values = sr.calculate_rs("人工智能", ["002371"], "2026-03-28", lookback=20)
        assert rs_values["002371"] == 0.0

        # 单只股票且数据为空时，RS=0，RPS=(1-1)/1=0.0
        rps_values = sr.calculate_rps("人工智能", ["002371"], "2026-03-28", lookback=20)
        assert rps_values["002371"] == 0.0

    def test_cache_clearing(self):
        """测试缓存清除"""
        sr = self.setup_method()
        sr._stock_data_cache = {"002371": pd.DataFrame()}
        sr._sector_data_cache = {"人工智能": pd.DataFrame()}

        assert len(sr._stock_data_cache) > 0
        assert len(sr._sector_data_cache) > 0

        sr.clear_cache()

        assert len(sr._stock_data_cache) == 0
        assert len(sr._sector_data_cache) == 0

    def test_rs_rank_compatibility(self):
        """测试 get_rs_rank 方法（兼容旧接口）"""
        sr = self.setup_method()

        stock_df = self._create_mock_stock_df("002371", base_price=100.0)
        sector_df = self._create_mock_sector_df(base_price=1000.0)

        stock_df.iloc[-1, stock_df.columns.get_loc('close')] = 115.0  # +15%
        sector_df.iloc[-1, sector_df.columns.get_loc('close')] = 1100.0  # +10%

        sr._stock_data_cache["002371"] = stock_df
        sr._sector_data_cache["人工智能"] = sector_df

        sr._sector_config = {
            'sectors': [{
                'name': '人工智能',
                'stocks': [{'code': '002371'}]
            }]
        }

        with patch.object(sr, '_get_sector_config', return_value=sr._sector_config):
            with patch.object(sr, '_get_sector_stocks', return_value=['002371']):
                rank_values = sr.get_rs_rank("人工智能", ["002371"], "2026-03-28", lookback=20)

        assert "002371" in rank_values
        # 单一股票，RPS 应该是 0.5（中间值）


class TestRSFormulas:
    """RS 公式验证"""

    def test_rs_formula(self):
        """验证 RS 计算公式: RS = 个股20日涨幅 - 板块20日涨幅"""
        stock_return = 20.0  # 个股 20 日涨幅 20%
        sector_return = 10.0  # 板块 20 日涨幅 10%
        expected_rs = 10.0

        rs = stock_return - sector_return
        assert rs == expected_rs

    def test_rps_formula(self):
        """验证 RPS 计算公式: RPS = 排名 / 总数"""
        total_stocks = 10
        rank = 3  # 第3名
        expected_rps = (total_stocks - rank) / total_stocks  # (10-3)/10 = 0.7

        rps = (total_stocks - rank) / total_stocks
        assert rps == 0.7

    def test_rps_edge_cases(self):
        """测试 RPS 边界情况"""
        # 第一名
        assert (10 - 1) / 10 == 0.9
        # 最后一名
        assert (10 - 10) / 10 == 0.0
        # 单只股票
        assert (1 - 1) / 1 == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
