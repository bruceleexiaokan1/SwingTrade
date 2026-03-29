"""价格转换器测试"""

import pytest
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from src.data.fetcher.price_converter import (
    convert_to_forward_adj,
    convert_to_post_adj,
    get_current_adj_factor
)


class TestConvertToForwardAdj:
    """前复权转换测试"""

    def test_basic_conversion(self):
        """基本转换测试"""
        df = pd.DataFrame({
            "date": ["2021-03-29", "2021-03-30", "2021-03-31"],
            "code": ["600519"] * 3,
            "adj_factor": [1.0, 1.0, 1.0],
            "close_adj": [100.0, 110.0, 120.0],
            "open_adj": [99.0, 108.0, 119.0],
            "high_adj": [101.0, 112.0, 122.0],
            "low_adj": [98.0, 107.0, 118.0]
        })

        result = convert_to_forward_adj(df)

        # adj_factor 都是 1.0，比率 = 1.0，前复权价 = 后复权价
        assert "forward_close" in result.columns
        assert result["forward_close"].iloc[0] == 100.0
        assert result["forward_close"].iloc[-1] == 120.0

    def test_split_adjustment(self):
        """模拟股票拆分（复权因子变化）"""
        # 假设：2021-03-29 的 adj_factor = 1.0, 2021-03-31 的 adj_factor = 2.0
        # 这表示中间发生了 2:1 拆分
        df = pd.DataFrame({
            "date": ["2021-03-29", "2021-03-31"],
            "code": ["600519"] * 2,
            "adj_factor": [1.0, 2.0],
            "close_adj": [100.0, 200.0],  # 后复权价保持连续
            "open_adj": [99.0, 198.0],
            "high_adj": [101.0, 202.0],
            "low_adj": [98.0, 197.0]
        })

        result = convert_to_forward_adj(df)

        # 最新 adj_factor = 2.0
        # 第一天：前复权 = 100 * (2.0 / 1.0) = 200
        # 第二天：前复权 = 200 * (2.0 / 2.0) = 200
        assert abs(result["forward_close"].iloc[0] - 200.0) < 0.01
        assert abs(result["forward_close"].iloc[1] - 200.0) < 0.01

        # 前复权价格连续（都是200），反映真实成本
        assert result["forward_close"].iloc[0] == result["forward_close"].iloc[1]

    def test_empty_dataframe(self):
        """空数据测试"""
        df = pd.DataFrame()
        result = convert_to_forward_adj(df)
        assert len(result) == 0

    def test_invalid_adj_factor(self):
        """无效复权因子测试"""
        df = pd.DataFrame({
            "date": ["2021-03-29"],
            "code": ["600519"],
            "adj_factor": [0.0],  # 无效
            "close_adj": [100.0],
            "open_adj": [99.0],
            "high_adj": [101.0],
            "low_adj": [98.0]
        })

        result = convert_to_forward_adj(df)
        # 应该使用 1.0 作为默认值
        assert result["forward_close"].iloc[0] == 100.0

    def test_missing_columns(self):
        """缺失列测试"""
        df = pd.DataFrame({
            "date": ["2021-03-29"],
            "code": ["600519"]
        })

        result = convert_to_forward_adj(df)
        # 应该直接返回，不添加列
        assert "forward_close" not in result.columns


class TestConvertToPostAdj:
    """后复权转换测试"""

    def test_basic_conversion(self):
        """基本转换测试"""
        df = pd.DataFrame({
            "date": ["2021-03-29", "2021-03-31"],
            "adj_factor": [1.0, 2.0],
            "forward_close": [200.0, 200.0],
            "forward_open": [198.0, 198.0],
            "forward_high": [202.0, 202.0],
            "forward_low": [197.0, 197.0]
        })

        result = convert_to_post_adj(df)

        # 最新 adj_factor = 2.0, base = 1.0
        # 第一天：后复权 = 200 / (2.0 / 1.0) = 100
        # 第二天：后复权 = 200 / (2.0 / 1.0) = 100
        assert abs(result["close_adj"].iloc[0] - 100.0) < 0.01


class TestGetCurrentAdjFactor:
    """获取当前复权因子测试"""

    def test_normal_case(self):
        """正常情况"""
        df = pd.DataFrame({
            "adj_factor": [1.0, 1.5, 2.0]
        })
        assert get_current_adj_factor(df) == 2.0

    def test_empty_dataframe(self):
        """空数据"""
        df = pd.DataFrame()
        assert get_current_adj_factor(df) == 1.0

    def test_missing_column(self):
        """缺失列"""
        df = pd.DataFrame({"date": ["2021-03-29"]})
        assert get_current_adj_factor(df) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
