"""数据合并工具测试"""

import pytest
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.fetcher.data_merger import (
    merge_daily_with_adj_factor,
    validate_date_freshness
)


class TestMergeDailyWithAdjFactor:
    """merge_daily_with_adj_factor 测试"""

    def test_normal_merge(self):
        """正常合并：日线和复权因子都完整"""
        daily = pd.DataFrame({
            "date": ["2026-03-27", "2026-03-28"],
            "open": [1800, 1810],
            "high": [1850, 1860],
            "low": [1790, 1800],
            "close": [1820, 1830],
            "volume": [1000000, 1100000]
        })
        adj = pd.DataFrame({
            "date": ["2026-03-27", "2026-03-28"],
            "adj_factor": [1.5, 1.5]
        })

        result = merge_daily_with_adj_factor(daily, adj, "600519")

        assert "adj_factor" in result.columns
        assert "close_adj" in result.columns
        assert result["adj_factor"].iloc[0] == 1.5
        assert result["close_adj"].iloc[0] == 1820 * 1.5

    def test_missing_adj_factor_partial(self):
        """部分缺失：用前向填充"""
        daily = pd.DataFrame({
            "date": ["2026-03-27", "2026-03-28"],
            "open": [1800, 1810],
            "high": [1850, 1860],
            "low": [1790, 1800],
            "close": [1820, 1830],
            "volume": [1000000, 1100000]
        })
        # 缺少 03-28 的复权因子
        adj = pd.DataFrame({
            "date": ["2026-03-27"],
            "adj_factor": [1.5]
        })

        result = merge_daily_with_adj_factor(daily, adj, "600519")

        # 03-28 应该用前一个值 1.5
        assert result["adj_factor"].iloc[1] == 1.5

    def test_missing_adj_factor_all(self):
        """全部缺失：使用默认值 1.0"""
        daily = pd.DataFrame({
            "date": ["2026-03-27", "2026-03-28"],
            "open": [1800, 1810],
            "high": [1850, 1860],
            "low": [1790, 1800],
            "close": [1820, 1830],
            "volume": [1000000, 1100000]
        })
        # 完全没有复权因子
        adj = pd.DataFrame(columns=["date", "adj_factor"])

        result = merge_daily_with_adj_factor(daily, adj, "600519")

        # 全部使用默认值 1.0
        assert result["adj_factor"].iloc[0] == 1.0
        assert result["adj_factor"].iloc[1] == 1.0

    def test_empty_daily(self):
        """空日线数据：直接返回"""
        daily = pd.DataFrame()
        adj = pd.DataFrame({
            "date": ["2026-03-27"],
            "adj_factor": [1.5]
        })

        result = merge_daily_with_adj_factor(daily, adj, "600519")

        assert len(result) == 0

    def test_adj_factor_first_is_na(self):
        """第一个复权因子是 NaN：应该用 1.0 填充"""
        daily = pd.DataFrame({
            "date": ["2026-03-27", "2026-03-28"],
            "open": [1800, 1810],
            "high": [1850, 1860],
            "low": [1790, 1800],
            "close": [1820, 1830],
            "volume": [1000000, 1100000]
        })
        # 第一个是 NaN
        adj = pd.DataFrame({
            "date": ["2026-03-27", "2026-03-28"],
            "adj_factor": [pd.NaT, 1.5]  # type: ignore
        })

        result = merge_daily_with_adj_factor(daily, adj, "600519")

        # 第一个应该变成 1.0
        assert result["adj_factor"].iloc[0] == 1.0


class TestValidateDateFreshness:
    """validate_date_freshness 测试"""

    def test_date_matches(self):
        """日期匹配"""
        df = pd.DataFrame({"date": ["2026-03-28"]})

        result = validate_date_freshness(df, "2026-03-28", "600519")

        assert result is True

    def test_date_mismatch(self):
        """日期不匹配"""
        df = pd.DataFrame({"date": ["2026-03-27"]})

        result = validate_date_freshness(df, "2026-03-28", "600519")

        assert result is False

    def test_empty_dataframe(self):
        """空数据"""
        df = pd.DataFrame()

        result = validate_date_freshness(df, "2026-03-28", "600519")

        assert result is False

    def test_multiple_dates(self):
        """多日数据（范围查询）：返回 True"""
        df = pd.DataFrame({
            "date": ["2026-03-26", "2026-03-27", "2026-03-28"]
        })

        result = validate_date_freshness(df, "2026-03-28", "600519")

        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
