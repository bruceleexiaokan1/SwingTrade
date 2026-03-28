"""验证器测试"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.fetcher.validators.stock_validator import (
    StockValidator,
    validate_stock_record,
    is_valid_company_name
)
from src.data.fetcher.validators.daily_validator import (
    DailyValidator,
    validate_daily_record
)


class TestStockValidator:
    """股票列表验证器测试"""

    def setup_method(self):
        self.validator = StockValidator()

    def test_valid_stock_record(self):
        """正常股票记录"""
        record = {
            "code": "600519",
            "name": "贵州茅台",
            "market": "sh",
            "list_date": "2001-08-27"
        }

        result = self.validator.validate(record)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_invalid_code_format(self):
        """无效代码格式"""
        record = {
            "code": "6005",  # 4位
            "name": "贵州茅台",
            "market": "sh"
        }

        result = self.validator.validate(record)

        assert result.is_valid is False
        assert any("invalid_code_format" in e for e in result.errors)

    def test_invalid_market(self):
        """无效市场标识"""
        record = {
            "code": "600519",
            "name": "贵州茅台",
            "market": "xx"
        }

        result = self.validator.validate(record)

        assert result.is_valid is False
        assert any("invalid_market" in e for e in result.errors)

    def test_code_market_mismatch(self):
        """代码与市场不匹配"""
        record = {
            "code": "600519",
            "name": "贵州茅台",
            "market": "sz"  # 600开头应该是上海
        }

        result = self.validator.validate(record)

        assert result.is_valid is False
        assert any("code_market_mismatch" in e for e in result.errors)

    def test_future_list_date(self):
        """未来上市日期"""
        record = {
            "code": "600519",
            "name": "贵州茅台",
            "market": "sh",
            "list_date": "2099-01-01"
        }

        result = self.validator.validate(record)

        assert result.is_valid is False
        assert any("future_list_date" in e for e in result.errors)


class TestIsValidCompanyName:
    """公司名称验证测试"""

    def test_valid_names(self):
        """合法名称"""
        assert is_valid_company_name("贵州茅台") is True
        assert is_valid_company_name("中国平安") is True
        assert is_valid_company_name("万科A") is True

    def test_invalid_names(self):
        """非法名称"""
        assert is_valid_company_name("") is False
        assert is_valid_company_name("A") is False  # 太短
        assert is_valid_company_name("1234567890" * 10) is False  # 太长


class TestDailyValidator:
    """日线数据验证器测试"""

    def setup_method(self):
        self.validator = DailyValidator()

    def test_valid_daily_record(self):
        """正常日线记录"""
        record = {
            "date": "2026-03-28",
            "code": "600519",
            "open": 1800.0,
            "high": 1850.0,
            "low": 1790.0,
            "close": 1820.0,
            "volume": 1000000
        }

        result = self.validator.validate(record)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_high_less_than_low(self):
        """最高价小于最低价"""
        record = {
            "date": "2026-03-28",
            "code": "600519",
            "open": 1800.0,
            "high": 1700.0,  # 高 < 低
            "low": 1790.0,
            "close": 1820.0,
            "volume": 1000000
        }

        result = self.validator.validate(record)

        assert result.is_valid is False
        assert any("high_less_than_low" in e for e in result.errors)

    def test_close_out_of_range(self):
        """收盘价超出范围"""
        record = {
            "date": "2026-03-28",
            "code": "600519",
            "open": 1800.0,
            "high": 1850.0,
            "low": 1790.0,
            "close": 50000.0,  # 超出 10000
            "volume": 1000000
        }

        result = self.validator.validate(record)

        assert result.is_valid is False
        assert any("close_out_of_range" in e for e in result.errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
