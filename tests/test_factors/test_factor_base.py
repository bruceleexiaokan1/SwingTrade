"""因子库单元测试

测试因子基类和注册表
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from factors.factor_base import FactorBase, FactorMetadata
from factors.registry import FactorRegistry
from factors.exceptions import (
    FactorNotFoundError,
    FactorValidationError,
    DataValidationError
)


class DummyFactor(FactorBase):
    """测试用Dummy因子"""
    name = "dummy"
    category = "test"
    description = "测试因子"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        result = data.groupby("code").apply(
            lambda g: pd.Series({
                "date": g["date"].iloc[-1],
                "code": g["code"].iloc[0],
                "factor_value": 1.0
            })
        ).reset_index(drop=True)

        return result[["date", "code", "factor_value"]]


class TestFactorMetadata:
    """测试因子元数据"""

    def test_metadata_creation(self):
        """测试元数据创建"""
        metadata = FactorMetadata(
            name="test_factor",
            category="test",
            description="测试描述"
        )

        assert metadata.name == "test_factor"
        assert metadata.category == "test"
        assert metadata.description == "测试描述"

    def test_metadata_to_dict(self):
        """测试转换为字典"""
        metadata = FactorMetadata(
            name="test_factor",
            category="test",
            description="测试描述",
            version="1.0"
        )

        d = metadata.to_dict()
        assert d["name"] == "test_factor"
        assert d["version"] == "1.0"


class TestFactorBase:
    """测试因子基类"""

    def test_factor_initialization(self):
        """测试因子初始化"""
        factor = DummyFactor()
        assert factor.name == "dummy"
        assert factor.category == "test"

    def test_factor_without_name_raises(self):
        """测试未定义名称时抛出异常"""

        class NoNameFactor(FactorBase):
            category = "test"

            def calculate(self, data):
                pass

        with pytest.raises(ValueError):
            NoNameFactor()

    def test_validate_data_success(self):
        """测试数据验证成功"""
        factor = DummyFactor()

        data = pd.DataFrame({
            "date": ["2026-03-28"],
            "code": ["600519"],
            "close": [1800.0]
        })

        assert factor.validate_data(data) is True

    def test_validate_data_missing_columns(self):
        """测试数据验证失败"""
        factor = DummyFactor()

        data = pd.DataFrame({
            "date": ["2026-03-28"],
            "code": ["600519"]
            # 缺少 close 列
        })

        with pytest.raises(DataValidationError) as exc_info:
            factor.validate_data(data)

        assert "close" in str(exc_info.value)


class TestFactorRegistry:
    """测试因子注册表"""

    def setup_method(self):
        """每个测试前清空注册表"""
        self.registry = FactorRegistry()
        self.registry.clear()

    def teardown_method(self):
        """每个测试后清空注册表"""
        self.registry.clear()

    def test_register_factor(self):
        """测试注册因子"""
        factor = DummyFactor()
        self.registry.register(factor)

        assert "dummy" in self.registry
        assert len(self.registry) == 1

    def test_register_duplicate_raises(self):
        """测试重复注册抛出异常"""
        factor1 = DummyFactor()
        factor2 = DummyFactor()

        self.registry.register(factor1)

        with pytest.raises(FactorValidationError):
            self.registry.register(factor2)

    def test_register_with_overwrite(self):
        """测试允许覆盖注册"""
        factor1 = DummyFactor()
        factor2 = DummyFactor()

        self.registry.register(factor1)
        self.registry.register(factor2, allow_overwrite=True)

        assert len(self.registry) == 1

    def test_get_factor(self):
        """测试获取因子"""
        factor = DummyFactor()
        self.registry.register(factor)

        retrieved = self.registry.get_factor("dummy")
        assert retrieved.name == "dummy"

    def test_get_factor_not_found(self):
        """测试获取不存在的因子"""
        with pytest.raises(FactorNotFoundError):
            self.registry.get_factor("nonexistent")

    def test_list_factors_by_category(self):
        """测试按类别列出因子"""

        class AnotherDummy(FactorBase):
            name = "another"
            category = "another_category"

            def calculate(self, data):
                pass

        self.registry.register(DummyFactor())
        self.registry.register(AnotherDummy())

        test_factors = self.registry.list_factors(category="test")
        assert "dummy" in test_factors
        assert "another" not in test_factors

    def test_calculate_single(self):
        """测试单因子计算"""
        factor = DummyFactor()
        self.registry.register(factor)

        data = pd.DataFrame({
            "date": ["2026-03-28", "2026-03-28"],
            "code": ["600519", "000001"],
            "close": [1800.0, 50.0]
        })

        result = self.registry.calculate_single("dummy", data)

        assert "date" in result.columns
        assert "code" in result.columns
        assert "factor_value" in result.columns
        assert len(result) == 2

    def test_calculate_all(self):
        """测试批量因子计算"""
        factor = DummyFactor()
        self.registry.register(factor)

        data = pd.DataFrame({
            "date": ["2026-03-28", "2026-03-28"],
            "code": ["600519", "000001"],
            "close": [1800.0, 50.0]
        })

        result = self.registry.calculate_all(data, ["dummy"])

        assert "dummy" in result.columns

    def test_unregister(self):
        """测试注销因子"""
        factor = DummyFactor()
        self.registry.register(factor)

        assert self.registry.unregister("dummy") is True
        assert "dummy" not in self.registry

    def test_clear(self):
        """测试清空注册表"""
        # 先清空确保干净状态
        self.registry.clear()

        # 注册两个不同名的因子
        class Dummy2(FactorBase):
            name = "dummy2"
            category = "test"
            def calculate(self, data):
                pass

        self.registry.register(DummyFactor())
        self.registry.register(Dummy2())

        assert len(self.registry) == 2

        count = self.registry.clear()
        assert count == 2
        assert len(self.registry) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
