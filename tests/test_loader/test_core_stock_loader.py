"""CoreStockLoader 测试"""

import pytest
import sys
import os
import tempfile
import shutil
import sqlite3
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.loader import CoreStockLoader


class TestCoreStockLoader:
    """CoreStockLoader 测试"""

    def setup_method(self):
        """使用临时目录"""
        self.temp_dir = tempfile.mkdtemp(prefix="core_stock_test_")
        os.makedirs(os.path.join(self.temp_dir, "sqlite"), exist_ok=True)

        # 创建测试配置文件
        self.config_path = os.path.join(self.temp_dir, "core_stocks.json")
        with open(self.config_path, 'w') as f:
            json.dump({
                "version": "1.0",
                "stocks": [
                    {"code": "600519", "name": "贵州茅台", "sector": "白酒"},
                    {"code": "000001", "name": "平安银行", "sector": "银行"},
                    {"code": "300001", "name": "创业板", "sector": "创业板"}
                ]
            }, f)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_from_config(self):
        """从配置文件加载"""
        loader = CoreStockLoader(self.temp_dir)
        df = loader.load_from_config(self.config_path)

        assert len(df) == 3
        assert "market" in df.columns
        assert df[df['code'] == '600519']['market'].iloc[0] == 'sh'
        assert df[df['code'] == '000001']['market'].iloc[0] == 'sz'
        assert df[df['code'] == '300001']['market'].iloc[0] == 'sz'

    def test_upsert_to_sqlite(self):
        """Upsert 到 SQLite"""
        loader = CoreStockLoader(self.temp_dir)
        df = loader.load_from_config(self.config_path)

        rows = loader.upsert_to_sqlite(df)

        assert rows == 3

        # 验证 SQLite
        db_path = os.path.join(self.temp_dir, "sqlite", "market.db")
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM stocks WHERE is_active = 1").fetchone()[0]
        conn.close()

        assert count == 3

    def test_load_and_upsert(self):
        """加载并 upsert"""
        loader = CoreStockLoader(self.temp_dir)
        # 使用测试配置而非默认配置
        df = loader.load_from_config(self.config_path)
        loader.upsert_to_sqlite(df)

        assert len(df) == 3
        assert loader.get_stock_count() == 3

    def test_infer_market(self):
        """市场推断"""
        loader = CoreStockLoader(self.temp_dir)

        assert loader._infer_market("600519") == "sh"
        assert loader._infer_market("000001") == "sz"
        assert loader._infer_market("300001") == "sz"
        assert loader._infer_market("400001") == "bj"
        assert loader._infer_market("800001") == "bj"

    def test_upsert_updates_existing(self):
        """Upsert 更新现有记录"""
        loader = CoreStockLoader(self.temp_dir)
        df = loader.load_from_config(self.config_path)

        # 第一次 upsert
        loader.upsert_to_sqlite(df)

        # 修改配置
        with open(self.config_path, 'w') as f:
            json.dump({
                "version": "1.0",
                "stocks": [
                    {"code": "600519", "name": "贵州茅台(新)", "sector": "白酒"},
                    {"code": "000001", "name": "平安银行", "sector": "银行"},
                    {"code": "300001", "name": "创业板", "sector": "创业板"}
                ]
            }, f)

        # 第二次 upsert
        df2 = loader.load_from_config(self.config_path)
        loader.upsert_to_sqlite(df2)

        # 验证更新
        db_path = os.path.join(self.temp_dir, "sqlite", "market.db")
        conn = sqlite3.connect(db_path)
        name = conn.execute("SELECT name FROM stocks WHERE code = '600519'").fetchone()[0]
        conn.close()

        assert name == "贵州茅台(新)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
