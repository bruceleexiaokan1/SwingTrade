"""
StockData Schema 迁移测试
"""

import pytest
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
from pathlib import Path
from datetime import datetime

# 导入被测试模块
import sys
sys.path.insert(0, 'scripts')

from utils.schema_registry import (
    SchemaRegistry,
    SchemaMigrator,
    write_parquet_with_metadata,
    read_parquet_with_version,
    SCHEMA_VERSION
)


class TestSchemaRegistry:
    """Schema 注册表测试"""

    def test_current_version(self):
        """验证当前版本"""
        assert SCHEMA_VERSION == 'v3', f"当前版本应为 v3: {SCHEMA_VERSION}"

    def test_get_latest_schema(self):
        """获取最新 schema"""
        schema = SchemaRegistry.get_schema('latest')
        assert schema is not None

        field_names = [f.name for f in schema]
        assert 'date' in field_names
        assert 'close' in field_names
        assert 'volume' in field_names

    def test_get_specific_version(self):
        """获取指定版本 schema"""
        schema_v1 = SchemaRegistry.get_schema('v1')
        schema_v2 = SchemaRegistry.get_schema('v2')
        schema_v3 = SchemaRegistry.get_schema('v3')

        # V1 有 6 个字段
        assert len(schema_v1) == 6

        # V2 有 9 个字段（V1 + amount + turnover + adj_factor）
        assert len(schema_v2) == 9

        # V3 有 11 个字段（V2 + pct_chg + is_halt）
        assert len(schema_v3) == 11


class TestSchemaMigration:
    """Schema 迁移测试"""

    def test_v1_to_v3_migration(self, temp_stockdata):
        """V1 文件迁移到 V3"""
        # 创建 V1 格式数据
        df_v1 = pd.DataFrame({
            'date': pd.to_datetime(['2026-03-01', '2026-03-02']),
            'open': [10.0, 10.2],
            'high': [10.5, 10.7],
            'low': [9.8, 10.0],
            'close': [10.2, 10.5],
            'volume': [1000000, 1100000],
        })

        # 写入为 V1
        v1_path = temp_stockdata / "test_v1.parquet"
        write_parquet_with_metadata(df_v1, str(v1_path), 'v1')

        # 按最新版本读取
        df_read, version = read_parquet_with_version(str(v1_path))

        # 验证：版本已更新
        assert version == SCHEMA_VERSION

        # 验证：新字段已补齐
        assert 'amount' in df_read.columns
        assert 'turnover' in df_read.columns
        assert 'adj_factor' in df_read.columns
        assert 'pct_chg' in df_read.columns
        assert 'is_halt' in df_read.columns

        # 验证：原始数据未丢失
        assert len(df_read) == len(df_v1)
        assert list(df_read['close']) == list(df_v1['close'])

    def test_v2_to_v3_migration(self, temp_stockdata):
        """V2 文件迁移到 V3"""
        # 创建 V2 格式数据
        df_v2 = pd.DataFrame({
            'date': pd.to_datetime(['2026-03-01']),
            'open': [10.0],
            'high': [10.5],
            'low': [9.8],
            'close': [10.2],
            'volume': [1000000],
            'amount': [10200000],
            'turnover': [0.05],
            'adj_factor': [1.0],
        })

        v2_path = temp_stockdata / "test_v2.parquet"
        write_parquet_with_metadata(df_v2, str(v2_path), 'v2')

        # 读取
        df_read, version = read_parquet_with_version(str(v2_path))

        # 验证：V3 新字段已补齐
        assert 'pct_chg' in df_read.columns
        assert 'is_halt' in df_read.columns

    def test_v3_read_unchanged(self, temp_stockdata):
        """V3 文件读取不变"""
        # 创建 V3 格式数据
        df_v3 = pd.DataFrame({
            'date': pd.to_datetime(['2026-03-01']),
            'open': [10.0],
            'high': [10.5],
            'low': [9.8],
            'close': [10.2],
            'volume': [1000000],
            'amount': [10200000],
            'turnover': [0.05],
            'adj_factor': [1.0],
            'pct_chg': [0.02],
            'is_halt': [False],
        })

        v3_path = temp_stockdata / "test_v3.parquet"
        write_parquet_with_metadata(df_v3, str(v3_path), 'v3')

        # 读取
        df_read, version = read_parquet_with_version(str(v3_path))

        # 验证：数据不变
        assert version == 'v3'
        assert list(df_read['close']) == list(df_v3['close'])

    def test_metadata_preserved(self, temp_stockdata):
        """元数据保留"""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-03-01']),
            'open': [10.0],
            'high': [10.5],
            'low': [9.8],
            'close': [10.2],
            'volume': [1000000],
        })

        path = temp_stockdata / "test_meta.parquet"
        write_parquet_with_metadata(df, str(path), 'v1')

        # 读取元数据
        pf = pq.ParquetFile(str(path))
        metadata = pf.schema_arrow.metadata

        # 验证：版本号已嵌入
        assert b'schema_version' in metadata
        assert metadata[b'schema_version'] == b'v1'

        # 验证：创建时间已记录
        assert b'created_at' in metadata

    def test_migration_preserves_data_integrity(self, temp_stockdata):
        """迁移保持数据完整性"""
        # 创建包含各种值的数据
        df_original = pd.DataFrame({
            'date': pd.to_datetime(['2026-03-01', '2026-03-02', '2026-03-03']),
            'open': [10.0, 10.2, 10.4],
            'high': [10.5, 10.7, 10.8],
            'low': [9.8, 10.0, 10.2],
            'close': [10.2, 10.5, 10.6],
            'volume': [1000000, 1100000, 1200000],
        })

        path = temp_stockdata / "test_integrity.parquet"
        write_parquet_with_metadata(df_original, str(path), 'v1')

        # 迁移后读取
        df_read, _ = read_parquet_with_version(str(path))

        # 验证：价格数据完全一致
        for col in ['open', 'high', 'low', 'close']:
            original_values = list(df_original[col])
            read_values = list(df_read[col])
            assert original_values == read_values, f"列 {col} 数据不一致"

        # 验证：成交量一致
        assert list(df_original['volume']) == list(df_read['volume'])


class TestMigrationChain:
    """迁移链测试"""

    def test_multi_step_migration(self):
        """多步迁移 v1 -> v2 -> v3"""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-03-01']),
            'open': [10.0],
            'high': [10.5],
            'low': [9.8],
            'close': [10.2],
            'volume': [1000000],
        })

        # v1 -> v2
        df_v2 = SchemaMigrator.migrate(df, 'v1', 'v2')
        assert 'amount' in df_v2.columns
        assert 'turnover' in df_v2.columns
        assert 'adj_factor' in df_v2.columns

        # v2 -> v3
        df_v3 = SchemaMigrator.migrate(df_v2, 'v2', 'v3')
        assert 'pct_chg' in df_v3.columns
        assert 'is_halt' in df_v3.columns

    def test_same_version_no_migration(self):
        """相同版本不迁移"""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-03-01']),
            'open': [10.0],
            'high': [10.5],
            'low': [9.8],
            'close': [10.2],
            'volume': [1000000],
        })

        df_result = SchemaMigrator.migrate(df, 'v1', 'v1')
        assert list(df.columns) == list(df_result.columns)
