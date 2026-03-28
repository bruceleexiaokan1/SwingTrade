"""
StockData Schema 注册与迁移模块

支持 Parquet 文件的版本管理和自动迁移
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Callable
import logging

logger = logging.getLogger(__name__)


# 当前 Schema 版本
SCHEMA_VERSION = 'v3'


class SchemaRegistry:
    """Schema 注册表"""

    SCHEMAS = {
        'v1': pa.schema([
            ('date', pa.string()),
            ('open', pa.float32()),
            ('high', pa.float32()),
            ('low', pa.float32()),
            ('close', pa.float32()),
            ('volume', pa.int64()),
        ]),
        'v2': pa.schema([
            ('date', pa.string()),
            ('open', pa.float32()),
            ('high', pa.float32()),
            ('low', pa.float32()),
            ('close', pa.float32()),
            ('volume', pa.int64()),
            ('amount', pa.float64()),
            ('turnover', pa.float32()),
            ('adj_factor', pa.float32()),
        ]),
        'v3': pa.schema([
            ('date', pa.string()),
            ('open', pa.float32()),
            ('high', pa.float32()),
            ('low', pa.float32()),
            ('close', pa.float32()),
            ('volume', pa.int64()),
            ('amount', pa.float64()),
            ('turnover', pa.float32()),
            ('adj_factor', pa.float32()),
            ('pct_chg', pa.float32()),
            ('is_halt', pa.bool_()),
        ]),
    }

    @classmethod
    def get_schema(cls, version: str = 'latest') -> pa.Schema:
        """获取指定版本的 schema"""
        if version == 'latest':
            return cls.SCHEMAS[SCHEMA_VERSION]
        return cls.SCHEMAS.get(version, cls.SCHEMAS['v1'])

    @classmethod
    def get_field_names(cls, version: str = 'latest') -> list:
        """获取指定版本的字段名列表"""
        schema = cls.get_schema(version)
        return [f.name for f in schema]


class SchemaMigrator:
    """Schema 迁移管理器"""

    # 迁移函数注册表
    MIGRATIONS: Dict[tuple, Callable] = {
        # v1 -> v2: 新增 amount, turnover, adj_factor
        ('v1', 'v2'): lambda df: df.assign(
            amount=None,
            turnover=None,
            adj_factor=None
        ),
        # v2 -> v3: 新增 pct_chg, is_halt
        ('v2', 'v3'): lambda df: df.assign(
            pct_chg=None,
            is_halt=False
        ),
    }

    @classmethod
    def get_next_version(cls, current: str) -> Optional[str]:
        """获取下一版本号"""
        version_order = ['v1', 'v2', 'v3']
        try:
            idx = version_order.index(current)
            if idx < len(version_order) - 1:
                return version_order[idx + 1]
        except ValueError:
            pass
        return None

    @classmethod
    def migrate(cls, df: pd.DataFrame, from_ver: str, to_ver: str) -> pd.DataFrame:
        """
        执行版本迁移

        支持多步迁移 (例如 v1 -> v2 -> v3)

        Args:
            df: 原始 DataFrame
            from_ver: 源版本
            to_ver: 目标版本

        Returns:
            pd.DataFrame: 迁移后的 DataFrame
        """
        if from_ver == to_ver:
            return df

        current = from_ver
        while current != to_ver:
            next_ver = cls.get_next_version(current)
            if next_ver is None:
                raise ValueError(f"无法迁移: {current} -> {to_ver}")

            migration = cls.MIGRATIONS.get((current, next_ver))
            if migration is None:
                raise ValueError(f"缺少迁移路径: {current} -> {next_ver}")

            df = migration(df)
            logger.info(f"Schema 迁移: {current} -> {next_ver}")
            current = next_ver

        return df


def write_parquet_with_metadata(
    df: pd.DataFrame,
    path: str,
    schema_version: str = SCHEMA_VERSION
) -> None:
    """
    写入 Parquet 并嵌入版本元数据

    Args:
        df: DataFrame 数据
        path: 文件路径
        schema_version: Schema 版本
    """
    # 确保目录存在
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # 准备元数据
    metadata = {
        'schema_version': schema_version,
        'created_at': datetime.now().isoformat(),
        'field_count': str(len(df.columns))
    }

    # 转换为 Arrow Table
    table = pa.Table.from_pandas(df, schema=SchemaRegistry.get_schema(schema_version))

    # 写入文件
    pq.write_table(
        table,
        path,
        metadata=metadata,
        compression='snappy'
    )


def read_parquet_with_version(path: str) -> Tuple[pd.DataFrame, str]:
    """
    读取 Parquet，自动检测并迁移版本

    Args:
        path: 文件路径

    Returns:
        Tuple[pd.DataFrame, str]: (数据, 版本号)
    """
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow

    # 检测版本
    metadata = schema.metadata or {}
    current_version = metadata.get(b'schema_version', b'v1').decode()

    # 读取数据
    df = pf.read().to_pandas()

    # 检查是否需要迁移
    if current_version != SCHEMA_VERSION:
        logger.info(f"Schema 迁移: {current_version} -> {SCHEMA_VERSION}")
        df = SchemaMigrator.migrate(df, current_version, SCHEMA_VERSION)

        # 重新写入以更新版本
        write_parquet_with_metadata(df, path, SCHEMA_VERSION)

    return df, current_version


def detect_schema_version(path: str) -> str:
    """
    检测 Parquet 文件的 Schema 版本

    Args:
        path: 文件路径

    Returns:
        str: 版本号
    """
    try:
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        metadata = schema.metadata or {}
        return metadata.get(b'schema_version', b'v1').decode()
    except Exception:
        return 'v1'  # 默认旧版本


def migrate_file(path: str) -> bool:
    """
    迁移单个 Parquet 文件

    Args:
        path: 文件路径

    Returns:
        bool: 是否执行了迁移
    """
    current_version = detect_schema_version(path)

    if current_version == SCHEMA_VERSION:
        return False

    df, _ = read_parquet_with_version(path)
    logger.info(f"文件已迁移: {path}, {current_version} -> {SCHEMA_VERSION}")
    return True


def migrate_directory(dir_path: str, pattern: str = "*.parquet") -> Dict[str, bool]:
    """
    批量迁移目录下的 Parquet 文件

    Args:
        dir_path: 目录路径
        pattern: 文件匹配模式

    Returns:
        Dict[str, bool]: 文件路径 -> 是否迁移
    """
    results = {}
    dir_path = Path(dir_path)

    for path in dir_path.rglob(pattern):
        try:
            results[str(path)] = migrate_file(str(path))
        except Exception as e:
            logger.error(f"迁移失败: {path}, {e}")
            results[str(path)] = False

    return results
