"""
StockData 单写入器模块
幂等写入 + 重试机制
"""

import os
import sqlite3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging
import time
import shutil
import fasteners

logger = logging.getLogger(__name__)


class WriteError(Exception):
    """写入异常"""
    pass


class QuarantineError(Exception):
    """数据质量过低，进入隔离"""
    pass


def get_stockdata_root() -> str:
    """获取 StockData 根目录"""
    return os.environ.get('STOCKDATA_ROOT', '/Users/bruce/workspace/trade/StockData')


def get_db_path() -> str:
    """获取数据库路径"""
    return os.path.join(get_stockdata_root(), 'sqlite', 'market.db')


class IdempotentWriter:
    """
    幂等写入器

    设计原则：
    1. 幂等检查 - 基于 (code, date) 判断是否已存在
    2. 失败重试 - 最多3次，指数退避
    3. 崩溃恢复 - 读取检查点，从检查点日期重新采集
    """

    def __init__(self, stockdata_root: str = None, db_path: str = None):
        self.stockdata_root = stockdata_root or get_stockdata_root()
        self.db_path = db_path or get_db_path()
        self.LOCK_DIR = Path("/tmp/stockdata_locks")

    def write(self, code: str, df: pd.DataFrame) -> bool:
        """
        幂等写入（线程安全，使用文件锁）

        Args:
            code: 股票代码
            df: 日线数据

        Returns:
            bool: 是否写入成功
        """
        # 按股票前缀分区锁（sh/sz/bj）
        lock_dir = self.LOCK_DIR / code[:2]
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file = lock_dir / f"{code}.lock"

        with fasteners.InterProcessLock(lock_file):
            return self._write_inner(code, df)

    def _write_inner(self, code: str, df: pd.DataFrame) -> bool:
        """写入内部逻辑（在锁内执行）"""
        if df.empty:
            logger.warning(f"空数据，跳过: {code}")
            return False

        # Step 1: 幂等检查
        latest_date = self._get_latest_date(code)
        if latest_date is not None:
            # 过滤已存在的日期
            new_df = df[df['date'] > pd.to_datetime(latest_date)]
            if len(new_df) == 0:
                logger.info(f"数据已存在，跳过: {code}")
                return True
            df = new_df

        # Step 2: 质量评估
        from utils.quality import validate_daily, calculate_quality_score
        validation = validate_daily(df)
        score = validation['score']

        if not score.usable:
            logger.error(f"数据质量过低: {score.total}分，隔离: {code}")
            self._save_to_quarantine(code, df, score, validation['anomalies'])
            return False

        # Step 3: 执行写入（带重试）
        try:
            self._write_with_retry(code, df)
        except Exception as e:
            logger.error(f"写入失败: {code}, {e}")
            raise WriteError(f"写入失败: {code}") from e

        # Step 4: 更新检查点
        self._update_checkpoint(code, df['date'].max())

        # Step 5: 触发告警（如果需要）
        if score.total < 80:
            self._send_quality_alert(code, score, validation['anomalies'])

        logger.info(f"写入成功: {code}, {len(df)} 行, 质量分: {score.total}")
        return True

    def _write_with_retry(self, code: str, df: pd.DataFrame, max_retries: int = 3):
        """带重试的写入"""
        for attempt in range(max_retries):
            try:
                self._write_atomic(code, df)
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt  # 指数退避: 1s, 2s, 4s
                    logger.warning(f"写入失败，重试 {attempt + 1}/{max_retries}: {code}, 等待 {sleep_time}s")
                    time.sleep(sleep_time)
                else:
                    raise

    def _write_atomic(self, code: str, df: pd.DataFrame):
        """原子写入"""
        daily_dir = Path(self.stockdata_root) / "raw/daily"
        daily_dir.mkdir(parents=True, exist_ok=True)

        temp_file = daily_dir / f"{code}_{datetime.now().strftime('%Y%m%d%H%M%S')}.tmp.parquet"
        target_file = daily_dir / f"{code}.parquet"

        try:
            # 读取现有数据
            if target_file.exists():
                existing_df = pd.read_parquet(str(target_file))
                # 合并并去重
                merged_df = pd.concat([existing_df, df]).drop_duplicates(subset=['date'])
                merged_df = merged_df.sort_values('date').reset_index(drop=True)
            else:
                merged_df = df

            # 写入临时文件
            merged_df.to_parquet(
                str(temp_file),
                engine='pyarrow',
                compression='snappy',
                version='2.6'  # 使用稳定的 Parquet 版本
            )

            # 原子替换
            os.replace(str(temp_file), str(target_file))

            # 更新 SQLite 索引
            self._update_daily_index(code, merged_df)

        except Exception as e:
            # 清理临时文件
            if temp_file.exists():
                temp_file.unlink()
            raise

    def _get_latest_date(self, code: str) -> Optional[str]:
        """获取已存在的最新日期"""
        try:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT latest_date FROM daily_index WHERE code = ?",
                [code]
            ).fetchone()
            conn.close()
            return row[0] if row else None
        except Exception as e:
            logger.error(f"SQLite 查询失败 {code}: {e}")
            return None

    def _update_daily_index(self, code: str, df: pd.DataFrame):
        """更新日线索引"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('PRAGMA journal_mode=WAL')

            date_range = df['date'].agg(['min', 'max'])
            latest_date = df['date'].max().strftime('%Y-%m-%d') if hasattr(df['date'].max(), 'strftime') else str(df['date'].max())

            conn.execute("""
                INSERT OR REPLACE INTO daily_index
                (code, latest_date, file_path, row_count, start_date, end_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                code,
                latest_date,
                f"raw/daily/{code}.parquet",
                len(df),
                date_range['min'].strftime('%Y-%m-%d') if hasattr(date_range['min'], 'strftime') else str(date_range['min']),
                latest_date
            ])
            conn.commit()
        finally:
            conn.close()

    def _update_checkpoint(self, code: str, date):
        """更新检查点"""
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR REPLACE INTO checkpoints (key, value, updated_at)
                VALUES (?, ?, datetime('now'))
            """, [f"daily_{code}_last_update", date_str])
            conn.commit()
        finally:
            conn.close()

    def _save_to_quarantine(self, code: str, df: pd.DataFrame, score, anomalies: list):
        """保存低质量数据到隔离区"""
        quarantine_dir = Path(self.stockdata_root) / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        # 保存数据
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        quarantine_file = quarantine_dir / f"{code}_{timestamp}.parquet"
        df.to_parquet(str(quarantine_file), engine='pyarrow')

        # 保存元数据
        meta_file = quarantine_dir / f"{code}_{timestamp}.meta.json"
        import json
        with open(meta_file, 'w') as f:
            json.dump({
                'code': code,
                'score': score.to_dict(),
                'anomalies': anomalies,
                'timestamp': timestamp
            }, f, indent=2, default=str)

        logger.info(f"低质量数据已隔离: {quarantine_file}")

    def _send_quality_alert(self, code: str, score, anomalies: list):
        """发送质量告警"""
        from utils.alert import send_alert

        level = "ERROR" if score.total < 50 else "WARNING"
        message = f"数据质量{score.grade}: {score.total}分, 代码: {code}"

        send_alert(level, message, {
            'code': code,
            'score': score.to_dict(),
            'anomalies': anomalies
        })


class AtomicWriter:
    """
    原子写入器（低级接口）

    用于单次原子写入操作，不做幂等检查
    """

    def __init__(self, stockdata_root: str = None):
        self.stockdata_root = stockdata_root or get_stockdata_root()

    def write(self, code: str, df: pd.DataFrame):
        """原子写入"""
        daily_dir = Path(self.stockdata_root) / "raw/daily"
        daily_dir.mkdir(parents=True, exist_ok=True)

        temp_file = daily_dir / f"{code}_{datetime.now().strftime('%Y%m%d%H%M%S')}.tmp.parquet"
        target_file = daily_dir / f"{code}.parquet"

        try:
            # 写入临时文件
            df.to_parquet(
                str(temp_file),
                engine='pyarrow',
                compression='snappy'
            )

            # 原子替换
            os.replace(str(temp_file), str(target_file))

        finally:
            # 清理临时文件（如果存在）
            if temp_file.exists():
                temp_file.unlink()
