"""历史数据回填模块"""

import os
import sys
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.fetcher.sources.tushare_source import TushareSource
from src.data.fetcher.quality_scorer import QualityScorer
from src.data.fetcher.data_merger import merge_daily_with_adj_factor
from src.data.fetcher.exceptions import NetworkError, SourceError

logger = logging.getLogger("backfill")

# 添加 scripts 到 path 以导入 utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'scripts'))
from utils.alert import send_alert


@dataclass
class BackfillResult:
    """回填结果"""
    code: str
    start_date: str
    end_date: str
    status: str  # success, failed, quarantined
    records_count: int = 0
    quality_score: Optional[float] = None
    error: Optional[str] = None
    quarantined_reason: Optional[str] = None


@dataclass
class BackfillReport:
    """回填报告"""
    start_date: str
    end_date: str
    total_stocks: int
    success_count: int = 0
    failed_count: int = 0
    quarantined_count: int = 0
    total_records: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    results: List[BackfillResult] = field(default_factory=list)

    @property
    def duration_minutes(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 60
        return 0

    def to_dict(self) -> dict:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_stocks": self.total_stocks,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "quarantined_count": self.quarantined_count,
            "total_records": self.total_records,
            "duration_minutes": round(self.duration_minutes, 1),
            "results": [r.__dict__ for r in self.results]
        }


class BackfillFetcher:
    """
    历史数据回填器

    支持：
    - 日期范围批量采集
    - 断点续传
    - 进度持久化
    - 速率限制
    - 低质量隔离 + 告警
    """

    # 速率限制：45 * 0.8 = 36 calls/min
    RATE_LIMIT = 36
    # 每批次处理天数（5年）
    CHUNK_DAYS = 1825  # ~5 years

    def __init__(
        self,
        stockdata_root: str,
        start_date: str,  # YYYY-MM-DD
        end_date: str,    # YYYY-MM-DD
        codes: Optional[List[str]] = None,
        rate_limit_buffer: float = 0.8,
        max_workers: int = 1,  # 默认单线程，稳定优先
        source=None,  # 可注入，用于测试
        scorer=None   # 可注入，用于测试
    ):
        """
        初始化回填器

        Args:
            stockdata_root: StockData 根目录
            start_date: 回填开始日期
            end_date: 回填结束日期
            codes: 股票代码列表，None = 全市场
            rate_limit_buffer: 速率限制安全系数
            max_workers: 最大并发数
            source: 数据源，可注入（用于测试）
            scorer: 评分器，可注入（用于测试）
        """
        self.stockdata_root = stockdata_root
        self.start_date = start_date
        self.end_date = end_date
        self.codes = codes
        self.max_workers = max_workers

        # 计算速率限制
        self.rate_limit = int(self.RATE_LIMIT * rate_limit_buffer)
        self.min_interval = 60.0 / self.rate_limit if self.rate_limit > 0 else 0

        # 组件（延迟初始化，支持注入）
        self._source = source
        self._scorer = scorer

        # 进度状态
        self._last_call_time = 0
        self._processed = 0
        self._failed = 0
        self._quarantined = 0

        # 进度文件
        self.progress_file = os.path.join(
            stockdata_root, "status", f"backfill_{start_date}_{end_date}.json"
        )

    @property
    def source(self):
        """延迟初始化数据源"""
        if self._source is None:
            self._source = TushareSource()
        return self._source

    @property
    def scorer(self):
        """延迟初始化评分器"""
        if self._scorer is None:
            self._scorer = QualityScorer()
        return self._scorer

    def fetch_all(self) -> BackfillReport:
        """
        执行全量回填

        Returns:
            BackfillReport
        """
        report = BackfillReport(
            start_date=self.start_date,
            end_date=self.end_date,
            total_stocks=len(self.codes) if self.codes else 0
        )

        logger.info(f"开始回填: {self.start_date} ~ {self.end_date}")
        logger.info(f"速率限制: {self.rate_limit} calls/min, 间隔: {self.min_interval:.2f}s")

        # 获取股票列表
        if not self.codes:
            self.codes = self._get_stock_list()
            report.total_stocks = len(self.codes)

        # 断点检查
        completed = self._load_progress()
        if completed:
            self.codes = [c for c in self.codes if c not in completed]
            logger.info(f"跳过已完成: {len(completed)} 只，剩余: {len(self.codes)} 只")

        # 单线程顺序处理（稳定优先）
        for i, code in enumerate(self.codes):
            logger.info(f"[{i+1}/{len(self.codes)}] 处理 {code}")

            result = self._fetch_stock(code)

            report.results.append(result)
            if result.status == "success":
                report.success_count += 1
                report.total_records += result.records_count
            elif result.status == "quarantined":
                report.quarantined_count += 1
                report.total_records += result.records_count
            else:
                report.failed_count += 1

            # 保存进度
            self._save_progress(code)

            # 速率限制
            self._rate_limit_sleep()

        report.end_time = datetime.now()

        # 最终报告
        self._send_report_alert(report)

        logger.info(
            f"回填完成: 成功 {report.success_count}, "
            f"隔离 {report.quarantined_count}, 失败 {report.failed_count}, "
            f"耗时 {report.duration_minutes:.1f} 分钟"
        )

        return report

    def _fetch_stock(self, code: str) -> BackfillResult:
        """
        采集单只股票历史数据

        Args:
            code: 股票代码

        Returns:
            BackfillResult
        """
        result = BackfillResult(
            code=code,
            start_date=self.start_date,
            end_date=self.end_date,
            status="failed"
        )

        try:
            # 1. 获取日线数据（5年一次请求）
            daily = self._call_with_rate_limit(
                self.source.fetch_daily,
                code, self.start_date, self.end_date
            )

            if daily is None or len(daily) == 0:
                result.error = "no_data"
                return result

            # 2. 获取复权因子
            adj = self._call_with_rate_limit(
                self.source.fetch_adj_factor,
                code, self.start_date, self.end_date
            )

            # 3. 合并数据
            merged = merge_daily_with_adj_factor(daily, adj, code)

            if merged.empty:
                result.error = "merge_empty"
                return result

            # 4. 质量评分
            quality_scores = []
            quarantined_records = []

            for _, row in merged.iterrows():
                record = row.to_dict()
                score = self.scorer.score(record)
                quality_scores.append(score.overall)

                # 质量 < 60 隔离
                if score.overall < 60:
                    quarantined_records.append((record, score.overall))

            # 5. 处理隔离数据
            if quarantined_records:
                self._quarantine_records(code, quarantined_records)
                result.status = "quarantined"
                result.quality_score = sum(quality_scores) / len(quality_scores)
                result.records_count = len(merged)
                result.quarantined_reason = f"{len(quarantined_records)} records < 60"
                return result

            # 6. 写入（只写入质量 >= 60 的数据）
            valid_records = merged[
                pd.Series(quality_scores) >= 60
            ].copy()

            if len(valid_records) > 0:
                self._write_records(code, valid_records)
                result.status = "success"
                result.records_count = len(valid_records)
                result.quality_score = sum(quality_scores) / len(quality_scores)
            else:
                result.error = "all_records_low_quality"

            return result

        except Exception as e:
            result.error = str(e)
            logger.error(f"回填失败 {code}: {e}")
            return result

    def _call_with_rate_limit(self, func, *args, **kwargs) -> Optional[pd.DataFrame]:
        """带速率限制的 API 调用"""
        now = time.time()
        elapsed = now - self._last_call_time

        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logger.debug(f"速率限制等待: {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self._last_call_time = time.time()
        return func(*args, **kwargs)

    def _rate_limit_sleep(self) -> None:
        """速率限制休眠"""
        now = time.time()
        elapsed = now - self._last_call_time

        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        self._last_call_time = time.time()

    def _get_stock_list(self) -> List[str]:
        """获取股票列表"""
        # 从 SQLite 读取
        sqlite_db = os.path.join(self.stockdata_root, "sqlite", "market.db")

        if os.path.exists(sqlite_db):
            import sqlite3
            with sqlite3.connect(sqlite_db) as conn:
                df = conn.execute(
                    "SELECT code FROM stocks WHERE is_active = 1"
                ).fetchall()
            return [row[0] for row in df]

        # 兜底：从 Tushare 获取
        logger.warning("股票列表为空，从 Tushare 获取")
        df = self.source.fetch_stock_list()
        return df["code"].tolist()

    def _write_records(self, code: str, df: pd.DataFrame) -> bool:
        """写入记录到 Parquet"""
        daily_dir = os.path.join(self.stockdata_root, "raw", "daily")
        os.makedirs(daily_dir, exist_ok=True)

        target_file = os.path.join(daily_dir, f"{code}.parquet")

        try:
            if os.path.exists(target_file):
                existing = pd.read_parquet(target_file)
                # 合并去重
                merged = pd.concat([existing, df]).drop_duplicates(subset=["date"])
                merged = merged.sort_values("date")
            else:
                merged = df

            # 原子写入
            temp_file = os.path.join(
                daily_dir, f"{code}_{datetime.now().strftime('%Y%m%d%H%M%S')}.tmp.parquet"
            )
            merged.to_parquet(temp_file, engine="pyarrow", compression="snappy")
            os.replace(temp_file, target_file)

            logger.info(f"写入成功 {code}: {len(df)} 条")

        except Exception as e:
            logger.error(f"写入失败 {code}: {e}")
            raise

    def _quarantine_records(self, code: str, records: List[tuple]) -> None:
        """隔离低质量记录"""
        quarantine_dir = os.path.join(self.stockdata_root, "quarantine", "backfill")
        os.makedirs(quarantine_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        for record, score in records:
            df = pd.DataFrame([record])
            qfile = os.path.join(quarantine_dir, f"{code}_{record['date']}_{score:.1f}_{timestamp}.parquet")
            df.to_parquet(qfile, engine="pyarrow")

        # 发送告警
        send_alert(
            "WARNING",
            f"回填数据隔离: {code}, {len(records)} 条记录质量 < 60",
            {"code": code, "count": len(records), "sample_score": records[0][1]}
        )

    def _load_progress(self) -> set:
        """加载进度"""
        if not os.path.exists(self.progress_file):
            return set()

        try:
            with open(self.progress_file, "r") as f:
                data = json.load(f)
            return set(data.get("completed_codes", []))
        except Exception as e:
            logger.warning(f"Failed to load backfill progress: {e}")
            return set()

    def _save_progress(self, code: str) -> None:
        """保存进度（原子写入，防止竞态条件）"""
        os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)

        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, "r") as f:
                    data = json.load(f)
            else:
                data = {"start_date": self.start_date, "end_date": self.end_date, "completed_codes": []}

            if code not in data["completed_codes"]:
                data["completed_codes"].append(code)

            # 原子写入：先写临时文件，再 rename
            temp_file = self.progress_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(temp_file, self.progress_file)

        except Exception as e:
            logger.error(f"保存进度失败: {e}")

    def _send_report_alert(self, report: "BackfillReport") -> None:
        """发送回填报告告警"""
        if report.failed_count > 0 or report.quarantined_count > 0:
            level = "ERROR" if report.failed_count > 10 else "WARNING"
            send_alert(
                level,
                f"回填完成: {report.success_count} 成功, {report.quarantined_count} 隔离, {report.failed_count} 失败",
                report.to_dict()
            )


def run_backfill(
    start_date: str,
    end_date: str,
    stockdata_root: str = "/Users/bruce/workspace/trade/StockData",
    codes: Optional[List[str]] = None,
    max_workers: int = 1
) -> BackfillReport:
    """
    执行回填

    Args:
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
        stockdata_root: StockData 根目录
        codes: 股票列表，None = 全市场
        max_workers: 并发数（默认1）

    Returns:
        BackfillReport
    """
    fetcher = BackfillFetcher(
        stockdata_root=stockdata_root,
        start_date=start_date,
        end_date=end_date,
        codes=codes,
        max_workers=max_workers
    )

    return fetcher.fetch_all()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="历史数据回填")
    parser.add_argument("--start", required=True, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--stockdata-root", default="/Users/bruce/workspace/trade/StockData")
    parser.add_argument("--codes", help="逗号分隔的股票代码")

    args = parser.parse_args()

    codes = args.codes.split(",") if args.codes else None

    report = run_backfill(
        start_date=args.start,
        end_date=args.end,
        stockdata_root=args.stockdata_root,
        codes=codes
    )

    print(f"\n=== 回填报告 ===")
    print(f"日期范围: {report.start_date} ~ {report.end_date}")
    print(f"总股票数: {report.total_stocks}")
    print(f"成功: {report.success_count}")
    print(f"隔离: {report.quarantined_count}")
    print(f"失败: {report.failed_count}")
    print(f"总记录: {report.total_records}")
    print(f"耗时: {report.duration_minutes:.1f} 分钟")
