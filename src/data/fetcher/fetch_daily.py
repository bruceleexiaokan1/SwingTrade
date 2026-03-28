"""日线数据采集主入口"""

import os
import sys
import logging
from datetime import datetime, date, timedelta
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "code": "%(name)s", "message": "%(message)s}',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("fetch_daily")

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.fetcher.sources.tushare_source import TushareSource
from src.data.fetcher.sources.akshare_source import AkShareSource
from src.data.fetcher.quality_scorer import QualityScorer
from src.data.fetcher.validators.daily_validator import DailyValidator
from src.data.fetcher.retry_handler import RetryHandler, FetchResult
from src.data.fetcher.report_generator import DailyReportGenerator
from src.data.fetcher.exceptions import ConfigurationError


class DailyFetcher:
    """
    日线数据采集器

    采集流程：
    1. 读取股票列表（从 SQLite 或数据源）
    2. 并行拉取 Tushare 日线数据
    3. 质量评分
    4. 质量不达标时触发 AkShare 验证
    5. 幂等写入 Parquet
    6. 生成日报
    """

    def __init__(
        self,
        stockdata_root: str,
        target_date: Optional[str] = None,
        use_akshare_verify: bool = True
    ):
        """
        Args:
            stockdata_root: StockData 根目录
            target_date: 目标日期，格式 YYYY-MM-DD，默认为上一个交易日
            use_akshare_verify: 是否使用 AkShare 验证质量可疑的数据
        """
        self.stockdata_root = stockdata_root
        self.target_date = target_date or self._get_previous_trading_day()
        self.use_akshare_verify = use_akshare_verify

        # 初始化组件
        self.scraper = None
        self.akshare = None
        self.scorer = QualityScorer()
        self.validator = DailyValidator()
        self.retry_handler = RetryHandler(max_attempts=2)
        self.report = DailyReportGenerator(self.target_date)

        # 状态
        self.success_count = 0
        self.fail_count = 0

    def _get_previous_trading_day(self) -> str:
        """获取上一个交易日（简化版，周一到周五）"""
        today = date.today()
        if today.weekday() == 0:  # Monday
            prev = today - timedelta(days=3)  # Friday
        else:
            prev = today - timedelta(days=1)
        return prev.strftime("%Y-%m-%d")

    def _init_sources(self):
        """初始化数据源"""
        if self.scraper is None:
            self.scraper = TushareSource()
        if self.use_akshare_verify and self.akshare is None:
            try:
                self.akshare = AkShareSource()
            except ConfigurationError:
                logger.warning("AkShare initialization failed, skipping verification")
                self.use_akshare_verify = False

    def _get_stock_list(self) -> list[dict]:
        """
        获取股票列表

        优先从 SQLite 读取，若无则从 Tushare 拉取
        """
        sqlite_db = os.path.join(self.stockdata_root, "sqlite", "market.db")

        if os.path.exists(sqlite_db):
            import sqlite3
            conn = sqlite3.connect(sqlite_db)
            df = conn.execute(
                "SELECT code, name, market FROM stocks WHERE is_active = 1"
            ).fetchall()
            conn.close()

            if df:
                return [{"code": row[0], "name": row[1], "market": row[2]} for row in df]

        # 兜底：从 Tushare 拉取
        logger.info("No stock list found in SQLite, fetching from Tushare")
        self._init_sources()
        df = self.scraper.fetch_stock_list()
        return df.to_dict("records")

    def _fetch_single(self, code: str, name: str) -> FetchResult:
        """
        采集单只股票的日线数据

        Args:
            code: 股票代码
            name: 股票名称（用于日志）

        Returns:
            FetchResult
        """
        self._init_sources()

        # Step 1: 从 Tushare 拉取
        result = self.retry_handler.fetch_with_retry(
            self.scraper, code, self.target_date, self.target_date
        )
        result.code = code

        if result.fetch_status != "success" or result.data is None:
            logger.warning(f"Fetch failed: {code} - {result.fail_reason}")
            return result

        # Step 2: 质量评分
        record = result.data.iloc[0].to_dict()
        quality_score = self.scorer.score(record)
        result.quality_score = quality_score.overall
        result.quality_dims = quality_score.to_dict()

        # Step 3: 质量检查
        if self.scorer.should_reject(quality_score):
            result.write_status = "rejected"
            result.fail_type = "quality"
            result.fail_reason = f"quality_score {quality_score.overall} < 60"
            logger.warning(f"Quality rejected: {code} score={quality_score.overall}")
            return result

        # Step 4: 质量可疑时触发 AkShare 验证
        if self.use_akshare_verify and self.scorer.should_verify(quality_score):
            verify_result = self._verify_with_akshare(code, record)
            if verify_result is not None:
                # 重新评分
                new_quality = self.scorer.score(record, verify_record=verify_result)
                result.quality_score = new_quality.overall
                result.quality_dims = new_quality.to_dict()

                if self.scorer.should_reject(new_quality):
                    result.write_status = "rejected"
                    result.fail_type = "quality"
                    result.fail_reason = f"quality_score {new_quality.overall} < 60 after verification"
                    return result

        # Step 5: 写入
        write_success = self._write_daily(code, result.data)
        if write_success:
            result.write_status = "success"
            logger.info(f"Success: {code} score={result.quality_score}")
        else:
            result.write_status = "failed"
            result.fail_reason = "write_failed"

        return result

    def _verify_with_akshare(self, code: str, record: dict) -> Optional[dict]:
        """
        使用 AkShare 验证数据

        Returns:
            AkShare 返回的记录，若失败返回 None
        """
        if self.akshare is None:
            return None

        try:
            df = self.akshare.fetch_daily(code, self.target_date, self.target_date)
            if df is not None and len(df) > 0:
                return df.iloc[0].to_dict()
        except Exception as e:
            logger.warning(f"AkShare verification failed for {code}: {e}")

        return None

    def _write_daily(self, code: str, df: "pd.DataFrame") -> bool:
        """
        写入日线数据到 Parquet

        Args:
            code: 股票代码
            df: 日线数据 DataFrame

        Returns:
            True if success
        """
        import pandas as pd

        daily_dir = os.path.join(self.stockdata_root, "raw", "daily")
        os.makedirs(daily_dir, exist_ok=True)

        target_file = os.path.join(daily_dir, f"{code}.parquet")

        try:
            if os.path.exists(target_file):
                # 读取现有数据
                existing_df = pd.read_parquet(target_file)

                # 过滤已存在的日期
                new_df = df[~df["date"].isin(existing_df["date"])]

                if len(new_df) == 0:
                    logger.info(f"Data already exists: {code}")
                    return True

                # 合并
                merged_df = pd.concat([existing_df, new_df]).drop_duplicates(["date"])
                merged_df = merged_df.sort_values("date")
            else:
                merged_df = df

            # 写入临时文件
            temp_file = os.path.join(
                daily_dir,
                f"{code}_{datetime.now().strftime('%Y%m%d%H%M%S')}.tmp.parquet"
            )
            merged_df.to_parquet(temp_file, engine="pyarrow", compression="snappy")

            # 原子替换
            os.replace(temp_file, target_file)

            # 更新索引
            self._update_daily_index(code, merged_df)

            return True

        except Exception as e:
            logger.error(f"Write failed: {code} - {e}")
            return False

    def _update_daily_index(self, code: str, df: "pd.DataFrame"):
        """更新 SQLite 日线索引"""
        import sqlite3

        sqlite_db = os.path.join(self.stockdata_root, "sqlite", "market.db")

        if not os.path.exists(os.path.dirname(sqlite_db)):
            return

        try:
            conn = sqlite3.connect(sqlite_db)
            conn.execute('PRAGMA journal_mode=WAL')

            date_range = df["date"].agg(["min", "max"])

            conn.execute("""
                INSERT OR REPLACE INTO daily_index
                (code, latest_date, file_path, row_count, start_date, end_date, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """, [
                code,
                date_range["max"],
                f"raw/daily/{code}.parquet",
                len(df),
                date_range["min"],
                date_range["max"]
            ])

            # 记录 update_log
            conn.execute("""
                INSERT INTO update_log
                (data_type, code, update_date, status, row_count, error_msg, created_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """, [
                "daily",
                code,
                self.target_date,
                "success",
                len(df),
                None
            ])

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update daily index: {e}")

    def fetch_all(self, codes: Optional[list] = None):
        """
        采集全市场日线数据

        Args:
            codes: 股票代码列表，若不提供则自动获取全市场列表
        """
        logger.info(f"Starting daily fetch for {self.target_date}")

        # 获取股票列表
        if codes is None:
            stocks = self._get_stock_list()
            codes = [s["code"] for s in stocks]
            names = {s["code"]: s["name"] for s in stocks}
        else:
            names = {c: "" for c in codes}

        # 逐只采集
        for i, code in enumerate(codes):
            logger.info(f"[{i+1}/{len(codes)}] Fetching {code}")

            result = self._fetch_single(code, names.get(code, ""))

            self.report.add_result(result)

            if result.fetch_status == "success" and result.write_status == "success":
                self.success_count += 1
            else:
                self.fail_count += 1

        # 生成日报
        report = self.report.generate()

        logger.info(f"Fetch completed: {self.success_count} success, {self.fail_count} failed")

        return report


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch daily stock data")
    parser.add_argument(
        "--date",
        type=str,
        help="Target date (YYYY-MM-DD), defaults to previous trading day"
    )
    parser.add_argument(
        "--stockdata-root",
        type=str,
        default="/Users/bruce/workspace/trade/StockData",
        help="StockData root directory"
    )
    parser.add_argument(
        "--codes",
        type=str,
        help="Comma-separated stock codes (for testing)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable AkShare verification"
    )

    args = parser.parse_args()

    fetcher = DailyFetcher(
        stockdata_root=args.stockdata_root,
        target_date=args.date,
        use_akshare_verify=not args.no_verify
    )

    codes = args.codes.split(",") if args.codes else None

    report = fetcher.fetch_all(codes=codes)

    print(f"\n=== Daily Report ===")
    print(f"Date: {report.date}")
    print(f"Total: {report.summary['total_stocks']}")
    print(f"Success: {report.summary['success_count']}")
    print(f"Network Failed: {report.summary['network_failed_count']}")
    print(f"Quality Rejected: {report.summary['quality_rejected_count']}")
    print(f"Success Rate: {report.summary['success_rate']:.2%}")


if __name__ == "__main__":
    main()
