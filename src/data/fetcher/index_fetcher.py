"""指数数据采集器

支持主要宽基指数的历史数据回填
"""

import os
import time
import logging
from datetime import datetime
from typing import Optional, List
from pathlib import Path

import pandas as pd

from .exceptions import NetworkError, SourceError, ConfigurationError

logger = logging.getLogger("index_fetcher")


# 主要宽基指数列表
INDEX_POOL = [
    {"code": "000001.SH", "name": "上证指数"},
    {"code": "000300.SH", "name": "沪深300"},
    {"code": "000016.SH", "name": "上证50"},
    {"code": "399001.SZ", "name": "深证成指"},
    {"code": "399006.SZ", "name": "创业板指"},
    {"code": "000852.SH", "name": "中证1000"},
]


class IndexFetcher:
    """
    指数数据采集器

    用于采集宽基指数的历史数据，支持：
    - 断点续传
    - 速率限制
    """

    # 速率限制：与 TushareSource 共享 36 calls/min
    RATE_LIMIT = 36

    def __init__(
        self,
        stockdata_root: str,
        start_date: str = "2021-03-29",
        end_date: str = None,
        rate_limit_buffer: float = 0.8
    ):
        """
        初始化指数采集器

        Args:
            stockdata_root: StockData 根目录
            start_date: 回填开始日期
            end_date: 回填结束日期，默认为今天
            rate_limit_buffer: 速率限制安全系数
        """
        self.stockdata_root = stockdata_root
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")

        # 速率限制
        self.rate_limit = int(self.RATE_LIMIT * rate_limit_buffer)
        self.min_interval = 60.0 / self.rate_limit if self.rate_limit > 0 else 0
        self._last_call_time = 0

        # 初始化 Tushare
        token = os.getenv("TUSHARE_TOKEN")
        if not token:
            raise ConfigurationError("TUSHARE_TOKEN not set", date=None)

        try:
            import tushare as ts
            self.pro = ts.pro_api(token)
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Tushare: {e}", date=None)

        # 索引存储目录
        self.index_dir = Path(stockdata_root) / "raw" / "index"
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def _rate_limit_sleep(self) -> None:
        """速率限制休眠"""
        now = time.time()
        elapsed = now - self._last_call_time

        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        self._last_call_time = time.time()

    def fetch_index(self, index_code: str) -> pd.DataFrame:
        """
        采集单个指数数据

        Args:
            index_code: 指数代码，如 "000300.SH"

        Returns:
            DataFrame
        """
        self._rate_limit_sleep()

        start_str = self.start_date.replace("-", "")
        end_str = self.end_date.replace("-", "")

        try:
            df = self.pro.index_daily(
                ts_code=index_code,
                start_date=start_str,
                end_date=end_str
            )
        except Exception as e:
            raise NetworkError(f"Failed to fetch index {index_code}: {e}", date=self.end_date)

        if df is None or len(df) == 0:
            logger.warning(f"No data for index {index_code}")
            return pd.DataFrame()

        # 字段映射
        df = df.rename(columns={
            "ts_code": "code",
            "trade_date": "date",
            "vol": "volume"
        })

        # 转换日期格式
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        # 确保字段类型
        df["volume"] = df["volume"].astype("int64")

        return df

    def fetch_all(self) -> dict:
        """
        采集所有指数数据

        Returns:
            dict: {index_code: result}
        """
        results = {}

        for idx in INDEX_POOL:
            code = idx["code"]
            name = idx["name"]
            logger.info(f"Fetching {name} ({code})...")

            try:
                df = self.fetch_index(code)

                if len(df) > 0:
                    # 写入 Parquet
                    self._write_index(code, df)
                    results[code] = {
                        "status": "success",
                        "records": len(df),
                        "name": name
                    }
                else:
                    results[code] = {
                        "status": "no_data",
                        "name": name
                    }

            except Exception as e:
                logger.error(f"Failed to fetch {code}: {e}")
                results[code] = {
                    "status": "error",
                    "error": str(e),
                    "name": name
                }

        return results

    def _write_index(self, code: str, df: pd.DataFrame) -> None:
        """写入指数数据"""
        target_file = self.index_dir / f"{code}.parquet"

        try:
            if target_file.exists():
                # 合并去重
                existing = pd.read_parquet(target_file)
                merged = pd.concat([existing, df]).drop_duplicates(subset=["date"])
                merged = merged.sort_values("date")
            else:
                merged = df

            # 原子写入
            temp_file = self.index_dir / f"{code}_{datetime.now().strftime('%Y%M%d%H%M%S')}.tmp.parquet"
            merged.to_parquet(temp_file, engine="pyarrow", compression="snappy")
            temp_file.replace(target_file)

            logger.info(f"Written {code}: {len(df)} records to {target_file}")

        except Exception as e:
            logger.error(f"Failed to write {code}: {e}")
            raise

    def get_existing_indices(self) -> List[str]:
        """获取已采集的指数列表"""
        return [f.stem for f in self.index_dir.glob("*.parquet")]


def run_index_backfill(
    stockdata_root: str = "/Users/bruce/workspace/trade/StockData",
    start_date: str = "2021-03-29",
    end_date: str = None
) -> dict:
    """
    执行指数回填

    Args:
        stockdata_root: StockData 根目录
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        dict: 回填结果
    """
    fetcher = IndexFetcher(
        stockdata_root=stockdata_root,
        start_date=start_date,
        end_date=end_date
    )

    return fetcher.fetch_all()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="指数数据回填")
    parser.add_argument("--start", default="2021-03-29", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--stockdata-root", default="/Users/bruce/workspace/trade/StockData")

    args = parser.parse_args()

    results = run_index_backfill(
        stockdata_root=args.stockdata_root,
        start_date=args.start,
        end_date=args.end
    )

    print("\n=== 指数回填结果 ===")
    for code, result in results.items():
        status = result["status"]
        name = result.get("name", code)
        if status == "success":
            print(f"  {name} ({code}): {result['records']} 条 ✓")
        else:
            print(f"  {name} ({code}): {status} ✗")
