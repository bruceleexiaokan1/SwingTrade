"""板块数据获取器

**只使用 EastMoney 数据源（akshare）**

数据来源：
- 板块历史数据：akshare stock_board_concept_hist_em
- 板块成分股：akshare stock_board_concept_cons_em
- 板块列表：akshare stock_board_concept_name_em

本地缓存：
- 使用 parquet 格式缓存
- 减少网络请求
- 支持增量更新
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


class SectorDataFetcher:
    """
    板块数据获取器 - EastMoney 专版

    只使用 EastMoney（akshare）数据源，不依赖其他数据。

    数据接口：
    - stock_board_concept_hist_em: 板块历史日线
    - stock_board_concept_cons_em: 板块成分股
    - stock_board_concept_name_em: 板块列表

    缓存策略：
    - 本地 parquet 缓存
    - 超过 24 小时自动更新
    - 支持强制刷新

    速率限制：
    - EastMoney API: 2次/秒（保守值，留有余量）
    """

    # EastMoney 板块代码前缀
    EASTMONEY_CODE_PREFIX = "BK"

    # 速率限制：2次/秒（保守值）
    RATE_LIMIT_PER_SECOND = 2
    MIN_INTERVAL = 1.0 / RATE_LIMIT_PER_SECOND

    def __init__(
        self,
        cache_dir: str = None,
        cache_ttl_hours: int = 24
    ):
        """
        初始化板块数据获取器

        Args:
            cache_dir: 缓存目录，默认使用 StockData/sector_cache
            cache_ttl_hours: 缓存有效期（小时），默认 24 小时
        """
        if cache_dir is None:
            stockdata_root = "/Users/bruce/workspace/trade/StockData"
            self.cache_dir = Path(stockdata_root) / "sector_cache"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_ttl_hours = cache_ttl_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # akshare 接口
        self._ak = None

        # 板块名称到代码的映射
        self._name_to_code: Dict[str, str] = {}
        self._code_to_name: Dict[str, str] = {}

        # 速率限制状态
        self._last_call_time = 0.0

    def _get_akshare(self):
        """懒加载 akshare"""
        if self._ak is None:
            try:
                import akshare as ak
                self._ak = ak
            except ImportError:
                raise ImportError(
                    "akshare is not installed. Run: pip install akshare"
                )
        return self._ak

    def _rate_limit_sleep(self) -> None:
        """速率限制休眠"""
        now = time.time()
        elapsed = now - self._last_call_time

        if elapsed < self.MIN_INTERVAL:
            sleep_time = self.MIN_INTERVAL - elapsed
            logger.debug(f"速率限制等待: {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self._last_call_time = time.time()

    def fetch_sector_daily(
        self,
        sector_name: str,
        start_date: str,
        end_date: str,
        force_update: bool = False
    ) -> pd.DataFrame:
        """
        获取板块日线数据（EastMoney）

        Args:
            sector_name: 板块名称，如 "半导体概念"
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            force_update: 是否强制更新缓存

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        cache_file = self.cache_dir / f"{sector_name}.parquet"

        # 检查缓存
        if not force_update and cache_file.exists():
            df_cache = pd.read_parquet(cache_file)
            if self._is_cache_valid(df_cache):
                logger.debug(f"使用缓存数据: {sector_name}")
                return df_cache[
                    (df_cache['date'] >= start_date) &
                    (df_cache['date'] <= end_date)
                ]

        # 获取新数据
        logger.info(f"获取板块数据: {sector_name}")
        df = self._fetch_from_eastmoney(sector_name, start_date, end_date)

        if df.empty:
            logger.warning(f"板块数据为空: {sector_name}")
            return df

        # 更新缓存
        self._save_cache(df, sector_name)

        return df[
            (df['date'] >= start_date) &
            (df['date'] <= end_date)
        ]

    def _fetch_from_eastmoney(
        self,
        sector_name: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        从 EastMoney 获取板块数据

        Args:
            sector_name: 板块名称
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        ak = self._get_akshare()

        # 速率限制
        self._rate_limit_sleep()

        # 转换日期格式：YYYY-MM-DD -> YYYYMMDD
        start_str = start_date.replace("-", "")
        end_str = end_date.replace("-", "")

        try:
            df = ak.stock_board_concept_hist_em(
                symbol=sector_name,
                period="daily",
                start_date=start_str,
                end_date=end_str
            )

            if df is None or df.empty:
                return pd.DataFrame()

            # 列名映射（EastMoney -> 标准）
            column_map = {
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change',
                '换手率': 'turnover'
            }

            df = df.rename(columns=column_map)

            # 转换日期格式
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            # 确保字段类型
            if 'volume' in df.columns:
                df['volume'] = df['volume'].astype('int64')

            # 按日期排序
            df = df.sort_values('date').reset_index(drop=True)

            # 返回标准列
            standard_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']
            existing_cols = [c for c in standard_cols if c in df.columns]
            return df[existing_cols]

        except Exception as e:
            logger.error(f"获取板块数据失败 {sector_name}: {e}")
            return pd.DataFrame()

    def get_constituents(
        self,
        sector_name: str,
        force_update: bool = False
    ) -> List[str]:
        """
        获取板块成分股（EastMoney）

        Args:
            sector_name: 板块名称，如 "半导体概念"
            force_update: 是否强制更新缓存

        Returns:
            股票代码列表，如 ["002371", "688012", ...]
        """
        cache_file = self.cache_dir / f"{sector_name}_constituents.parquet"

        # 检查缓存（成分股用文件修改时间验证）
        if not force_update and cache_file.exists():
            file_age_hours = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
            if file_age_hours < self.cache_ttl_hours:
                df_cache = pd.read_parquet(cache_file)
                return df_cache['code'].tolist()

        # 获取新数据
        ak = self._get_akshare()

        # 速率限制
        self._rate_limit_sleep()

        try:
            df = ak.stock_board_concept_cons_em(symbol=sector_name)

            if df is None or df.empty:
                return []

            # 提取股票代码（东方财富代码格式）
            codes = df['代码'].astype(str).tolist()

            # 更新缓存
            pd.DataFrame({'code': codes}).to_parquet(cache_file, index=False)

            return codes

        except Exception as e:
            logger.error(f"获取板块成分股失败 {sector_name}: {e}")
            return []

    def get_all_sectors(self) -> List[Dict[str, str]]:
        """
        获取所有概念板块（EastMoney）

        Returns:
            List of dicts with 'name' and 'code' keys
        """
        ak = self._get_akshare()

        # 速率限制
        self._rate_limit_sleep()

        try:
            df = ak.stock_board_concept_name_em()

            if df is None or df.empty:
                return []

            # 构建映射
            self._name_to_code = dict(zip(df['板块名称'], df['板块代码']))
            self._code_to_name = dict(zip(df['板块代码'], df['板块名称']))

            return [
                {'name': row['板块名称'], 'code': row['板块代码']}
                for _, row in df.iterrows()
            ]

        except Exception as e:
            logger.error(f"获取板块列表失败: {e}")
            return []

    def get_sector_code(self, sector_name: str) -> Optional[str]:
        """
        根据板块名称获取 EastMoney 代码

        Args:
            sector_name: 板块名称

        Returns:
            EastMoney 板块代码，如 "BK0917"
        """
        if not self._name_to_code:
            self.get_all_sectors()

        return self._name_to_code.get(sector_name)

    def get_sector_name(self, sector_code: str) -> Optional[str]:
        """
        根据 EastMoney 代码获取板块名称

        Args:
            sector_code: EastMoney 板块代码，如 "BK0917"

        Returns:
            板块名称
        """
        if not self._code_to_name:
            self.get_all_sectors()

        return self._code_to_name.get(sector_code)

    def backfill_sector_data(
        self,
        sector_name: str,
        start_date: str,
        end_date: str,
        force_update: bool = False
    ) -> pd.DataFrame:
        """
        回填板块历史数据

        EastMoney 通常只有约 1 年的历史数据。
        如果需要更长的历史，使用 synthesize_sector_index() 合成。

        Args:
            sector_name: 板块名称
            start_date: 开始日期
            end_date: 结束日期
            force_update: 是否强制更新

        Returns:
            板块日线数据
        """
        return self.fetch_sector_daily(
            sector_name=sector_name,
            start_date=start_date,
            end_date=end_date,
            force_update=force_update
        )

    def synthesize_sector_index(
        self,
        sector_name: str,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        stockdata_root: str = "/Users/bruce/workspace/trade/StockData"
    ) -> pd.DataFrame:
        """
        合成板块指数（当官方数据不足时）

        使用板块成分股的加权（等权）日收益率计算累计指数。

        Args:
            sector_name: 板块名称（用于缓存）
            stock_codes: 成分股代码列表
            start_date: 开始日期
            end_date: 结束日期
            stockdata_root: StockData 根目录

        Returns:
            DataFrame with columns: date, close
        """
        from ..loader import StockDataLoader

        loader = StockDataLoader(stockdata_root)

        # 收集所有成分股的日收益率
        all_returns = []
        valid_codes = []

        for code in stock_codes:
            df = loader.load_daily(code, start_date, end_date)
            if df.empty or len(df) < 2:
                continue

            # 计算日收益率
            df = df.copy()
            df['return'] = df['close'].pct_change()
            df = df[['date', 'return']].rename(columns={'return': code})
            df = df.set_index('date')
            all_returns.append(df)
            valid_codes.append(code)

        if not all_returns:
            logger.warning(f"没有有效的成分股数据: {sector_name}")
            return pd.DataFrame()

        # 合并所有收益率
        combined = pd.concat(all_returns, axis=1)

        # 等权平均
        combined['avg_return'] = combined.mean(axis=1)

        # 计算累计净值（基准 1000 点）
        net_value = (1 + combined['avg_return']).cumprod() * 1000

        result = pd.DataFrame({
            'date': net_value.index,
            'close': net_value.values,
            'n_stocks': len(valid_codes)
        })

        result['date'] = pd.to_datetime(result['date']).dt.strftime('%Y-%m-%d')

        logger.info(f"合成板块指数: {sector_name}, 有效成分股: {len(valid_codes)}")

        return result

    def _is_cache_valid(self, df: pd.DataFrame) -> bool:
        """检查缓存是否有效"""
        if df.empty:
            return False

        try:
            latest_date = pd.to_datetime(df['date'].max())
            age_hours = (datetime.now() - latest_date).total_seconds() / 3600
            return age_hours < self.cache_ttl_hours
        except Exception as e:
            logger.warning(f"检查缓存有效性失败: {e}")
            return False

    def _save_cache(self, df: pd.DataFrame, sector_name: str):
        """保存缓存"""
        cache_file = self.cache_dir / f"{sector_name}.parquet"
        try:
            df.to_parquet(cache_file, index=False)
            logger.debug(f"更新缓存: {sector_name}")
        except Exception as e:
            logger.warning(f"更新缓存失败: {e}")

    def clear_cache(self, sector_name: str = None):
        """
        清除缓存

        Args:
            sector_name: None 表示清除所有缓存
        """
        if sector_name:
            for pattern in [f"{sector_name}.parquet", f"{sector_name}_constituents.parquet"]:
                cache_file = self.cache_dir / pattern
                if cache_file.exists():
                    cache_file.unlink()
                    logger.info(f"清除缓存: {sector_name}")
        else:
            for f in self.cache_dir.glob("*.parquet"):
                f.unlink()
            logger.info("清除所有缓存")

    def get_cache_info(self) -> Dict[str, any]:
        """
        获取缓存信息

        Returns:
            缓存统计信息
        """
        cache_files = list(self.cache_dir.glob("*.parquet"))

        total_size = sum(f.stat().st_size for f in cache_files)
        latest_update = None

        for f in cache_files:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if latest_update is None or mtime > latest_update:
                latest_update = mtime

        return {
            'cache_dir': str(self.cache_dir),
            'n_files': len(cache_files),
            'total_size_mb': round(total_size / 1024 / 1024, 2),
            'latest_update': latest_update
        }
