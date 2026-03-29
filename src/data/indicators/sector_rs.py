"""板块相对强度排名系统 (Sector RS/RPS)

RS (相对强度) = 个股20日涨幅 - 板块20日涨幅
RPS = 板块内相对强度排名 (0-100)
RS 创 20 日新高 = 强势板块信号

欧奈尔 RPS 体系: 相对强度 = 个股涨幅 / 大盘涨幅 × 100
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..fetcher.sector_fetcher import SectorDataFetcher
from ..loader import StockDataLoader

logger = logging.getLogger(__name__)


class SectorRelativeStrength:
    """
    板块相对强度排名系统

    提供 RS 和 RPS 计算，用于：
    1. 个股相对强度排名（板块内）
    2. 板块相对强度排名（跨板块）
    3. 强势板块筛选

    使用示例：
        sr = SectorRelativeStrength(
            stockdata_root="/Users/bruce/workspace/trade/StockData"
        )

        # 计算个股RS
        rs_values = sr.calculate_rs("人工智能", ["002371", "688012"], "2026-03-28")

        # 计算个股RPS
        rps_values = sr.calculate_rps("人工智能", ["002371", "688012"], "2026-03-28")

        # 获取最强板块
        top = sr.get_top_sectors(["人工智能", "半导体"], "2026-03-28")
    """

    def __init__(
        self,
        stockdata_root: str = "/Users/bruce/workspace/trade/StockData",
        cache_dir: str = None
    ):
        """
        初始化板块相对强度计算器

        Args:
            stockdata_root: StockData 根目录
            cache_dir: 板块缓存目录
        """
        self.stock_loader = StockDataLoader(stockdata_root)
        self.sector_fetcher = SectorDataFetcher(cache_dir=cache_dir)

        # 缓存
        self._stock_data_cache: Dict[str, pd.DataFrame] = {}
        self._sector_data_cache: Dict[str, pd.DataFrame] = {}

    def _get_stock_data(
        self,
        stock_code: str,
        end_date: str,
        lookback: int = 60
    ) -> pd.DataFrame:
        """
        获取个股数据（带缓存）

        Args:
            stock_code: 股票代码
            end_date: 截止日期
            lookback: 回溯天数

        Returns:
            DataFrame
        """
        if stock_code in self._stock_data_cache:
            return self._stock_data_cache[stock_code]

        start_date = pd.Timestamp(end_date) - pd.DateOffset(days=lookback * 2)
        df = self.stock_loader.load_daily(
            code=stock_code,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date
        )

        if not df.empty:
            self._stock_data_cache[stock_code] = df

        return df

    def _get_sector_data(
        self,
        sector_name: str,
        end_date: str,
        lookback: int = 60
    ) -> pd.DataFrame:
        """
        获取板块数据（带缓存）

        Args:
            sector_name: 板块名称
            end_date: 截止日期
            lookback: 回溯天数

        Returns:
            DataFrame
        """
        if sector_name in self._sector_data_cache:
            return self._sector_data_cache[sector_name]

        start_date = pd.Timestamp(end_date) - pd.DateOffset(days=lookback * 2)
        df = self.sector_fetcher.fetch_sector_daily(
            sector_name=sector_name,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date
        )

        if not df.empty:
            self._sector_data_cache[sector_name] = df

        return df

    def calculate_rs(
        self,
        sector_name: str,
        stock_codes: List[str],
        date: str,
        lookback: int = 20
    ) -> Dict[str, float]:
        """
        计算个股相对强度

        RS = 个股20日涨幅 - 板块20日涨幅

        Args:
            sector_name: 板块名称
            stock_codes: 个股代码列表
            date: 日期
            lookback: 回溯周期（默认20日）

        Returns:
            {code: rs_value} - RS 值，正值表示强于板块
        """
        # 获取板块20日涨幅
        sector_df = self._get_sector_data(sector_name, date, lookback + 10)
        if sector_df.empty or len(sector_df) < lookback + 1:
            logger.warning(f"板块数据不足: {sector_name}")
            return {code: 0.0 for code in stock_codes}

        sector_return = (sector_df['close'].iloc[-1] / sector_df['close'].iloc[-lookback] - 1) * 100

        rs_values = {}

        for code in stock_codes:
            stock_df = self._get_stock_data(code, date, lookback + 10)
            if stock_df.empty or len(stock_df) < lookback + 1:
                logger.warning(f"个股数据不足: {code}")
                rs_values[code] = 0.0
                continue

            stock_return = (stock_df['close'].iloc[-1] / stock_df['close'].iloc[-lookback] - 1) * 100
            rs = stock_return - sector_return
            rs_values[code] = rs

        return rs_values

    def calculate_rps(
        self,
        sector_name: str,
        stock_codes: List[str],
        date: str,
        lookback: int = 20
    ) -> Dict[str, float]:
        """
        计算板块内相对强度排名 RPS (0.0 ~ 1.0)

        RPS = 排名 / 总数

        Args:
            sector_name: 板块名称
            stock_codes: 个股代码列表
            date: 日期
            lookback: 回溯周期（默认20日）

        Returns:
            {code: rps_value} - RPS 值，1.0 表示板块最强
        """
        rs_values = self.calculate_rs(sector_name, stock_codes, date, lookback)

        # 按 RS 值排序
        sorted_stocks = sorted(rs_values.items(), key=lambda x: x[1], reverse=True)

        rps_values = {}
        total = len(sorted_stocks)

        for rank, (code, rs) in enumerate(sorted_stocks, 1):
            # RPS = 排名 / 总数 (0.0 ~ 1.0)
            rps = (total - rank) / total if total > 0 else 0.0
            rps_values[code] = rps

        return rps_values

    def get_rs_rank(
        self,
        sector_name: str,
        stock_codes: List[str],
        date: str,
        lookback: int = 20
    ) -> Dict[str, float]:
        """
        获取个股 RS 排名（兼容旧接口）

        等同于 calculate_rps()
        """
        return self.calculate_rps(sector_name, stock_codes, date, lookback)

    def get_top_sectors(
        self,
        sector_names: List[str],
        date: str,
        lookback: int = 20,
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        获取最强板块（按 RS 排名）

        Args:
            sector_names: 板块名称列表
            date: 日期
            lookback: 回溯周期（默认20日）
            top_n: 返回前 N 个板块

        Returns:
            [(sector_name, rs_score), ...] - 按 RS 降序排列
        """
        sector_rs_scores = {}

        for sector_name in sector_names:
            try:
                # 获取板块的代表性RS（使用成分股等权平均）
                config = self._get_sector_config()
                stock_codes = self._get_sector_stocks(sector_name, config)

                if not stock_codes:
                    # 尝试获取成分股
                    stock_codes = self.sector_fetcher.get_constituents(sector_name)

                if not stock_codes:
                    logger.warning(f"未找到板块成分股: {sector_name}")
                    sector_rs_scores[sector_name] = 0.0
                    continue

                # 计算板块内所有成分股的平均RS
                rs_values = self.calculate_rs(sector_name, stock_codes, date, lookback)
                avg_rs = sum(rs_values.values()) / len(rs_values) if rs_values else 0.0
                sector_rs_scores[sector_name] = avg_rs

            except Exception as e:
                logger.error(f"计算板块RS失败 {sector_name}: {e}")
                sector_rs_scores[sector_name] = 0.0

        # 按 RS 降序排列
        sorted_sectors = sorted(sector_rs_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_sectors[:top_n]

    def is_rs_20d_high(
        self,
        sector_name: str,
        date: str,
        lookback: int = 20
    ) -> bool:
        """
        判断板块 RS 是否创 20 日新高

        Args:
            sector_name: 板块名称
            date: 日期
            lookback: 回溯周期

        Returns:
            True if RS 创 20 日新高
        """
        sector_df = self._get_sector_data(sector_name, date, lookback * 2 + 10)
        if sector_df.empty or len(sector_df) < lookback * 2:
            return False

        # 获取历史 RS 序列
        rs_series = []
        for i in range(lookback, len(sector_df) - lookback):
            current_return = (sector_df['close'].iloc[i] / sector_df['close'].iloc[i - lookback] - 1) * 100
            prev_return = (sector_df['close'].iloc[i - lookback] / sector_df['close'].iloc[i - lookback * 2] - 1) * 100
            rs = current_return - prev_return
            rs_series.append(rs)

        if not rs_series:
            return False

        # 当前 RS
        current_idx = len(sector_df) - 1
        current_return = (sector_df['close'].iloc[current_idx] / sector_df['close'].iloc[current_idx - lookback] - 1) * 100
        prev_return = (sector_df['close'].iloc[current_idx - lookback] / sector_df['close'].iloc[current_idx - lookback * 2] - 1) * 100
        current_rs = current_return - prev_return

        # 检查是否创 20 日新高
        return current_rs == max(rs_series[-lookback:])

    def _get_sector_config(self) -> Dict:
        """获取板块配置"""
        from pathlib import Path
        config_path = Path(__file__).parent.parent.parent / "config" / "sectors" / "sector_portfolio.json"
        if not hasattr(self, '_sector_config'):
            import json
            with open(config_path) as f:
                self._sector_config = json.load(f)
        return self._sector_config

    def _get_sector_stocks(self, sector_name: str, config: Dict) -> List[str]:
        """从配置获取板块成分股"""
        for sector in config.get('sectors', []):
            if sector.get('name') == sector_name:
                return [s['code'] for s in sector.get('stocks', [])]
        return []

    def clear_cache(self):
        """清除缓存"""
        self._stock_data_cache.clear()
        self._sector_data_cache.clear()
