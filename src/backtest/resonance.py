"""共振检测器

核心职责：
1. 加载板块配置和成分股
2. 获取板块和个股数据
3. 计算板块和个股指标
4. 检测共振条件
5. 生成共振候选列表

配合 SectorSignals 和 SwingSignals 进行共振检测。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import pandas as pd
import numpy as np

from ..data.loader import StockDataLoader
from ..data.fetcher.sector_fetcher import SectorDataFetcher
from ..data.indicators.sector_signals import SectorSignals, SectorSignalResult
from ..data.indicators.signals import SwingSignals
from ..data.indicators.sector_rs import SectorRelativeStrength
from ..data.indicators.ma import golden_cross
from ..data.indicators.resonance import ResonanceResult, create_resonance_result, ResonanceLevel, ResonanceDataError

logger = logging.getLogger(__name__)


class ResonanceDetector:
    """
    共振检测器

    使用示例：
        detector = ResonanceDetector(
            sector_config_path="config/sectors/sector_portfolio.json",
            stockdata_root="/Users/bruce/workspace/trade/StockData"
        )

        # 检测单只股票
        result = detector.check_resonance("002371", "半导体概念", "2026-03-28")

        # 检测板块内所有股票
        results = detector.check_sector_resonance("半导体概念", "2026-03-28")
    """

    # 金叉即将确认的阈值（MA5 距离 MA20 < 2%）
    GOLDEN_CROSS_APPROACHING_THRESHOLD = 0.02

    def __init__(
        self,
        sector_config_path: str = "config/sectors/sector_portfolio.json",
        stockdata_root: str = "/Users/bruce/workspace/trade/StockData",
        params = None,
        n_workers: int = None
    ):
        """
        初始化共振检测器

        Args:
            sector_config_path: 板块配置路径
            stockdata_root: StockData 根目录
            params: 策略参数
            n_workers: 并行工作进程数，默认为 CPU 核数
        """
        self.sector_config_path = Path(sector_config_path)
        self.stockdata_root = stockdata_root
        self.params = params
        self.n_workers = n_workers or mp.cpu_count()

        # 数据加载器
        self.stock_loader = StockDataLoader(stockdata_root)
        self.sector_fetcher = SectorDataFetcher()

        # 信号检测器
        self.sector_signals = SectorSignals(params)
        self.stock_signals = SwingSignals(params)

        # RS/RPS 计算器
        self.sector_rs = SectorRelativeStrength(stockdata_root=stockdata_root)

        # 缓存
        self._sector_data_cache: Dict[str, pd.DataFrame] = {}
        self._stock_data_cache: Dict[str, pd.DataFrame] = {}

    def load_sector_config(self) -> Dict:
        """加载板块配置"""
        with open(self.sector_config_path) as f:
            return json.load(f)

    def get_sector_config(self) -> Dict:
        """获取板块配置（带缓存）"""
        if not hasattr(self, '_sector_config'):
            self._sector_config = self.load_sector_config()
        return self._sector_config

    def get_sector_data(
        self,
        sector_name: str,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取板块数据（带缓存）

        Args:
            sector_name: 板块名称
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            板块日线数据
        """
        # 缓存键（只按板块名，去掉日期参数避免每天产生新条目）
        cache_key = sector_name

        if cache_key not in self._sector_data_cache:
            # 如果没有指定日期，使用过去一年
            if end_date is None:
                end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (pd.Timestamp.now() - pd.DateOffset(years=1)).strftime('%Y-%m-%d')

            df = self.sector_fetcher.fetch_sector_daily(
                sector_name=sector_name,
                start_date=start_date,
                end_date=end_date
            )

            self._sector_data_cache[cache_key] = df

        # 从缓存中按日期过滤返回（避免每次都重新获取）
        cached_df = self._sector_data_cache[cache_key]
        if start_date is not None or end_date is not None:
            mask = True
            if start_date is not None:
                mask = mask & (cached_df['date'] >= start_date)
            if end_date is not None:
                mask = mask & (cached_df['date'] <= end_date)
            return cached_df[mask].copy()
        return cached_df.copy()

    def get_stock_data(
        self,
        stock_code: str,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取个股数据（带缓存）

        Args:
            stock_code: 股票代码
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            个股日线数据
        """
        # 缓存键（只按股票代码，去掉日期避免每天产生新条目）
        cache_key = stock_code

        if cache_key not in self._stock_data_cache:
            # 个股数据一次性加载完整区间（由调用方确保足够长）
            df = self.stock_loader.load_daily(
                code=stock_code,
                start_date=None,
                end_date=None
            )
            self._stock_data_cache[cache_key] = df

        # 从缓存中按日期过滤返回
        cached_df = self._stock_data_cache[cache_key]
        if start_date is not None or end_date is not None:
            mask = True
            if start_date is not None:
                mask = mask & (cached_df['date'] >= start_date)
            if end_date is not None:
                mask = mask & (cached_df['date'] <= end_date)
            return cached_df[mask].copy()
        return cached_df.copy()

    def check_resonance(
        self,
        stock_code: str,
        sector_name: str,
        date: str
    ) -> ResonanceResult:
        """
        检查个股-板块共振

        Args:
            stock_code: 股票代码
            sector_name: 板块名称
            date: 日期

        Returns:
            ResonanceResult
        """
        # 1. 获取板块数据
        sector_df = self.get_sector_data(sector_name, end_date=date)
        if sector_df.empty:
            raise ResonanceDataError(stock_code, sector_name, "板块数据为空")

        # 取到 date 为止的数据
        sector_df = sector_df[sector_df['date'] <= date].copy()
        if len(sector_df) < 60:
            raise ResonanceDataError(stock_code, sector_name, f"板块数据不足，仅有 {len(sector_df)} 条")

        # 2. 获取个股数据
        stock_df = self.get_stock_data(stock_code, end_date=date)
        if stock_df.empty:
            raise ResonanceDataError(stock_code, sector_name, "个股数据为空")

        stock_df = stock_df[stock_df['date'] <= date].copy()
        if len(stock_df) < 60:
            raise ResonanceDataError(stock_code, sector_name, f"个股数据不足，仅有 {len(stock_df)} 条")

        # 3. 计算板块指标
        sector_df = self.sector_signals.calculate_all(sector_df)
        sector_signal = self.sector_signals.detect_trend(sector_df)
        sector_trend, sector_trend_conf = sector_signal
        sector_rsi = sector_df['rsi14'].iloc[-1]
        sector_momentum = sector_df['momentum_20d'].iloc[-1]

        # 4. 计算个股指标
        stock_df = self.stock_signals.calculate_all(stock_df)
        stock_signal = self.stock_signals.detect_trend(stock_df)
        stock_trend, stock_trend_conf = stock_signal
        stock_rsi = stock_df['rsi14'].iloc[-1]

        # 5. 检测金叉
        stock_gc = golden_cross(stock_df['ma5'], stock_df['ma20']) if 'ma5' in stock_df.columns else False

        # 6. 检测即将金叉
        stock_gc_approaching = self._check_golden_cross_approaching(stock_df)

        # 7. 计算相对强度排名
        stock_rs_rank, stock_rs_score = self._calculate_rs_rank(stock_code, sector_name, date)

        # 8. 市场趋势（简化：使用上证指数）
        market_trend, market_trend_conf = self._get_market_trend(date)

        # 9. 创建共振结果
        result = create_resonance_result(
            date=date,
            stock_code=stock_code,
            sector_name=sector_name,
            sector_trend=sector_trend,
            sector_rsi=sector_rsi,
            sector_momentum=sector_momentum,
            stock_trend=stock_trend,
            stock_rsi=stock_rsi,
            stock_golden_cross=stock_gc,
            stock_gc_approaching=stock_gc_approaching,
            stock_rs_rank=stock_rs_rank,
            stock_rs_score=stock_rs_score,
            market_trend=market_trend
        )

        return result

    def check_sector_resonance(
        self,
        sector_name: str,
        date: str
    ) -> List[ResonanceResult]:
        """
        检查板块内所有股票的共振状态

        Args:
            sector_name: 板块名称
            date: 日期

        Returns:
            共振结果列表
        """
        # 获取板块成分股
        config = self.get_sector_config()
        stock_codes = []

        for sector in config.get('sectors', []):
            if sector.get('name') == sector_name:
                stock_codes = [s['code'] for s in sector.get('stocks', [])]
                break

        if not stock_codes:
            logger.warning(f"未找到板块成分股: {sector_name}")
            return []

        # 批量检查共振
        results = []
        for code in stock_codes:
            result = self.check_resonance(code, sector_name, date)
            results.append(result)

        return results

    def get_resonance_stocks(
        self,
        sector_name: str,
        date: str,
        min_level: ResonanceLevel = ResonanceLevel.C
    ) -> List[ResonanceResult]:
        """
        获取共振股票列表

        Args:
            sector_name: 板块名称
            date: 日期
            min_level: 最低共振等级

        Returns:
            符合条件的共振结果列表
        """
        results = self.check_sector_resonance(sector_name, date)

        # 过滤
        filtered = [
            r for r in results
            if r.is_resonance and r.resonance_level.value >= min_level.value  # noqa: IntEnum supports >= comparison
        ]

        # 按置信度排序
        filtered.sort(key=lambda x: x.resonance_confidence, reverse=True)

        return filtered

    def _check_golden_cross_approaching(self, df: pd.DataFrame) -> bool:
        """
        检测是否即将金叉

        条件：MA5 < MA20 但距离 < 2%
        """
        if 'ma5' not in df.columns or 'ma20' not in df.columns:
            return False

        ma5 = df['ma5'].iloc[-1]
        ma20 = df['ma20'].iloc[-1]

        if ma5 > ma20:
            return False

        # MA5 距离 MA20 < 2%
        distance = (ma20 - ma5) / ma20
        return distance < self.GOLDEN_CROSS_APPROACHING_THRESHOLD

    def _calculate_rs_rank(
        self,
        stock_code: str,
        sector_name: str,
        date: str,
        lookback: int = 20
    ) -> Tuple[float, float]:
        """
        计算个股相对强度在板块内的排名和原始RS值

        RS = 个股 20 日涨幅 - 板块 20 日涨幅
        RPS = 板块内排名 (0.0 ~ 1.0)

        Args:
            stock_code: 股票代码
            sector_name: 板块名称
            date: 日期
            lookback: 回溯周期

        Returns:
            (rps_rank, rs_score) - (排名 0.0~1.0, 原始RS值)
        """
        try:
            # 获取板块成分股
            config = self.get_sector_config()
            stock_codes = []
            for sector in config.get('sectors', []):
                if sector.get('name') == sector_name:
                    stock_codes = [s['code'] for s in sector.get('stocks', [])]
                    break

            if not stock_codes:
                # 简化为二元判断
                stock_df = self.get_stock_data(stock_code, end_date=date)
                if stock_df.empty or len(stock_df) < lookback:
                    return (0.5, 0.0)
                sector_df = self.get_sector_data(sector_name, end_date=date)
                if sector_df.empty or len(sector_df) < lookback:
                    return (0.5, 0.0)

                stock_return = (stock_df['close'].iloc[-1] / stock_df['close'].iloc[-lookback] - 1) * 100
                sector_return = (sector_df['close'].iloc[-1] / sector_df['close'].iloc[-lookback] - 1) * 100
                rs = stock_return - sector_return
                return (1.0 if rs > 0 else 0.3, rs)

            # 计算板块内所有成分股的 RPS
            rps_values = self.sector_rs.calculate_rps(sector_name, stock_codes, date, lookback)
            rs_values = self.sector_rs.calculate_rs(sector_name, stock_codes, date, lookback)

            rps_rank = rps_values.get(stock_code, 0.5)
            rs_score = rs_values.get(stock_code, 0.0)

            return (rps_rank, rs_score)

        except Exception as e:
            logger.debug(f"计算 RS 排名失败: {e}")
            return (0.5, 0.0)

    def _get_market_trend(self, date: str) -> Tuple[str, float]:
        """
        获取市场趋势（简化版：使用上证指数）

        Args:
            date: 日期

        Returns:
            (趋势, 置信度)
        """
        try:
            # 使用 000001（上证指数）
            index_df = self.stock_loader.load_daily("000001", end_date=date)
            if index_df.empty or len(index_df) < 60:
                return ("sideways", 0.0)

            # 计算 MA
            index_df = self.sector_signals.calculate_all(index_df)

            # 检测趋势
            trend, conf = self.sector_signals.detect_trend(index_df)
            return (trend, conf)

        except Exception as e:
            logger.debug(f"获取市场趋势失败: {e}")
            return ("sideways", 0.0)

    def clear_cache(self):
        """清除缓存"""
        self._sector_data_cache.clear()
        self._stock_data_cache.clear()
