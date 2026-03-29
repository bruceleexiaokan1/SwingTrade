"""共振回测器

基于板块共振的波段交易回测系统。

继承/组合 SwingBacktester，增加：
1. 板块数据加载
2. 共振过滤
3. 分级仓位管理
"""

import logging
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd

from .engine import SwingBacktester
from .resonance import ResonanceDetector
from .resonance_position import ResonancePositionManager
from .multi_cycle import MultiCycleResonance, MultiCycleLevel
from .models import BacktestResult
from ..data.indicators.resonance import ResonanceLevel, ResonanceResult, ResonanceDataError
from ..data.loader import StockDataLoader

logger = logging.getLogger(__name__)


class ResonanceBacktester:
    """
    共振回测器

    使用板块共振作为选股和仓位管理的核心逻辑：

    1. 加载板块配置和成分股
    2. 每日检测共振状态
    3. 根据共振等级分配仓位
    4. 执行波段交易

    使用方式：
        backtester = ResonanceBacktester(
            sector_config_path="config/sectors/sector_portfolio.json",
            initial_capital=1_000_000
        )

        result = backtester.run(
            sector_names=["半导体概念", "人工智能"],
            start_date="2025-01-01",
            end_date="2026-03-28"
        )
    """

    def __init__(
        self,
        sector_config_path: str = "config/sectors/sector_portfolio.json",
        initial_capital: float = 1_000_000.0,
        commission_rate: float = 0.0003,
        stamp_tax: float = 0.0001,
        max_positions: int = 5,
        trial_position_pct: float = 0.10,
        max_single_loss_pct: float = 0.02,
        atr_stop_multiplier: float = 2.0,
        strategy_params = None,
        stockdata_root: str = "/Users/bruce/workspace/trade/StockData"
    ):
        """
        初始化共振回测器

        Args:
            sector_config_path: 板块配置路径
            initial_capital: 初始资金
            commission_rate: 佣金率
            stamp_tax: 印花税率
            max_positions: 最大持仓数
            trial_position_pct: 试探仓比例
            max_single_loss_pct: 单笔最大亏损
            atr_stop_multiplier: ATR止损倍数
            strategy_params: 策略参数
            stockdata_root: StockData 根目录
        """
        self.sector_config_path = Path(sector_config_path)
        self.stockdata_root = stockdata_root
        self.initial_capital = initial_capital

        # 板块配置
        self.sector_config = None

        # 共振检测器
        self.resonance_detector = ResonanceDetector(
            sector_config_path=str(sector_config_path),
            stockdata_root=stockdata_root,
            params=strategy_params
        )

        # 多周期共振检测器
        self.multi_cycle_resonance = MultiCycleResonance(stockdata_root=stockdata_root)

        # 仓位管理器
        self.position_manager = ResonancePositionManager(
            base_position_value=initial_capital / max_positions,
            max_positions=max_positions,
            trial_upgrade_days=5,
            trial_upgrade_profit=0.05
        )

        # 基础回测器
        self.base_backtester = SwingBacktester(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            stamp_tax=stamp_tax,
            max_open_positions=max_positions,
            trial_position_pct=trial_position_pct,
            max_single_loss_pct=max_single_loss_pct,
            atr_stop_multiplier=atr_stop_multiplier,
            strategy_params=strategy_params
        )

        # 数据加载器
        self.stock_loader = StockDataLoader(stockdata_root)

        # 回测结果
        self.result: Optional[BacktestResult] = None

    def run(
        self,
        sector_names: List[str] = None,
        stock_codes: List[str] = None,
        start_date: str = "2025-01-01",
        end_date: str = "2026-03-28",
        min_resonance_level: ResonanceLevel = ResonanceLevel.C
    ) -> BacktestResult:
        """
        执行共振回测

        Args:
            sector_names: 要回测的板块名称列表，如 ["半导体概念", "人工智能"]
            stock_codes: 直接指定股票代码（覆盖 sector_names）
            start_date: 开始日期
            end_date: 结束日期
            min_resonance_level: 最低共振等级

        Returns:
            BacktestResult
        """
        # 加载板块配置
        self.sector_config = self.resonance_detector.get_sector_config()

        # 确定要回测的股票
        if stock_codes is None:
            stock_codes = []
            for sector in self.sector_config.get('sectors', []):
                if sector_names and sector['name'] not in sector_names:
                    continue
                for stock in sector.get('stocks', []):
                    if stock['code'] not in stock_codes:
                        stock_codes.append(stock['code'])

        logger.info(f"共振回测：{len(stock_codes)} 只股票，日期范围 {start_date} ~ {end_date}")

        # 预加载板块数据
        self._preload_sector_data(start_date, end_date, sector_names)

        # 预计算每日共振映射（关键修复：不是只在最后一天检查）
        resonance_map = self._compute_daily_resonance_map(
            stock_codes,
            start_date,
            end_date,
            min_resonance_level
        )

        # 统计有多少股票在任一天有共振
        stocks_with_resonance = set()
        for date_resonance in resonance_map.values():
            stocks_with_resonance.update(code for code, has_res in date_resonance.items() if has_res)

        logger.info(f"共振过滤后：{len(stocks_with_resonance)} 只股票在回测期内有过共振")

        if not stocks_with_resonance:
            logger.warning("没有符合共振条件的股票")
            return self._create_empty_result(start_date, end_date)

        # 设置共振映射到基础回测器（在调用run之前）
        self.base_backtester.resonance_map = resonance_map

        # 执行基础回测
        result = self.base_backtester.run(
            stock_codes=list(stocks_with_resonance),
            start_date=start_date,
            end_date=end_date,
            resonance_checker=None  # 使用预计算的 resonance_map，不需要 checker
        )

        self.result = result
        return result

    def _preload_sector_data(
        self,
        start_date: str,
        end_date: str,
        sector_names: List[str] = None
    ):
        """预加载板块数据"""
        if not self.sector_config:
            return

        sectors_to_preload = []
        for sector in self.sector_config.get('sectors', []):
            if sector_names and sector['name'] not in sector_names:
                continue
            sectors_to_preload.append(sector['name'])

        logger.info(f"预加载 {len(sectors_to_preload)} 个板块数据")

        for sector_name in sectors_to_preload:
            try:
                self.resonance_detector.get_sector_data(sector_name, start_date, end_date)
            except Exception as e:
                logger.warning(f"预加载板块数据失败 {sector_name}: {e}")

    def _compute_daily_resonance_map(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        min_level: ResonanceLevel
    ) -> Dict[str, Dict[str, bool]]:
        """
        预计算每日共振映射

        关键：不是只在最后一天检查，而是对回测期内的每一天检查共振

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            min_level: 最低共振等级

        Returns:
            {date: {code: has_resonance}}
        """
        # 获取所有交易日（使用 stock_codes 以确保覆盖）
        trading_dates = self._get_trading_dates(start_date, end_date, stock_codes)

        if not trading_dates:
            logger.warning("没有交易日")
            return {}

        # 预热：获取板块配置和映射
        sector_of_stock: Dict[str, str] = {}
        for code in stock_codes:
            sector = self._find_stock_sector(code)
            if sector:
                sector_of_stock[code] = sector

        # 初始化结果映射
        resonance_map: Dict[str, Dict[str, bool]] = {date: {} for date in trading_dates}

        # 进度持久化
        progress_file = self._get_progress_file(stock_codes, start_date, end_date)
        completed_dates = self._load_progress(progress_file, trading_dates)

        logger.info(f"预计算 {len(trading_dates)} 个交易日的共振状态...")
        if completed_dates:
            logger.info(f"  从进度文件恢复，已完成 {len(completed_dates)} 个日期")

        for i, date in enumerate(trading_dates):
            if date in completed_dates:
                continue  # 跳过已完成的日期

            if (i + 1) % 20 == 0:
                logger.info(f"  进度: {i+1}/{len(trading_dates)}")
                self._save_progress(progress_file, list(completed_dates))

            for code in stock_codes:
                sector_name = sector_of_stock.get(code)
                if sector_name is None:
                    resonance_map[date][code] = False
                    continue

                try:
                    result = self.resonance_detector.check_resonance(code, sector_name, date)
                    has_resonance = (
                        result.is_resonance and
                        result.resonance_level.value >= min_level.value  # noqa:_RESonanceLevel is int now
                    )
                    resonance_map[date][code] = has_resonance
                except ResonanceDataError as e:
                    logger.debug(f"共振数据获取失败 {code} on {date}: {e}")
                    resonance_map[date][code] = False  # 数据失败不算有共振
                except Exception as e:
                    logger.warning(f"共振检测异常 {code} on {date}: {e}")
                    resonance_map[date][code] = False

            completed_dates.add(date)

        # 最终清理进度文件
        self._save_progress(progress_file, list(completed_dates))
        if progress_file.exists():
            progress_file.unlink()

        logger.info(f"共振预计算完成")

        return resonance_map

    def _get_progress_file(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str
    ) -> Path:
        """获取进度文件路径"""
        # 用回测参数生成唯一键
        key = f"{','.join(sorted(stock_codes))}_{start_date}_{end_date}"
        hash_key = hashlib.md5(key.encode()).hexdigest()[:12]
        progress_dir = Path("/tmp/swingtrade_resonance_progress")
        progress_dir.mkdir(parents=True, exist_ok=True)
        return progress_dir / f"progress_{hash_key}.json"

    def _load_progress(self, progress_file: Path, trading_dates: List[str]) -> set:
        """从进度文件加载已完成的日期"""
        if not progress_file.exists():
            return set()
        try:
            with open(progress_file) as f:
                data = json.load(f)
            completed = set(data.get("completed_dates", []))
            # 只保留在当前交易日列表中的日期
            return completed & set(trading_dates)
        except (json.JSONDecodeError, IOError):
            return set()

    def _save_progress(self, progress_file: Path, completed_dates: List[str]):
        """保存进度到文件"""
        try:
            with open(progress_file, 'w') as f:
                json.dump({"completed_dates": completed_dates}, f)
        except IOError as e:
            logger.warning(f"进度保存失败: {e}")

    def _find_stock_sector(self, stock_code: str) -> Optional[str]:
        """找到股票所属板块"""
        if not self.sector_config:
            return None

        for sector in self.sector_config.get('sectors', []):
            for stock in sector.get('stocks', []):
                if stock['code'] == stock_code:
                    return sector['name']

        return None

    def _get_trading_dates(
        self,
        start_date: str,
        end_date: str,
        stock_codes: List[str] = None
    ) -> List[str]:
        """
        获取交易日列表

        Args:
            start_date: 开始日期
            end_date: 结束日期
            stock_codes: 如果提供，优先使用这些股票获取交易日
        """
        # 优先使用传入的股票列表
        codes_to_use = stock_codes if stock_codes else []

        # 如果没有传入或为空，回退到代表性指数股票
        if not codes_to_use:
            codes_to_use = ['000001', '600519']

        all_dates = set()
        for code in codes_to_use:
            try:
                df = self.stock_loader.load_daily(code, start_date, end_date)
                if not df.empty:
                    dates = df['date'].tolist()
                    all_dates.update(dates)
            except Exception as e:
                logger.warning(f"获取交易日数据失败 {code}: {e}")
                continue

        return sorted([d for d in all_dates if start_date <= d <= end_date])

    def _create_empty_result(self, start_date: str, end_date: str) -> BacktestResult:
        """创建空结果"""
        from .models import BacktestResult
        return BacktestResult(
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            total_trades=0,
            start_date=start_date,
            end_date=end_date
        )

    def get_resonance_report(
        self,
        stock_code: str,
        date: str = None
    ) -> Optional[ResonanceResult]:
        """
        获取个股共振报告

        Args:
            stock_code: 股票代码
            date: 日期（默认为最后一个交易日）

        Returns:
            ResonanceResult
        """
        sector_name = self._find_stock_sector(stock_code)
        if sector_name is None:
            return None

        if date is None:
            dates = self._get_trading_dates("2020-01-01", "2030-12-31")
            if not dates:
                return None
            date = dates[-1]

        return self.resonance_detector.check_resonance(stock_code, sector_name, date)

    def get_multi_cycle_resonance(
        self,
        stock_code: str,
        date: str = None
    ) -> Optional["MultiCycleResult"]:
        """
        获取个股多周期共振报告

        Args:
            stock_code: 股票代码
            date: 日期（默认为最后一个交易日）

        Returns:
            MultiCycleResult
        """
        if date is None:
            dates = self._get_trading_dates("2020-01-01", "2030-12-31")
            if not dates:
                return None
            date = dates[-1]

        return self.multi_cycle_resonance.check_resonance(stock_code, date)

    def get_combined_resonance_report(
        self,
        stock_code: str,
        date: str = None
    ) -> dict:
        """
        获取综合共振报告（板块共振 + 多周期共振）

        Args:
            stock_code: 股票代码
            date: 日期

        Returns:
            dict: 包含板块共振和多周期共振的结果
        """
        sector_name = self._find_stock_sector(stock_code)

        # 板块共振
        sector_resonance = None
        if sector_name:
            if date is None:
                dates = self._get_trading_dates("2020-01-01", "2030-12-31")
                date = dates[-1] if dates else None
            if date:
                try:
                    sector_resonance = self.resonance_detector.check_resonance(
                        stock_code, sector_name, date
                    )
                except ResonanceDataError:
                    pass

        # 多周期共振
        multi_cycle_result = self.multi_cycle_resonance.check_resonance(stock_code, date) if date else None

        return {
            "stock_code": stock_code,
            "date": date,
            "sector_name": sector_name,
            "sector_resonance": sector_resonance,
            "multi_cycle": multi_cycle_result
        }

