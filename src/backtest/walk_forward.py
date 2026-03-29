"""Walk-Forward 分析模块

Walk-Forward 分析用于验证策略参数的稳健性：
- 训练窗口（In-Sample）：用历史数据优化参数
- 测试窗口（Out-of-Sample）：验证优化后参数的效果
- 滚动窗口：每月滚动，持续验证

WFR (Walk-Forward Ratio) = Out-of-sample Sharpe / In-sample Sharpe
- WFR > 0.6: 参数稳健
- WFR > 0.8: 参数非常稳健
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """单个 Walk-Forward 窗口"""
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: Dict[str, Any]
    is_sharpe: float
    oos_sharpe: float
    wfr: float
    is_robust: bool
    oos_return: float
    oos_trades: int


@dataclass
class WalkForwardResult:
    """Walk-Forward 分析结果"""
    total_windows: int
    robust_windows: int
    avg_wfr: float
    avg_is_sharpe: float
    avg_oos_sharpe: float
    windows: List[WalkForwardWindow] = field(default_factory=list)

    @property
    def robustness_ratio(self) -> float:
        """稳健窗口比例"""
        return self.robust_windows / self.total_windows if self.total_windows > 0 else 0.0

    def summary(self) -> str:
        """生成摘要"""
        return f"""=== Walk-Forward Analysis ===
Windows: {self.total_windows} | Robust: {self.robust_windows} ({self.robustness_ratio:.1%})
Avg WFR: {self.avg_wfr:.3f} | Avg IS Sharpe: {self.avg_is_sharpe:.3f} | Avg OOS Sharpe: {self.avg_oos_sharpe:.3f}
"""


class WalkForwardAnalyzer:
    """
    Walk-Forward 参数稳健性分析

    策略：
    - 每半年 → 用过去3年数据优化 → 指导未来半年交易
    - 每月滚动一次
    - WFR = OOS Sharpe / IS Sharpe
    """

    def __init__(
        self,
        backtester: Any,
        train_window: int = 756,  # 3年 ~756交易日
        test_window: int = 126,   # 半年 ~126交易日
        step: int = 63,           # 每月滚动 ~63交易日
        robust_threshold: float = 0.6,
        very_robust_threshold: float = 0.8,
    ):
        """
        初始化 Walk-Forward 分析器

        Args:
            backtester: SwingBacktester 实例
            train_window: 训练窗口大小（交易日数），默认 756 (~3年)
            test_window: 测试窗口大小（交易日数），默认 126 (~6个月)
            step: 滚动步长（交易日数），默认 63 (~1个月)
            robust_threshold: 稳健阈值，默认 0.6
            very_robust_threshold: 非常稳健阈值，默认 0.8
        """
        self.backtester = backtester
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        self.robust_threshold = robust_threshold
        self.very_robust_threshold = very_robust_threshold

    def run_walk_forward(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        param_ranges: Dict[str, List[Any]],
        metric: str = "sharpe_ratio",
    ) -> WalkForwardResult:
        """
        执行 Walk-Forward 分析

        步骤:
        1. 用 train_window 数据优化参数
        2. 在 test_window 验证
        3. 滚动到下一个窗口

        Args:
            stock_codes: 股票代码列表
            start_date: 分析开始日期 (YYYY-MM-DD)
            end_date: 分析结束日期 (YYYY-MM-DD)
            param_ranges: 参数网格，如 {"ma_short": [10, 20], "ma_long": [30, 60]}
            metric: 优化目标指标

        Returns:
            WalkForwardResult: Walk-Forward 分析结果
        """
        from .optimizer import ParameterOptimizer

        windows: List[WalkForwardWindow] = []
        current_date = start_date

        # 计算训练和测试窗口的日期范围
        while True:
            # 计算训练期结束日期（训练窗口起点 + 训练窗口大小）
            train_start = self._offset_trading_days(stock_codes, current_date, 0)
            train_end = self._offset_trading_days(stock_codes, train_start, self.train_window - 1)

            # 测试期
            test_start = self._offset_trading_days(stock_codes, train_end, 1)
            test_end = self._offset_trading_days(stock_codes, test_start, self.test_window - 1)

            # 检查是否超出范围
            if test_end > end_date:
                # 最后一个月不足，跳过
                break

            logger.info(f"WF Window: Train {train_start}~{train_end}, Test {test_start}~{test_end}")

            try:
                # Step 1: 在训练窗口优化参数
                optimizer = ParameterOptimizer(
                    backtest_fn=self._create_backtest_fn(),
                    n_workers=4
                )

                opt_result = optimizer.grid_search(
                    param_grid=param_ranges,
                    stock_codes=stock_codes,
                    start_date=train_start,
                    end_date=train_end,
                    metric=metric,
                    minimize=False,
                )

                # Step 2: 用最优参数在测试窗口回测
                test_result = self.backtester.run(
                    stock_codes=stock_codes,
                    start_date=test_start,
                    end_date=test_end,
                    data_loader=None,
                )

                # Step 3: 计算各项指标
                is_sharpe = opt_result.best_metric if opt_result.best_metric else 0.0
                oos_sharpe = test_result.sharpe_ratio
                wfr = self.calculate_wfr(oos_sharpe, is_sharpe)
                is_robust = self.is_robust(wfr, self.robust_threshold)

                window = WalkForwardWindow(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    best_params=opt_result.best_params,
                    is_sharpe=is_sharpe,
                    oos_sharpe=oos_sharpe,
                    wfr=wfr,
                    is_robust=is_robust,
                    oos_return=test_result.total_return,
                    oos_trades=test_result.total_trades,
                )
                windows.append(window)

                logger.info(
                    f"  IS Sharpe: {is_sharpe:.3f}, OOS Sharpe: {oos_sharpe:.3f}, "
                    f"WFR: {wfr:.3f}, Robust: {is_robust}"
                )

            except Exception as e:
                logger.warning(f"Walk-Forward window failed: {e}")

            # 滚动到下一个窗口
            current_date = self._offset_trading_days(stock_codes, current_date, self.step)

            # 防止无限循环
            if self._date_to_int(current_date) >= self._date_to_int(end_date):
                break

        # 汇总结果
        return self._aggregate_results(windows)

    def calculate_wfr(self, oos_sharpe: float, is_sharpe: float) -> float:
        """
        计算 Walk-Forward Ratio

        WFR = Out-of-sample Sharpe / In-sample Sharpe

        Args:
            oos_sharpe: 样本外夏普比率
            is_sharpe: 样本内夏普比率

        Returns:
            WFR 值
        """
        if is_sharpe == 0:
            return 0.0
        return oos_sharpe / is_sharpe

    def is_robust(self, wfr: float, threshold: float = 0.6) -> bool:
        """
        判断参数是否稳健

        Args:
            wfr: Walk-Forward Ratio
            threshold: 稳健阈值

        Returns:
            是否稳健
        """
        return wfr >= threshold

    def get_very_robust(self, wfr: float) -> bool:
        """判断是否非常稳健 (WFR > 0.8)"""
        return wfr > self.very_robust_threshold

    def _create_backtest_fn(self) -> Callable:
        """创建回测函数"""
        def backtest_fn(
            stock_codes: List[str],
            start_date: str,
            end_date: str,
            params: Any
        ):
            # 使用深拷贝避免修改原始参数
            import copy
            params_copy = copy.deepcopy(params)

            # 创建回测器实例
            backtester = copy.deepcopy(self.backtester)
            backtester.strategy_params = params_copy

            return backtester.run(
                stock_codes=stock_codes,
                start_date=start_date,
                end_date=end_date,
            )

        return backtest_fn

    def _offset_trading_days(
        self,
        stock_codes: List[str],
        start_date: str,
        offset: int
    ) -> str:
        """
        计算偏移指定交易日数后的日期

        Args:
            stock_codes: 股票代码列表（用于获取交易日）
            start_date: 起始日期
            offset: 偏移量（可为负数）

        Returns:
            偏移后的日期 (YYYY-MM-DD)
        """
        from ..data.loader import StockDataLoader

        if offset == 0:
            return start_date

        # 获取所有交易日
        loader = StockDataLoader()
        all_dates: List[str] = []

        for code in stock_codes[:5]:  # 最多取5只股票
            try:
                df = loader.load_daily(code, "2015-01-01", "2030-12-31")
                all_dates.extend(df["date"].tolist())
            except Exception:
                continue

        if not all_dates:
            # 如果无法获取数据，使用简单的时间偏移
            dt = datetime.strptime(start_date, "%Y-%m-%d")
            dt = dt + timedelta(days=offset * 2)  # 粗略估计
            return dt.strftime("%Y-%m-%d")

        all_dates = sorted(set(all_dates))

        if start_date not in all_dates:
            # 找到最接近的日期
            start_idx = 0
            for i, d in enumerate(all_dates):
                if d >= start_date:
                    start_idx = i
                    break
        else:
            start_idx = all_dates.index(start_date)

        new_idx = start_idx + offset
        if new_idx < 0:
            new_idx = 0
        elif new_idx >= len(all_dates):
            new_idx = len(all_dates) - 1

        return all_dates[new_idx]

    def _date_to_int(self, date_str: str) -> int:
        """将日期转换为整数用于比较"""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.year * 10000 + dt.month * 100 + dt.day

    def _aggregate_results(self, windows: List[WalkForwardWindow]) -> WalkForwardResult:
        """汇总所有窗口的结果"""
        if not windows:
            return WalkForwardResult(
                total_windows=0,
                robust_windows=0,
                avg_wfr=0.0,
                avg_is_sharpe=0.0,
                avg_oos_sharpe=0.0,
                windows=[],
            )

        total = len(windows)
        robust = sum(1 for w in windows if w.is_robust)

        return WalkForwardResult(
            total_windows=total,
            robust_windows=robust,
            avg_wfr=sum(w.wfr for w in windows) / total,
            avg_is_sharpe=sum(w.is_sharpe for w in windows) / total,
            avg_oos_sharpe=sum(w.oos_sharpe for w in windows) / total,
            windows=windows,
        )
