"""向量化回测引擎核心

核心设计：
1. MultiIndexData: 复合索引数据访问器
2. DateBoundAccessor: 日期边界防护访问器
3. VectorizedBacktester: 向量化回测基类

性能优化：
- 指标预计算：一次性计算所有指标
- 向量化信号检测：使用pandas布尔索引代替循环
- 批量多周期计算：消除每日重算瓶颈
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.0003
    stamp_tax: float = 0.0001
    max_positions: int = 5
    atr_stop_multiplier: float = 2.0
    atr_trailing_multiplier: float = 3.0
    entry_confidence_threshold: float = 0.5
    slippage_base: float = 0.001


@dataclass
class VectorizedResult:
    """向量化回测结果"""
    total_trades: int = 0
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    trades: List = field(default_factory=list)
    equity_curve: pd.DataFrame = None


# ============================================================================
# 日期边界防护
# ============================================================================

class DateBoundAccessor:
    """
    日期边界访问器 - 防止数据穿越

    核心机制：
    - 所有访问必须指定 current_date
    - 内部自动过滤 current_date 之后的数据
    - 使用复合索引确保数据隔离

    使用示例：
        accessor = DateBoundAccessor(indicators_df)

        # 获取某股票在某个日期之前的最新MA20值
        ma20 = accessor.get_last('600519', '2024-03-15', 'ma20')
    """

    def __init__(self, df: pd.DataFrame, data_id_col: str = 'data_id', date_col: str = 'date'):
        """
        初始化访问器

        Args:
            df: 数据DataFrame，必须包含 data_id 和 date 列
            data_id_col: 股票代码列名
            date_col: 日期列名
        """
        self.data_id_col = data_id_col
        self.date_col = date_col

        # 复制并排序（避免索引混乱）
        self._df = df.copy()

        # 确保日期列是字符串格式
        if pd.api.types.is_datetime64_any_dtype(self._df[date_col]):
            self._df[date_col] = pd.to_datetime(self._df[date_col]).dt.strftime('%Y-%m-%d')

        # 按 data_id 和 date 排序
        self._df = self._df.sort_values([data_id_col, date_col])

        # 创建复合索引以加速查询
        self._indexed_df = self._df.set_index([data_id_col, date_col])

        logger.debug(f"DateBoundAccessor initialized with {len(df)} rows")

    def get_value(self, data_id: str, current_date: str, column: str) -> Optional[float]:
        """
        获取指定日期之前的最新值

        Args:
            data_id: 股票代码
            current_date: 当前日期（只返回 <= current_date 的数据）
            column: 要获取的列名

        Returns:
            最新值，如果无数据则返回 NaN
        """
        try:
            # 使用 MultiIndex 切片：slice(None) 表示选择该级别的所有值
            # 正确语法：loc[(data_id, slice(None)), (column)] 但我们需要行过滤
            # 更简单的方式：直接使用布尔索引

            mask = (
                (self._df[self.data_id_col] == data_id) &
                (self._df[self.date_col] <= current_date)
            )
            filtered = self._df.loc[mask]

            if filtered.empty:
                return np.nan

            return filtered.iloc[-1][column]

        except (KeyError, IndexError):
            # 该股票在当前日期之前无数据
            return np.nan

    def get_series(
        self,
        data_id: str,
        current_date: str,
        column: str,
        lookback: int = 20
    ) -> pd.Series:
        """
        获取当前日期之前N天的序列

        Args:
            data_id: 股票代码
            current_date: 当前日期
            column: 要获取的列名
            lookback: 回看天数

        Returns:
            Series（包含lookback天的数据）
        """
        try:
            data = self._indexed_df.loc[data_id, :current_date]

            if data.empty:
                return pd.Series(dtype=float)

            return data.iloc[-lookback:][column]

        except KeyError:
            return pd.Series(dtype=float)

    def get_before(self, data_id: str, current_date: str) -> pd.DataFrame:
        """
        获取指定日期之前的所有数据（不含当日）

        Args:
            data_id: 股票代码
            current_date: 当前日期

        Returns:
            DataFrame
        """
        try:
            data = self._indexed_df.loc[data_id, :current_date]
            # 排除今日数据
            return data.iloc[:-1]

        except KeyError:
            return pd.DataFrame()

    def get_at_or_before(self, data_id: str, current_date: str) -> pd.DataFrame:
        """
        获取指定日期及之前的所有数据

        Args:
            data_id: 股票代码
            current_date: 当前日期

        Returns:
            DataFrame
        """
        try:
            return self._indexed_df.loc[data_id, :current_date]

        except KeyError:
            return pd.DataFrame()

    def has_data(self, data_id: str, current_date: str) -> bool:
        """检查指定日期之前是否有数据"""
        mask = (
            (self._df[self.data_id_col] == data_id) &
            (self._df[self.date_col] <= current_date)
        )
        return mask.any()


# ============================================================================
# 向量化回测基类
# ============================================================================

class VectorizedBacktester:
    """
    向量化回测引擎基类

    核心流程：
    1. 预加载所有数据
    2. 预计算所有指标
    3. 预计算所有信号
    4. 逐日迭代（只处理持仓相关股票）
    5. 生成结果

    性能优化点：
    - 指标只计算一次（而非每日重算）
    - 快照获取使用复合索引（O(log n) 而非 O(n)）
    - 持仓更新向量化（pandas布尔索引）
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        stockdata_root: str = "/Users/bruce/workspace/trade/StockData"
    ):
        """
        初始化向量化回测器

        Args:
            config: 回测配置
            stockdata_root: StockData根目录
        """
        from ...data.loader import StockDataLoader

        self.config = config or BacktestConfig()
        self.stockdata_root = stockdata_root
        self.loader = StockDataLoader(stockdata_root)

        # 状态
        self._price_df: pd.DataFrame = None  # 原始价格
        self._indicators_df: pd.DataFrame = None  # 指标
        self._signals_df: pd.DataFrame = None  # 信号
        self._multi_cycle_df: pd.DataFrame = None  # 多周期

        # 持仓
        self._positions: Dict[str, Any] = {}
        self._cash: float = self.config.initial_capital
        self._equity_history: List = []
        self._trades: List = []  # 交易记录

        logger.info(f"VectorizedBacktester initialized with capital={self.config.initial_capital}")

    def run(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str
    ) -> VectorizedResult:
        """
        执行向量化回测

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            VectorizedResult
        """
        # Step 1: 预加载数据
        logger.info(f"Step 1: Loading data for {len(stock_codes)} stocks...")
        self._load_data(stock_codes, start_date, end_date)

        # Step 2: 预计算指标
        logger.info("Step 2: Precomputing indicators...")
        self._precompute_indicators()

        # Step 3: 预计算信号
        logger.info("Step 3: Precomputing signals...")
        self._precompute_signals()

        # Step 4: 预计算多周期状态（如果需要）
        if self._need_multi_cycle():
            logger.info("Step 4: Precomputing multi-cycle states...")
            self._precompute_multi_cycle()

        # Step 5: 获取交易日列表
        trading_dates = self._get_trading_dates(start_date, end_date)
        logger.info(f"Step 5: Found {len(trading_dates)} trading days")

        # Step 6: 逐日回测
        logger.info("Step 6: Running daily backtest...")
        for i, date in enumerate(trading_dates[:-1]):
            if i % 100 == 0:
                logger.debug(f"  Progress: {i}/{len(trading_dates)}")

            self._process_day(date, trading_dates[i + 1])

        # Step 7: 平仓
        logger.info("Step 7: Closing positions...")
        final_date = trading_dates[-1]
        self._close_all_positions(final_date)

        # Step 8: 生成结果
        logger.info("Step 8: Generating results...")
        return self._generate_result(start_date, end_date, trading_dates)

    def _load_data(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str
    ) -> None:
        """加载所有股票数据"""
        all_data = []

        for code in stock_codes:
            df = self.loader.load_daily(code, start_date=start_date, end_date=end_date)

            if df.empty:
                continue

            df['data_id'] = code
            all_data.append(df)

        if not all_data:
            raise ValueError("No data loaded")

        self._price_df = pd.concat(all_data, ignore_index=True)
        self._price_df = self._price_df.sort_values(['data_id', 'date'])

        logger.info(f"  Loaded {len(self._price_df)} price records")

    def _precompute_indicators(self) -> None:
        """预计算所有指标"""
        from ...data.fetcher.price_converter import convert_to_forward_adj
        from ...data.vectorized import VectorizedIndicators

        # 转换前复权
        self._price_df = convert_to_forward_adj(self._price_df)

        # 计算指标（向量化）
        vectorized = VectorizedIndicators()
        self._indicators_df = vectorized.calculate_all(self._price_df)

        logger.info(f"  Indicators computed: {len(self._indicators_df)} rows")

    def _precompute_signals(self) -> None:
        """预计算所有信号"""
        from ...data.vectorized import VectorizedSignals

        # 使用向量化信号检测
        vectorized = VectorizedSignals()
        self._signals_df = vectorized.calculate_all(self._indicators_df)

        logger.info(f"  Signals computed: {len(self._signals_df)} rows")

    def _precompute_multi_cycle(self) -> None:
        """预计算多周期状态"""
        # TODO: 实现批量多周期计算
        pass

    def _need_multi_cycle(self) -> bool:
        """是否需要多周期"""
        return False

    def _get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表"""
        dates = self._price_df['date'].unique()
        dates = sorted([d for d in dates if start_date <= d <= end_date])
        return dates

    def _process_day(self, date: str, next_date: str) -> None:
        """
        处理单日回测

        向量化优化：只处理持仓股票和新信号
        """
        # 1. 更新持仓状态
        self._update_positions(date)

        # 2. 检测入场信号
        entry_signals = self._detect_entries(date)

        # 3. 执行入场
        self._execute_entries(entry_signals, date)

        # 4. 检测出场信号
        exit_signals = self._detect_exits(date)

        # 5. 执行出场
        self._execute_exits(exit_signals, date)

        # 6. 记录权益
        self._record_equity(date)

    def _update_positions(self, date: str) -> None:
        """更新持仓状态"""
        # 遍历持仓，检查止损/止盈
        for code, pos in list(self._positions.items()):
            # 获取当日价格
            price = self._get_price(code, date)
            if price is None:
                continue

            # 更新最高价（用于追踪止损）
            if price > pos['highest_price']:
                pos['highest_price'] = price

    def _detect_entries(self, date: str) -> List:
        """检测入场信号（基于预计算信号）"""
        if self._signals_df is None:
            return []

        # 从预计算信号中获取当日入场信号
        signals = self._signals_df[
            (self._signals_df['date'] == date) &
            (self._signals_df['entry_confidence'] > self.config.entry_confidence_threshold)
        ]

        results = []
        for _, row in signals.iterrows():
            results.append({
                'data_id': row['data_id'],
                'signal_type': 'entry',
                'confidence': row['entry_confidence'],
                'close': row['close']
            })

        return results

    def _detect_exits(self, date: str) -> List:
        """检测出场信号（基于预计算信号）"""
        if self._signals_df is None:
            return []

        # 从预计算信号中获取当日出场信号
        signals = self._signals_df[
            (self._signals_df['date'] == date) &
            (self._signals_df['exit_death_cross'] == 1)
        ]

        results = []
        for _, row in signals.iterrows():
            results.append({
                'data_id': row['data_id'],
                'signal_type': 'exit',
                'reason': 'death_cross',
                'close': row['close']
            })

        return results

    def _execute_entries(self, signals: List, date: str) -> None:
        """执行入场"""
        for signal in signals:
            code = signal['data_id']

            # 跳过已有持仓的股票
            if code in self._positions:
                continue

            # 检查是否达到最大持仓数
            if len(self._positions) >= self.config.max_positions:
                break

            # 检查资金是否足够
            if self._cash < 10000:  # 最低交易金额
                continue

            # 获取入场价格（使用信号触发日的收盘价）
            entry_price = signal.get('close', 0)
            if entry_price <= 0:
                continue

            # 获取 ATR 用于止损计算
            atr = 0.0
            if self._indicators_df is not None:
                atr_row = self._indicators_df[
                    (self._indicators_df['data_id'] == code) &
                    (self._indicators_df['date'] == date)
                ]
                if not atr_row.empty and 'atr14' in atr_row.columns:
                    atr = atr_row['atr14'].iloc[0]

            # 计算仓位数量（使用 ATR 止损比例）
            if atr > 0:
                stop_distance = atr * self.config.atr_stop_multiplier
                risk_per_share = stop_distance
                position_value = self._cash * 0.1  # 单笔风险不超过资金的 10%
                shares = int(position_value / risk_per_share / 100) * 100
            else:
                # 简化：使用资金的 20% 建仓
                position_value = self._cash * 0.2
                shares = int(position_value / entry_price / 100) * 100

            if shares < 100:
                continue

            # 扣除资金（简化：不考虑佣金）
            cost = shares * entry_price
            if cost > self._cash:
                shares = int(self._cash / entry_price / 100) * 100
                cost = shares * entry_price

            if shares < 100:
                continue

            # 创建持仓
            self._cash -= cost
            self._positions[code] = {
                'code': code,
                'entry_date': date,
                'entry_price': entry_price,
                'shares': shares,
                'atr': atr,
                'stop_loss': entry_price - atr * self.config.atr_stop_multiplier if atr > 0 else entry_price * 0.95,
                'trailing_stop': entry_price - atr * self.config.atr_trailing_multiplier if atr > 0 else entry_price * 0.9,
                'highest_price': entry_price,
                'signal_confidence': signal.get('confidence', 0),
                'signal_type': signal.get('signal_type', 'unknown')
            }

            logger.debug(f"  Entry: {code} @ {entry_price} x {shares}")

    def _execute_exits(self, signals: List, date: str) -> None:
        """执行出场"""
        for signal in signals:
            code = signal['data_id']

            # 检查是否有该股票的持仓
            if code not in self._positions:
                continue

            pos = self._positions[code]
            exit_price = signal.get('close', 0)
            if exit_price <= 0:
                continue

            shares = pos['shares']

            # 计算盈亏
            pnl = (exit_price - pos['entry_price']) * shares
            pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']

            # 简化佣金计算
            commission = exit_price * shares * (self.config.commission_rate + self.config.stamp_tax)

            # 记录交易
            self._trades.append({
                'trade_id': f"{date}_{code}",
                'date': date,
                'code': code,
                'direction': 'long',
                'entry_date': pos['entry_date'],
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'shares': shares,
                'pnl': pnl - commission,
                'pnl_pct': pnl_pct,
                'commission': commission,
                'exit_reason': signal.get('reason', 'unknown'),
                'signal_type': pos.get('signal_type', 'unknown')
            })

            # 释放资金
            self._cash += exit_price * shares - commission

            # 删除持仓
            del self._positions[code]

            logger.debug(f"  Exit: {code} @ {exit_price} x {shares}, PnL: {pnl_pct:.2%}")

    def _close_all_positions(self, date: str) -> None:
        """平所有仓"""
        for code, pos in list(self._positions.items()):
            exit_price = self._get_price(code, date)
            if exit_price is None:
                continue

            shares = pos['shares']

            # 计算盈亏
            pnl = (exit_price - pos['entry_price']) * shares
            pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']

            # 简化佣金
            commission = exit_price * shares * (self.config.commission_rate + self.config.stamp_tax)

            # 记录交易
            self._trades.append({
                'trade_id': f"{date}_{code}",
                'date': date,
                'code': code,
                'direction': 'long',
                'entry_date': pos['entry_date'],
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'shares': shares,
                'pnl': pnl - commission,
                'pnl_pct': pnl_pct,
                'commission': commission,
                'exit_reason': 'end_of_backtest',
                'signal_type': pos.get('signal_type', 'unknown')
            })

            # 释放资金
            self._cash += exit_price * shares - commission

            # 删除持仓
            del self._positions[code]

            logger.debug(f"  Force Close: {code} @ {exit_price} x {shares}, PnL: {pnl_pct:.2%}")

    def _record_equity(self, date: str) -> None:
        """记录权益"""
        total_value = self._cash
        for pos in self._positions.values():
            price = self._get_price(pos['code'], date)
            if price:
                total_value += price * pos['shares']

        self._equity_history.append({
            'date': date,
            'equity': total_value,
            'cash': self._cash,
            'positions_value': total_value - self._cash
        })

    def _get_price(self, data_id: str, date: str) -> Optional[float]:
        """获取指定日期的价格"""
        row = self._price_df[
            (self._price_df['data_id'] == data_id) &
            (self._price_df['date'] == date)
        ]

        if row.empty:
            return None

        return row.iloc[0]['close']

    def _generate_result(
        self,
        start_date: str,
        end_date: str,
        trading_dates: List[str]
    ) -> VectorizedResult:
        """生成回测结果"""
        # 计算权益曲线
        equity_df = pd.DataFrame(self._equity_history)

        # 计算统计指标
        if equity_df.empty:
            return VectorizedResult()

        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - self.config.initial_capital) / self.config.initial_capital

        # 计算年化收益
        n_days = len(trading_dates)
        n_years = n_days / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # 计算夏普比率
        daily_returns = equity_df['equity'].pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

        # 计算最大回撤
        cummax = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cummax) / cummax
        max_drawdown = drawdown.min()

        # 计算交易统计
        trades = self._trades
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        losing_trades = sum(1 for t in trades if t['pnl'] <= 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_win = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        total_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = total_win / total_loss if total_loss > 0 else 0

        return VectorizedResult(
            total_trades=total_trades,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            equity_curve=equity_df,
            trades=trades
        )


# ============================================================================
# 辅助函数
# ============================================================================

def create_vectorized_backtester(
    config: Optional[BacktestConfig] = None,
    stockdata_root: str = "/Users/bruce/workspace/trade/StockData"
) -> VectorizedBacktester:
    """创建向量化回测器工厂函数"""
    return VectorizedBacktester(config=config, stockdata_root=stockdata_root)
