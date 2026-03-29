"""回测引擎

波段交易回测核心逻辑：
1. 加载数据 + 计算指标
2. 逐日迭代：信号检测 → 订单撮合 → 持仓更新 → 权益记录
3. 生成 BacktestResult
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..data.loader import StockDataLoader
from ..data.fetcher.price_converter import convert_to_forward_adj
from ..data.indicators import SwingSignals

from .models import (
    Trade, Position, BacktestResult, EquityRecord,
    EntrySignal, ExitSignal, generate_id
)
from .matching import OrderMatcher
from .performance import PerformanceAnalyzer
from .strategy_params import StrategyParams
from .position_sizer import KellyPositionSizer
from .market_state import MarketState, detect_market_state


class SwingBacktester:
    """
    波段交易回测引擎

    使用三屏系统信号：
    1. 方向（趋势）：MA20/MA60, MACD 零轴
    2. 时机（信号）：RSI, 布林带
    3. 确认（量价）：成交量

    止损策略：
    - ATR止损：入场价 - 2~3×ATR
    - 追踪止损：持仓最高价 - 3×ATR
    - RSI超买出场
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission_rate: float = 0.0003,
        stamp_tax: float = 0.0001,
        position_size_type: str = "fixed",
        fixed_position_value: float = 100_000.0,
        max_position_pct: float = 0.2,
        atr_stop_multiplier: float = 2.0,
        atr_trailing_multiplier: float = 3.0,
        min_trade_amount: float = 1000.0,
        entry_confidence_threshold: float = 0.5,
        slippage_base: float = 0.001,
        # === 新增：风险/仓位参数 ===
        trial_position_pct: float = 0.10,       # 试探仓位比例（10%）
        max_single_loss_pct: float = 0.02,      # 单笔最大亏损限制（2%）
        min_profit_loss_ratio: float = 3.0,    # 最小盈亏比要求（中长线 >= 3:1）
        max_open_positions: int = 5,           # 最大同时持仓数
        atr_circuit_breaker: float = 3.0,      # ATR熔断倍数
        # === 新增：策略参数 ===
        strategy_params: Optional[StrategyParams] = None,
        # === 共振检查器（可选）===
        resonance_checker=None,
    ):
        """
        初始化回测引擎

        Args:
            initial_capital: 初始资金，默认 100万
            commission_rate: 佣金率，默认 0.03%
            stamp_tax: 印花税率，默认 0.01%（卖出时）
            position_size_type: 仓位管理类型，"fixed" / "percent"
            fixed_position_value: 固定仓位金额，默认 10万
            max_position_pct: 最大仓位占比，默认 20%
            atr_stop_multiplier: ATR止损倍数，默认 2.0
            atr_trailing_multiplier: ATR追踪止损倍数，默认 3.0
            min_trade_amount: 最小交易金额，默认 1000
            entry_confidence_threshold: 入场置信度阈值，默认 0.5
            slippage_base: 基准滑点，默认 0.1%
            trial_position_pct: 试探仓位比例，默认 10%（首笔建仓使用较小仓位）
            max_single_loss_pct: 单笔最大亏损限制，默认 2%（单笔亏损不超过总资金的2%）
            min_profit_loss_ratio: 最小盈亏比，默认 3.0（中长线 >= 3:1）
            max_open_positions: 最大同时持仓数，默认 5
            atr_circuit_breaker: ATR熔断倍数，默认 3.0（当前ATR超过入场时ATR的3倍时禁止开仓）
            strategy_params: 策略参数对象（推荐方式，可一次性配置所有参数）
        """
        # 保存策略参数引用
        self.strategy_params = strategy_params
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_tax = stamp_tax
        self.position_size_type = position_size_type
        self.fixed_position_value = fixed_position_value
        self.max_position_pct = max_position_pct
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_trailing_multiplier = atr_trailing_multiplier
        self.min_trade_amount = min_trade_amount
        self.entry_confidence_threshold = entry_confidence_threshold

        # === 新增：风险/仓位参数 ===
        self.trial_position_pct = trial_position_pct
        self.max_single_loss_pct = max_single_loss_pct
        self.min_profit_loss_ratio = min_profit_loss_ratio
        self.max_open_positions = max_open_positions
        self.atr_circuit_breaker = atr_circuit_breaker

        # 内部组件
        if strategy_params is not None:
            self.signals = SwingSignals(params=strategy_params)
        else:
            self.signals = SwingSignals()
        self.matcher = OrderMatcher(
            slippage_base=slippage_base,
            commission_rate=commission_rate,
            stamp_tax=stamp_tax,
        )
        self.analyzer = PerformanceAnalyzer()

        # 回测状态
        self.cash: float = initial_capital
        self.positions: Dict[str, Position] = {}  # code -> Position
        self.equity_history: List[EquityRecord] = []
        self.trades: List[Trade] = []
        self.stock_data: Dict[str, pd.DataFrame] = {}

        # 共振检查器（可选）
        self.resonance_checker = resonance_checker
        self.resonance_map: Dict[str, Dict[str, bool]] = {}  # date -> {code -> has_resonance}

        # 仓位管理器（凯利公式 + 波动率调整）
        self.position_sizer = KellyPositionSizer(
            max_risk_pct=max_single_loss_pct,  # 单笔最大风险 2%
            max_position_pct=max_position_pct   # 最大持仓 20%（外部会再限制）
        )

    def run(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        data_loader: Optional[StockDataLoader] = None,
        stockdata_root: str = "/Users/bruce/workspace/trade/StockData"
    ) -> BacktestResult:
        """
        执行回测

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            data_loader: 数据加载器（可选）
            stockdata_root: StockData 根目录

        Returns:
            BacktestResult: 回测结果
        """
        # 重置状态
        self.cash = self.initial_capital
        self.positions = {}
        self.equity_history = []
        self.trades = []
        self.stock_data = {}

        # 重置共振映射（避免上次运行的残留数据）
        self.resonance_map = {}

        # 获取数据加载器
        if data_loader is None:
            data_loader = StockDataLoader(stockdata_root)

        # Step 1: 加载并计算所有数据
        for code in stock_codes:
            df = data_loader.load_daily(code, start_date, end_date)
            if len(df) < 60:  # 需要足够的数据计算指标
                continue
            df = convert_to_forward_adj(df)
            df = self.signals.calculate_all(df)
            self.stock_data[code] = df

        if not self.stock_data:
            return self._generate_empty_result(start_date, end_date)

        # Step 2: 获取合并交易日期
        trading_dates = self._get_trading_dates(start_date, end_date)

        # Step 3: 逐日迭代
        for i, date in enumerate(trading_dates[:-1]):  # 最后一天不出信号
            # 获取当日全市场快照
            snapshots = self._get_snapshots(date)

            # 信号检测
            entry_signals = self._detect_entries(snapshots, date)
            exit_signals = self._detect_exits(snapshots, date)

            # 订单执行
            self._execute_entry_orders(entry_signals, date)
            self._execute_exit_orders(exit_signals, date)

            # 更新持仓 & 权益记录
            self._update_and_record(date)

        # 最终平仓（回测结束时）
        self._close_all_positions(trading_dates[-1])

        # Step 4: 生成结果
        return self._generate_result(start_date, end_date, trading_dates)

    def _get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取所有交易日"""
        all_dates = set()
        for df in self.stock_data.values():
            dates = df["date"].tolist()
            all_dates.update(dates)

        dates = sorted([d for d in all_dates if start_date <= d <= end_date])
        return dates

    def _get_snapshots(self, date: str) -> Dict[str, pd.DataFrame]:
        """获取指定日期的全市场快照"""
        snapshots = {}
        for code, df in self.stock_data.items():
            # 使用向量化过滤替代 O(n) 的 `in` 检查
            mask = df["date"] == date
            if mask.any():
                idx = mask.idxmax()
                # 取从开始到当前的所有历史数据（用于计算指标）
                snapshots[code] = df.loc[:idx].copy()
        return snapshots

    def _get_next_day_data(self, code: str, date: str) -> pd.DataFrame:
        """获取 T+1 日数据"""
        df = self.stock_data.get(code)
        if df is None:
            return pd.DataFrame()

        dates = df["date"].tolist()
        if date not in dates:
            return pd.DataFrame()

        idx = dates.index(date)
        if idx + 1 >= len(dates):
            return pd.DataFrame()

        next_date = dates[idx + 1]
        return df[df["date"] == next_date].copy()

    def _detect_entries(
        self,
        snapshots: Dict[str, pd.DataFrame],
        date: str
    ) -> List[EntrySignal]:
        """检测入场信号"""
        signals = []

        for code, df in snapshots.items():
            if code in self.positions:
                continue  # 跳过已持仓的股票

            if len(df) < 20:
                continue

            # 共振检查：只有当天有共振才允许入场
            if self.resonance_checker is not None:
                has_resonance = self._check_resonance_on_date(code, date)
                if not has_resonance:
                    continue

            # === 市场状态检测 ===
            market_state_result = detect_market_state(df)
            market_state = market_state_result.state

            # 分析信号
            result = self.signals.analyze(df)

            # 趋势过滤：仅在上涨趋势入场
            if result.trend != "uptrend":
                continue

            # 入场信号检测
            if result.entry_signal in ("golden", "breakout") and result.entry_confidence >= self.entry_confidence_threshold:
                atr = result.atr if result.atr else df["atr"].iloc[-1]
                if pd.isna(atr) or atr <= 0:
                    continue

                entry_price = df["close"].iloc[-1]
                stop_loss = entry_price - (self.atr_stop_multiplier * atr)

                # === 新增：ATR熔断检查 ===
                # 当前ATR超过入场时ATR的 atr_circuit_breaker 倍时禁止开仓
                current_atr = df["atr"].iloc[-1]
                if not pd.isna(current_atr) and current_atr > atr * self.atr_circuit_breaker:
                    continue

                # === 新增：最小盈亏比检查 ===
                # 预期涨幅 = 5%（波段交易典型目标），止损距离 = atr_stop_multiplier * atr
                expected_profit_pct = 0.05  # 5% 目标涨幅
                expected_profit = entry_price * expected_profit_pct
                stop_distance = self.atr_stop_multiplier * atr
                profit_loss_ratio = expected_profit / stop_distance if stop_distance > 0 else 0
                if profit_loss_ratio < self.min_profit_loss_ratio:
                    continue

                signals.append(EntrySignal(
                    code=code,
                    signal_type=result.entry_signal,
                    confidence=result.entry_confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    atr=atr,
                    reason=result.entry_reason,
                    market_state=market_state.value
                ))

        # 按置信度排序
        signals.sort(key=lambda x: x.confidence, reverse=True)
        return signals

    def _check_resonance_on_date(self, code: str, date: str) -> bool:
        """
        检查个股在指定日期是否有共振

        Args:
            code: 股票代码
            date: 日期

        Returns:
            是否有共振（无 resonance_map 时默认允许）
        """
        if not self.resonance_map:
            return True  # 没有共振映射时默认允许入场
        date_resonance = self.resonance_map.get(date, {})
        return date_resonance.get(code, False)

    def _detect_exits(
        self,
        snapshots: Dict[str, pd.DataFrame],
        date: str
    ) -> List[ExitSignal]:
        """检测出场信号（增强版含分批止盈）"""
        signals = []

        for code, position in list(self.positions.items()):
            df = snapshots.get(code)
            if df is None or len(df) < 2:
                continue

            current_price = df["close"].iloc[-1]

            # 更新持仓最高价
            if current_price > position.highest_price:
                position.highest_price = current_price

            # 追踪止损更新
            trailing_stop = position.highest_price - (self.atr_trailing_multiplier * position.atr)
            position.trailing_stop = trailing_stop

            # 计算浮动盈亏
            unrealized_pnl_pct = (current_price - position.entry_price) / position.entry_price if position.entry_price > 0 else 0

            # 综合分析
            result = self.signals.analyze(
                df,
                entry_price=position.entry_price,
                highest_price=position.highest_price
            )

            exit_signal = None
            exit_reason = None
            reduce_only = False
            reduce_ratio = 1.0

            # === 分批止盈检测（优先级介于追踪止损和MA死叉之间）===

            # T1: 触及 20 日前高阻力 且 有足够浮动盈利（5%以上）
            if not position.t1_triggered and unrealized_pnl_pct > 0.05:
                # 计算 20 日前高
                recent_high = df["high"].rolling(20).max().iloc[-1]
                # 价格接近前高 2% 以内
                if current_price >= recent_high * 0.98:
                    exit_signal = "take_profit_1"
                    exit_reason = f"T1止盈@{recent_high:.2f}(20日高点)"
                    reduce_only = True
                    reduce_ratio = 0.5  # 减仓50%

            # T2: 跌破 10 日均线（知识库：独立于 T1 触发）
            # 注意：知识库定义 T2 是"跌破 10 日均线减仓 50%"，不依赖 T1
            if not position.t2_triggered:
                ma10 = df["ma10"].iloc[-1] if "ma10" in df.columns else None
                if ma10 is not None and current_price < ma10:
                    exit_signal = "take_profit_2"
                    exit_reason = f"T2止盈(MA10@{ma10:.2f})"
                    reduce_only = True
                    reduce_ratio = 0.5  # 减仓50%

            # 优先级判断
            # 知识库止损优先级（满足任一即触发）：
            # 1. 跌破入场后前一根K线最低点（结构止损1）
            # 2. 跌破前3日最低点（结构止损2）
            # 3. 跌破入场价 - 2倍ATR（ATR止损）

            # 结构止损1: 跌破入场后前一根K线最低点
            if position.entry_prev_low > 0 and current_price <= position.entry_prev_low:
                exit_signal = "structure_stop_1"
                exit_reason = f"结构止损1@{position.entry_prev_low:.2f}(入场后前一根K线最低)"
                reduce_only = False
                reduce_ratio = 1.0
            # 结构止损2: 跌破前3日最低点
            elif position.lowest_3d_low > 0 and current_price <= position.lowest_3d_low:
                exit_signal = "structure_stop_2"
                exit_reason = f"结构止损2@{position.lowest_3d_low:.2f}(前3日最低)"
                reduce_only = False
                reduce_ratio = 1.0
            # ATR止损
            elif current_price <= position.stop_loss:
                exit_signal = "stop_loss"
                exit_reason = f"ATR止损@{position.stop_loss:.2f}"
                reduce_only = False
                reduce_ratio = 1.0
            # 追踪止损
            elif current_price <= trailing_stop:
                exit_signal = "trailing_stop"
                exit_reason = f"追踪止损@{trailing_stop:.2f}"
                reduce_only = False
                reduce_ratio = 1.0
            elif result.exit_signal == "ma_cross":
                exit_signal = "ma_cross"
                exit_reason = result.exit_reason
                reduce_only = False
                reduce_ratio = 1.0
            elif result.exit_signal == "rsi_overbought":
                exit_signal = "rsi_overbought"
                exit_reason = result.exit_reason
                reduce_only = False
                reduce_ratio = 1.0

            if exit_signal:
                signals.append(ExitSignal(
                    position_id=position.position_id,
                    code=code,
                    exit_signal=exit_signal,
                    exit_price=current_price,
                    reason=exit_reason,
                    reduce_only=reduce_only,
                    reduce_ratio=reduce_ratio
                ))

        return signals

    def _execute_entry_orders(self, entry_signals: List[EntrySignal], date: str) -> None:
        """执行入场订单（使用凯利公式 + 波动率调整）"""
        # 计算可用资金
        available_cash = self.cash * self.max_position_pct

        for signal in entry_signals:
            if self.cash < self.min_trade_amount:
                break

            if signal.code in self.positions:
                continue

            # === 新增：最大同时持仓数检查 ===
            if len(self.positions) >= self.max_open_positions:
                break  # 达到上限，不再开新仓

            # === 使用凯利仓位计算器 + 波动率调整 ===
            # 计算 ATR% (用于波动率调整)
            atr_pct = (signal.atr / signal.entry_price) * 100 if signal.entry_price > 0 else 0

            # 使用 position_sizer 计算仓位
            # 注意：这里没有传凯利参数（win_rate, avg_win, avg_loss），
            # 因为回测过程中还没有足够的统计数据
            # position_sizer 会使用波动率调整 + 单笔风险限制来计算仓位
            if self.position_size_type == "fixed":
                # 固定仓位模式下，先计算固定金额
                position_value = min(self.fixed_position_value, available_cash)
            else:
                # 波动率调整仓位计算
                position_value = self.position_sizer.calculate_position(
                    account_value=self.cash,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    atr_pct=atr_pct,
                    win_rate=None,  # 暂不使用凯利公式（需要历史数据）
                    avg_win=None,
                    avg_loss=None
                )

            # === 新增：试探仓位处理 ===
            # 第一笔交易使用试探仓位（较小型）
            is_trial = len(self.positions) == 0
            if is_trial:
                position_value = min(position_value, self.cash * self.trial_position_pct)

            # === 市场状态仓位调整 ===
            # 震荡市：减少50%仓位
            if signal.market_state == MarketState.VOLATILE.value:
                position_value *= 0.5

            shares = int(position_value / signal.entry_price / 100) * 100
            if shares < 100:
                continue

            # 获取 T+1 日数据
            df_next = self._get_next_day_data(signal.code, date)
            if df_next.empty:
                continue

            # 撮合买入
            match_result = self.matcher.match_buy(
                signal_date=date,
                code=signal.code,
                target_price=signal.entry_price,
                df_next=df_next,
                shares=shares,
                avg_daily_turnover=0.0  # 简化
            )

            if match_result.success:
                # 计算结构止损字段
                # entry_prev_low: 入场后前一根K线最低点（取df_next的最低价）
                entry_prev_low = df_next["low"].min() if "low" in df_next.columns else match_result.match_price
                # lowest_3d_low: 前3日最低点
                if len(df_next) >= 3:
                    lowest_3d_low = df_next["low"].iloc[-3:].min()
                else:
                    lowest_3d_low = df_next["low"].min()

                # 创建持仓
                position = Position(
                    position_id=generate_id(),
                    code=signal.code,
                    direction="long",
                    entry_date=match_result.match_date,
                    entry_price=match_result.match_price,
                    shares=shares,
                    original_shares=shares,  # 记录原始持股数量
                    atr=signal.atr,
                    stop_loss=signal.stop_loss,
                    trailing_stop=signal.stop_loss,
                    highest_price=match_result.match_price,
                    entry_prev_low=entry_prev_low,
                    lowest_3d_low=lowest_3d_low,
                    status="open",
                )

                self.positions[signal.code] = position
                self.cash -= (match_result.match_price * shares + match_result.commission)

    def _execute_exit_orders(self, exit_signals: List[ExitSignal], date: str) -> None:
        """执行出场订单（支持分批止盈部分平仓）"""
        for signal in exit_signals:
            position = self.positions.get(signal.code)
            if position is None:
                continue

            # 计算实际平仓数量
            if signal.reduce_only and signal.reduce_ratio < 1.0:
                # 部分平仓（分批止盈）
                shares_to_close = int(position.shares * signal.reduce_ratio)
                if shares_to_close < 100:
                    continue  # 不足以手为单位交易，跳过
            else:
                # 全部平仓
                shares_to_close = position.shares

            # 获取 T+1 日数据
            df_next = self._get_next_day_data(signal.code, date)
            if df_next.empty:
                # 如果没有 T+1 数据，使用当前价格
                exit_price = signal.exit_price
                match_date = date
            else:
                # 撮合卖出
                match_result = self.matcher.match_sell(
                    signal_date=date,
                    code=signal.code,
                    target_price=signal.exit_price,
                    df_next=df_next,
                    shares=shares_to_close,
                    avg_daily_turnover=0.0
                )

                if not match_result.success:
                    continue

                exit_price = match_result.match_price
                match_date = match_result.match_date

            # 生成 Trade
            trade = Trade(
                trade_id=generate_id(),
                date=position.entry_date,
                code=position.code,
                direction=position.direction,
                entry_price=position.entry_price,
                exit_price=exit_price,
                shares=shares_to_close,
                turnover=exit_price * shares_to_close,
                commission=0.0,  # 已在match_sell中计算
                signal_type=signal.exit_signal,
                signal_reason=signal.reason,
                position_id=position.position_id,
            )
            self.trades.append(trade)

            # 处理持仓
            if signal.reduce_only and signal.reduce_ratio < 1.0:
                # 部分平仓：更新持仓状态
                self.cash += exit_price * shares_to_close
                position.shares -= shares_to_close

                # 标记止盈状态
                if signal.exit_signal == "take_profit_1":
                    position.t1_triggered = True
                elif signal.exit_signal == "take_profit_2":
                    position.t2_triggered = True
                    # T2 后清空持仓（全部平掉）
                    position.status = "closed"
                    position.exit_date = match_date
                    position.exit_reason = signal.reason
                    del self.positions[signal.code]
            else:
                # 全部平仓
                self.cash += exit_price * position.shares
                position.status = "closed"
                position.exit_date = match_date
                position.exit_reason = signal.reason
                del self.positions[signal.code]

    def _update_and_record(self, date: str) -> None:
        """更新持仓市值并记录权益"""
        market_value = 0.0
        for position in self.positions.values():
            df = self.stock_data.get(position.code)
            if df is not None:
                mask = df["date"] == date
                if mask.any():
                    current_price = df.loc[mask.idxmax(), "close"]
                    position.current_price = current_price
                    market_value += current_price * position.shares

                    # 更新前3日最低点（知识库：结构止损2）
                    if len(df) >= 3:
                        position.lowest_3d_low = df["low"].iloc[-3:].min()

        total_equity = self.cash + market_value

        # 计算日收益率
        if self.equity_history:
            prev_equity = self.equity_history[-1].equity
            daily_return = (total_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
        else:
            daily_return = 0.0

        self.equity_history.append(EquityRecord(
            date=date,
            equity=total_equity,
            cash=self.cash,
            market_value=market_value,
            daily_return=daily_return,
        ))

    def _close_all_positions(self, date: str) -> None:
        """回测结束时平所有持仓"""
        for code, position in list(self.positions.items()):
            df = self.stock_data.get(code)
            if df is None:
                continue

            last_row = df.iloc[-1]
            exit_price = last_row["close"]

            trade = Trade(
                trade_id=generate_id(),
                date=position.entry_date,
                code=position.code,
                direction=position.direction,
                entry_price=position.entry_price,
                exit_price=exit_price,
                shares=position.shares,
                turnover=exit_price * position.shares,
                commission=0.0,
                signal_type="end_backtest",
                signal_reason="回测结束强制平仓",
                position_id=position.position_id,
            )
            self.trades.append(trade)

            self.cash += exit_price * position.shares
            position.status = "closed"
            position.exit_date = last_row["date"]
            position.exit_reason = "回测结束"

        self.positions.clear()

    def _generate_result(
        self,
        start_date: str,
        end_date: str,
        trading_dates: List[str]
    ) -> BacktestResult:
        """生成回测结果"""
        # 生成权益曲线
        equity_df = pd.DataFrame([
            {
                "date": e.date,
                "equity": e.equity,
                "cash": e.cash,
                "market_value": e.market_value,
                "daily_return": e.daily_return,
            }
            for e in self.equity_history
        ])

        # 计算绩效
        if equity_df.empty:
            equity_df = pd.DataFrame({
                "date": [start_date],
                "equity": [self.initial_capital],
                "cash": [self.initial_capital],
                "market_value": [0.0],
                "daily_return": [0.0],
            })

        metrics = self.analyzer.analyze(
            equity_df,
            self.trades,
            list(self.positions.values()),  # 当前持仓
            self.initial_capital
        )

        # 统计
        winning_trades = [t for t in self.trades if t.exit_price > t.entry_price]
        losing_trades = [t for t in self.trades if t.exit_price <= t.entry_price]

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=self.cash,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=len(winning_trades) / len(self.trades) if self.trades else 0.0,
            total_return=(self.cash - self.initial_capital) / self.initial_capital,
            annualized_return=metrics.annualized_return,
            profit_factor=metrics.profit_factor,
            avg_win=metrics.avg_win,
            avg_loss=metrics.avg_loss,
            sharpe_ratio=metrics.sharpe_ratio,
            sortino_ratio=metrics.sortino_ratio,
            max_drawdown=metrics.max_drawdown,
            max_drawdown_duration=metrics.max_drawdown_duration,
            calmar_ratio=metrics.calmar_ratio,
            avg_holding_days=metrics.avg_holding_days,
            total_trading_days=len(trading_dates),
            trades_per_year=metrics.trades_per_year,
            equity_curve=equity_df,
            trades=self.trades,
            positions=list(self.positions.values()),
        )

    def _generate_empty_result(self, start_date: str, end_date: str) -> BacktestResult:
        """生成空结果"""
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            equity_curve=pd.DataFrame({
                "date": [start_date],
                "equity": [self.initial_capital],
                "cash": [self.initial_capital],
                "market_value": [0.0],
                "daily_return": [0.0],
            }),
            trades=[],
            positions=[],
        )
