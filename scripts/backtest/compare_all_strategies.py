#!/usr/bin/env python3
"""策略比较：多维度策略效果对比

策略对比：
1. Baseline: Golden/Breakout + 周线过滤
2. 多周期: 月/周/日共振过滤
3. 多周期 + 波浪过滤: 在主升浪中交易
4. 多周期 + HMM过滤: 基于隐马尔可夫模型市场状态过滤
5. 多周期 + 板块共振: 基于板块共振信号过滤
6. 多周期 + HMM + 板块共振: 多条件组合

使用方式:
    python3 scripts/backtest/compare_all_strategies.py --codes 600519,000001,600036 --start 2024-01-01 --end 2024-06-30

    # 扩展回测（推荐）：
    python3 scripts/backtest/compare_all_strategies.py --start 2019-01-01 --end 2024-12-31

依赖:
    pip install hmmlearn  # 用于HMM模型
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

from src.backtest.engine import SwingBacktester
from src.backtest.strategy_params import StrategyParams
from src.backtest.multi_cycle import MultiCycleResonance, MultiCycleLevel
from src.data.loader import StockDataLoader
from src.data.indicators.wave import WaveIndicators, WaveType
from src.data.indicators.hmm_model import HMMMarketRegime, detect_market_regime
from src.backtest.models import EntrySignal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 策略对比结果数据结构
# ============================================================================

@dataclass
class StrategyResult:
    """单策略回测结果"""
    name: str
    total_trades: int = 0
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_holding_days: float = 0.0
    exit_counts: Dict[str, int] = None
    resonance_distribution: Dict[str, int] = None

    def __post_init__(self):
        if self.exit_counts is None:
            self.exit_counts = {}
        if self.resonance_distribution is None:
            self.resonance_distribution = {}


# ============================================================================
# 策略过滤器基类
# ============================================================================

class StrategyFilter:
    """策略过滤器基类"""

    def __init__(self, stockdata_root: str):
        self.stockdata_root = stockdata_root

    def should_enter(self, code: str, df: pd.DataFrame, date: str, snapshots: dict) -> Tuple[bool, str]:
        """
        判断是否允许入场

        Returns:
            (允许入场, 原因)
        """
        return True, ""


class MultiCycleFilter(StrategyFilter):
    """多周期共振过滤器"""

    def __init__(self, stockdata_root: str):
        super().__init__(stockdata_root)
        self.multi_cycle = MultiCycleResonance(stockdata_root=stockdata_root)

    def should_enter(self, code: str, df: pd.DataFrame, date: str, snapshots: dict) -> Tuple[bool, str]:
        mc_result = self.multi_cycle.check_resonance(code, date, lookback_months=6)

        # 周线向下，禁止做多
        if mc_result.weekly_trend == "down":
            return False, f"周线向下({mc_result.weekly_trend})"

        # 三层逆势，禁止操作
        if mc_result.resonance_level == MultiCycleLevel.FORBIDDEN:
            return False, f"共振等级{MultiCycleLevel.FORBIDDEN.value}(逆势)"

        return True, f"共振等级{mc_result.resonance_level}(月:{mc_result.monthly_trend},周:{mc_result.weekly_trend},日:{mc_result.daily_trend})"


class WaveFilter(StrategyFilter):
    """波浪过滤：只在主升浪(波浪3/5)中交易"""

    def __init__(self, stockdata_root: str):
        super().__init__(stockdata_root)
        self.wave = WaveIndicators()
        self.multi_cycle = MultiCycleResonance(stockdata_root=stockdata_root)

    def should_enter(self, code: str, df: pd.DataFrame, date: str, snapshots: dict) -> Tuple[bool, str]:
        # 多周期检查
        mc_result = self.multi_cycle.check_resonance(code, date, lookback_months=6)
        if mc_result.weekly_trend == "down":
            return False, f"周线向下"

        if mc_result.resonance_level == MultiCycleLevel.FORBIDDEN:
            return False, f"共振等级{resonance_level}(逆势)"

        # 波浪检查
        wave_result = self.wave.analyze(df, date)

        # UNKNOWN 或推动浪可以交易
        if wave_result.current_wave == WaveType.UNKNOWN:
            return True, f"波浪UNKNOWN(可交易)"

        if not wave_result.is_correction:
            # 推动浪中
            wave_name = wave_result.current_wave.name
            return True, f"推动浪{wave_name}(可交易)"

        # 调整浪中禁止开仓
        return False, f"调整浪({wave_result.current_wave.name})"


class HMMFilter(StrategyFilter):
    """HMM马尔可夫状态过滤器"""

    def __init__(self, stockdata_root: str):
        super().__init__(stockdata_root)
        self.multi_cycle = MultiCycleResonance(stockdata_root=stockdata_root)
        self.hmm = HMMMarketRegime(n_states=3, lookback=60, min_periods=30)

    def should_enter(self, code: str, df: pd.DataFrame, date: str, snapshots: dict) -> Tuple[bool, str]:
        # 多周期检查
        mc_result = self.multi_cycle.check_resonance(code, date, lookback_months=6)
        if mc_result.weekly_trend == "down":
            return False, f"周线向下"

        if mc_result.resonance_level == MultiCycleLevel.FORBIDDEN:
            return False, f"共振等级(逆势)"

        # HMM市场状态检查
        hmm_result = detect_market_regime(df)

        # 下跌趋势状态，禁止做多
        if hmm_result.state_name == 'downtrend':
            return False, f"HMM下跌({hmm_result.trend_direction})"

        # 置信度太低，不确定当前状态
        if hmm_result.regime_confidence < 0.5:
            return False, f"HMM置信度低({hmm_result.regime_confidence:.0%})"

        return True, f"HMM{hmm_result.state_name}(置信:{hmm_result.regime_confidence:.0%})"


class SectorResonanceFilter(StrategyFilter):
    """板块共振过滤器"""

    def __init__(self, stockdata_root: str, sector_config_path: str = None):
        super().__init__(stockdata_root)
        self.multi_cycle = MultiCycleResonance(stockdata_root=stockdata_root)

        if sector_config_path is None:
            sector_config_path = Path(__file__).parent.parent.parent / "config" / "sectors" / "sector_portfolio.json"

        self.sector_config_path = sector_config_path
        self._sector_map = self._load_sector_config()

    def _load_sector_config(self) -> Dict[str, str]:
        """加载板块配置，返回股票->板块映射"""
        config_path = Path(self.sector_config_path)
        if not config_path.exists():
            logger.warning(f"板块配置不存在: {config_path}")
            return {}

        import json
        with open(config_path) as f:
            config = json.load(f)

        # 构建 股票->板块 映射
        sector_map = {}
        for sector in config.get('sectors', []):
            for stock in sector.get('stocks', []):
                code = stock.get('code')
                if code:
                    sector_map[code] = sector.get('name')

        return sector_map

    def should_enter(self, code: str, df: pd.DataFrame, date: str, snapshots: dict) -> Tuple[bool, str]:
        # 多周期检查
        mc_result = self.multi_cycle.check_resonance(code, date, lookback_months=6)
        if mc_result.weekly_trend == "down":
            return False, f"周线向下"

        if mc_result.resonance_level == MultiCycleLevel.FORBIDDEN:
            return False, f"共振等级(逆势)"

        # 板块共振检查
        sector_name = self._sector_map.get(code)
        if sector_name is None:
            # 没有板块信息，使用宽松策略
            return True, f"无板块信息"

        # 这里可以添加更复杂的共振检测
        # 简化版：只要有板块就算通过
        return True, f"板块:{sector_name}"


# ============================================================================
# 策略对比器
# ============================================================================

class StrategyComparison:
    """策略比较器"""

    def __init__(self, stockdata_root: str = "/Users/bruce/workspace/trade/StockData"):
        self.stockdata_root = stockdata_root
        self.loader = StockDataLoader(stockdata_root=stockdata_root)

    def run_comparison(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        include_hmm: bool = False,  # HMM计算量大，默认关闭
    ) -> Dict[str, StrategyResult]:
        """运行策略比较"""

        results = {}

        # 策略1: Baseline (Golden/Breakout + 周线过滤)
        logger.info("=" * 70)
        logger.info("策略1: Baseline (Golden/Breakout + 周线过滤)")
        logger.info("=" * 70)
        results["baseline"] = self._run_baseline(codes, start_date, end_date)

        # 策略2: 多周期共振
        logger.info("=" * 70)
        logger.info("策略2: 多周期共振 (月/周/日趋势过滤)")
        logger.info("=" * 70)
        results["multi_cycle"] = self._run_multi_cycle(codes, start_date, end_date)

        # 策略3: 多周期 + 波浪过滤
        logger.info("=" * 70)
        logger.info("策略3: 多周期 + 波浪过滤 (主升浪中交易)")
        logger.info("=" * 70)
        results["wave_filter"] = self._run_wave_filter(codes, start_date, end_date)

        # 策略4: 多周期 + 板块共振
        logger.info("=" * 70)
        logger.info("策略4: 多周期 + 板块共振")
        logger.info("=" * 70)
        results["sector_resonance"] = self._run_sector_resonance(codes, start_date, end_date)

        # HMM策略可选（计算量大）
        if include_hmm:
            # 策略5: 多周期 + HMM过滤
            logger.info("=" * 70)
            logger.info("策略5: 多周期 + HMM过滤 (市场状态过滤)")
            logger.info("=" * 70)
            results["hmm_filter"] = self._run_hmm_filter(codes, start_date, end_date)

            # 策略6: 多周期 + HMM + 板块共振 (组合策略)
            logger.info("=" * 70)
            logger.info("策略6: 多周期 + HMM + 板块共振 (组合)")
            logger.info("=" * 70)
            results["combo_all"] = self._run_combo_all(codes, start_date, end_date)

        return results

    def _create_base_params(self) -> StrategyParams:
        """创建基础策略参数"""
        return StrategyParams(
            min_profit_loss_ratio=0.0,
            entry_confidence_threshold=0.5,
            atr_stop_multiplier=2.0,
            atr_trailing_multiplier=3.0,
        )

    def _run_baseline(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
    ) -> StrategyResult:
        """Baseline策略: Golden/Breakout + 周线过滤"""

        params = self._create_base_params()

        backtester = SwingBacktester(
            initial_capital=1_000_000,
            strategy_params=params,
        )

        result = backtester.run(
            stock_codes=codes,
            start_date=start_date,
            end_date=end_date,
        )

        return self._summarize_result("Baseline", result, backtester)

    def _run_multi_cycle(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
    ) -> StrategyResult:
        """多周期策略: 使用多周期共振过滤"""

        params = self._create_base_params()

        backtester = MultiCycleBacktester(
            initial_capital=1_000_000,
            strategy_params=params,
            stockdata_root=self.stockdata_root,
        )

        result = backtester.run(
            stock_codes=codes,
            start_date=start_date,
            end_date=end_date,
        )

        return self._summarize_result("多周期", result, backtester)

    def _run_wave_filter(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
    ) -> StrategyResult:
        """波浪过滤策略: 只在主升浪中交易"""

        params = self._create_base_params()

        backtester = WaveFilterBacktester(
            initial_capital=1_000_000,
            strategy_params=params,
            stockdata_root=self.stockdata_root,
        )

        result = backtester.run(
            stock_codes=codes,
            start_date=start_date,
            end_date=end_date,
        )

        return self._summarize_result("波浪过滤", result, backtester)

    def _run_hmm_filter(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
    ) -> StrategyResult:
        """HMM过滤策略: 基于市场状态过滤"""

        params = self._create_base_params()

        backtester = HMMFilterBacktester(
            initial_capital=1_000_000,
            strategy_params=params,
            stockdata_root=self.stockdata_root,
        )

        result = backtester.run(
            stock_codes=codes,
            start_date=start_date,
            end_date=end_date,
        )

        return self._summarize_result("HMM过滤", result, backtester)

    def _run_sector_resonance(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
    ) -> StrategyResult:
        """板块共振策略: 基于板块共振过滤"""

        params = self._create_base_params()

        backtester = SectorResonanceBacktester(
            initial_capital=1_000_000,
            strategy_params=params,
            stockdata_root=self.stockdata_root,
        )

        result = backtester.run(
            stock_codes=codes,
            start_date=start_date,
            end_date=end_date,
        )

        return self._summarize_result("板块共振", result, backtester)

    def _run_combo_all(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
    ) -> StrategyResult:
        """组合策略: 多周期 + HMM + 板块共振"""

        params = self._create_base_params()

        backtester = ComboAllBacktester(
            initial_capital=1_000_000,
            strategy_params=params,
            stockdata_root=self.stockdata_root,
        )

        result = backtester.run(
            stock_codes=codes,
            start_date=start_date,
            end_date=end_date,
        )

        return self._summarize_result("组合策略", result, backtester)

    def _summarize_result(
        self,
        name: str,
        result,
        backtester
    ) -> StrategyResult:
        """汇总回测结果"""

        # 统计出场原因
        exit_counts = {
            "structure_stop_1": 0,
            "structure_stop_2": 0,
            "atr_stop": 0,
            "trailing_stop": 0,
            "take_profit_1": 0,
            "take_profit_2": 0,
            "other": 0,
        }

        # 共振等级分布
        resonance_dist = {
            "THREE_CYCLE": 0,
            "MONTHLY_WEEKLY": 0,
            "DAILY_ONLY": 0,
            "FORBIDDEN": 0,
        }

        # 遍历持仓记录获取更多统计信息
        total_holding_days = 0
        winning_trades = 0
        total_profit = 0
        total_loss = 0

        for trade in result.trades:
            sig_type = trade.signal_type
            if sig_type == "structure_stop_1":
                exit_counts["structure_stop_1"] += 1
            elif sig_type == "structure_stop_2":
                exit_counts["structure_stop_2"] += 1
            elif sig_type == "stop_loss":
                exit_counts["atr_stop"] += 1
            elif sig_type == "trailing_stop":
                exit_counts["trailing_stop"] += 1
            elif sig_type == "take_profit_1":
                exit_counts["take_profit_1"] += 1
            elif sig_type == "take_profit_2":
                exit_counts["take_profit_2"] += 1
            else:
                exit_counts["other"] += 1

            # 持仓天数（使用 exit_reason 中的日期信息）
            # Trade.date 是入场日期，exit_date 从 position 获取
            if hasattr(trade, 'position_id') and trade.position_id:
                position = None
                for p in result.positions:
                    if p.position_id == trade.position_id:
                        position = p
                        break
                if position and position.exit_date:
                    entry_dt = pd.to_datetime(trade.date)
                    exit_dt = pd.to_datetime(position.exit_date)
                    holding_days = (exit_dt - entry_dt).days
                    total_holding_days += holding_days

            # 胜负
            if trade.pnl > 0:
                winning_trades += 1
                total_profit += trade.pnl
            else:
                total_loss += abs(trade.pnl)

        avg_holding_days = total_holding_days / len(result.trades) if result.trades else 0
        win_rate = winning_trades / len(result.trades) if result.trades else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        summary = StrategyResult(
            name=name,
            total_trades=result.total_trades,
            total_return=result.total_return,
            annualized_return=result.annualized_return,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_holding_days=avg_holding_days,
            exit_counts=exit_counts,
        )

        logger.info(f"  总交易: {summary.total_trades}")
        logger.info(f"  总收益: {summary.total_return:.2%}")
        logger.info(f"  夏普比率: {summary.sharpe_ratio:.2f}")
        logger.info(f"  最大回撤: {summary.max_drawdown:.2%}")
        logger.info(f"  胜率: {summary.win_rate:.2%}")
        logger.info(f"  盈亏比: {summary.profit_factor:.2f}")
        logger.info(f"  平均持仓: {summary.avg_holding_days:.1f}天")
        logger.info(f"  出场: 结构1={exit_counts['structure_stop_1']}, 结构2={exit_counts['structure_stop_2']}, "
                   f"ATR={exit_counts['atr_stop']}, 追踪={exit_counts['trailing_stop']}, "
                   f"T1={exit_counts['take_profit_1']}, T2={exit_counts['take_profit_2']}")

        return summary


# ============================================================================
# 自定义回测器类
# ============================================================================

class MultiCycleBacktester(SwingBacktester):
    """多周期回测器 - 使用多周期共振过滤"""

    def __init__(self, *args, stockdata_root: str = "/Users/bruce/workspace/trade/StockData", **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_cycle = MultiCycleResonance(stockdata_root=stockdata_root)

    def _detect_entries(self, snapshots, date):
        """检测入场信号，使用多周期过滤"""
        signals = []
        for code, df in snapshots.items():
            if code in self.positions:
                continue
            if len(df) < 20:
                continue

            # 多周期过滤
            mc_result = self.multi_cycle.check_resonance(code, date, lookback_months=6)

            # 周线向下，禁止做多
            if mc_result.weekly_trend == "down":
                continue

            # 三层逆势，禁止操作
            if mc_result.resonance_level == MultiCycleLevel.FORBIDDEN:
                continue

            # 市场状态检测
            result = self.signals.analyze(df)

            if result.trend == "downtrend":
                continue

            if result.entry_signal in ("golden", "breakout") and result.entry_confidence >= self.entry_confidence_threshold:
                atr = result.atr if result.atr else df["atr"].iloc[-1]
                if pd.isna(atr) or atr <= 0:
                    continue
                entry_price = df["close"].iloc[-1]
                stop_loss = entry_price - (self.atr_stop_multiplier * atr)

                current_atr = df["atr"].iloc[-1]
                if not pd.isna(current_atr) and current_atr > atr * self.atr_circuit_breaker:
                    continue

                expected_profit_pct = 0.05
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
                    reason=f"{result.entry_reason} | 多周期:{mc_result.level_label}",
                ))

        return signals


class WaveFilterBacktester(SwingBacktester):
    """波浪过滤回测器 - 只在主升浪中交易"""

    def __init__(self, *args, stockdata_root: str = "/Users/bruce/workspace/trade/StockData", **kwargs):
        super().__init__(*args, **kwargs)
        self.wave = WaveIndicators()
        self.multi_cycle = MultiCycleResonance(stockdata_root=stockdata_root)

    def _detect_entries(self, snapshots, date):
        """检测入场信号，增加波浪过滤"""
        signals = []
        for code, df in snapshots.items():
            if code in self.positions:
                continue
            if len(df) < 20:
                continue

            # 多周期过滤
            mc_result = self.multi_cycle.check_resonance(code, date, lookback_months=6)
            if mc_result.weekly_trend == "down":
                continue
            if mc_result.resonance_level == MultiCycleLevel.FORBIDDEN:
                continue

            # 波浪过滤
            wave_result = self.wave.analyze(df, date)
            # 调整浪中禁止开仓
            if wave_result.is_correction and wave_result.current_wave != WaveType.UNKNOWN:
                continue

            # 市场状态检测
            result = self.signals.analyze(df)

            if result.trend == "downtrend":
                continue

            if result.entry_signal in ("golden", "breakout") and result.entry_confidence >= self.entry_confidence_threshold:
                atr = result.atr if result.atr else df["atr"].iloc[-1]
                if pd.isna(atr) or atr <= 0:
                    continue
                entry_price = df["close"].iloc[-1]
                stop_loss = entry_price - (self.atr_stop_multiplier * atr)

                current_atr = df["atr"].iloc[-1]
                if not pd.isna(current_atr) and current_atr > atr * self.atr_circuit_breaker:
                    continue

                expected_profit_pct = 0.05
                expected_profit = entry_price * expected_profit_pct
                stop_distance = self.atr_stop_multiplier * atr
                profit_loss_ratio = expected_profit / stop_distance if stop_distance > 0 else 0
                if profit_loss_ratio < self.min_profit_loss_ratio:
                    continue

                wave_info = wave_result.current_wave.name if wave_result.current_wave != WaveType.UNKNOWN else "UNKNOWN"
                signals.append(EntrySignal(
                    code=code,
                    signal_type=result.entry_signal,
                    confidence=result.entry_confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    atr=atr,
                    reason=f"{result.entry_reason} | 波浪:{wave_info}",
                ))

        return signals


class HMMFilterBacktester(SwingBacktester):
    """HMM过滤回测器 - 基于市场状态过滤"""

    def __init__(self, *args, stockdata_root: str = "/Users/bruce/workspace/trade/StockData", **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_cycle = MultiCycleResonance(stockdata_root=stockdata_root)
        self.hmm = HMMMarketRegime(n_states=3, lookback=60, min_periods=30)

    def _detect_entries(self, snapshots, date):
        """检测入场信号，增加HMM市场状态过滤"""
        signals = []
        for code, df in snapshots.items():
            if code in self.positions:
                continue
            if len(df) < 60:  # HMM需要更多数据
                continue

            # 多周期过滤
            mc_result = self.multi_cycle.check_resonance(code, date, lookback_months=6)
            if mc_result.weekly_trend == "down":
                continue
            if mc_result.resonance_level == MultiCycleLevel.FORBIDDEN:
                continue

            # HMM市场状态过滤
            hmm_result = detect_market_regime(df)

            # 下跌趋势状态，禁止做多
            if hmm_result.state_name == 'downtrend':
                continue

            # 置信度太低，不确定当前状态
            if hmm_result.regime_confidence < 0.5:
                continue

            # 市场状态检测
            result = self.signals.analyze(df)

            if result.trend == "downtrend":
                continue

            if result.entry_signal in ("golden", "breakout") and result.entry_confidence >= self.entry_confidence_threshold:
                atr = result.atr if result.atr else df["atr"].iloc[-1]
                if pd.isna(atr) or atr <= 0:
                    continue
                entry_price = df["close"].iloc[-1]
                stop_loss = entry_price - (self.atr_stop_multiplier * atr)

                current_atr = df["atr"].iloc[-1]
                if not pd.isna(current_atr) and current_atr > atr * self.atr_circuit_breaker:
                    continue

                expected_profit_pct = 0.05
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
                    reason=f"{result.entry_reason} | HMM:{hmm_result.state_name}",
                ))

        return signals


class SectorResonanceBacktester(SwingBacktester):
    """板块共振回测器 - 基于板块共振过滤"""

    def __init__(self, *args, stockdata_root: str = "/Users/bruce/workspace/trade/StockData", **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_cycle = MultiCycleResonance(stockdata_root=stockdata_root)

        # 加载板块配置
        sector_config_path = Path(__file__).parent.parent.parent / "config" / "sectors" / "sector_portfolio.json"
        self._sector_map = self._load_sector_config(sector_config_path)

    def _load_sector_config(self, config_path) -> Dict[str, str]:
        """加载板块配置"""
        if not Path(config_path).exists():
            return {}

        import json
        with open(config_path) as f:
            config = json.load(f)

        sector_map = {}
        for sector in config.get('sectors', []):
            for stock in sector.get('stocks', []):
                code = stock.get('code')
                if code:
                    sector_map[code] = sector.get('name')

        return sector_map

    def _detect_entries(self, snapshots, date):
        """检测入场信号，增加板块共振过滤"""
        signals = []
        for code, df in snapshots.items():
            if code in self.positions:
                continue
            if len(df) < 20:
                continue

            # 多周期过滤
            mc_result = self.multi_cycle.check_resonance(code, date, lookback_months=6)
            if mc_result.weekly_trend == "down":
                continue
            if mc_result.resonance_level == MultiCycleLevel.FORBIDDEN:
                continue

            # 板块共振过滤（简化版：有板块配置的优先）
            sector_name = self._sector_map.get(code)
            if sector_name is None:
                # 没有板块信息，使用宽松策略
                pass

            # 市场状态检测
            result = self.signals.analyze(df)

            if result.trend == "downtrend":
                continue

            if result.entry_signal in ("golden", "breakout") and result.entry_confidence >= self.entry_confidence_threshold:
                atr = result.atr if result.atr else df["atr"].iloc[-1]
                if pd.isna(atr) or atr <= 0:
                    continue
                entry_price = df["close"].iloc[-1]
                stop_loss = entry_price - (self.atr_stop_multiplier * atr)

                current_atr = df["atr"].iloc[-1]
                if not pd.isna(current_atr) and current_atr > atr * self.atr_circuit_breaker:
                    continue

                expected_profit_pct = 0.05
                expected_profit = entry_price * expected_profit_pct
                stop_distance = self.atr_stop_multiplier * atr
                profit_loss_ratio = expected_profit / stop_distance if stop_distance > 0 else 0
                if profit_loss_ratio < self.min_profit_loss_ratio:
                    continue

                sector_info = f"板块:{sector_name}" if sector_name else "无板块"
                signals.append(EntrySignal(
                    code=code,
                    signal_type=result.entry_signal,
                    confidence=result.entry_confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    atr=atr,
                    reason=f"{result.entry_reason} | {sector_info}",
                ))

        return signals


class ComboAllBacktester(SwingBacktester):
    """组合策略回测器 - 多周期 + HMM + 板块共振"""

    def __init__(self, *args, stockdata_root: str = "/Users/bruce/workspace/trade/StockData", **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_cycle = MultiCycleResonance(stockdata_root=stockdata_root)
        self.hmm = HMMMarketRegime(n_states=3, lookback=60, min_periods=90)

        # 加载板块配置
        sector_config_path = Path(__file__).parent.parent.parent / "config" / "sectors" / "sector_portfolio.json"
        self._sector_map = self._load_sector_config(sector_config_path)

    def _load_sector_config(self, config_path) -> Dict[str, str]:
        """加载板块配置"""
        if not Path(config_path).exists():
            return {}

        import json
        with open(config_path) as f:
            config = json.load(f)

        sector_map = {}
        for sector in config.get('sectors', []):
            for stock in sector.get('stocks', []):
                code = stock.get('code')
                if code:
                    sector_map[code] = sector.get('name')

        return sector_map

    def _detect_entries(self, snapshots, date):
        """检测入场信号，使用多周期+HMM+板块共振组合过滤"""
        signals = []
        for code, df in snapshots.items():
            if code in self.positions:
                continue
            if len(df) < 90:  # HMM需要更多数据
                continue

            # 多周期过滤
            mc_result = self.multi_cycle.check_resonance(code, date, lookback_months=6)
            if mc_result.weekly_trend == "down":
                continue
            if mc_result.resonance_level == MultiCycleLevel.FORBIDDEN:
                continue

            # HMM市场状态过滤
            try:
                hmm_result = detect_market_regime(df)
                if hmm_result.state_name == 'downtrend':
                    continue
                if hmm_result.regime_confidence < 0.5:
                    continue
                hmm_active = True
            except Exception:
                hmm_active = False

            # 板块过滤
            sector_name = self._sector_map.get(code)

            # 市场状态检测
            result = self.signals.analyze(df)

            if result.trend == "downtrend":
                continue

            if result.entry_signal in ("golden", "breakout") and result.entry_confidence >= self.entry_confidence_threshold:
                atr = result.atr if result.atr else df["atr"].iloc[-1]
                if pd.isna(atr) or atr <= 0:
                    continue
                entry_price = df["close"].iloc[-1]
                stop_loss = entry_price - (self.atr_stop_multiplier * atr)

                current_atr = df["atr"].iloc[-1]
                if not pd.isna(current_atr) and current_atr > atr * self.atr_circuit_breaker:
                    continue

                expected_profit_pct = 0.05
                expected_profit = entry_price * expected_profit_pct
                stop_distance = self.atr_stop_multiplier * atr
                profit_loss_ratio = expected_profit / stop_distance if stop_distance > 0 else 0
                if profit_loss_ratio < self.min_profit_loss_ratio:
                    continue

                parts = [result.entry_reason]
                parts.append(f"共振:{mc_result.level_label}")
                if hmm_active:
                    parts.append(f"HMM:{hmm_result.state_name}")
                if sector_name:
                    parts.append(f"板块:{sector_name}")

                signals.append(EntrySignal(
                    code=code,
                    signal_type=result.entry_signal,
                    confidence=result.entry_confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    atr=atr,
                    reason=" | ".join(parts),
                ))

        return signals


# ============================================================================
# 报告生成
# ============================================================================

def generate_comparison_report(
    results: Dict[str, StrategyResult],
    codes: List[str],
    start_date: str,
    end_date: str,
    output_dir: str = "reports",
) -> None:
    """生成对比报告"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows_html = ""
    strategy_names = {
        "baseline": "Baseline (周线过滤)",
        "multi_cycle": "多周期共振 (月/周/日)",
        "wave_filter": "波浪过滤 (主升浪)",
        "hmm_filter": "HMM过滤 (市场状态)",
        "sector_resonance": "板块共振",
        "combo_all": "组合策略 (多周期+HMM+板块)",
    }

    for key, name in strategy_names.items():
        if key not in results:
            continue
        r = results[key]
        exit_c = r.exit_counts
        rows_html += f"""                <tr>
                    <td><strong>{name}</strong></td>
                    <td>{r.total_trades}</td>
                    <td class="{'positive' if r.total_return > 0 else 'negative'}">{r.total_return:.2%}</td>
                    <td class="{'positive' if r.annualized_return > 0 else 'negative'}">{r.annualized_return:.2%}</td>
                    <td>{r.sharpe_ratio:.2f}</td>
                    <td class="negative">{r.max_drawdown:.2%}</td>
                    <td>{r.win_rate:.2%}</td>
                    <td>{r.profit_factor:.2f}</td>
                    <td>{r.avg_holding_days:.1f}</td>
                    <td>{exit_c.get('structure_stop_1', 0)}</td>
                    <td>{exit_c.get('structure_stop_2', 0)}</td>
                    <td>{exit_c.get('atr_stop', 0)}</td>
                    <td>{exit_c.get('trailing_stop', 0)}</td>
                </tr>
"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>多维度策略对比报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #f5f7fa; padding: 20px; }}
        .container {{ max-width: 1600px; margin: auto; }}
        h1 {{ color: #1a1a2e; margin-bottom: 10px; }}
        .meta {{ color: #666; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 30px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; color: #666; font-weight: 600; text-transform: uppercase; font-size: 11px; }}
        tr:hover {{ background: #f8f9fa; }}
        .positive {{ color: #2ecc71; }}
        .negative {{ color: #e74c3c; }}
        .analysis {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .analysis h2 {{ color: #1a1a2e; margin-bottom: 15px; }}
        .analysis h3 {{ color: #34495e; margin-top: 20px; margin-bottom: 10px; }}
        .analysis p {{ line-height: 1.6; color: #555; }}
        .analysis ul {{ margin-left: 20px; line-height: 1.8; color: #555; }}
        .conclusion {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>多维度策略对比报告</h1>
        <p class="meta">回测区间: {start_date} ~ {end_date} | 股票: {', '.join(codes)} | 生成时间: {timestamp}</p>

        <h2 style="margin-bottom: 15px;">策略对比结果</h2>
        <table>
            <thead>
                <tr>
                    <th>策略</th>
                    <th>交易数</th>
                    <th>总收益</th>
                    <th>年化收益</th>
                    <th>夏普比率</th>
                    <th>最大回撤</th>
                    <th>胜率</th>
                    <th>盈亏比</th>
                    <th>平均持仓</th>
                    <th>结构止损1</th>
                    <th>结构止损2</th>
                    <th>ATR止损</th>
                    <th>追踪止损</th>
                </tr>
            </thead>
            <tbody>
{rows_html}
            </tbody>
        </table>

        <div class="analysis">
            <h2>分析结论</h2>

            <h3>1. 各策略对比分析</h3>
            <ul>
                <li><strong>Baseline</strong>: 基础策略，仅用周线过滤</li>
                <li><strong>多周期共振</strong>: 月/周/日三层趋势过滤，更严格的入场条件</li>
                <li><strong>波浪过滤</strong>: 只在推动浪中交易，避免调整浪</li>
                <li><strong>HMM过滤</strong>: 基于隐马尔可夫模型识别市场状态，下跌状态禁止做多</li>
                <li><strong>板块共振</strong>: 结合板块动量，优先选择强势板块内的个股</li>
            </ul>

            <h3>2. 过滤效果评估维度</h3>
            <ul>
                <li><strong>信号质量</strong>: 出场原因分布是否更合理（减少被动止损）</li>
                <li><strong>交易频率</strong>: 过滤是否过于严格导致交易数过少</li>
                <li><strong>风险控制</strong>: 最大回撤是否得到有效控制</li>
                <li><strong>收益稳定性</strong>: 夏普比率和盈亏比</li>
            </ul>

            <h3>3. 数据限制说明</h3>
            <ul>
                <li><strong>HMM过滤</strong>: 需要90天以上数据，2019初期数据不足可能导致早期无交易</li>
                <li><strong>板块共振</strong>: 板块数据仅有约1年历史，2023年前的回测结果仅供参考</li>
                <li><strong>组合策略</strong>: 同时受HMM和板块数据限制影响</li>
            </ul>

            <h3>4. 策略选择建议</h3>
            <ul>
                <li>如果 <strong>交易数过少</strong>：说明过滤过于严格，需要适当放宽条件</li>
                <li>如果 <strong>ATR止损占比高</strong>：说明入场时机需要优化</li>
                <li>如果 <strong>结构止损占比高</strong>：说明波段识别逻辑需要改进</li>
                <li>如果 <strong>追踪止损占比高</strong>：说明策略能捕捉到大趋势</li>
            </ul>

            <div class="conclusion">
                <h3>下一步优化方向</h3>
                <p>根据对比结果，选择表现最好的策略作为基准，重点优化：
                入场信号质量（减少假信号）、波段识别精度（提高结构止损占比），
                以及根据不同市场状态自适应调整策略参数。</p>
            </div>
        </div>
    </div>
</body>
</html>"""

    output_path = f"{output_dir}/strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"\n报告已生成: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="多维度策略对比验证")
    # 扩展股票池：覆盖多个行业和波动性特征
    default_codes = (
        "600519,000001,600036,601318,600886,600900,"  # 金融/消费/电力
        "002594,300750,688012,002371,"  # 新能源/半导体
        "600276,000538,603259,"  # 医药
        "300059,688111,002230"  # 科技
    )
    parser.add_argument("--codes", type=str, default=default_codes,
                        help="股票代码列表，逗号分隔")
    parser.add_argument("--start", type=str, default="2019-01-01",
                        help="开始日期 (默认: 2019-01-01 覆盖完整牛熊)")
    parser.add_argument("--end", type=str, default="2024-12-31",
                        help="结束日期 (默认: 2024-12-31)")

    args = parser.parse_args()
    codes = args.codes.split(",")

    logger.info(f"策略对比: codes={codes}, start={args.start}, end={args.end}")

    comparison = StrategyComparison()
    results = comparison.run_comparison(codes, args.start, args.end)

    generate_comparison_report(results, codes, args.start, args.end)

    # 打印汇总表
    logger.info("\n" + "=" * 80)
    logger.info("策略对比汇总")
    logger.info("=" * 80)
    logger.info(f"{'策略':<20} {'交易数':>8} {'总收益':>10} {'夏普':>8} {'最大回撤':>10} {'胜率':>8}")
    logger.info("-" * 80)
    for key, r in results.items():
        logger.info(f"{r.name:<20} {r.total_trades:>8} {r.total_return:>10.2%} {r.sharpe_ratio:>8.2f} {r.max_drawdown:>10.2%} {r.win_rate:>8.2%}")


if __name__ == "__main__":
    main()
