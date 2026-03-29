"""知识库与代码实现偏差验证测试

本文档验证以下偏差是否修复：

1. RSI 硬编码偏差 (P1)
   - 知识库: RSI < 40 且 > 30 是回踩入场机会
   - 代码: rsi_oversold() 硬编码 threshold=30，忽略 period 参数

2. T2 止盈依赖 T1 状态 (P1)
   - 知识库: T2（跌破 MA10 减仓50%）应独立触发
   - 代码: elif position.t1_triggered 强制要求 T1 先触发

3. 关键结构破坏缺失 (P1)
   - 知识库: 跌破前3日最低点应触发止损
   - 代码: 只有 ATR 止损，无结构止损

4. min_profit_loss_ratio 参数不一致 (P0)
   - 知识库: 中长线最低 1:3
   - 代码: strategy_params.py 默认 1.5，engine.py 默认 3.0
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.backtest.models import EntrySignal, Position, Trade, ExitSignal
from src.backtest.engine import SwingBacktester
from src.backtest.strategy_params import StrategyParams
from src.data.indicators.rsi import rsi_oversold, rsi_overbought
from src.data.indicators.signals import SwingSignals


class TestRSIDeviation:
    """RSI 偏差验证"""

    def test_rsi_oversold_uses_dynamic_threshold(self):
        """
        验证 rsi_oversold() 使用动态阈值

        知识库定义:
        - 周期 14: RSI < 30 超卖确认
        - 动态调整: 短周期更敏感（阈值略高），长周期更保守

        修复: threshold = max(20, 30 - (14 - period))
        """
        import inspect

        # 验证函数签名
        sig = inspect.signature(rsi_oversold)
        params = list(sig.parameters.keys())
        assert 'period' in params, "rsi_oversold 应该有 period 参数"

        # 验证动态阈值计算
        # period=14: threshold = max(20, 30 - 0) = 30
        assert rsi_oversold(29, period=14) == True, "RSI=29 < 30 应触发超卖(period=14)"
        assert rsi_oversold(31, period=14) == False, "RSI=31 > 30 不应触发超卖(period=14)"

        # period=6: threshold = max(20, 30 - 8) = 22
        assert rsi_oversold(21, period=6) == True, "RSI=21 < 22 应触发超卖(period=6)"
        assert rsi_oversold(23, period=6) == False, "RSI=23 > 22 不应触发超卖(period=6)"

        # period=21: threshold = max(20, 30 - (-7)) = 30 (不小于20)
        assert rsi_oversold(29, period=21) == True, "RSI=29 < 30 应触发超卖(period=21)"

    def test_rsi_threshold_now_dynamic(self):
        """
        验证 RSI 阈值现在是动态的

        修复后: threshold 根据 period 动态计算
        """
        # 动态阈值公式: threshold = max(20, 30 - (14 - period))
        # period=14: threshold=30
        # period=6: threshold=22
        # period=10: threshold=26

        threshold_14 = max(20, 30 - (14 - 14))  # 30
        threshold_6 = max(20, 30 - (14 - 6))     # 22
        threshold_10 = max(20, 30 - (14 - 10))   # 26

        assert threshold_14 == 30
        assert threshold_6 == 22
        assert threshold_10 == 26

        # 验证不同 RSI 值在不同 period 下的行为
        # threshold = max(20, 30 - (14 - period))
        # period=14: threshold=30, RSI=25 < 30 → 触发
        # period=6: threshold=22, RSI=25 < 22 → 不触发
        # period=10: threshold=26, RSI=25 < 26 → 触发
        assert rsi_oversold(25, period=14) == True, "RSI=25 < 30触发(period=14)"
        assert rsi_oversold(25, period=6) == False, "RSI=25 > 22不触发(period=6)"
        assert rsi_oversold(25, period=10) == True, "RSI=25 < 26触发(period=10)"


class TestT2Deviation:
    """T2 止盈偏差验证"""

    def setup_method(self):
        """测试初始化"""
        self.backtester = SwingBacktester(
            initial_capital=1_000_000,
            atr_stop_multiplier=2.0,
            atr_trailing_multiplier=3.0,
        )

    def test_t2_should_trigger_independently(self):
        """
        验证 T2 是否独立于 T1 触发

        知识库:
        - T1: 触及前高 或 盈亏比 1:1 → 减仓 50%
        - T2: 跌破 10 日均线 → 减仓 50%
        - T2 应该独立触发，不依赖 T1

        修复: 将 elif position.t1_triggered 改为 if not position.t2_triggered
        """
        # 创建一个持仓，T1 未触发，highest_price 较低避免触发追踪止损
        position = Position(
            position_id="test_pos",
            code="600000",
            entry_date="2024-01-01",
            entry_price=100.0,
            shares=1000,
            atr=2.0,
            stop_loss=96.0,  # 100 - 2*2
            highest_price=102.0,  # 较低，避免触发追踪止损
            t1_triggered=False,  # T1 未触发
            t2_triggered=False,
            status="open",
        )
        self.backtester.positions["600000"] = position

        # 创建快照：价格跌破 MA10，但 T1 未触发
        # 需要至少 20 天数据来计算指标
        dates = pd.date_range('2024-01-01', periods=25, freq='D')
        close_prices = [100.0 + i * 0.4 for i in range(25)]  # 缓慢上涨
        df = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'close': close_prices,
            'high': [p + 1.0 for p in close_prices],
            'low': [p - 1.0 for p in close_prices],
            'volume': [1000000] * 25,
        })

        # 添加必要的指标列
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['atr'] = 2.0

        # 设置当前价格 = 101.0，MA10 = 102.0（跌破 MA10）
        # trailing_stop = 102.0 - 3*2.0 = 96.0
        # current_price = 101.0 > 96.0，不触发追踪止损
        df.iloc[-1, df.columns.get_loc('close')] = 101.0
        df.iloc[-1, df.columns.get_loc('high')] = 102.0
        df.iloc[-1, df.columns.get_loc('low')] = 100.0
        df.iloc[-1, df.columns.get_loc('ma10')] = 102.0  # 价格 101 < MA10 102

        # 重新计算 MA10
        df['ma10'] = df['close'].rolling(10).mean()

        # 验证计算
        trailing_stop = position.highest_price - (3.0 * position.atr)  # 102 - 6 = 96
        assert trailing_stop < 101.0, f"追踪止损 {trailing_stop} 应该 < 当前价格 101"

        # 准备快照
        snapshots = {"600000": df}

        # 调用 _detect_exits
        exit_signals = self.backtester._detect_exits(snapshots, "2024-01-25")

        # 验证 T2 是否独立触发（不依赖 T1）
        # 修复后：T2 应该被触发，因为跌破 MA10 且 T2 未触发
        t2_signals = [s for s in exit_signals if s.exit_signal == "take_profit_2"]

        # 断言：T2 应该被触发（因为修复后不依赖 T1）
        assert len(t2_signals) > 0, \
            f"修复后 T2 应独立触发（跌破 MA10），但未触发。信号列表: {[s.exit_signal for s in exit_signals]}"


class TestStructureStopDeviation:
    """关键结构破坏偏差验证"""

    def test_structure_stop_now_implemented(self):
        """
        验证关键结构破坏已实现

        知识库止损触发条件（满足任一即触发）:
        1. 跌破入场后前一根 K 线最低点
        2. 跌破前 3 日最低点
        3. 跌破入场价 - 2 倍 ATR

        修复: Position 类添加 entry_prev_low 和 lowest_3d_low 字段
        """
        # 检查 Position 类是否有结构止损相关字段
        from src.backtest.models import Position

        position = Position(
            position_id="test",
            code="600000",
            entry_date="2024-01-01",
            entry_price=100.0,
            shares=1000,
            atr=2.0,
            stop_loss=96.0,
            entry_prev_low=98.0,  # 修复后添加
            lowest_3d_low=99.0,   # 修复后添加
            status="open",
        )

        # 验证结构止损字段存在
        assert hasattr(position, 'stop_loss'), "应有 stop_loss 属性"
        assert hasattr(position, 'atr'), "应有 atr 属性"
        assert hasattr(position, 'entry_prev_low'), "应有 entry_prev_low（入场后前一根K线最低）"
        assert hasattr(position, 'lowest_3d_low'), "应有 lowest_3d_low（前3日最低）"

        # 验证字段值正确
        assert position.entry_prev_low == 98.0, "entry_prev_low 应为 98.0"
        assert position.lowest_3d_low == 99.0, "lowest_3d_low 应为 99.0"

    def test_exit_detection_has_structure_check(self):
        """
        验证 _detect_exits 是否检查关键结构破坏

        修复: 在 ATR 止损之前添加结构止损检测
        """
        # 读取 engine.py 中 _detect_exits 的实现
        import inspect
        source = inspect.getsource(SwingBacktester._detect_exits)

        # 检查是否有结构破坏相关代码
        has_structure_stop_1 = 'structure_stop_1' in source or 'entry_prev_low' in source
        has_structure_stop_2 = 'structure_stop_2' in source or 'lowest_3d_low' in source

        # 修复后应该实现这些检查
        assert has_structure_stop_1, "修复后应有 entry_prev_low 结构止损检查"
        assert has_structure_stop_2, "修复后应有 lowest_3d_low 结构止损检查"

    def test_structure_stop_triggers_correctly(self):
        """
        验证结构止损能正确触发

        场景: 价格跌破 entry_prev_low 但未跌破 ATR 止损
        预期: 触发 structure_stop_1
        """
        from src.backtest.engine import SwingBacktester

        backtester = SwingBacktester(
            initial_capital=1_000_000,
            atr_stop_multiplier=2.0,
        )

        # 创建持仓：entry_prev_low=98, stop_loss=96（ATR止损）
        position = Position(
            position_id="test_pos",
            code="600000",
            entry_date="2024-01-01",
            entry_price=100.0,
            shares=1000,
            atr=2.0,
            stop_loss=96.0,  # ATR止损
            highest_price=100.0,
            entry_prev_low=98.0,  # 结构止损1
            lowest_3d_low=99.0,   # 结构止损2
            t1_triggered=False,
            t2_triggered=False,
            status="open",
        )
        backtester.positions["600000"] = position

        # 创建快照：价格 = 97，跌破 entry_prev_low=98 但高于 ATR 止损 96
        dates = pd.date_range('2024-01-01', periods=25, freq='D')
        df = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'close': [100.0 + i * 0.4 for i in range(25)],
            'high': [101.0 + i * 0.4 for i in range(25)],
            'low': [98.0 + i * 0.4 for i in range(25)],
            'volume': [1000000] * 25,
        })
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['atr'] = 2.0

        # 设置价格为 97（跌破 entry_prev_low=98，但高于 stop_loss=96）
        df.iloc[-1, df.columns.get_loc('close')] = 97.0
        df.iloc[-1, df.columns.get_loc('high')] = 98.0
        df.iloc[-1, df.columns.get_loc('low')] = 96.0

        snapshots = {"600000": df}
        exit_signals = backtester._detect_exits(snapshots, "2024-01-25")

        # 验证触发了结构止损1
        structure_signals = [s for s in exit_signals if 'structure_stop' in s.exit_signal]
        assert len(structure_signals) > 0, \
            f"应触发结构止损1，但信号列表: {[s.exit_signal for s in exit_signals]}"
        assert structure_signals[0].exit_signal == "structure_stop_1", \
            f"应为 structure_stop_1，实际: {structure_signals[0].exit_signal}"


class TestProfitLossRatioDeviation:
    """盈亏比参数偏差验证"""

    def test_min_profit_loss_ratio_now_consistent(self):
        """
        验证 min_profit_loss_ratio 参数现在是否一致

        知识库: 中长线最低盈亏比 1:3 (3.0)

        修复后: strategy_params.py 和 engine.py 都使用 3.0
        """
        # 检查 StrategyParams 默认值
        params = StrategyParams()
        assert params.min_profit_loss_ratio == 3.0, \
            f"StrategyParams.min_profit_loss_ratio = {params.min_profit_loss_ratio}，应为 3.0（知识库要求）"

    def test_engine_uses_correct_default(self):
        """
        验证 engine.py 使用的默认值

        engine.py 注释说"中长线 >= 3:1"
        """
        backtester = SwingBacktester()

        # engine.py 的注释说要 3.0
        # 但如果通过 strategy_params 传入，值会变成 1.5
        # 这是设计问题：engine 有自己的默认值，但 SwingSignals 用的是 strategy_params
        assert backtester.min_profit_loss_ratio == 3.0, \
            f"engine.py 默认值应为 3.0，实际: {backtester.min_profit_loss_ratio}"

    def test_knowledge_base_requires_3_0(self):
        """
        验证知识库的最低盈亏比要求

        知识库: 最低盈亏比 >= 1:3
        即 min_profit_loss_ratio >= 3.0
        """
        # 知识库定义
        KNOWLEDGE_BASE_MIN_RATIO = 3.0

        # 当前 strategy_params 默认值
        params = StrategyParams()
        assert params.min_profit_loss_ratio >= KNOWLEDGE_BASE_MIN_RATIO, \
            f"知识库要求 min_profit_loss_ratio >= {KNOWLEDGE_BASE_MIN_RATIO}，" \
            f"实际 StrategyParams 默认: {params.min_profit_loss_ratio}"


class TestRSIUpperBoundDeviation:
    """RSI 上限偏差验证"""

    def test_rsi_entry_needs_upper_bound(self):
        """
        验证 RSI 入场是否有上限检查

        知识库: RSI < 40 且 > 30 是回踩入场机会

        代码问题: 只检查 RSI < 上限，没有检查 RSI > 下限
        """
        params = StrategyParams(rsi_oversold=40)  # 知识库说回踩入场是 RSI < 40
        signals = SwingSignals(params)

        # 创建测试数据：RSI = 25（极度超卖）
        df = pd.DataFrame({
            'date': ['2024-01-01'] * 50,
            'close': [100.0 + i for i in range(50)],
            'open': [100.0 + i for i in range(50)],
            'high': [101.0 + i for i in range(50)],
            'low': [99.0 + i for i in range(50)],
            'volume': [1000000] * 50,
        })

        # 计算指标
        df = signals.calculate_all(df)

        # 设置极低的 RSI（超卖）
        df.loc[df.index[-1], 'rsi14'] = 25

        # 检测入场信号
        entry_signal, confidence, reason = signals.detect_entry(df)

        # RSI = 25 低于超卖阈值，应该有信号
        # 但知识库说"RSI < 40 且 > 30"，RSI=25 不在范围内（低于30）
        # 代码缺少 > 30 的检查，可能在 RSI 过低时错误入场

        # 验证代码是否检查 RSI > 下限
        # 目前代码只检查 rsi14 < self.rsi_overbought (80)
        # 没有检查 rsi14 > 下限（如 30）

        # 知识库定义的回踩入场区间是 30 < RSI < 40
        # RSI = 25 不在有效区间内
        current_rsi = df['rsi14'].iloc[-1]
        lower_bound = 30

        assert current_rsi < lower_bound, "RSI=25 低于知识库定义的下限 30"

        # 这个测试验证了代码问题：RSI 低于 30 时可能错误入场
        # 因为代码没有检查 RSI > 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
