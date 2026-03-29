#!/usr/bin/env python3
"""修复验证脚本

验证以下修复是否正常工作：
1. min_profit_loss_ratio = 3.0
2. T2 独立于 T1 触发
3. 结构止损实现
4. RSI 动态阈值
"""

import sys
sys.path.insert(0, '.')

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.backtest.engine import SwingBacktester
from src.backtest.strategy_params import StrategyParams
from src.backtest.models import Position
from src.data.indicators.rsi import rsi_oversold


def test_min_profit_loss_ratio():
    """验证 min_profit_loss_ratio = 3.0"""
    params = StrategyParams()
    assert params.min_profit_loss_ratio == 3.0, \
        f"min_profit_loss_ratio 应为 3.0，实际: {params.min_profit_loss_ratio}"

    backtester = SwingBacktester()
    assert backtester.min_profit_loss_ratio == 3.0, \
        f"backtester.min_profit_loss_ratio 应为 3.0，实际: {backtester.min_profit_loss_ratio}"
    print("✓ min_profit_loss_ratio = 3.0")


def test_t2_independent():
    """验证 T2 独立于 T1 触发"""
    backtester = SwingBacktester()

    # 创建持仓
    position = Position(
        position_id="test",
        code="600000",
        entry_date="2024-01-01",
        entry_price=100.0,
        shares=1000,
        atr=2.0,
        stop_loss=96.0,
        highest_price=102.0,
        t1_triggered=False,
        t2_triggered=False,
        status="open",
    )
    backtester.positions["600000"] = position

    # 创建快照
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

    # 设置价格跌破 MA10
    df.iloc[-1, df.columns.get_loc('close')] = 101.0
    df.iloc[-1, df.columns.get_loc('low')] = 100.0
    df.iloc[-1, df.columns.get_loc('ma10')] = 102.0

    snapshots = {"600000": df}
    exit_signals = backtester._detect_exits(snapshots, "2024-01-25")

    t2_signals = [s for s in exit_signals if s.exit_signal == "take_profit_2"]
    assert len(t2_signals) > 0, "T2 应独立触发"
    print("✓ T2 独立于 T1 触发")


def test_structure_stop():
    """验证结构止损实现"""
    position = Position(
        position_id="test",
        code="600000",
        entry_date="2024-01-01",
        entry_price=100.0,
        shares=1000,
        atr=2.0,
        stop_loss=96.0,
        entry_prev_low=98.0,
        lowest_3d_low=99.0,
        status="open",
    )

    assert hasattr(position, 'entry_prev_low'), "应有 entry_prev_low"
    assert hasattr(position, 'lowest_3d_low'), "应有 lowest_3d_low"
    assert position.entry_prev_low == 98.0
    assert position.lowest_3d_low == 99.0
    print("✓ 结构止损字段已实现")


def test_structure_stop_triggers():
    """验证结构止损触发"""
    backtester = SwingBacktester()

    position = Position(
        position_id="test",
        code="600000",
        entry_date="2024-01-01",
        entry_price=100.0,
        shares=1000,
        atr=2.0,
        stop_loss=96.0,
        highest_price=100.0,
        entry_prev_low=98.0,
        lowest_3d_low=99.0,
        t1_triggered=False,
        t2_triggered=False,
        status="open",
    )
    backtester.positions["600000"] = position

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

    # 设置价格跌破 entry_prev_low 但高于 stop_loss
    df.iloc[-1, df.columns.get_loc('close')] = 97.0
    df.iloc[-1, df.columns.get_loc('low')] = 96.0

    snapshots = {"600000": df}
    exit_signals = backtester._detect_exits(snapshots, "2024-01-25")

    structure_signals = [s for s in exit_signals if 'structure_stop' in s.exit_signal]
    assert len(structure_signals) > 0, "应触发结构止损"
    assert structure_signals[0].exit_signal == "structure_stop_1", \
        f"应为 structure_stop_1，实际: {structure_signals[0].exit_signal}"
    print("✓ 结构止损正确触发")


def test_rsi_dynamic_threshold():
    """验证 RSI 动态阈值"""
    # period=14: threshold = max(20, 30 - 0) = 30
    assert rsi_oversold(29, period=14) == True, "RSI=29 < 30 应触发(period=14)"
    assert rsi_oversold(31, period=14) == False, "RSI=31 > 30 不应触发(period=14)"

    # period=6: threshold = max(20, 30 - 8) = 22
    assert rsi_oversold(21, period=6) == True, "RSI=21 < 22 应触发(period=6)"
    assert rsi_oversold(23, period=6) == False, "RSI=23 > 22 不应触发(period=6)"
    print("✓ RSI 动态阈值生效")


if __name__ == "__main__":
    print("=" * 50)
    print("修复验证测试")
    print("=" * 50)

    tests = [
        test_min_profit_loss_ratio,
        test_t2_independent,
        test_structure_stop,
        test_structure_stop_triggers,
        test_rsi_dynamic_threshold,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print("=" * 50)
    print(f"结果: {passed} passed, {failed} failed")
    print("=" * 50)

    sys.exit(0 if failed == 0 else 1)
