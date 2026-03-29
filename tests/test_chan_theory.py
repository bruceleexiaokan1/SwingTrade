"""缠论指标测试"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

from src.data.indicators.chan_theory import (
    ChanTheory,
    Direction,
    KLineWithDirection,
    Pen,
    Segment,
    Center,
    ChanBuySignal,
    calculate_chan,
    detect_chan_signals
)


def create_test_df(dates=None, base_price=10.0, trend='random'):
    """
    创建测试数据

    Args:
        dates: 日期列表
        base_price: 基础价格
        trend: 'up', 'down', 'random'
    """
    if dates is None:
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                 for i in range(30, 0, -1)]

    n = len(dates)
    prices = [base_price]

    if trend == 'up':
        # 上涨趋势
        for i in range(1, n):
            prices.append(prices[-1] * (1 + 0.02 + np.random.random() * 0.01))
    elif trend == 'down':
        # 下跌趋势
        for i in range(1, n):
            prices.append(prices[-1] * (1 - 0.02 - np.random.random() * 0.01))
    else:
        # 随机
        for i in range(1, n):
            change = (np.random.random() - 0.5) * 0.04
            prices.append(prices[-1] * (1 + change))

    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'code': 'TEST',
        'open': [p - 0.2 for p in prices],
        'high': [p + 0.3 for p in prices],
        'low': [p - 0.3 for p in prices],
        'close': prices,
        'volume': [1000000 + i * 100000 for i in range(n)],
        'amount': [p * 1000000 for p in prices],
        'adj_factor': [1.0] * n,
        'turnover': [0.05] * n,
        'is_halt': [False] * n,
        'pct_chg': [0.02] * n,
    })

    return df


def create_uptrend_df():
    """创建上涨趋势数据（便于形成笔和线段）"""
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
             for i in range(50, 0, -1)]

    # 模拟上涨：高低点交替上升
    data = []
    for i, date in enumerate(dates):
        phase = i % 10
        base = 10.0 + (i // 10) * 2  # 每10天抬升一次

        if phase < 5:
            # 上涨段
            offset = phase * 0.3
            open_p = base + offset
            high = base + offset + 0.3
            low = base + offset - 0.1
            close = base + offset + 0.2
        else:
            # 回调段
            offset = (10 - phase) * 0.15
            open_p = base + 1.5 - offset
            high = base + 1.5 - offset + 0.2
            low = base + 1.5 - offset - 0.2
            close = base + 1.5 - offset - 0.1

        data.append({
            'date': date,
            'open': open_p,
            'high': high,
            'low': low,
            'close': close,
            'volume': 1000000,
            'amount': close * 1000000,
            'adj_factor': 1.0,
            'turnover': 0.05,
            'is_halt': False,
            'pct_chg': 0.02,
        })

    df = pd.DataFrame(data)
    df['code'] = 'TEST'
    return df


class TestChanTheoryInit:
    """测试 ChanTheory 初始化"""

    def test_default_init(self):
        """测试默认初始化"""
        chan = ChanTheory()
        assert chan.MIN_PEN_K == 5
        assert chan.MIN_SEGMENT_PENS == 3
        assert chan.MIN_CENTER_SEGMENTS == 3

    def test_custom_init(self):
        """测试自定义参数初始化"""
        chan = ChanTheory(min_pen_k=7, min_segment_pens=5, min_center_segments=4)
        assert chan.MIN_PEN_K == 7
        assert chan.MIN_SEGMENT_PENS == 5
        assert chan.MIN_CENTER_SEGMENTS == 4


class TestContainmentProcessing:
    """测试包含关系处理"""

    def test_no_containment(self):
        """测试无包含关系的情况"""
        chan = ChanTheory()

        df = create_test_df(trend='up')
        klines = chan._process_containment(df)

        assert len(klines) > 0
        assert all(hasattr(k, 'high') for k in klines)
        assert all(hasattr(k, 'low') for k in klines)

    def test_containment_merge(self):
        """测试包含关系合并"""
        chan = ChanTheory()

        # 创建有包含关系的数据
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                 for i in range(10, 0, -1)]

        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'code': 'TEST',
            'open': [10.0, 10.1, 10.05, 10.08, 10.12, 10.15, 10.18, 10.20, 10.22, 10.25],
            'high': [10.3, 10.4, 10.35, 10.38, 10.42, 10.45, 10.48, 10.50, 10.52, 10.55],
            'low': [9.9, 10.0, 9.95, 9.98, 10.02, 10.05, 10.08, 10.10, 10.12, 10.15],
            'close': [10.2, 10.3, 10.25, 10.28, 10.32, 10.35, 10.38, 10.40, 10.42, 10.45],
            'volume': [1000000] * 10,
            'amount': [p * 1000000 for p in [10.2, 10.3, 10.25, 10.28, 10.32, 10.35, 10.38, 10.40, 10.42, 10.45]],
            'adj_factor': [1.0] * 10,
            'turnover': [0.05] * 10,
            'is_halt': [False] * 10,
            'pct_chg': [0.02] * 10,
        })

        klines = chan._process_containment(df)
        assert len(klines) <= len(df)  # 合并后应该减少或相等


class TestFenXing:
    """测试分型识别"""

    def test_top_fen_xing(self):
        """测试顶分型"""
        chan = ChanTheory()

        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                 for i in range(5, 0, -1)]

        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'code': 'TEST',
            'open': [10.0, 10.2, 10.1, 10.3, 10.2],
            'high': [10.1, 10.3, 10.2, 10.4, 10.3],  # 中间最高 = 顶分型
            'low': [9.8, 9.9, 9.85, 9.95, 9.9],
            'close': [10.0, 10.2, 10.1, 10.3, 10.2],
            'volume': [1000000] * 5,
            'amount': [p * 1000000 for p in [10.0, 10.2, 10.1, 10.3, 10.2]],
            'adj_factor': [1.0] * 5,
            'turnover': [0.05] * 5,
            'is_halt': [False] * 5,
            'pct_chg': [0.02] * 5,
        })

        klines = chan._process_containment(df)
        fen_xings = chan._find_fen_xing(klines)

        # 应该识别到顶分型
        assert any(fx[1] == "顶" for fx in fen_xings)

    def test_bottom_fen_xing(self):
        """测试底分型"""
        chan = ChanTheory()

        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                 for i in range(5, 0, -1)]

        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'code': 'TEST',
            'open': [10.2, 10.0, 10.1, 9.9, 10.0],
            'high': [10.3, 10.1, 10.2, 10.0, 10.1],
            'low': [10.0, 9.8, 9.9, 9.7, 9.8],  # 中间最低 = 底分型
            'close': [10.2, 10.0, 10.1, 9.9, 10.0],
            'volume': [1000000] * 5,
            'amount': [p * 1000000 for p in [10.2, 10.0, 10.1, 9.9, 10.0]],
            'adj_factor': [1.0] * 5,
            'turnover': [0.05] * 5,
            'is_halt': [False] * 5,
            'pct_chg': [0.02] * 5,
        })

        klines = chan._process_containment(df)
        fen_xings = chan._find_fen_xing(klines)

        # 应该识别到底分型
        assert any(fx[1] == "底" for fx in fen_xings)


class TestFindPens:
    """测试笔识别"""

    def test_insufficient_data(self):
        """测试数据不足"""
        chan = ChanTheory()
        df = create_test_df(dates=[(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                                     for i in range(3, 0, -1)])
        pens = chan.find_pens(df)
        assert len(pens) == 0

    def test_find_pens_uptrend(self):
        """测试上涨趋势找笔"""
        chan = ChanTheory()
        df = create_uptrend_df()
        pens = chan.find_pens(df)

        # 上涨趋势应该有笔
        if len(pens) > 0:
            assert all(p.direction == Direction.UP or p.direction == Direction.DOWN
                      for p in pens)

    def test_pen_structure(self):
        """测试笔的结构"""
        chan = ChanTheory()
        df = create_uptrend_df()
        pens = chan.find_pens(df)

        if len(pens) > 0:
            for pen in pens:
                assert pen.start_idx >= 0
                assert pen.end_idx >= pen.start_idx
                assert pen.high >= pen.low
                assert pen.kline_count >= chan.MIN_PEN_K


class TestFindSegments:
    """测试线段识别"""

    def test_insufficient_pens(self):
        """测试笔不足"""
        chan = ChanTheory()
        df = create_test_df(dates=[(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                                     for i in range(10, 0, -1)])
        pens = chan.find_pens(df)
        segments = chan.find_segments(pens)
        assert len(segments) == 0

    def test_segment_structure(self):
        """测试线段结构"""
        chan = ChanTheory()
        df = create_uptrend_df()
        pens = chan.find_pens(df)
        segments = chan.find_segments(pens)

        if len(segments) > 0:
            for seg in segments:
                assert len(seg.pens) >= chan.MIN_SEGMENT_PENS
                assert seg.high >= seg.low


class TestFindCenters:
    """测试中枢识别"""

    def test_insufficient_segments(self):
        """测试线段不足"""
        chan = ChanTheory()
        df = create_test_df()
        pens = chan.find_pens(df)
        segments = chan.find_segments(pens)
        centers = chan.find_centers(segments)
        # 可能为空，因为数据可能不足以形成中枢
        assert isinstance(centers, list)

    def test_center_structure(self):
        """测试中枢结构"""
        chan = ChanTheory()
        df = create_uptrend_df()
        pens = chan.find_pens(df)
        segments = chan.find_segments(pens)
        centers = chan.find_centers(segments)

        if len(centers) > 0:
            for center in centers:
                assert len(center.segments) >= chan.MIN_CENTER_SEGMENTS
                assert center.zg >= center.zd
                assert center.high >= center.low


class TestFindBuySignals:
    """测试买卖点识别"""

    def test_no_signals_insufficient_data(self):
        """测试数据不足时无信号"""
        chan = ChanTheory()
        df = create_test_df(dates=[(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                                     for i in range(5, 0, -1)])
        signals = chan.find_buy_signals(df)
        assert isinstance(signals, list)

    def test_signals_sorted_by_priority(self):
        """测试信号按优先级排序"""
        chan = ChanTheory()
        df = create_uptrend_df()
        signals = chan.find_buy_signals(df)

        if len(signals) > 1:
            # 类二买 > 类三买 > 类一买
            priority_map = {"类二买": 0, "类三买": 1, "类一买": 2}
            for i in range(len(signals) - 1):
                assert priority_map[signals[i].signal_type] <= priority_map[signals[i+1].signal_type]


class TestDetectChanSignals:
    """测试 detect_chan_signals 函数"""

    def test_detect_chan_signals_basic(self):
        """测试基本功能"""
        df = create_uptrend_df()
        result = detect_chan_signals(df)

        assert 'has_buy_signal' in result
        assert 'primary_signal' in result
        assert 'all_signals' in result
        assert 'centers' in result
        assert 'pens' in result
        assert 'segments' in result

    def test_detect_chan_signals_insufficient_data(self):
        """测试数据不足"""
        df = create_test_df(dates=[(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                                     for i in range(5, 0, -1)])
        result = detect_chan_signals(df)

        assert result['has_buy_signal'] == False
        assert result['primary_signal'] is None


class TestCalculateChan:
    """测试 calculate_chan 函数"""

    def test_calculate_chan_basic(self):
        """测试基本功能"""
        df = create_uptrend_df()
        result_df = calculate_chan(df)

        assert 'chan_has_pen' in result_df.columns
        assert 'chan_has_center' in result_df.columns
        assert 'chan_has_buy_signal' in result_df.columns

    def test_calculate_chan_insufficient_data(self):
        """测试数据不足"""
        df = create_test_df(dates=[(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                                     for i in range(5, 0, -1)])
        result_df = calculate_chan(df)

        assert 'chan_signal_type' in result_df.columns
        assert 'chan_signal_price' in result_df.columns
        assert 'chan_signal_confidence' in result_df.columns


class TestChanTheoryAnalyze:
    """测试 ChanTheory.analyze 方法"""

    def test_analyze_insufficient_data(self):
        """测试数据不足"""
        chan = ChanTheory()
        df = create_test_df(dates=[(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                                     for i in range(5, 0, -1)])
        result = chan.analyze(df)

        assert result['pens'] == []
        assert result['segments'] == []
        assert result['centers'] == []
        assert result['buy_signals'] == []
        assert result['has_buy_signal'] == False
        assert result['primary_buy_signal'] is None

    def test_analyze_returns_dict(self):
        """测试返回字典"""
        chan = ChanTheory()
        df = create_uptrend_df()
        result = chan.analyze(df)

        assert isinstance(result, dict)
        assert 'pens' in result
        assert 'segments' in result
        assert 'centers' in result
        assert 'buy_signals' in result
        assert 'has_buy_signal' in result
        assert 'primary_buy_signal' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
