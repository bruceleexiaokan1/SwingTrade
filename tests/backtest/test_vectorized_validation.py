"""向量化回测验证测试

用真实/模拟数据验证向量化回测引擎的正确性：
1. 信号一致性验证
2. 执行逻辑验证
3. 结果正确性验证
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import sys
sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

from src.backtest.vectorized.engine import (
    VectorizedBacktester,
    BacktestConfig,
    VectorizedResult
)
from src.data.vectorized import VectorizedIndicators, IndicatorConfig
from src.data.vectorized import VectorizedSignals, SignalConfig


class TestSignalConsistency:
    """信号一致性验证"""

    def _create_golden_cross_data(self):
        """创建明确产生金叉的数据"""
        # 构造数据：MA20 从低于 MA60 变为高于 MA60
        dates = pd.date_range('2025-01-01', periods=100, freq='B')

        data = []
        for i, date in enumerate(dates):
            if i < 50:
                # 前期：MA20 < MA60
                base = 100 + i * 0.3
            elif i < 60:
                # 交叉期间：MA20 逐渐接近并超过 MA60
                offset = i - 50
                base = 115 + offset * 1.0  # 快速上涨
            else:
                # 后期：MA20 > MA60
                base = 125 + (i - 60) * 0.2

            # 添加随机波动
            noise = np.random.randn() * 0.5
            data.append({
                'data_id': '600519',
                'date': date.strftime('%Y-%m-%d'),
                'open': base + noise,
                'high': base + abs(noise) + 1,
                'low': base - abs(noise) - 1,
                'close': base + noise,
                'volume': 1000000 + np.random.randint(-200000, 200000)
            })

        return pd.DataFrame(data)

    def _create_death_cross_data(self):
        """创建明确产生死叉的数据"""
        dates = pd.date_range('2025-01-01', periods=100, freq='B')

        data = []
        for i, date in enumerate(dates):
            if i < 50:
                # 前期：MA20 > MA60
                base = 100 + (50 - i) * 0.3
            elif i < 60:
                # 交叉期间：MA20 逐渐低于 MA60
                offset = i - 50
                base = 85 - offset * 1.0  # 快速下跌
            else:
                # 后期：MA20 < MA60
                base = 75 - (i - 60) * 0.2

            noise = np.random.randn() * 0.5
            data.append({
                'data_id': '600519',
                'date': date.strftime('%Y-%m-%d'),
                'open': base + noise,
                'high': base + abs(noise) + 1,
                'low': base - abs(noise) - 1,
                'close': base + noise,
                'volume': 1000000 + np.random.randint(-200000, 200000)
            })

        return pd.DataFrame(data)

    def test_golden_cross_signal_occurs(self):
        """测试：金叉信号确实产生"""
        df = self._create_golden_cross_data()

        # 计算指标
        indicators = VectorizedIndicators(IndicatorConfig(
            ma_periods=[20, 60],
            rsi_period=14
        ))
        df = indicators.calculate_all(df)

        # 计算信号
        signals = VectorizedSignals(SignalConfig(
            ma_short_period=20,
            ma_long_period=60
        ))
        df = signals.calculate_all(df)

        # 验证金叉确实产生
        golden_cross_count = df['golden_cross'].sum()
        print(f"\n金叉信号数量: {golden_cross_count}")

        # 在这个构造的数据中，应该有金叉
        assert golden_cross_count > 0, "应该产生金叉信号"

    def test_death_cross_signal_occurs(self):
        """测试：死叉信号确实产生"""
        df = self._create_death_cross_data()

        indicators = VectorizedIndicators(IndicatorConfig(
            ma_periods=[20, 60],
            rsi_period=14
        ))
        df = indicators.calculate_all(df)

        signals = VectorizedSignals(SignalConfig(
            ma_short_period=20,
            ma_long_period=60
        ))
        df = signals.calculate_all(df)

        death_cross_count = df['death_cross'].sum()
        print(f"\n死叉信号数量: {death_cross_count}")

        assert death_cross_count > 0, "应该产生死叉信号"

    def test_golden_cross_at_expected_location(self):
        """测试：金叉在预期位置产生"""
        df = self._create_golden_cross_data()

        indicators = VectorizedIndicators(IndicatorConfig(
            ma_periods=[20, 60]
        ))
        df = indicators.calculate_all(df)

        signals = VectorizedSignals(SignalConfig(
            ma_short_period=20,
            ma_long_period=60
        ))
        df = signals.calculate_all(df)

        # 找到金叉位置
        golden_indices = df[df['golden_cross'] == 1].index.tolist()

        if golden_indices:
            first_golden = golden_indices[0]
            print(f"\n第一个金叉位置: {first_golden}")
            # 金叉应该在数据范围内产生
            assert first_golden >= 20, f"金叉位置 {first_golden} 太早（需要足够数据计算 MA60）"

    def test_signal_no_duplicate_on_same_day(self):
        """测试：同一交易日不会重复产生信号"""
        df = self._create_golden_cross_data()

        indicators = VectorizedIndicators(IndicatorConfig(
            ma_periods=[20, 60]
        ))
        df = indicators.calculate_all(df)

        signals = VectorizedSignals(SignalConfig(
            ma_short_period=20,
            ma_long_period=60
        ))
        df = signals.calculate_all(df)

        # 检查同一天是否有多个金叉
        golden_by_date = df[df['golden_cross'] == 1].groupby('date').size()
        duplicates = golden_by_date[golden_by_date > 1]

        assert len(duplicates) == 0, f"发现重复信号: {duplicates}"


class TestExecutionLogic:
    """执行逻辑验证"""

    def _create_simple_trend_data(self):
        """创建简单趋势数据"""
        dates = pd.date_range('2025-01-01', periods=50, freq='B')

        data = []
        for i, date in enumerate(dates):
            # 持续上涨趋势
            base = 100 + i * 0.5
            data.append({
                'data_id': '600519',
                'date': date.strftime('%Y-%m-%d'),
                'open': base,
                'high': base + 1,
                'low': base - 1,
                'close': base,
                'volume': 1000000
            })

        return pd.DataFrame(data)

    def test_entry_executes_on_golden_cross(self):
        """测试：金叉时执行入场"""
        df = self._create_simple_trend_data()

        # 修改数据确保产生金叉
        # 在第30天，让 MA20 穿过 MA60
        for i in range(30, 50):
            df.loc[i, 'close'] = 100 + i * 2  # 快速上涨

        # 重新计算指标
        indicators = VectorizedIndicators(IndicatorConfig(
            ma_periods=[20, 60]
        ))
        df = indicators.calculate_all(df)

        signals = VectorizedSignals(SignalConfig(
            ma_short_period=20,
            ma_long_period=60
        ))
        df = signals.calculate_all(df)

        # 模拟回测引擎
        config = BacktestConfig(
            initial_capital=1_000_000,
            max_positions=5,
            entry_confidence_threshold=0.0
        )
        engine = VectorizedBacktester(config=config)
        engine._signals_df = df
        engine._indicators_df = df
        engine._price_df = df

        # 运行单日处理
        engine._process_day(df['date'].iloc[30], df['date'].iloc[31])

        # 验证是否有持仓
        if engine._positions:
            print(f"\n持仓: {list(engine._positions.keys())}")
            print(f"现金: {engine._cash}")

        # 在上涨趋势中，应该有入场

    def test_max_position_limit(self):
        """测试：最大持仓数限制"""
        # 创建多只股票的数据
        dates = pd.date_range('2025-01-01', periods=50, freq='B')

        all_data = []
        for code in ['600519', '000001', '600036', '601318', '000002', '600000']:
            for i, date in enumerate(dates):
                base = 100 + i * 0.5
                all_data.append({
                    'data_id': code,
                    'date': date.strftime('%Y-%m-%d'),
                    'open': base,
                    'high': base + 1,
                    'low': base - 1,
                    'close': base,
                    'volume': 1000000
                })

        df = pd.DataFrame(all_data)

        # 计算指标和信号
        indicators = VectorizedIndicators(IndicatorConfig(
            ma_periods=[20, 60]
        ))
        df = indicators.calculate_all(df)

        signals = VectorizedSignals(SignalConfig(
            ma_short_period=20,
            ma_long_period=60
        ))
        df = signals.calculate_all(df)

        # 设置最大持仓为 3
        config = BacktestConfig(
            initial_capital=1_000_000,
            max_positions=3,
            entry_confidence_threshold=0.0  # 最低阈值，确保产生信号
        )
        engine = VectorizedBacktester(config=config)
        engine._signals_df = df
        engine._indicators_df = df
        engine._price_df = df
        engine._load_data(['600519', '000001', '600036', '601318', '000002', '600000'], '2025-01-01', '2025-03-31')

        # 触发多只股票的金叉信号
        for i in range(30, 50):
            date = df['date'].iloc[i]
            entry_signals = engine._detect_entries(date)
            engine._execute_entries(entry_signals, date)

        # 验证持仓数不超过限制
        print(f"\n实际持仓数: {len(engine._positions)}")
        print(f"最大持仓限制: {config.max_positions}")

        assert len(engine._positions) <= config.max_positions

    def test_exit_executes_on_death_cross(self):
        """测试：死叉时执行出场"""
        # 先创建一个有持仓的情况
        config = BacktestConfig(
            initial_capital=1_000_000,
            max_positions=5,
            entry_confidence_threshold=0.0
        )
        engine = VectorizedBacktester(config=config)

        # 手动添加一个持仓
        engine._positions['600519'] = {
            'code': '600519',
            'entry_date': '2025-01-15',
            'entry_price': 100.0,
            'shares': 1000,
            'atr': 2.0,
            'stop_loss': 96.0,
            'trailing_stop': 94.0,
            'highest_price': 110.0,
            'signal_confidence': 0.7,
            'signal_type': 'golden_cross'
        }
        engine._cash = 900_000

        # 创建死叉信号
        signals = [{
            'data_id': '600519',
            'signal_type': 'exit',
            'reason': 'death_cross',
            'close': 108.0
        }]

        # 执行出场
        engine._execute_exits(signals, '2025-03-15')

        # 验证持仓已平
        assert '600519' not in engine._positions
        print(f"\n出场后现金: {engine._cash}")

        # 验证有交易记录
        assert len(engine._trades) > 0
        print(f"交易记录: {engine._trades[-1]}")


class TestResultCalculation:
    """结果计算验证"""

    def test_win_rate_from_trades(self):
        """测试：从交易记录计算胜率"""
        trades = [
            {'pnl': 1000, 'code': '600519'},
            {'pnl': -500, 'code': '000001'},
            {'pnl': 2000, 'code': '600036'},
            {'pnl': -300, 'code': '601318'},
            {'pnl': 1500, 'code': '000002'},
        ]

        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = winning_trades / total_trades

        assert winning_trades == 3
        assert win_rate == 0.6

    def test_profit_factor_from_trades(self):
        """测试：从交易记录计算盈亏比"""
        trades = [
            {'pnl': 1000},
            {'pnl': -500},
            {'pnl': 2000},
            {'pnl': -300},
        ]

        total_win = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        total_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = total_win / total_loss

        assert total_win == 3000
        assert total_loss == 800
        assert profit_factor == 3.75

    def test_equity_curve_max_drawdown(self):
        """测试：权益曲线计算最大回撤"""
        equity_history = [
            {'date': '2025-01-01', 'equity': 1_000_000},
            {'date': '2025-01-02', 'equity': 1_010_000},  # 新高
            {'date': '2025-01-03', 'equity': 1_005_000},  # 回撤
            {'date': '2025-01-04', 'equity': 1_015_000},  # 新高
            {'date': '2025-01-05', 'equity': 1_000_000},  # 大回撤
        ]

        df = pd.DataFrame(equity_history)

        cummax = df['equity'].cummax()
        drawdown = (df['equity'] - cummax) / cummax
        max_drawdown = drawdown.min()

        # 最大回撤发生在 2025-01-05
        expected_max_dd = (1_000_000 - 1_015_000) / 1_015_000

        assert abs(max_drawdown - expected_max_dd) < 0.0001

    def test_sharpe_ratio_calculation(self):
        """测试：夏普比率计算"""
        equity_history = [
            {'date': '2025-01-01', 'equity': 1_000_000},
            {'date': '2025-01-02', 'equity': 1_010_000},
            {'date': '2025-01-03', 'equity': 1_020_000},
            {'date': '2025-01-04', 'equity': 1_030_000},
            {'date': '2025-01-05', 'equity': 1_040_000},
        ]

        df = pd.DataFrame(equity_history)
        daily_returns = df['equity'].pct_change().dropna()

        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)

        # 稳定上涨的策略应该有正夏普比率
        assert sharpe > 0, f"夏普比率应该为正，实际为 {sharpe}"

    def test_annualized_return_calculation(self):
        """测试：年化收益计算"""
        initial = 1_000_000
        final = 1_200_000
        n_days = 252  # 一年交易日

        total_return = (final - initial) / initial
        n_years = n_days / 252
        annualized = (1 + total_return) ** (1 / n_years) - 1

        expected = (1.2 ** (1/1)) - 1

        assert abs(annualized - expected) < 0.0001


class TestVectorizedVsReferenceOutput:
    """向量化输出与参考输出对比"""

    def test_signal_output_format(self):
        """测试：信号输出格式正确"""
        dates = pd.date_range('2025-01-01', periods=100, freq='B')

        data = []
        for i, date in enumerate(dates):
            base = 100 + i * 0.5 + np.random.randn() * 2
            data.append({
                'data_id': '600519',
                'date': date.strftime('%Y-%m-%d'),
                'open': base + np.random.randn() * 0.5,
                'high': base + abs(np.random.randn()) * 0.5 + 2,
                'low': base - abs(np.random.randn()) * 0.5 - 2,
                'close': base,
                'volume': 1000000 + np.random.randint(-200000, 200000)
            })

        df = pd.DataFrame(data)

        # 计算完整流程
        indicators = VectorizedIndicators(IndicatorConfig(
            ma_periods=[5, 10, 20, 60],
            rsi_period=14,
            atr_period=14,
            bollinger_period=20,
            bollinger_std=2
        ))
        df = indicators.calculate_all(df)

        signals = VectorizedSignals(SignalConfig(
            ma_short_period=20,
            ma_long_period=60,
            rsi_overbought=80
        ))
        df = signals.calculate_all(df)

        # 验证所有必要列存在
        required_signal_cols = [
            'golden_cross', 'death_cross',
            'rsi_oversold', 'rsi_overbought',
            'entry_confidence', 'exit_death_cross'
        ]

        for col in required_signal_cols:
            assert col in df.columns, f"缺少信号列: {col}"

        # 验证数据类型
        assert df['golden_cross'].dtype in [np.int64, np.int32, int]
        assert df['death_cross'].dtype in [np.int64, np.int32, int]
        assert 0 <= df['entry_confidence'].min()
        assert df['entry_confidence'].max() <= 1

    def test_entry_confidence_weighted(self):
        """测试：入场置信度加权正确"""
        dates = pd.date_range('2025-01-01', periods=100, freq='B')

        data = []
        for i, date in enumerate(dates):
            base = 100 + i * 0.5 + np.random.randn() * 2
            data.append({
                'data_id': '600519',
                'date': date.strftime('%Y-%m-%d'),
                'open': base + np.random.randn() * 0.5,
                'high': base + abs(np.random.randn()) * 0.5 + 2,
                'low': base - abs(np.random.randn()) * 0.5 - 2,
                'close': base,
                'volume': 1000000 + np.random.randint(-200000, 200000)
            })

        df = pd.DataFrame(data)

        indicators = VectorizedIndicators(IndicatorConfig(
            ma_periods=[20, 60],
            rsi_period=14
        ))
        df = indicators.calculate_all(df)

        signals = VectorizedSignals(SignalConfig(
            ma_short_period=20,
            ma_long_period=60,
            rsi_overbought=80
        ))
        df = signals.calculate_all(df)

        # 验证置信度计算逻辑
        # entry_golden_cross * 0.7 + entry_breakout * 0.6 + entry_rsi_bounce * 0.4

        # 找到有金叉的日子
        golden_days = df[df['golden_cross'] == 1]

        if len(golden_days) > 0:
            # 金叉产生的置信度至少是 0.7
            for idx in golden_days.index:
                confidence = df.loc[idx, 'entry_confidence']
                entry_golden = df.loc[idx, 'entry_golden_cross']
                if entry_golden == 1:
                    assert confidence >= 0.7, f"金叉入场置信度应该 >= 0.7，实际为 {confidence}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
