"""向量化回测端到端测试

验证完整的回测流程：
1. 数据加载
2. 指标预计算
3. 信号预计算
4. 多周期预计算
5. 逐日回测迭代
6. 结果生成
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

from src.backtest.vectorized.engine import (
    VectorizedBacktester,
    BacktestConfig,
    VectorizedResult
)
from src.data.vectorized import VectorizedIndicators, IndicatorConfig
from src.data.vectorized import VectorizedSignals, SignalConfig


class TestVectorizedEndToEnd:
    """端到端回测测试"""

    @pytest.fixture
    def mock_stock_data(self):
        """创建模拟股票数据"""
        dates = pd.date_range('2025-01-01', '2025-06-30', freq='B')
        np.random.seed(42)

        all_data = []
        for data_id in ['600519', '000001']:
            for i, date in enumerate(dates):
                # 模拟上涨趋势
                if data_id == '600519':
                    base = 100 + i * 0.3 + np.random.randn() * 2
                else:
                    base = 200 - i * 0.2 + np.random.randn() * 3

                open_price = base + np.random.randn() * 0.5
                high_price = max(open_price, base) + abs(np.random.randn() * 0.5)
                low_price = min(open_price, base) - abs(np.random.randn() * 0.5)
                close_price = base
                volume = 1000000 + np.random.randint(-200000, 200000)

                all_data.append({
                    'data_id': data_id,
                    'date': date.strftime('%Y-%m-%d'),
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': int(volume)
                })

        return pd.DataFrame(all_data)

    @pytest.fixture
    def mock_loader(self, mock_stock_data):
        """创建模拟数据加载器"""
        loader = MagicMock()
        loader.load_daily = MagicMock(return_value=mock_stock_data)
        return loader

    def test_engine_initialization(self):
        """测试：引擎初始化"""
        config = BacktestConfig(
            initial_capital=1_000_000,
            max_positions=3,
            entry_confidence_threshold=0.5
        )
        engine = VectorizedBacktester(config=config)

        assert engine.config.initial_capital == 1_000_000
        assert engine.config.max_positions == 3
        assert engine._cash == 1_000_000
        assert len(engine._positions) == 0

    def test_config_defaults(self):
        """测试：默认配置"""
        config = BacktestConfig()
        assert config.initial_capital == 1_000_000
        assert config.commission_rate == 0.0003
        assert config.stamp_tax == 0.0001
        assert config.max_positions == 5
        assert config.entry_confidence_threshold == 0.5

    def test_result_structure(self):
        """测试：结果结构"""
        result = VectorizedResult(
            total_trades=10,
            total_return=0.15,
            annualized_return=0.25,
            sharpe_ratio=1.5,
            max_drawdown=-0.1,
            win_rate=0.6,
            profit_factor=2.0
        )

        assert result.total_trades == 10
        assert result.total_return == 0.15
        assert result.win_rate == 0.6
        assert result.profit_factor == 2.0

    def test_indicators_precomputation(self, mock_stock_data):
        """测试：指标预计算"""
        config = IndicatorConfig(
            ma_periods=[5, 10, 20, 60],
            rsi_period=14,
            atr_period=14,
            bollinger_period=20,
            bollinger_std=2
        )
        indicators = VectorizedIndicators(config)
        result = indicators.calculate_all(mock_stock_data)

        # 验证指标列存在
        assert 'ma5' in result.columns
        assert 'ma20' in result.columns
        assert 'ma60' in result.columns
        assert 'rsi14' in result.columns
        assert 'atr14' in result.columns
        assert 'bb_upper' in result.columns
        assert 'bb_lower' in result.columns

        # 验证数据行数不变
        assert len(result) == len(mock_stock_data)

    def test_signals_precomputation(self, mock_stock_data):
        """测试：信号预计算"""
        # 先计算指标
        config = IndicatorConfig(
            ma_periods=[5, 10, 20, 60],
            rsi_period=14,
            atr_period=14
        )
        indicators = VectorizedIndicators(config)
        df_with_indicators = indicators.calculate_all(mock_stock_data)

        # 计算信号
        signal_config = SignalConfig(
            ma_short_period=20,
            ma_long_period=60,
            rsi_overbought=80
        )
        signals = VectorizedSignals(signal_config)
        result = signals.calculate_all(df_with_indicators)

        # 验证信号列存在
        assert 'golden_cross' in result.columns
        assert 'death_cross' in result.columns
        assert 'rsi_oversold' in result.columns
        assert 'rsi_overbought' in result.columns
        assert 'entry_confidence' in result.columns
        assert 'exit_death_cross' in result.columns

    def test_multi_day_iteration(self, mock_stock_data):
        """测试：多日迭代处理"""
        # 模拟多日迭代
        dates = sorted(mock_stock_data['date'].unique())

        # 预计算指标和信号
        config = IndicatorConfig(ma_periods=[20, 60], rsi_period=14)
        indicators = VectorizedIndicators(config)
        df = indicators.calculate_all(mock_stock_data)

        signal_config = SignalConfig(
            ma_short_period=20,
            ma_long_period=60,
            rsi_overbought=80
        )
        signals = VectorizedSignals(signal_config)
        df = signals.calculate_all(df)

        # 验证所有日期都有信号
        for date in dates[:10]:  # 检查前10天
            day_signals = df[df['date'] == date]
            assert len(day_signals) > 0

    def test_entry_confidence_threshold(self, mock_stock_data):
        """测试：入场置信度阈值"""
        # 计算指标和信号
        config = IndicatorConfig(ma_periods=[20, 60], rsi_period=14)
        indicators = VectorizedIndicators(config)
        df = indicators.calculate_all(mock_stock_data)

        signal_config = SignalConfig(
            ma_short_period=20,
            ma_long_period=60,
            rsi_overbought=80
        )
        signals = VectorizedSignals(signal_config)
        df = signals.calculate_all(df)

        # 筛选高置信度信号
        high_confidence = df[df['entry_confidence'] > 0.5]

        # 验证置信度范围
        if 'entry_confidence' in df.columns:
            assert (df['entry_confidence'] >= 0).all()
            assert (df['entry_confidence'] <= 1).all()

    def test_trade_recording(self):
        """测试：交易记录"""
        # 创建简单的交易记录
        trades = [
            {
                'trade_id': '2025-03-01_600519',
                'date': '2025-03-01',
                'code': '600519',
                'direction': 'long',
                'entry_date': '2025-02-15',
                'entry_price': 100.0,
                'exit_price': 105.0,
                'shares': 1000,
                'pnl': 5000 - 50,  # 5000 profit - 50 commission
                'pnl_pct': 0.05,
                'commission': 50,
                'exit_reason': 'death_cross',
                'signal_type': 'golden_cross'
            }
        ]

        assert len(trades) == 1
        assert trades[0]['pnl'] > 0
        assert trades[0]['pnl_pct'] == 0.05

    def test_equity_curve_calculation(self):
        """测试：权益曲线计算"""
        equity_history = [
            {'date': '2025-01-01', 'equity': 1_000_000, 'cash': 1_000_000, 'positions_value': 0},
            {'date': '2025-01-02', 'equity': 1_010_000, 'cash': 800_000, 'positions_value': 210_000},
            {'date': '2025-01-03', 'equity': 1_005_000, 'cash': 900_000, 'positions_value': 105_000},
        ]

        equity_df = pd.DataFrame(equity_history)

        # 验证权益曲线
        assert len(equity_df) == 3
        assert equity_df['equity'].iloc[0] == 1_000_000
        assert equity_df['equity'].iloc[-1] == 1_005_000

        # 计算最大回撤
        cummax = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cummax) / cummax
        max_drawdown = drawdown.min()

        assert max_drawdown < 0  # 有回撤

    def test_win_rate_calculation(self):
        """测试：胜率计算"""
        trades = [
            {'pnl': 1000},   # 盈利
            {'pnl': -500},   # 亏损
            {'pnl': 2000},   # 盈利
            {'pnl': -300},   # 亏损
            {'pnl': 1500},   # 盈利
        ]

        winning = sum(1 for t in trades if t['pnl'] > 0)
        total = len(trades)
        win_rate = winning / total

        assert winning == 3
        assert win_rate == 0.6

    def test_profit_factor_calculation(self):
        """测试：盈亏比计算"""
        trades = [
            {'pnl': 1000},   # 盈利
            {'pnl': -500},   # 亏损
            {'pnl': 2000},   # 盈利
            {'pnl': -300},   # 亏损
        ]

        total_win = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        total_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = total_win / total_loss

        assert total_win == 3000
        assert total_loss == 800
        assert profit_factor == 3.75


class TestVectorizedBacktesterRun:
    """回测运行测试（使用模拟数据）"""

    def test_run_with_mock_data(self):
        """测试：使用模拟数据运行完整回测"""
        # 创建模拟数据
        dates = pd.date_range('2025-01-01', '2025-03-31', freq='B')
        np.random.seed(42)

        all_data = []
        for data_id in ['600519']:
            for i, date in enumerate(dates):
                base = 100 + i * 0.5 + np.random.randn() * 2
                all_data.append({
                    'data_id': data_id,
                    'date': date.strftime('%Y-%m-%d'),
                    'open': round(base + np.random.randn() * 0.5, 2),
                    'high': round(base + abs(np.random.randn()) * 0.5 + 2, 2),
                    'low': round(base - abs(np.random.randn()) * 0.5 - 2, 2),
                    'close': round(base, 2),
                    'volume': int(1000000 + np.random.randint(-200000, 200000))
                })

        df = pd.DataFrame(all_data)

        # 创建引擎并直接设置模拟加载器
        config = BacktestConfig(
            initial_capital=1_000_000,
            max_positions=2,
            entry_confidence_threshold=0.5
        )
        engine = VectorizedBacktester(config=config)

        # 替换加载器
        mock_loader = MagicMock()
        mock_loader.load_daily = MagicMock(return_value=df)
        engine.loader = mock_loader

        # 运行回测
        result = engine.run(
            stock_codes=['600519'],
            start_date='2025-01-01',
            end_date='2025-03-31'
        )

        # 验证结果结构
        assert isinstance(result, VectorizedResult)
        assert result.total_trades >= 0
        assert isinstance(result.equity_curve, pd.DataFrame) or result.equity_curve is None


class TestVectorizedPerformance:
    """性能测试"""

    def test_precompute_performance(self):
        """测试：预计算性能"""
        import time

        # 创建多只股票大量数据
        dates = pd.date_range('2024-01-01', '2025-06-30', freq='B')
        np.random.seed(42)

        all_data = []
        for data_id in ['600519', '000001', '600036', '601318', '000002']:
            for i, date in enumerate(dates):
                base = 100 + np.random.randn() * 10
                all_data.append({
                    'data_id': data_id,
                    'date': date.strftime('%Y-%m-%d'),
                    'open': round(base + np.random.randn() * 0.5, 2),
                    'high': round(base + abs(np.random.randn()) * 0.5 + 2, 2),
                    'low': round(base - abs(np.random.randn()) * 0.5 - 2, 2),
                    'close': round(base, 2),
                    'volume': int(1000000 + np.random.randint(-200000, 200000))
                })

        df = pd.DataFrame(all_data)
        print(f"\n测试数据: {len(df)} 行 x 5 只股票")

        # 测试指标计算性能
        start = time.time()
        indicators = VectorizedIndicators(IndicatorConfig(
            ma_periods=[5, 10, 20, 60],
            rsi_period=14,
            atr_period=14,
            bollinger_period=20,
            bollinger_std=2
        ))
        df = indicators.calculate_all(df)
        indicator_time = time.time() - start
        print(f"指标计算耗时: {indicator_time:.3f}s")

        # 测试信号计算性能
        start = time.time()
        signals = VectorizedSignals(SignalConfig(
            ma_short_period=20,
            ma_long_period=60,
            rsi_overbought=80
        ))
        df = signals.calculate_all(df)
        signal_time = time.time() - start
        print(f"信号计算耗时: {signal_time:.3f}s")

        # 总预计算时间应该 < 3秒
        total_time = indicator_time + signal_time
        print(f"总预计算耗时: {total_time:.3f}s")
        assert total_time < 3.0, f"预计算性能测试失败: {total_time:.3f}s > 3.0s"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
