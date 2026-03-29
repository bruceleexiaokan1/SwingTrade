"""Walk-Forward Analysis Tests"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.backtest.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardWindow,
    WalkForwardResult,
)


class TestWalkForwardRatio:
    """Walk-Forward Ratio 计算测试"""

    def setup_method(self):
        """测试初始化"""
        self.analyzer = WalkForwardAnalyzer(
            backtester=None,
            train_window=756,
            test_window=126,
            step=63,
        )

    def test_wfr_calculation(self):
        """WFR 计算测试"""
        # 完美参数：样本外 = 样本内
        wfr = self.analyzer.calculate_wfr(oos_sharpe=1.0, is_sharpe=1.0)
        assert wfr == 1.0

        # 样本外是样本内的一半
        wfr = self.analyzer.calculate_wfr(oos_sharpe=0.5, is_sharpe=1.0)
        assert wfr == 0.5

        # 样本外是样本内的 60%
        wfr = self.analyzer.calculate_wfr(oos_sharpe=0.6, is_sharpe=1.0)
        assert wfr == 0.6

    def test_wfr_with_zero_in_sample(self):
        """样本内为0时的WFR计算"""
        wfr = self.analyzer.calculate_wfr(oos_sharpe=0.5, is_sharpe=0.0)
        assert wfr == 0.0

    def test_wfr_negative_values(self):
        """负夏普比率测试"""
        wfr = self.analyzer.calculate_wfr(oos_sharpe=-0.2, is_sharpe=0.5)
        assert wfr == -0.4

    def test_is_robust_threshold_06(self):
        """稳健阈值 0.6 测试"""
        assert self.analyzer.is_robust(0.7) == True
        assert self.analyzer.is_robust(0.6) == True
        assert self.analyzer.is_robust(0.59) == False

    def test_is_robust_custom_threshold(self):
        """自定义稳健阈值测试"""
        assert self.analyzer.is_robust(0.9, threshold=0.8) == True
        assert self.analyzer.is_robust(0.85, threshold=0.8) == True
        assert self.analyzer.is_robust(0.79, threshold=0.8) == False

    def test_get_very_robust(self):
        """非常稳健判断测试 (WFR > 0.8)"""
        assert self.analyzer.get_very_robust(0.9) == True
        assert self.analyzer.get_very_robust(0.8) == False
        assert self.analyzer.get_very_robust(0.79) == False


class TestWalkForwardWindow:
    """Walk-Forward Window 测试"""

    def test_window_creation(self):
        """窗口创建测试"""
        window = WalkForwardWindow(
            train_start="2020-01-01",
            train_end="2023-01-01",
            test_start="2023-01-02",
            test_end="2023-07-01",
            best_params={"ma_short": 20, "ma_long": 60},
            is_sharpe=1.0,
            oos_sharpe=0.7,
            wfr=0.7,
            is_robust=True,
            oos_return=0.15,
            oos_trades=10,
        )

        assert window.train_start == "2020-01-01"
        assert window.wfr == 0.7
        assert window.is_robust == True


class TestWalkForwardResult:
    """Walk-Forward Result 测试"""

    def test_robustness_ratio(self):
        """稳健窗口比例计算"""
        result = WalkForwardResult(
            total_windows=10,
            robust_windows=6,
            avg_wfr=0.65,
            avg_is_sharpe=1.0,
            avg_oos_sharpe=0.65,
        )

        assert result.robustness_ratio == 0.6

    def test_robustness_ratio_zero_windows(self):
        """零窗口时的稳健比例"""
        result = WalkForwardResult(
            total_windows=0,
            robust_windows=0,
            avg_wfr=0.0,
            avg_is_sharpe=0.0,
            avg_oos_sharpe=0.0,
        )

        assert result.robustness_ratio == 0.0

    def test_summary(self):
        """摘要生成测试"""
        result = WalkForwardResult(
            total_windows=5,
            robust_windows=4,
            avg_wfr=0.72,
            avg_is_sharpe=1.05,
            avg_oos_sharpe=0.756,
        )

        summary = result.summary()
        assert "5" in summary
        assert "4" in summary
        assert "0.72" in summary


class TestWalkForwardAnalyzerInit:
    """Walk-Forward Analyzer 初始化测试"""

    def test_default_parameters(self):
        """默认参数测试"""
        analyzer = WalkForwardAnalyzer(backtester=None)

        assert analyzer.train_window == 756  # 3年
        assert analyzer.test_window == 126    # 半年
        assert analyzer.step == 63            # 每月
        assert analyzer.robust_threshold == 0.6
        assert analyzer.very_robust_threshold == 0.8

    def test_custom_parameters(self):
        """自定义参数测试"""
        analyzer = WalkForwardAnalyzer(
            backtester=None,
            train_window=504,  # 2年
            test_window=252,   # 1年
            step=126,          # 半年
            robust_threshold=0.7,
        )

        assert analyzer.train_window == 504
        assert analyzer.test_window == 252
        assert analyzer.step == 126
        assert analyzer.robust_threshold == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
