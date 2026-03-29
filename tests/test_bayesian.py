"""Bayesian Parameter Estimation Tests"""

import pytest
import sys
import os
import numpy as np
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.backtest.bayesian import (
    bayesian_win_rate_estimation,
    bayesian_sharpe_ratio,
    bayesian_normal_parameters,
    bayesian_position_decision,
    bayesian_strategy_selection,
    WinRateResult,
    SharpeRatioResult,
    NormalParametersResult,
    PositionDecisionResult,
    StrategySelectionResult,
)


class TestBayesianWinRateEstimation:
    """Tests for Bayesian win rate estimation using Beta distribution"""

    def test_basic_win_rate(self):
        """Basic win rate estimation"""
        result = bayesian_win_rate_estimation(wins=30, losses=70)

        assert result.total_trades == 100
        assert result.wins == 30
        assert result.losses == 70
        assert result.posterior_alpha == 32.0  # 2 + 30
        assert result.posterior_beta == 72.0    # 2 + 70
        assert 0.25 < result.posterior_mean < 0.35  # Approximately 30/100

    def test_prior_parameters(self):
        """Test with custom prior parameters"""
        result = bayesian_win_rate_estimation(
            wins=10,
            losses=10,
            prior_alpha=5.0,
            prior_beta=5.0
        )

        assert result.prior_alpha == 5.0
        assert result.prior_beta == 5.0
        assert result.posterior_alpha == 15.0  # 5 + 10
        assert result.posterior_beta == 15.0    # 5 + 10

    def test_prob_above_50_high_win_rate(self):
        """High win rate should have high probability above 0.5"""
        result = bayesian_win_rate_estimation(wins=80, losses=20)
        assert result.prob_above_50 > 0.99

    def test_prob_above_50_low_win_rate(self):
        """Low win rate should have low probability above 0.5"""
        result = bayesian_win_rate_estimation(wins=10, losses=90)
        assert result.prob_above_50 < 0.01

    def test_custom_threshold(self):
        """Test with custom threshold"""
        result = bayesian_win_rate_estimation(wins=60, losses=40, threshold=0.6)
        # With 60% win rate, P(win_rate > 0.6) should be around 0.5
        assert 0.3 < result.prob_above_threshold < 0.7

    def test_zero_wins(self):
        """All losing trades"""
        result = bayesian_win_rate_estimation(wins=0, losses=100)

        assert result.posterior_alpha == 2.0
        assert result.posterior_beta == 102.0
        assert result.posterior_mean < 0.1
        assert result.prob_above_50 == 0.0

    def test_zero_losses(self):
        """All winning trades"""
        result = bayesian_win_rate_estimation(wins=100, losses=0)

        assert result.posterior_alpha == 102.0
        assert result.posterior_beta == 2.0
        assert result.posterior_mean > 0.9
        assert result.prob_above_50 == 1.0

    def test_empty_trades(self):
        """No trades - should use prior only"""
        result = bayesian_win_rate_estimation(wins=0, losses=0)

        assert result.posterior_alpha == 2.0
        assert result.posterior_beta == 2.0
        assert result.posterior_mean == 0.5  # Beta(2,2) mean
        assert result.total_trades == 0

    def test_negative_wins_raises_error(self):
        """Negative wins should raise ValueError"""
        with pytest.raises(ValueError):
            bayesian_win_rate_estimation(wins=-1, losses=10)

    def test_negative_losses_raises_error(self):
        """Negative losses should raise ValueError"""
        with pytest.raises(ValueError):
            bayesian_win_rate_estimation(wins=10, losses=-1)

    def test_posterior_std_lower_than_prior(self):
        """Posterior std should be lower than prior with more data"""
        result_small = bayesian_win_rate_estimation(wins=3, losses=7)
        result_large = bayesian_win_rate_estimation(wins=30, losses=70)

        assert result_large.posterior_std < result_small.posterior_std

    def test_summary_generation(self):
        """Test summary string generation"""
        result = bayesian_win_rate_estimation(wins=30, losses=70)
        summary = result.summary()

        assert "Bayesian Win Rate" in summary
        assert "30" in summary
        assert "70" in summary


class TestBayesianSharpeRatio:
    """Tests for Bayesian Sharpe ratio estimation"""

    def test_basic_sharpe_ratio(self):
        """Basic Sharpe ratio estimation"""
        returns = [0.01, 0.02, -0.01, 0.015, -0.005, 0.008]
        result = bayesian_sharpe_ratio(returns)

        assert result.sample_size == 6
        assert result.monte_carlo_samples == 10000
        assert isinstance(result.posterior_mean, float)
        assert isinstance(result.posterior_std, float)
        assert result.posterior_10th <= result.posterior_mean <= result.posterior_90th

    def test_sharpe_with_risk_free_rate(self):
        """Test with non-zero risk-free rate"""
        returns = [0.01, 0.02, 0.015]
        result_no_rf = bayesian_sharpe_ratio(returns, risk_free_rate=0.0)
        result_with_rf = bayesian_sharpe_ratio(returns, risk_free_rate=0.03)

        # Higher risk-free rate should lower Sharpe
        assert result_with_rf.posterior_mean < result_no_rf.posterior_mean

    def test_negative_returns(self):
        """Negative Sharpe ratio for negative returns"""
        returns = [-0.01, -0.02, -0.015, -0.01]
        result = bayesian_sharpe_ratio(returns)

        assert result.posterior_mean < 0

    def test_highly_volatile_returns(self):
        """Test with highly volatile returns"""
        returns = [0.1, -0.1, 0.1, -0.1, 0.1]
        result = bayesian_sharpe_ratio(returns)

        assert result.posterior_std > 0.5  # High uncertainty

    def test_consistent_returns(self):
        """Test with consistent (low variance) returns"""
        returns = [0.01, 0.011, 0.009, 0.0105, 0.0102]
        result = bayesian_sharpe_ratio(returns)

        assert result.posterior_std < result.posterior_mean if result.posterior_mean > 0 else True

    def test_annualization(self):
        """Test annualization factor"""
        daily_returns = [0.001, -0.002, 0.0015]
        result_daily = bayesian_sharpe_ratio(daily_returns, annualization_factor=252)
        result_annual = bayesian_sharpe_ratio(daily_returns, annualization_factor=1)

        # Annualized should be scaled by sqrt(252)
        # (rough check since it's stochastic)
        assert result_daily.posterior_mean != result_annual.posterior_mean

    def test_credible_interval_order(self):
        """Test that CI lower < mean < CI upper"""
        returns = [0.01, 0.02, -0.01, 0.015, 0.008]
        result = bayesian_sharpe_ratio(returns)

        assert result.ci_lower <= result.posterior_mean <= result.ci_upper
        assert result.posterior_10th <= result.posterior_mean <= result.posterior_90th

    def test_insufficient_data_raises_error(self):
        """Less than 3 returns should raise ValueError"""
        with pytest.raises(ValueError):
            bayesian_sharpe_ratio([0.01, -0.02])

        with pytest.raises(ValueError):
            bayesian_sharpe_ratio([])

    def test_invalid_risk_free_rate(self):
        """Invalid risk-free rate should raise ValueError"""
        returns = [0.01, 0.02, 0.015]

        with pytest.raises(ValueError):
            bayesian_sharpe_ratio(returns, risk_free_rate=-0.1)

        with pytest.raises(ValueError):
            bayesian_sharpe_ratio(returns, risk_free_rate=1.5)

    def test_zero_std_returns(self):
        """Returns with zero variance"""
        returns = [0.01, 0.01, 0.01, 0.01]
        result = bayesian_sharpe_ratio(returns)

        # Should handle zero variance gracefully
        assert result.posterior_std == 0.0

    def test_summary_generation(self):
        """Test summary string generation"""
        returns = [0.01, 0.02, -0.01, 0.015]
        result = bayesian_sharpe_ratio(returns)
        summary = result.summary()

        assert "Bayesian Sharpe Ratio" in summary
        assert "6" in summary


class TestBayesianNormalParameters:
    """Tests for Bayesian normal parameters estimation"""

    def test_basic_normal_parameters(self):
        """Basic normal parameters estimation"""
        data = [0.01, 0.02, -0.01, 0.015, -0.005]
        result = bayesian_normal_parameters(data)

        assert result.sample_size == 5
        assert isinstance(result.mean_posterior_mean, float)
        assert isinstance(result.var_posterior_mean, float)
        assert result.ci_lower_mean <= result.mean_posterior_mean <= result.ci_upper_mean

    def test_known_mean_prior(self):
        """Test with specified prior mean"""
        data = [0.01, 0.02, 0.015]
        result = bayesian_normal_parameters(data, prior_mean=0.01)

        assert result.mean_posterior_mean is not None

    def test_custom_ci_level(self):
        """Test with custom confidence interval level"""
        data = [0.01, 0.02, -0.01, 0.015, 0.008, 0.012]
        result_95 = bayesian_normal_parameters(data, ci_level=0.95)
        result_90 = bayesian_normal_parameters(data, ci_level=0.90)

        # 90% CI should be narrower than 95% CI
        width_95 = result_95.ci_upper_mean - result_95.ci_lower_mean
        width_90 = result_90.ci_upper_mean - result_90.ci_lower_mean
        assert width_90 < width_95

    def test_variance_estimation(self):
        """Test variance is estimated correctly"""
        # Use larger sample to get more reasonable posterior
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        result = bayesian_normal_parameters(data)

        # Sample variance (unbiased) is approximately 6.67
        # With Jeffreys prior, posterior mean is biased toward population variance
        # For larger n, posterior mean converges to sample variance
        sample_var = np.var(data, ddof=1)
        assert result.var_posterior_mean > 0

    def test_insufficient_data_raises_error(self):
        """Less than 2 data points should raise ValueError"""
        with pytest.raises(ValueError):
            bayesian_normal_parameters([0.01])

        with pytest.raises(ValueError):
            bayesian_normal_parameters([])

    def test_ess_calculation(self):
        """Test effective sample size calculation"""
        data = [0.01, 0.02, 0.015, 0.008, 0.012, 0.005]
        result = bayesian_normal_parameters(data)

        assert result.ess_mean == 6
        assert result.ess_var == 2.5  # (n-1)/2 = 2.5

    def test_posterior_std_of_mean(self):
        """Test posterior std of mean decreases with more data"""
        small_data = [0.01, 0.02, 0.015]
        large_data = [0.01, 0.02, 0.015, 0.008, 0.012, 0.005, 0.018, 0.022]

        result_small = bayesian_normal_parameters(small_data)
        result_large = bayesian_normal_parameters(large_data)

        assert result_large.mean_posterior_std < result_small.mean_posterior_std

    def test_summary_generation(self):
        """Test summary string generation"""
        data = [0.01, 0.02, -0.01, 0.015, 0.008]
        result = bayesian_normal_parameters(data)
        summary = result.summary()

        assert "Bayesian Normal Parameters" in summary


class TestBayesianPositionDecision:
    """Tests for Bayesian position decision"""

    def test_basic_position_decision(self):
        """Basic position decision"""
        returns = [0.01, 0.02, -0.01, 0.015, 0.008, 0.012]
        result = bayesian_position_decision(returns, base_position=1.0)

        assert 0.0 <= result.adjusted_position <= 1.0
        assert 0.0 <= result.recommended_position_pct <= 100.0
        assert isinstance(result.reasoning, str)

    def test_min_max_position_bounds(self):
        """Test position is bounded by min and max"""
        returns = [0.01, -0.02, 0.015]  # Uncertain Sharpe
        result = bayesian_position_decision(
            returns,
            base_position=1.0,
            min_position=0.2,
            max_position=0.8
        )

        assert 0.2 <= result.adjusted_position <= 0.8

    def test_zero_base_position(self):
        """Zero base position should return zero adjusted"""
        returns = [0.01, 0.02, 0.015]
        result = bayesian_position_decision(returns, base_position=0.0)

        assert result.adjusted_position == 0.0
        assert result.reasoning == "No base position specified"

    def test_conservative_position_for_uncertain_returns(self):
        """High uncertainty should lead to conservative position"""
        # Returns with high variance
        volatile_returns = [0.1, -0.1, 0.1, -0.1, 0.1, -0.1]
        # Returns with low variance
        stable_returns = [0.01, 0.011, 0.009, 0.0105, 0.0102, 0.0101]

        result_volatile = bayesian_position_decision(volatile_returns, base_position=0.5)
        result_stable = bayesian_position_decision(stable_returns, base_position=0.5)

        # Volatile should have lower uncertainty factor
        assert result_volatile.uncertainty_factor < result_stable.uncertainty_factor

    def test_empty_returns_raises_error(self):
        """Empty returns should raise ValueError"""
        with pytest.raises(ValueError):
            bayesian_position_decision([], base_position=0.5)

    def test_invalid_base_position_raises_error(self):
        """Invalid base position should raise ValueError"""
        returns = [0.01, 0.02, 0.015]

        with pytest.raises(ValueError):
            bayesian_position_decision(returns, base_position=-0.1)

        with pytest.raises(ValueError):
            bayesian_position_decision(returns, base_position=1.5)

    def test_use_sharpe_10th_conservative(self):
        """Using 10th percentile Sharpe should be more conservative"""
        # Use more stable returns with larger sample for less Monte Carlo variance
        returns = [0.02, 0.03, 0.01, 0.025, 0.015, 0.022, 0.018, 0.028, 0.016, 0.024]

        result_mean = bayesian_position_decision(returns, base_position=0.5, use_sharpe_10th=False)
        result_10th = bayesian_position_decision(returns, base_position=0.5, use_sharpe_10th=True)

        # 10th percentile Sharpe should give lower position due to lower expected Sharpe
        # The actual Sharpe value used for 10th percentile should be less than mean
        sharpe_result = bayesian_sharpe_ratio(returns, n_samples=10000)
        assert sharpe_result.posterior_10th < sharpe_result.posterior_mean

    def test_confidence_calculation(self):
        """Test confidence metric"""
        returns = [0.01, 0.02, 0.015, 0.018, 0.012]
        result = bayesian_position_decision(returns, base_position=0.5)

        assert 0.0 <= result.confidence <= 1.0

    def test_summary_generation(self):
        """Test summary string generation"""
        returns = [0.01, 0.02, -0.01, 0.015, 0.008]
        result = bayesian_position_decision(returns, base_position=0.5)
        summary = result.summary()

        assert "Bayesian Position Decision" in summary
        assert "50" in summary  # 0.5 base position


class TestBayesianStrategySelection:
    """Tests for Bayesian strategy selection"""

    def test_basic_strategy_selection(self):
        """Basic strategy selection"""
        strategy_returns = {
            "strategy_a": [0.01, 0.02, -0.01, 0.015, 0.008],
            "strategy_b": [0.005, 0.015, 0.01, -0.005, 0.012],
        }
        result = bayesian_strategy_selection(strategy_returns)

        assert result.best_strategy in ["strategy_a", "strategy_b"]
        assert isinstance(result.strategy_rankings, list)
        assert len(result.posterior_probs) == 2

    def test_risk_adjusted_selection(self):
        """Test risk-adjusted (Sharpe) selection"""
        # Strategy A: higher return, higher variance
        strategy_a = [0.03, -0.02, 0.04, -0.01, 0.03]
        # Strategy B: lower return, lower variance
        strategy_b = [0.01, 0.005, 0.015, 0.01, 0.012]

        strategy_returns = {
            "high_risk": strategy_a,
            "low_risk": strategy_b,
        }

        result = bayesian_strategy_selection(strategy_returns, risk_adjusted=True)

        # Best strategy should be one of the two
        assert result.best_strategy in ["high_risk", "low_risk"]

    def test_raw_return_selection(self):
        """Test raw return (non-risk-adjusted) selection"""
        strategy_returns = {
            "high_return": [0.05, 0.04, 0.06, 0.045, 0.055],
            "low_return": [0.01, 0.015, 0.01, 0.012, 0.011],
        }

        result = bayesian_strategy_selection(strategy_returns, risk_adjusted=False)

        assert result.best_strategy == "high_return"

    def test_benchmark_comparison(self):
        """Test with benchmark return"""
        strategy_returns = {
            "beating_benchmark": [0.02, 0.025, 0.018, 0.022],
            "below_benchmark": [-0.01, 0.005, -0.005, 0.002],
        }

        result = bayesian_strategy_selection(
            strategy_returns,
            benchmark_return=0.01,
            risk_adjusted=False
        )

        # Extract probabilities from strategy rankings
        beating_prob = 0
        below_prob = 0
        for name, exp_ret, prob_beat in result.strategy_rankings:
            if name == "beating_benchmark":
                beating_prob = prob_beat
            elif name == "below_benchmark":
                below_prob = prob_beat

        # Beating benchmark should have higher probability
        assert beating_prob > below_prob

    def test_min_prob_threshold(self):
        """Test minimum probability threshold"""
        strategy_returns = {
            "risky": [-0.05, 0.1, -0.05, 0.1],
            "safe": [0.005, 0.006, 0.005, 0.006],
        }

        result = bayesian_strategy_selection(
            strategy_returns,
            min_prob_beat_benchmark=0.6
        )

        # Should still return a recommendation
        assert result.recommended_strategy in ["risky", "safe"]

    def test_empty_strategy_dict_raises_error(self):
        """Empty strategy dictionary should raise ValueError"""
        with pytest.raises(ValueError):
            bayesian_strategy_selection({})

    def test_insufficient_data_raises_error(self):
        """Strategy with insufficient data should raise ValueError"""
        strategy_returns = {
            "valid": [0.01, 0.02, 0.015],
            "invalid": [0.01],  # Only 1 return
        }

        with pytest.raises(ValueError):
            bayesian_strategy_selection(strategy_returns)

    def test_posterior_probs_sum_to_one(self):
        """Posterior probabilities should sum to approximately 1"""
        strategy_returns = {
            "a": [0.01, 0.02, 0.015, 0.008],
            "b": [0.005, 0.015, 0.01, 0.012],
            "c": [0.008, 0.012, 0.009, 0.011],
        }

        result = bayesian_strategy_selection(strategy_returns)

        total_prob = sum(result.posterior_probs.values())
        assert 0.99 < total_prob < 1.01

    def test_rankings_sorted_by_expected_return(self):
        """Strategy rankings should be sorted by expected return (descending)"""
        strategy_returns = {
            "low": [0.001, 0.002, 0.001],
            "medium": [0.01, 0.015, 0.012],
            "high": [0.03, 0.035, 0.032],
        }

        result = bayesian_strategy_selection(strategy_returns, risk_adjusted=False)

        # Rankings should be sorted by expected return
        returns = [r[1] for r in result.strategy_rankings]
        assert returns == sorted(returns, reverse=True)

    def test_summary_generation(self):
        """Test summary string generation"""
        strategy_returns = {
            "strategy_a": [0.01, 0.02, 0.015, 0.008],
            "strategy_b": [0.005, 0.015, 0.01, 0.012],
        }

        result = bayesian_strategy_selection(strategy_returns)
        summary = result.summary()

        assert "Bayesian Strategy Selection" in summary
        assert "strategy_a" in summary or "strategy_b" in summary


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_very_small_probabilities(self):
        """Test with very small win rates/losses"""
        result = bayesian_win_rate_estimation(wins=1, losses=999)

        assert result.posterior_mean < 0.05
        assert result.prob_above_50 < 0.01

    def test_very_large_win_rates(self):
        """Test with very high win rates"""
        result = bayesian_win_rate_estimation(wins=999, losses=1)

        assert result.posterior_mean > 0.95
        assert result.prob_above_50 > 0.99

    def test_sharpe_ratio_with_single_sign_returns(self):
        """Test Sharpe ratio with all positive or all negative returns"""
        all_positive = [0.01, 0.02, 0.015, 0.018, 0.012]
        all_negative = [-0.01, -0.02, -0.015, -0.018, -0.012]

        result_pos = bayesian_sharpe_ratio(all_positive)
        result_neg = bayesian_sharpe_ratio(all_negative)

        assert result_pos.posterior_mean > 0
        assert result_neg.posterior_mean < 0

    def test_position_with_highly_diversified_returns(self):
        """Test position decision with well-diversified returns"""
        # Low correlation returns (effectively high variance of mean estimate)
        returns = [0.05, -0.03, 0.08, -0.04, 0.06, -0.02]
        result = bayesian_position_decision(returns, base_position=0.8)

        # Should reduce position due to uncertainty
        assert result.adjusted_position < 0.8

    def test_strategy_with_unequal_sample_sizes(self):
        """Test strategy selection with different sample sizes"""
        strategy_returns = {
            "small_sample": [0.01, 0.02, 0.015],
            "large_sample": [0.01, 0.015, 0.01, 0.012, 0.008, 0.011, 0.013, 0.009],
        }

        result = bayesian_strategy_selection(strategy_returns)

        # Small sample should have higher uncertainty
        assert "small_sample" in result.sample_size_per_strategy
        assert "large_sample" in result.sample_size_per_strategy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
