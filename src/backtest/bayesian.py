"""Bayesian Parameter Estimation Module

Bayesian estimation for trading strategy parameters:
- Win rate estimation using Beta distribution
- Sharpe ratio estimation using t-distribution with Jeffreys prior
- Normal parameters (mean and variance) joint estimation
- Position sizing based on uncertainty
- Strategy selection via Bayesian model comparison

Key Formulas:
    Posterior ∝ Likelihood × Prior

    Win Rate (Beta Distribution):
        Prior: Beta(α=2, β=2)
        Posterior: Beta(α+wins, β=losses)
        P(win_rate > 0.5) = 1 - posterior.cdf(0.5)

    Sharpe Ratio:
        Jeffreys prior for normal likelihood
        Monte Carlo sampling for posterior
        Return 10th percentile as conservative position size
"""

from __future__ import annotations

import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable

logger = logging.getLogger(__name__)


# Type alias for returns array
ReturnsArray = List[float]


@dataclass
class WinRateResult:
    """Bayesian win rate estimation result"""
    prior_alpha: float           # Prior alpha parameter
    prior_beta: float            # Prior beta parameter
    posterior_alpha: float       # Posterior alpha
    posterior_beta: float        # Posterior beta
    posterior_mean: float        # Posterior mean (mode if mode exists)
    posterior_std: float         # Posterior standard deviation
    prob_above_50: float         # P(win_rate > 0.5)
    prob_above_threshold: float  # P(win_rate > threshold)
    wins: int                    # Number of winning trades
    losses: int                  # Number of losing trades
    total_trades: int            # Total trades

    def summary(self) -> str:
        """Generate summary string"""
        return (
            f"=== Bayesian Win Rate ===\n"
            f"Trades: {self.total_trades} (Wins: {self.wins}, Losses: {self.losses})\n"
            f"Prior: Beta({self.prior_alpha:.1f}, {self.prior_beta:.1f})\n"
            f"Posterior: Beta({self.posterior_alpha:.1f}, {self.posterior_beta:.1f})\n"
            f"Posterior Mean: {self.posterior_mean:.4f}\n"
            f"Posterior Std: {self.posterior_std:.4f}\n"
            f"P(win_rate > 0.5): {self.prob_above_50:.4f}\n"
            f"P(win_rate > threshold): {self.prob_above_threshold:.4f}"
        )


@dataclass
class SharpeRatioResult:
    """Bayesian Sharpe ratio estimation result"""
    posterior_mean: float        # Posterior mean of Sharpe ratio
    posterior_std: float         # Posterior standard deviation
    posterior_10th: float        # 10th percentile (conservative)
    posterior_90th: float        # 90th percentile
    ci_lower: float              # 95% credible interval lower bound
    ci_upper: float              # 95% credible interval upper bound
    sample_size: int             # Number of returns used
    monte_carlo_samples: int     # Number of MC samples

    def summary(self) -> str:
        """Generate summary string"""
        return (
            f"=== Bayesian Sharpe Ratio ===\n"
            f"Sample Size: {self.sample_size}\n"
            f"MC Samples: {self.monte_carlo_samples}\n"
            f"Posterior Mean: {self.posterior_mean:.4f}\n"
            f"Posterior Std: {self.posterior_std:.4f}\n"
            f"10th Percentile: {self.posterior_10th:.4f}\n"
            f"90th Percentile: {self.posterior_90th:.4f}\n"
            f"95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        )


@dataclass
class NormalParametersResult:
    """Bayesian normal parameters estimation result"""
    mean_posterior_mean: float   # Posterior mean of mean
    mean_posterior_std: float    # Posterior std of mean
    var_posterior_mean: float    # Posterior mean of variance
    var_posterior_std: float     # Posterior std of variance
    std_posterior_mean: float    # Posterior mean of std
    ci_lower_mean: float         # 95% CI lower for mean
    ci_upper_mean: float         # 95% CI upper for mean
    sample_size: int             # Number of samples
    ess_mean: float              # Effective sample size for mean
    ess_var: float               # Effective sample size for variance

    def summary(self) -> str:
        """Generate summary string"""
        return (
            f"=== Bayesian Normal Parameters ===\n"
            f"Sample Size: {self.sample_size}\n"
            f"Mean: {self.mean_posterior_mean:.6f} ± {self.mean_posterior_std:.6f}\n"
            f"Std: {self.std_posterior_mean:.6f} ± {self.var_posterior_std:.6f}\n"
            f"95% CI for Mean: [{self.ci_lower_mean:.6f}, {self.ci_upper_mean:.6f}]"
        )


@dataclass
class PositionDecisionResult:
    """Bayesian position decision result"""
    base_position: float         # Base position size (0-1)
    adjusted_position: float     # Uncertainty-adjusted position
    uncertainty_factor: float     # Uncertainty multiplier
    confidence: float             # Confidence in the estimate
    recommended_position_pct: float  # Recommended position as percentage
    reasoning: str                # Human-readable reasoning

    def summary(self) -> str:
        """Generate summary string"""
        return (
            f"=== Bayesian Position Decision ===\n"
            f"Base Position: {self.base_position:.2%}\n"
            f"Adjusted Position: {self.adjusted_position:.2%}\n"
            f"Uncertainty Factor: {self.uncertainty_factor:.4f}\n"
            f"Confidence: {self.confidence:.4f}\n"
            f"Recommended: {self.recommended_position_pct:.2%}\n"
            f"Reasoning: {self.reasoning}"
        )


@dataclass
class StrategySelectionResult:
    """Bayesian strategy selection result"""
    best_strategy: str           # Name of best strategy
    best_expected_return: float   # Expected return of best strategy
    best_prob_beat_benchmark: float  # P(outperform benchmark)
    strategy_rankings: List[Tuple[str, float, float]]  # (name, expected_return, prob_beat_benchmark)
    posterior_probs: Dict[str, float]  # Posterior probability for each strategy
    recommended_strategy: str     # Recommended strategy based on risk-adjusted returns
    sample_size_per_strategy: Dict[str, int]  # Sample sizes

    def summary(self) -> str:
        """Generate summary string"""
        lines = [
            f"=== Bayesian Strategy Selection ===",
            f"Best Strategy: {self.best_strategy}",
            f"Recommended: {self.recommended_strategy}",
            f"",
            f"Rankings (Expected Return, P(beat benchmark)):",
        ]
        for name, exp_ret, prob in self.strategy_rankings:
            lines.append(f"  {name}: {exp_ret:.4f}, {prob:.4f}")

        return "\n".join(lines)


def bayesian_win_rate_estimation(
    wins: int,
    losses: int,
    prior_alpha: float = 2.0,
    prior_beta: float = 2.0,
    threshold: float = 0.5,
) -> WinRateResult:
    """
    Bayesian win rate estimation using Beta distribution.

    Uses a Beta prior (default Beta(2,2) which is weakly informative)
    and updates to posterior Beta(prior_alpha + wins, prior_beta + losses).

    Args:
        wins: Number of winning trades
        losses: Number of losing trades
        prior_alpha: Prior alpha parameter (default 2.0)
        prior_beta: Prior beta parameter (default 2.0)
        threshold: Threshold for probability calculation (default 0.5)

    Returns:
        WinRateResult with posterior parameters and probabilities

    Raises:
        ValueError: If wins or losses are negative

    Example:
        >>> result = bayesian_win_rate_estimation(30, 70)
        >>> print(f"P(win_rate > 0.5) = {result.prob_above_50:.4f}")
    """
    if wins < 0 or losses < 0:
        raise ValueError("wins and losses must be non-negative")

    total_trades = wins + losses

    # Posterior parameters
    posterior_alpha = prior_alpha + wins
    posterior_beta = prior_beta + losses

    # Posterior mean and std for Beta distribution
    # Mean = alpha / (alpha + beta)
    # Std = sqrt(alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1)))
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    denominator = (posterior_alpha + posterior_beta) ** 2 * (posterior_alpha + posterior_beta + 1)
    posterior_std = np.sqrt(posterior_alpha * posterior_beta / denominator) if denominator > 0 else 0.0

    # Calculate P(win_rate > threshold) = 1 - CDF(threshold)
    # Using scipy's beta CDF is more accurate, but we can approximate
    # For Beta distribution: P(X > x) = 1 - I_x(alpha, beta) where I is regularized incomplete beta
    try:
        from scipy import stats
        prob_above_threshold = 1 - stats.beta.cdf(threshold, posterior_alpha, posterior_beta)
        prob_above_50 = 1 - stats.beta.cdf(0.5, posterior_alpha, posterior_beta)
    except ImportError:
        # Fallback approximation using Monte Carlo if scipy not available
        logger.warning("scipy not available, using Monte Carlo approximation")
        samples = np.random.beta(posterior_alpha, posterior_beta, size=10000)
        prob_above_threshold = np.mean(samples > threshold)
        prob_above_50 = np.mean(samples > 0.5)

    return WinRateResult(
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        posterior_alpha=posterior_alpha,
        posterior_beta=posterior_beta,
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        prob_above_50=prob_above_50,
        prob_above_threshold=prob_above_threshold,
        wins=wins,
        losses=losses,
        total_trades=total_trades,
    )


def bayesian_sharpe_ratio(
    returns: ReturnsArray,
    risk_free_rate: float = 0.0,
    n_samples: int = 10000,
    annualization_factor: float = 252.0,
) -> SharpeRatioResult:
    """
    Bayesian Sharpe ratio estimation using Jeffreys prior.

    Uses Jeffreys prior for normal likelihood and Monte Carlo sampling
    to compute the posterior distribution of the Sharpe ratio.

    Args:
        returns: List or array of historical returns
        risk_free_rate: Risk-free rate (default 0.0)
        n_samples: Number of Monte Carlo samples (default 10000)
        annualization_factor: Factor to annualize returns (default 252 for daily)

    Returns:
        SharpeRatioResult with posterior statistics

    Raises:
        ValueError: If returns has fewer than 3 elements
        ValueError: If risk_free_rate is not between 0 and 1

    Example:
        >>> returns = [0.01, -0.02, 0.03, 0.015, -0.01]
        >>> result = bayesian_sharpe_ratio(returns)
        >>> print(f"Conservative Sharpe (10th): {result.posterior_10th:.4f}")
    """
    if len(returns) < 3:
        raise ValueError("Need at least 3 returns for Bayesian Sharpe estimation")

    if not 0 <= risk_free_rate <= 1:
        raise ValueError("risk_free_rate should be between 0 and 1")

    returns = np.array(returns)
    n = len(returns)

    # Excess returns
    excess_returns = returns - risk_free_rate / annualization_factor

    # Sample statistics
    sample_mean = np.mean(excess_returns)
    sample_var = np.var(excess_returns, ddof=1)
    sample_std = np.sqrt(sample_var)

    if sample_std == 0:
        logger.warning("Sample std is zero, returning zeros")
        return SharpeRatioResult(
            posterior_mean=0.0,
            posterior_std=0.0,
            posterior_10th=0.0,
            posterior_90th=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            sample_size=n,
            monte_carlo_samples=n_samples,
        )

    # Jeffreys prior for normal likelihood:
    # Prior on variance: p(sigma^2) ∝ 1/sigma^2 (Scale-invariant)
    # This corresponds to Inverse-Chi-squared with 0 degrees of freedom
    # Posterior: sigma^2 | data ~ Inverse-Chi-squared(df=n-1, scale^2=(n-1)*s^2/(n-1))
    # which simplifies to: sigma^2 | data ~ Inverse-Gamma((n-1)/2, (n-1)*s^2/2)

    # For mean with unknown variance:
    # Posterior mean ~ t-distributed with n-1 degrees of freedom
    # Posterior precision ~ Gamma((n-1)/2, (n-1)*s^2/2)

    # Monte Carlo sampling from posterior
    # Sample from posterior variance
    df = n - 1  # degrees of freedom
    scale_sq = df * sample_var  # scale parameter for inverse-chi-squared

    # Sample variance from Inverse-Chi-squared
    # X ~ Inv-Chi2(df, scale) ==> X = df * scale / Y where Y ~ Chi2(df)
    chi2_samples = np.random.chisquare(df, size=n_samples)
    variance_samples = df * scale_sq / chi2_samples
    std_samples = np.sqrt(variance_samples)

    # Sample mean from t-distribution: mean | data, var ~ t(n-1, x_bar, s/sqrt(n))
    # Equivalent: mean ~ Normal(x_bar, s/sqrt(n)) with variance sampled from posterior
    std_error = sample_std / np.sqrt(n)
    mean_samples = np.random.normal(sample_mean, std_error, size=n_samples)

    # Sharpe ratio = mean / std (for the period)
    sharpe_samples = mean_samples / std_samples

    # Annualize
    sharpe_samples_annualized = sharpe_samples * np.sqrt(annualization_factor)

    # Posterior statistics
    posterior_mean = np.mean(sharpe_samples_annualized)
    posterior_std = np.std(sharpe_samples_annualized)
    posterior_10th = np.percentile(sharpe_samples_annualized, 10)
    posterior_90th = np.percentile(sharpe_samples_annualized, 90)
    ci_lower = np.percentile(sharpe_samples_annualized, 2.5)
    ci_upper = np.percentile(sharpe_samples_annualized, 97.5)

    return SharpeRatioResult(
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        posterior_10th=posterior_10th,
        posterior_90th=posterior_90th,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        sample_size=n,
        monte_carlo_samples=n_samples,
    )


def _t_critical_value(df: int, ci_level: float) -> float:
    """
    Calculate t critical value for confidence interval.

    Uses scipy if available, otherwise approximates using normal distribution.

    Args:
        df: Degrees of freedom
        ci_level: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        t critical value
    """
    # Quantile corresponding to (1 + ci_level) / 2
    # e.g., for 95% CI: (1 + 0.95) / 2 = 0.975
    quantile = (1 + ci_level) / 2

    try:
        from scipy import stats as scipy_stats
        return scipy_stats.t.ppf(quantile, df)
    except ImportError:
        # Approximate using normal distribution for large df
        # For df > 30, t approaches normal
        if df > 30:
            import scipy.stats as norm_stats
            return norm_stats.norm.ppf(quantile)
        # For small df, use a rough approximation based on quantile
        # Using Wilson-Hilferty transformation or simple scaling
        # z for 95% is 1.96, for 90% is 1.645
        z_for_95 = 1.96
        z_quantile = z_for_95 * np.sqrt(quantile / 0.975)
        return z_quantile * (1 + 1 / (4 * df))


def _normal_cdf(x: float, mean: float, std: float) -> float:
    """
    Calculate normal CDF.

    Uses scipy if available, otherwise uses math.erf.

    Args:
        x: Value to evaluate
        mean: Distribution mean
        std: Distribution standard deviation

    Returns:
        CDF value
    """
    try:
        from scipy import stats as scipy_stats
        return scipy_stats.norm.cdf(x, loc=mean, scale=std)
    except ImportError:
        import math
        return 0.5 * (1 + math.erf((x - mean) / (std * np.sqrt(2))))


def bayesian_normal_parameters(
    data: ReturnsArray,
    prior_mean: Optional[float] = None,
    prior_var_alpha: float = 0.01,  # Jeffreys prior for variance
    prior_var_beta: float = 0.01,
    ci_level: float = 0.95,
) -> NormalParametersResult:
    """
    Bayesian estimation of normal distribution parameters (mean and variance).

    Uses a normal-inverse-gamma conjugate prior (or Jeffreys prior for
    variance with a flat prior on mean).

    Args:
        data: List or array of observations
        prior_mean: Prior mean (if None, use data mean with weak prior)
        prior_var_alpha: Prior alpha for inverse-gamma on variance
        prior_var_beta: Prior beta for inverse-gamma on variance
        ci_level: Credible interval level (default 0.95 for 95% CI)

    Returns:
        NormalParametersResult with posterior statistics

    Raises:
        ValueError: If data has fewer than 2 elements

    Example:
        >>> data = [0.01, -0.02, 0.03, 0.015, -0.01, 0.02]
        >>> result = bayesian_normal_parameters(data)
        >>> print(f"Mean: {result.mean_posterior_mean:.6f}")
    """
    if len(data) < 2:
        raise ValueError("Need at least 2 data points for Bayesian normal estimation")

    data = np.array(data)
    n = len(data)

    # Sample statistics
    sample_mean = np.mean(data)
    sample_var = np.var(data, ddof=1)

    # Jeffreys prior for variance: p(sigma^2) ∝ 1/sigma^2
    # This is achieved with prior_var_alpha = prior_var_beta = 0 (in some parametrizations)
    # or equivalently, using the scale-invariant prior

    # Posterior for variance with Jeffreys prior:
    # Variance ~ Inverse-Gamma((n-1)/2, (n-1)*s^2/2)
    # where s^2 is the sample variance

    df = n - 1  # degrees of freedom

    # Posterior variance parameters (Inverse-Gamma)
    var_alpha_posterior = df / 2 + prior_var_alpha
    var_beta_posterior = df * sample_var / 2 + prior_var_beta

    # Posterior mean of variance
    var_posterior_mean = var_beta_posterior / (var_alpha_posterior - 1) if var_alpha_posterior > 1 else float('inf')
    std_posterior_mean = np.sqrt(var_posterior_mean) if var_posterior_mean != float('inf') else np.sqrt(sample_var)

    # Posterior for mean (with known or unknown variance)
    # With unknown variance, mean follows a t-distribution
    # Posterior mean ~ t(df, x_bar, s/sqrt(n))

    # Credible interval for mean
    # Using t-distribution approximation

    # Standard error of mean
    se_mean = np.sqrt(sample_var / n)

    # t critical value for CI
    t_crit = _t_critical_value(df, ci_level)

    ci_lower_mean = sample_mean - t_crit * se_mean
    ci_upper_mean = sample_mean + t_crit * se_mean

    # Effective sample size (approximation)
    ess_mean = n  # Full information for mean
    ess_var = df / 2  # Reduced for variance

    # Calculate posterior std of variance
    # Var(Inv-Gamma(α, β)) = β² / ((α-1)² * (α-2)) for α > 2
    if var_alpha_posterior > 2:
        var_posterior_var = 2 * var_beta_posterior**2 / ((var_alpha_posterior - 1)**2 * (var_alpha_posterior - 2))
        var_posterior_std = np.sqrt(var_posterior_var) if var_posterior_var > 0 else 0.0
    else:
        var_posterior_std = 0.0

    return NormalParametersResult(
        mean_posterior_mean=sample_mean,
        mean_posterior_std=se_mean,
        var_posterior_mean=var_posterior_mean,
        var_posterior_std=var_posterior_std,
        std_posterior_mean=std_posterior_mean,
        ci_lower_mean=ci_lower_mean,
        ci_upper_mean=ci_upper_mean,
        sample_size=n,
        ess_mean=ess_mean,
        ess_var=ess_var,
    )


def bayesian_position_decision(
    returns: ReturnsArray,
    base_position: float = 1.0,
    min_position: float = 0.1,
    max_position: float = 1.0,
    confidence_threshold: float = 0.7,
    use_sharpe_10th: bool = True,
) -> PositionDecisionResult:
    """
    Position sizing decision based on Bayesian uncertainty.

    Uses Bayesian Sharpe ratio estimation to adjust position size
    based on uncertainty. More uncertain estimates lead to smaller positions.

    Args:
        returns: Historical returns for the strategy
        base_position: Base position size before adjustment (0-1)
        min_position: Minimum allowed position (default 0.1 = 10%)
        max_position: Maximum allowed position (default 1.0 = 100%)
        confidence_threshold: Threshold for confidence level (default 0.7)
        use_sharpe_10th: If True, use 10th percentile Sharpe for conservative sizing

    Returns:
        PositionDecisionResult with recommended position size

    Raises:
        ValueError: If returns is empty
        ValueError: If base_position is not between 0 and 1

    Example:
        >>> returns = [0.01, -0.02, 0.03, 0.015, -0.01, 0.02, 0.005]
        >>> result = bayesian_position_decision(returns, base_position=0.5)
        >>> print(f"Recommended: {result.adjusted_position:.2%}")
    """
    if len(returns) == 0:
        raise ValueError("Returns cannot be empty")

    if not 0 <= base_position <= 1:
        raise ValueError("base_position must be between 0 and 1")

    if base_position == 0:
        return PositionDecisionResult(
            base_position=0.0,
            adjusted_position=0.0,
            uncertainty_factor=0.0,
            confidence=0.0,
            recommended_position_pct=0.0,
            reasoning="No base position specified",
        )

    # Get Bayesian Sharpe ratio estimate
    try:
        sharpe_result = bayesian_sharpe_ratio(returns, n_samples=10000)
    except ValueError as e:
        logger.warning(f"Bayesian Sharpe estimation failed: {e}, using conservative estimate")
        sharpe_result = None

    if sharpe_result is None or sharpe_result.posterior_std == 0:
        # Very uncertain, use minimum position
        uncertainty_factor = 0.0
        confidence = 0.0
        recommended_position = min_position
        reasoning = "Unable to estimate uncertainty, using minimum position"
    else:
        # Calculate uncertainty factor based on coefficient of variation
        # Higher relative uncertainty -> lower position
        cv = sharpe_result.posterior_std / abs(sharpe_result.posterior_mean) if sharpe_result.posterior_mean != 0 else float('inf')

        # Uncertainty factor: higher CV means more uncertainty
        # Map CV to [0, 1] where 1 means low uncertainty (high confidence)
        # Using a sigmoid-like transformation
        uncertainty_factor = 1 / (1 + cv)

        # Confidence in the Sharpe ratio estimate
        confidence = 1 / (1 + sharpe_result.posterior_std)

        # Calculate adjusted position
        if use_sharpe_10th:
            # Use 10th percentile Sharpe for conservative sizing
            # Scale Sharpe percentile to position: higher Sharpe -> higher position
            sharpe_for_position = sharpe_result.posterior_10th
        else:
            sharpe_for_position = sharpe_result.posterior_mean

        # Position adjustment based on Sharpe ratio
        # Positive Sharpe -> increase position, negative -> decrease
        sharpe_factor = np.clip(sharpe_for_position / 2, -1, 2)  # Limit adjustment range

        # Combine uncertainty and Sharpe factors
        adjusted_position = base_position * uncertainty_factor * (1 + sharpe_factor) / 2

        # Ensure within bounds
        recommended_position = np.clip(adjusted_position, min_position, max_position)

        # Generate reasoning
        if sharpe_for_position < 0:
            reasoning = f"Negative Sharpe ({sharpe_for_position:.4f}), reducing position"
        elif uncertainty_factor < 0.5:
            reasoning = f"High uncertainty (factor={uncertainty_factor:.4f}), reducing position"
        elif sharpe_for_position > 1:
            reasoning = f"Strong Sharpe ({sharpe_for_position:.4f}), increasing position"
        else:
            reasoning = f"Moderate conditions, position adjusted to {recommended_position:.2%}"

    # Calculate recommended position as percentage
    recommended_position_pct = recommended_position * 100

    return PositionDecisionResult(
        base_position=base_position,
        adjusted_position=recommended_position,
        uncertainty_factor=uncertainty_factor if sharpe_result else 0.0,
        confidence=confidence if sharpe_result else 0.0,
        recommended_position_pct=recommended_position_pct,
        reasoning=reasoning,
    )


def bayesian_strategy_selection(
    strategy_returns: Dict[str, ReturnsArray],
    benchmark_return: float = 0.0,
    risk_adjusted: bool = True,
    min_prob_beat_benchmark: float = 0.5,
) -> StrategySelectionResult:
    """
    Bayesian strategy selection using posterior probabilities.

    Compares strategies using their posterior distributions and selects
    based on expected return and probability of beating the benchmark.

    Args:
        strategy_returns: Dictionary mapping strategy names to their return series
        benchmark_return: Benchmark return to compare against (default 0.0)
        risk_adjusted: If True, use Sharpe ratio for comparison; else use raw returns
        min_prob_beat_benchmark: Minimum probability of beating benchmark to be considered

    Returns:
        StrategySelectionResult with rankings and recommendations

    Raises:
        ValueError: If strategy_returns is empty
        ValueError: If any strategy has fewer than 3 returns

    Example:
        >>> strategy_returns = {
        ...     "momentum": [0.01, 0.02, -0.01, 0.03],
        ...     "mean_reversion": [0.005, 0.015, 0.01, -0.005],
        ... }
        >>> result = bayesian_strategy_selection(strategy_returns)
        >>> print(f"Best: {result.best_strategy}")
    """
    if not strategy_returns:
        raise ValueError("strategy_returns cannot be empty")

    for name, returns in strategy_returns.items():
        if len(returns) < 3:
            raise ValueError(f"Strategy '{name}' needs at least 3 returns, got {len(returns)}")

    # Calculate posterior metrics for each strategy
    posterior_probs = {}
    expected_returns = {}
    prob_beat_benchmarks = {}
    sample_sizes = {}

    for name, returns in strategy_returns.items():
        returns_arr = np.array(returns)

        # Get Bayesian normal parameters
        norm_result = bayesian_normal_parameters(returns_arr)

        # Expected return (mean of posterior)
        expected_return = norm_result.mean_posterior_mean

        # For Sharpe ratio based selection
        if risk_adjusted:
            try:
                sharpe_result = bayesian_sharpe_ratio(returns, n_samples=5000)
                expected_return = sharpe_result.posterior_mean
                posterior_std = sharpe_result.posterior_std
            except ValueError:
                # Fall back to mean if Sharpe fails
                expected_return = norm_result.mean_posterior_mean
                posterior_std = norm_result.mean_posterior_std
        else:
            posterior_std = norm_result.mean_posterior_std

        # Probability of beating benchmark
        # P(mean > benchmark) using posterior distribution
        if posterior_std > 0:
            z_score = (expected_return - benchmark_return) / posterior_std
            prob_beat = 1 - _normal_cdf(0, z_score, 1)  # P(Z > z_score) = 1 - Phi(z_score)
        else:
            prob_beat = 1.0 if expected_return > benchmark_return else 0.0

        expected_returns[name] = expected_return
        prob_beat_benchmarks[name] = prob_beat
        sample_sizes[name] = len(returns)

        # Approximate posterior probability (unnormalized)
        # Using log-likelihood approximation
        log_prob = -0.5 * ((expected_return - benchmark_return) / max(posterior_std, 1e-6)) ** 2
        posterior_probs[name] = np.exp(log_prob)

    # Normalize posterior probabilities
    total_prob = sum(posterior_probs.values())
    if total_prob > 0:
        posterior_probs = {k: v / total_prob for k, v in posterior_probs.items()}

    # Create rankings
    rankings = [
        (name, expected_returns[name], prob_beat_benchmarks[name])
        for name in strategy_returns.keys()
    ]

    # Sort by expected return (descending)
    rankings.sort(key=lambda x: x[1], reverse=True)

    # Best strategy (highest expected return)
    best_strategy = rankings[0][0] if rankings else None
    best_expected_return = rankings[0][1] if rankings else 0.0
    best_prob_beat_benchmark = prob_beat_benchmarks.get(best_strategy, 0.0) if best_strategy else 0.0

    # Recommended strategy: consider both expected return and probability of beating benchmark
    # Use a weighted score
    scored_strategies = []
    for name, exp_ret, prob_beat in rankings:
        # Score = expected_return * prob_beat
        # Higher score is better
        score = exp_ret * prob_beat if prob_beat >= min_prob_beat_benchmark else exp_ret * prob_beat * 0.5
        scored_strategies.append((name, score))

    scored_strategies.sort(key=lambda x: x[1], reverse=True)
    recommended_strategy = scored_strategies[0][0] if scored_strategies else best_strategy

    return StrategySelectionResult(
        best_strategy=best_strategy,
        best_expected_return=best_expected_return,
        best_prob_beat_benchmark=best_prob_beat_benchmark,
        strategy_rankings=rankings,
        posterior_probs=posterior_probs,
        recommended_strategy=recommended_strategy,
        sample_size_per_strategy=sample_sizes,
    )
