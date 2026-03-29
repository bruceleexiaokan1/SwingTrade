"""HMM隐马尔可夫模型：市场状态自动识别

基于hmmlearn库实现市场状态检测，支持：
- 3状态市场识别（震荡/上涨/下跌）
- 状态概率输出
- 状态转移矩阵分析
- 滚动训练更新
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# 可选依赖检查
try:
    from hmmlearn import hmm
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False
    logger.warning("hmmlearn not installed. HMM functionality will be limited.")


@dataclass
class HMMState:
    """HMM状态结果"""
    state_id: int
    state_name: str
    probability: float
    mean_return: float
    mean_volatility: float


@dataclass
class HMMResult:
    """HMM检测结果"""
    date: str
    current_state: int
    state_name: str
    state_probs: Dict[str, float]
    transition_matrix: Optional[np.ndarray]
    state_means: Optional[np.ndarray]
    regime_confidence: float  # 0-1, 状态概率的确定性
    volatility_level: float   # 当前波动率水平
    trend_direction: str      # up/down/sideways
    recommended_action: str   # 买入/观望/卖出
    position_limit: float     # 建议仓位上限


class HMMModel:
    """隐马尔可夫模型市场状态检测器"""

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = 'full',
        n_iter: int = 100,
        window: int = 252,
        update_freq: int = 60
    ):
        """
        Args:
            n_states: 隐状态数量（默认3：震荡/上涨/下跌）
            covariance_type: 协方差类型 ('full', 'tied', 'diag', 'spherical')
            n_iter: Baum-Welch算法最大迭代次数
            window: 训练窗口大小
            update_freq: 模型更新频率（天）
        """
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.window = window
        self.update_freq = update_freq

        self.model: Optional[hmm.GaussianHMM] = None
        self.fitted = False
        self.last_update_day = 0
        self.state_names_: Optional[Dict[int, str]] = None
        self._trained_features: Optional[np.ndarray] = None

    def prepare_features(
        self,
        prices: pd.DataFrame,
        returns_col: str = 'close',
        volume_col: Optional[str] = 'volume'
    ) -> np.ndarray:
        """
        准备HMM特征

        特征组成：
        1. 收益率（带方向）
        2. 波动率（绝对收益）
        3. 成交量变化率

        Args:
            prices: 价格数据（含收盘价和可选成交量）
            returns_col: 收盘价列名
            volume_col: 成交量列名

        Returns:
            特征数组 (n_features, n_samples)
        """
        # 计算收益率
        prices_series = prices[returns_col] if isinstance(prices, pd.DataFrame) else prices
        returns = prices_series.pct_change().fillna(0).values

        # 计算已实现波动率（5日滚动标准差）
        realized_vol = pd.Series(returns).rolling(5).std().fillna(0).values

        # 组合特征
        features_list = [returns, realized_vol]

        # 如果有成交量，加入成交量变化率
        if volume_col is not None and volume_col in prices.columns:
            volume_change = prices[volume_col].pct_change().fillna(0).values
            features_list.append(volume_change)

        # 堆叠特征
        features = np.column_stack(features_list)

        # 转置：hmmlearn需要 (n_features, n_samples)
        return features.T

    def _name_states_by_characteristics(self, means: np.ndarray) -> Dict[int, str]:
        """
        根据状态特征命名状态

        Args:
            means: 各状态的均值数组 (n_states, n_features)

        Returns:
            状态ID到名称的映射
        """
        state_names = {}

        # 第一列是收益率，第二列是波动率
        returns = means[:, 0]
        vols = np.abs(means[:, 1])  # 波动率取绝对值

        # 按波动率排序（低波动=震荡，中波动和高波动=趋势）
        vol_order = np.argsort(vols)

        # 最低波动的是震荡
        lowest_vol_idx = vol_order[0]
        state_names[lowest_vol_idx] = 'sideways'

        # 剩下两个按收益率正负分
        remaining_indices = vol_order[1:]
        remaining_returns = returns[remaining_indices]

        if remaining_returns[0] > 0:
            state_names[remaining_indices[0]] = 'uptrend'
            state_names[remaining_indices[1]] = 'downtrend'
        else:
            state_names[remaining_indices[0]] = 'downtrend'
            state_names[remaining_indices[1]] = 'uptrend'

        return state_names

    def fit(self, features: np.ndarray) -> 'HMMModel':
        """
        拟合HMM模型

        Args:
            features: 特征数组 (n_features, n_samples)
        """
        if not HAS_HMMLEARN:
            raise ImportError("hmmlearn is required. Install with: pip install hmmlearn")

        # 创建模型
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=42
        )

        # 拟合
        self.model.fit(features)
        self.fitted = True
        self._trained_features = features

        # 推断状态名称
        self.state_names_ = self._name_states_by_characteristics(self.model.means_)

        return self

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测状态序列和概率

        Args:
            features: 特征数组 (n_features, n_samples)

        Returns:
            (hidden_states, state_probs)
        """
        if not self.fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # 预测最可能的状态序列
        hidden_states = self.model.predict(features)

        # 获取每个状态的概率
        state_probs = self.model.predict_proba(features)

        return hidden_states, state_probs

    def get_current_state(self, features: np.ndarray) -> HMMState:
        """
        获取当前市场状态

        Args:
            features: 特征数组 (n_features, n_samples)

        Returns:
            HMMState对象
        """
        if not self.fitted or self.model is None:
            # 用历史数据训练
            raise ValueError("Model not fitted. Call fit() first.")

        hidden_states, state_probs = self.predict(features)

        current_state = hidden_states[-1]
        current_probs = state_probs[-1]

        # 获取当前状态的特征
        current_means = self.model.means_[current_state]

        return HMMState(
            state_id=int(current_state),
            state_name=self.state_names_.get(current_state, f'state_{current_state}'),
            probability=float(current_probs[current_state]),
            mean_return=float(current_means[0]),
            mean_volatility=float(np.abs(current_means[1]))
        )

    def should_update(self, current_day: int) -> bool:
        """判断是否需要更新模型"""
        return current_day - self.last_update_day >= self.update_freq

    def rolling_fit(self, prices: pd.DataFrame, date_col: str = 'date') -> 'HMMModel':
        """
        使用滚动窗口训练/更新模型

        Args:
            prices: 价格数据
            date_col: 日期列名

        Returns:
            self
        """
        current_day = len(prices)

        # 如果需要更新
        if self.should_update(current_day):
            # 使用最近window天的数据
            recent_prices = prices.iloc[-self.window:] if len(prices) > self.window else prices

            features = self.prepare_features(recent_prices)
            self.fit(features)
            self.last_update_day = current_day

        return self


class HMMMarketRegime:
    """HMM市场状态检测器（高层接口）"""

    def __init__(
        self,
        n_states: int = 3,
        lookback: int = 60,
        min_periods: int = 30
    ):
        """
        Args:
            n_states: 状态数量
            lookback: 回看天数
            min_periods: 最少需要的数据点数
        """
        self.n_states = n_states
        self.lookback = lookback
        self.min_periods = min_periods
        self.hmm_model = HMMModel(n_states=n_states)
        self._is_fitted = False

    def calculate(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        volume_col: Optional[str] = 'volume'
    ) -> pd.DataFrame:
        """
        计算市场状态序列

        Args:
            df: 价格数据
            price_col: 价格列名
            volume_col: 成交量列名

        Returns:
            包含状态信息的DataFrame
        """
        if len(df) < self.min_periods:
            logger.warning(f"Insufficient data: {len(df)} < {self.min_periods}")
            return self._empty_result(df)

        # 准备特征
        features = self.hmm_model.prepare_features(df, price_col, volume_col)

        # 如果数据不够，回退到min_periods
        if features.shape[1] < self.min_periods:
            return self._empty_result(df)

        # 滚动拟合
        results = []
        for i in range(self.lookback, len(df)):
            window_features = features[:, :i]
            self.hmm_model.fit(window_features)
            _, state_probs = self.hmm_model.predict(features[:, :i+1])

            current_probs = state_probs[-1]
            current_state = np.argmax(current_probs)
            state_name = self.hmm_model.state_names_.get(current_state, 'unknown')

            results.append({
                'date': df.index[i] if hasattr(df.index, '__iter__') and not isinstance(df.index, str) else df.iloc[i].get('date', str(i)),
                'state_id': current_state,
                'state_name': state_name,
                'state_prob': float(current_probs[current_state]),
                'regime_confidence': float(np.max(current_probs)),
                'up_prob': float(current_probs[list(self.hmm_model.state_names_.values()).index('uptrend')]) if 'uptrend' in self.hmm_model.state_names_.values() else 0.0,
                'down_prob': float(current_probs[list(self.hmm_model.state_names_.values()).index('downtrend')]) if 'downtrend' in self.hmm_model.state_names_.values() else 0.0,
                'sideways_prob': float(current_probs[list(self.hmm_model.state_names_.values()).index('sideways')]) if 'sideways' in self.hmm_model.state_names_.values() else 0.0,
            })

        return pd.DataFrame(results)

    def detect_current_regime(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        volume_col: Optional[str] = 'volume'
    ) -> HMMResult:
        """
        检测当前市场状态

        Args:
            df: 价格数据
            price_col: 价格列名
            volume_col: 成交量列名

        Returns:
            HMMResult对象
        """
        if len(df) < self.min_periods:
            return self._default_result()

        # 准备特征
        features = self.hmm_model.prepare_features(df, price_col, volume_col)

        # 训练（使用历史数据）
        train_features = features[:, :-1]
        if train_features.shape[1] < self.min_periods:
            return self._default_result()

        try:
            self.hmm_model.fit(train_features)

            # 预测当前状态
            _, state_probs = self.hmm_model.predict(features)
            current_probs = state_probs[-1]
            current_state = int(np.argmax(current_probs))
            state_name = self.hmm_model.state_names_.get(current_state, 'unknown')

            # 计算波动率水平
            current_vol = float(np.abs(features[1, -1])) if features.shape[1] > 0 else 0.0

            # 判断趋势方向
            if 'uptrend' in self.hmm_model.state_names_.values():
                up_idx = list(self.hmm_model.state_names_.values()).index('uptrend')
                down_idx = list(self.hmm_model.state_names_.values()).index('downtrend')
                trend_direction = 'up' if current_probs[up_idx] > current_probs[down_idx] else 'down'
            else:
                trend_direction = 'sideways'

            # 建议操作和仓位
            action, position_limit = self._get_action_and_position(state_name, current_probs)

            return HMMResult(
                date=str(df.index[-1]) if hasattr(df.index, '__iter__') else df.iloc[-1].get('date', 'unknown'),
                current_state=current_state,
                state_name=state_name,
                state_probs={name: float(prob) for name, prob in zip(self.hmm_model.state_names_.values(), current_probs)},
                transition_matrix=self.hmm_model.model.transmat_ if self.hmm_model.model else None,
                state_means=self.hmm_model.model.means_ if self.hmm_model.model else None,
                regime_confidence=float(np.max(current_probs)),
                volatility_level=current_vol,
                trend_direction=trend_direction,
                recommended_action=action,
                position_limit=position_limit
            )

        except Exception as e:
            logger.warning(f"HMM detection failed: {e}")
            return self._default_result()

    def _empty_result(self, df: pd.DataFrame) -> pd.DataFrame:
        """返回空结果"""
        return pd.DataFrame(columns=[
            'date', 'state_id', 'state_name', 'state_prob',
            'regime_confidence', 'up_prob', 'down_prob', 'sideways_prob'
        ])

    def _default_result(self) -> HMMResult:
        """返回默认结果"""
        return HMMResult(
            date='unknown',
            current_state=-1,
            state_name='unknown',
            state_probs={},
            transition_matrix=None,
            state_means=None,
            regime_confidence=0.0,
            volatility_level=0.0,
            trend_direction='unknown',
            recommended_action='观望',
            position_limit=0.0
        )

    def _get_action_and_position(self, state_name: str, probs: np.ndarray) -> Tuple[str, float]:
        """
        根据状态获取建议操作和仓位

        Args:
            state_name: 状态名称
            probs: 各状态概率

        Returns:
            (action, position_limit)
        """
        dominant_prob = float(np.max(probs))

        if state_name == 'uptrend':
            return '买入', min(0.6 + dominant_prob * 0.4, 1.0)
        elif state_name == 'downtrend':
            return '卖出/对冲', max(0.2 - dominant_prob * 0.2, 0.0)
        else:  # sideways
            return '观望', max(0.3 - dominant_prob * 0.3, 0.0)


def calculate_hmm_regime(
    df: pd.DataFrame,
    price_col: str = 'close',
    volume_col: Optional[str] = 'volume',
    n_states: int = 3,
    lookback: int = 60
) -> pd.DataFrame:
    """
    计算HMM市场状态序列（便捷函数）

    Args:
        df: 价格数据
        price_col: 价格列名
        volume_col: 成交量列名
        n_states: 状态数量
        lookback: 回看天数

    Returns:
        包含状态信息的DataFrame
    """
    detector = HMMMarketRegime(n_states=n_states, lookback=lookback)
    return detector.calculate(df, price_col, volume_col)


def detect_market_regime(
    df: pd.DataFrame,
    price_col: str = 'close',
    volume_col: Optional[str] = 'volume'
) -> HMMResult:
    """
    检测当前市场状态（便捷函数）

    Args:
        df: 价格数据
        price_col: 价格列名
        volume_col: 成交量列名

    Returns:
        HMMResult对象
    """
    detector = HMMMarketRegime()
    return detector.detect_current_regime(df, price_col, volume_col)
