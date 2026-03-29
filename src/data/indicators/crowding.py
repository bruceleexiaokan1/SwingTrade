"""因子拥挤度检测模块

提供多维度拥挤度指标计算，用于检测市场局部过热和潜在反转风险。

主要指标:
- 换手率拥挤度 (turnover_crowding)
- 动量拥挤度 (momentum_crowding)
- 资金流拥挤度 (fund_flow_crowding)
- 持仓集中度 HHI (position_concentration_hhi)
- 相关性崩溃检测 (correlation_breakdown_detection)
- A股特色拥挤度综合指标 (a_share_crowding_indicator)
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any


def turnover_crowding(
    df: pd.DataFrame,
    short_period: int = 5,
    long_period: int = 20,
    threshold: float = 1.5,
    column: str = "turnover"
) -> pd.DataFrame:
    """
    计算换手率拥挤度

    换手率过高表明市场局部过热，可能面临回调风险。

    Args:
        df: 包含换手率数据的 DataFrame
        short_period: 短期周期，默认 5
        long_period: 长期周期，默认 20
        threshold: 拥挤阈值，默认 1.5（短期/长期 > 1.5 为拥挤）
        column: 换手率列名，默认 turnover

    Returns:
        添加以下列的 DataFrame:
        - turnover_short_ma: 短期换手率均线
        - turnover_long_ma: 长期换手率均线
        - turnover_ratio: 短期/长期比值
        - turnover_crowding_signal: 拥挤信号 (1=拥挤, 0=正常, -1=冷清)

    Formula:
        Turnover Ratio = turnover_short_ma / turnover_long_ma
        > threshold 视为拥挤

    Example:
        >>> df = turnover_crowding(df)
        >>> df[df['turnover_crowding_signal'] == 1]['date'].tail()  # 获取拥挤日期
    """
    df = df.copy()

    # 处理空DataFrame
    if df.empty or column not in df.columns:
        df["turnover_short_ma"] = np.nan
        df["turnover_long_ma"] = np.nan
        df["turnover_ratio"] = np.nan
        df["turnover_crowding_signal"] = 0
        return df

    # 计算短期和长期换手率均线
    df["turnover_short_ma"] = df[column].rolling(window=short_period, min_periods=1).mean()
    df["turnover_long_ma"] = df[column].rolling(window=long_period, min_periods=1).mean()

    # 计算比值
    df["turnover_ratio"] = df["turnover_short_ma"] / df["turnover_long_ma"].replace(0, np.nan)

    # 生成信号
    # 1 = 拥挤 (短期 >> 长期)
    # 0 = 正常
    # -1 = 冷清 (短期 << 长期)
    df["turnover_crowding_signal"] = 0
    df.loc[df["turnover_ratio"] > threshold, "turnover_crowding_signal"] = 1
    df.loc[df["turnover_ratio"] < (1 / threshold), "turnover_crowding_signal"] = -1

    return df


def momentum_crowding(
    df: pd.DataFrame,
    lookback_short: int = 20,
    lookback_long: int = 60,
    threshold: float = 2.0
) -> pd.DataFrame:
    """
    计算动量拥挤度

    当短期动量远强于长期动量时，表明市场一致性过强，可能反转。

    Args:
        df: 包含收益率数据的 DataFrame
        lookback_short: 短期回溯期，默认 20
        lookback_long: 长期回溯期，默认 60
        threshold: 拥挤阈值，默认 2.0（短期/长期 > 2 为极度拥挤）

    Returns:
        添加以下列的 DataFrame:
        - momentum_short: 短期累计收益
        - momentum_long: 长期累计收益
        - momentum_ratio: 动量比率
        - momentum_crowding_signal: 拥挤信号

    Formula:
        Momentum Short = Σ(close[i] - close[i-1]) / close[i-1] over lookback_short
        Momentum Long = Σ(close[i] - close[i-1]) / close[i-1] over lookback_long
        Momentum Ratio = momentum_short / momentum_long
        > threshold 视为拥挤

    Example:
        >>> df = momentum_crowding(df)
        >>> crowded = df[df['momentum_crowding_signal'] == 1]
    """
    df = df.copy()

    # 计算收益率
    returns = df["close"].pct_change()

    # 计算短期和长期累计动量
    df["momentum_short"] = returns.rolling(window=lookback_short, min_periods=1).sum()
    df["momentum_long"] = returns.rolling(window=lookback_long, min_periods=1).sum()

    # 计算动量比率
    df["momentum_ratio"] = df["momentum_short"] / df["momentum_long"].replace(0, np.nan)

    # 生成信号
    # 1 = 拥挤 (短期动量 >> 长期)
    # 0 = 正常
    # -1 = 反向拥挤 (短期动量 << 长期)
    df["momentum_crowding_signal"] = 0
    df.loc[df["momentum_ratio"] > threshold, "momentum_crowding_signal"] = 1
    df.loc[df["momentum_ratio"] < (1 / threshold), "momentum_crowding_signal"] = -1

    return df


def fund_flow_crowding(
    df: pd.DataFrame,
    inflow_column: str = "inflow",
    short_period: int = 20,
    long_period: int = 60,
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    计算资金流拥挤度

    通过比较短期和长期资金流入速度，检测资金面的拥挤程度。

    Args:
        df: 包含资金流数据的 DataFrame
        inflow_column: 资金流入列名（正数=流入，负数=流出），默认 inflow
        short_period: 短期周期，默认 20
        long_period: 长期周期，默认 60
        threshold: 流入加速度阈值，默认 1.5

    Returns:
        添加以下列的 DataFrame:
        - inflow_short_avg: 短期日均流入
        - inflow_long_avg: 长期日均流入
        - inflow_acceleration: 流入加速度 (短期/长期)
        - fund_flow_crowding_signal: 拥挤信号

    Formula:
        Inflow Short Avg = 20d流入 / 20
        Inflow Long Avg = 60d流入 / 60
        Inflow Acceleration = (20d流入/20) / (60d流入/60)
        > threshold 视为资金拥挤

    Note:
        资金流入加速度过高可能预示：
        - 杠杆资金涌入
        - 散户追高
        - 潜在踩踏风险

    Example:
        >>> df = fund_flow_crowding(df)
        >>> df[df['fund_flow_crowding_signal'] == 1].tail()
    """
    df = df.copy()

    # 计算累计流入
    df["inflow_cumsum"] = df[inflow_column].cumsum()

    # 计算N日前累计流入
    df["inflow_shifted_20"] = df[inflow_column].shift(short_period).cumsum()
    df["inflow_shifted_60"] = df[inflow_column].shift(long_period).cumsum()

    # 计算周期内净流入
    df["inflow_20d"] = df["inflow_cumsum"] - df["inflow_shifted_20"]
    df["inflow_60d"] = df["inflow_cumsum"] - df["inflow_shifted_60"]

    # 计算日均流入
    df["inflow_short_avg"] = df["inflow_20d"] / short_period
    df["inflow_long_avg"] = df["inflow_60d"] / long_period

    # 计算流入加速度
    df["inflow_acceleration"] = df["inflow_short_avg"] / df["inflow_long_avg"].replace(0, np.nan)

    # 生成信号
    df["fund_flow_crowding_signal"] = 0
    df.loc[df["inflow_acceleration"] > threshold, "fund_flow_crowding_signal"] = 1
    df.loc[df["inflow_acceleration"] < (1 / threshold), "fund_flow_crowding_signal"] = -1

    return df


def position_concentration_hhi(
    market_cap: pd.Series,
    threshold_high: float = 2500,
    threshold_extreme: float = 4000
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算持仓集中度 HHI 指数

    HHI (Herfindahl-Hirschman Index) 衡量市场集中度，常用于检测板块/风格拥挤。

    Args:
        market_cap: 各标的市值/持仓权重序列
        threshold_high: 高集中度阈值，默认 2500
        threshold_extreme: 极度集中阈值，默认 4000

    Returns:
        Tuple containing:
        - hhi: HHI 指数序列
        - hhi_signal: 信号序列 (0=正常, 1=高集中, 2=极度集中)
        - concentration_rank: 当前集中度历史分位 (0-100)

    Formula:
        HHI = Σ(市场份额²) * 10000
        市场份额 = 该标的市值 / 总市值

    Note:
        HHI 取值范围 0-10000:
        - < 1500: 分散市场
        - 1500-2500: 中等集中
        - 2500-4000: 高度集中
        - > 4000: 极度集中

    Example:
        >>> sector_caps = df.groupby('sector')['market_cap'].sum()
        >>> hhi, signal, rank = position_concentration_hhi(sector_caps)
    """
    if len(market_cap) == 0:
        return pd.Series(), pd.Series(), pd.Series()

    # 计算总市值
    total_cap = market_cap.sum()

    if total_cap <= 0:
        return pd.Series(), pd.Series(), pd.Series()

    # 计算各标的的市场份额
    market_share = market_cap / total_cap

    # 计算 HHI
    hhi_value = (market_share ** 2).sum() * 10000

    # 返回标量值（HHI已经是聚合后的单一指标）
    concentration_rank = 50.0  # 单值无法计算历史分位，返回50作为默认值

    # 生成信号
    hhi_signal = 0
    if hhi_value > threshold_high:
        hhi_signal = 1
    if hhi_value > threshold_extreme:
        hhi_signal = 2

    return hhi_value, hhi_signal, concentration_rank


def correlation_breakdown_detection(
    price_df: pd.DataFrame,
    lookback: int = 60,
    correlation_threshold: float = 0.5,
    spike_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    检测相关性崩溃

    当市场个股/板块相关性急剧升高时，通常预示着市场顶部或流动性危机。

    Args:
        price_df: 价格 DataFrame，每列是一个标的
        lookback: 回溯期，默认 60
        correlation_threshold: 相关性阈值，默认 0.5
        spike_threshold: 相关性突增阈值，默认 0.3

    Returns:
        Dict containing:
        - avg_correlation: 平均相关性序列
        - correlation_spike: 相关性突增信号
        - max_correlation: 最大相关性序列
        - crisis_flag: 危机标志 (True/False)

    Formula:
        Correlation Spike: avg_corr > 0.5 为危机信号
        Correlation Acceleration: corr_t - corr_{t-20} > spike_threshold

    Warning:
        相关性崩溃通常发生在：
        - 市场顶部（所有标的一起跌）
        - 流动性危机（不计成本抛售）
        - 黑天鹅事件

    Example:
        >>> prices = df.pivot_table(index='date', columns='code', values='close')
        >>> result = correlation_breakdown_detection(prices)
        >>> if result['crisis_flag']:
        ...     print("警告：市场相关性异常升高")
    """
    if len(price_df) < 20 or price_df.shape[1] < 2:
        return {
            "avg_correlation": pd.Series(),
            "correlation_spike": pd.Series(),
            "max_correlation": pd.Series(),
            "crisis_flag": False
        }

    # 计算收益率
    returns = price_df.pct_change().dropna()

    # 计算滚动相关性矩阵
    rolling_corrs = []
    for i in range(lookback, len(returns)):
        window = returns.iloc[i-lookback:i]
        corr_matrix = window.corr()
        # 获取上三角矩阵的相关性（排除对角线）
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_values = corr_matrix.where(mask).stack()
        rolling_corrs.append({
            "avg_correlation": corr_values.mean(),
            "max_correlation": corr_values.max(),
        })

    if not rolling_corrs:
        return {
            "avg_correlation": pd.Series(),
            "correlation_spike": pd.Series(),
            "max_correlation": pd.Series(),
            "crisis_flag": False
        }

    result_df = pd.DataFrame(rolling_corrs, index=returns.index[lookback:])

    # 计算相关性加速度
    avg_corr = result_df["avg_correlation"]
    corr_20d_ago = avg_corr.shift(20)
    correlation_spike = avg_corr - corr_20d_ago

    # 危机标志
    crisis_flag = (avg_corr > correlation_threshold).any()

    return {
        "avg_correlation": avg_corr,
        "correlation_spike": correlation_spike,
        "max_correlation": result_df["max_correlation"],
        "crisis_flag": crisis_flag
    }


def a_share_crowding_indicator(
    df: pd.DataFrame,
    margin_balance: Optional[pd.Series] = None,
    limit_up_count: Optional[pd.Series] = None,
    etf_flow: Optional[pd.Series] = None,
    market_cap: Optional[float] = None
) -> pd.DataFrame:
    """
    计算 A股特色拥挤度综合指标

    综合融资余额、涨停家数、ETF资金流等A股特有数据，计算综合拥挤度。

    Args:
        df: 包含基本交易数据的 DataFrame（需有 turnover 列）
        margin_balance: 融资余额序列（可选）
        limit_up_count: 涨停家数序列（可选）
        etf_flow: ETF资金净流入序列（可选）
        market_cap: 总市值（可选），用于计算融资余额/市值比

    Returns:
        添加以下列的 DataFrame:
        - margin_balance_ratio: 融资余额/市值比
        - margin_ratio_percentile: 融资比例历史分位
        - limit_up_percentile: 涨停家数历史分位
        - etf_flow_acceleration: ETF流入加速度
        - a_share_crowding_score: A股综合拥挤度 (0-100)
        - a_share_crowding_level: 拥挤等级 (low/medium/high/extreme)

    Formula:
        融资余额/MarketCap 比值:
        - 高位分位 > 80% 视为拥挤

        涨停家数历史分位:
        - > 90% 分位视为极度拥挤

        ETF资金流入加速度:
        - (20d流入/20) / (60d流入/60) > 1.5 为拥挤

        综合评分:
        - 融资分位 * 0.4 + 涨停分位 * 0.3 + ETF分位 * 0.3

    Note:
        A股市场特点：
        - 散户占比高，杠杆资金影响大
        - 涨跌停板制度导致涨停家数是情绪指标
        - ETF申赎数据反映机构动向

    Example:
        >>> df = a_share_crowding_indicator(df, margin_balance=mb, limit_up_count=lu, etf_flow=etf)
        >>> print(df['a_share_crowding_level'].iloc[-1])
    """
    df = df.copy()

    def _calc_percentile(series: pd.Series) -> pd.Series:
        """计算历史分位的辅助函数"""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            if i < 20:  # 数据不足时返回50
                result.iloc[i] = 50.0
            else:
                window = series.iloc[:i+1]
                result.iloc[i] = (window <= window.iloc[-1]).sum() / len(window) * 100
        return result

    # 1. 融资余额/市值比
    if margin_balance is not None and market_cap is not None and market_cap > 0:
        margin_ratio = margin_balance / market_cap
        df["margin_balance_ratio"] = margin_ratio
        df["margin_ratio_percentile"] = _calc_percentile(margin_ratio)
    else:
        df["margin_balance_ratio"] = np.nan
        df["margin_ratio_percentile"] = 50.0  # 默认中等

    # 2. 涨停家数历史分位
    if limit_up_count is not None:
        df["limit_up_count"] = limit_up_count
        df["limit_up_percentile"] = _calc_percentile(limit_up_count)
    else:
        # 如果没有涨停数据，使用涨跌家数比估算
        if "pct_chg" in df.columns:
            # 粗略估算：涨幅>9%视为涨停
            limit_up_est = (df["pct_chg"] > 0.09).rolling(60, min_periods=1).sum()
            df["limit_up_percentile"] = _calc_percentile(limit_up_est)
        else:
            df["limit_up_percentile"] = 50.0

    # 3. ETF资金流入加速度
    if etf_flow is not None:
        df["etf_flow"] = etf_flow
        etf_20d = etf_flow.rolling(20, min_periods=1).sum()
        etf_60d = etf_flow.rolling(60, min_periods=1).sum()
        df["etf_flow_acceleration"] = (etf_20d / 20) / (etf_60d / 60).replace(0, np.nan)
        df["etf_flow_percentile"] = _calc_percentile(df["etf_flow_acceleration"])
    else:
        # 使用换手率变化估算ETF申赎压力
        turnover_change = df["turnover"].diff().rolling(20, min_periods=1).sum()
        df["etf_flow_acceleration"] = turnover_change
        df["etf_flow_percentile"] = 50.0

    # 4. 综合拥挤度评分
    # 填充缺失值
    margin_pct = df["margin_ratio_percentile"].fillna(50)
    limit_pct = df["limit_up_percentile"].fillna(50)
    etf_pct = df["etf_flow_percentile"].fillna(50)

    # 加权平均
    df["a_share_crowding_score"] = (
        margin_pct * 0.4 +
        limit_pct * 0.3 +
        etf_pct * 0.3
    )

    # 限制在 0-100
    df["a_share_crowding_score"] = df["a_share_crowding_score"].clip(0, 100)

    # 5. 拥挤等级
    def get_crowding_level(score: float) -> str:
        if score < 30:
            return "low"
        elif score < 60:
            return "medium"
        elif score < 80:
            return "high"
        else:
            return "extreme"

    df["a_share_crowding_level"] = df["a_share_crowding_score"].apply(get_crowding_level)

    return df


# 向后兼容的别名
def calculate_turnover_crowding(
    df: pd.DataFrame,
    short_period: int = 5,
    long_period: int = 20,
    threshold: float = 1.5,
    column: str = "turnover"
) -> pd.DataFrame:
    """calculate_turnover_crowding - 换手率拥挤度（别名）"""
    return turnover_crowding(df, short_period, long_period, threshold, column)


def calculate_momentum_crowding(
    df: pd.DataFrame,
    lookback_short: int = 20,
    lookback_long: int = 60,
    threshold: float = 2.0
) -> pd.DataFrame:
    """calculate_momentum_crowding - 动量拥挤度（别名）"""
    return momentum_crowding(df, lookback_short, lookback_long, threshold)


def calculate_fund_flow_crowding(
    df: pd.DataFrame,
    inflow_column: str = "inflow",
    short_period: int = 20,
    long_period: int = 60,
    threshold: float = 1.5
) -> pd.DataFrame:
    """calculate_fund_flow_crowding - 资金流拥挤度（别名）"""
    return fund_flow_crowding(df, inflow_column, short_period, long_period, threshold)


def calculate_position_concentration_hhi(
    market_cap: pd.Series,
    threshold_high: float = 2500,
    threshold_extreme: float = 4000
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """calculate_position_concentration_hhi - 持仓集中度 HHI（别名）"""
    return position_concentration_hhi(market_cap, threshold_high, threshold_extreme)


def calculate_correlation_breakdown_detection(
    price_df: pd.DataFrame,
    lookback: int = 60,
    correlation_threshold: float = 0.5,
    spike_threshold: float = 0.3
) -> Dict[str, Any]:
    """calculate_correlation_breakdown_detection - 相关性崩溃检测（别名）"""
    return correlation_breakdown_detection(price_df, lookback, correlation_threshold, spike_threshold)


def calculate_a_share_crowding_indicator(
    df: pd.DataFrame,
    margin_balance: Optional[pd.Series] = None,
    limit_up_count: Optional[pd.Series] = None,
    etf_flow: Optional[pd.Series] = None,
    market_cap: Optional[float] = None
) -> pd.DataFrame:
    """calculate_a_share_crowding_indicator - A股特色拥挤度（别名）"""
    return a_share_crowding_indicator(df, margin_balance, limit_up_count, etf_flow, market_cap)


__all__ = [
    "turnover_crowding",
    "momentum_crowding",
    "fund_flow_crowding",
    "position_concentration_hhi",
    "correlation_breakdown_detection",
    "a_share_crowding_indicator",
    # 别名
    "calculate_turnover_crowding",
    "calculate_momentum_crowding",
    "calculate_fund_flow_crowding",
    "calculate_position_concentration_hhi",
    "calculate_correlation_breakdown_detection",
    "calculate_a_share_crowding_indicator",
]
