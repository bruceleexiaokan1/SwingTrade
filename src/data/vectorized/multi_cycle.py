"""向量化多周期共振计算

核心优化：一次性计算所有股票所有日期的多周期状态
消除原有实现中每日每股票重算月线/周线的瓶颈

性能优势：
- 预计算所有周期状态 O(1) vs 原有的 O(n_stocks × n_dates)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ...data.loader import StockDataLoader


@dataclass
class MultiCycleConfig:
    """多周期配置"""
    # 各周期MA参数
    monthly_ma_short: int = 5
    monthly_ma_long: int = 10
    weekly_ma_short: int = 10
    weekly_ma_long: int = 20
    daily_ma_short: int = 20
    daily_ma_long: int = 60

    # 最低数据量要求
    min_monthly_bars: int = 6
    min_weekly_bars: int = 10
    min_daily_bars: int = 20

    # 趋势判断置信度阈值
    trend_threshold_pct: float = 3.0


class VectorizedMultiCycle:
    """
    向量化多周期共振计算器

    核心优化：
    1. 一次性加载所有股票数据
    2. 批量转换月线/周线
    3. 向量化趋势检测
    4. 预计算所有日期的共振状态

    输出格式：
        multi_cycle_df: [data_id, date, monthly_trend, monthly_conf,
                        weekly_trend, weekly_conf, daily_trend, daily_conf,
                        resonance_level, position_limit, is_bullish]
    """

    def __init__(
        self,
        config: Optional[MultiCycleConfig] = None,
        stockdata_root: str = "/Users/bruce/workspace/trade/StockData"
    ):
        self.config = config or MultiCycleConfig()
        self.stockdata_root = stockdata_root
        self.loader = StockDataLoader(stockdata_root)

    def precompute_all(
        self,
        data_ids: List[str],
        end_date: str,
        lookback_months: int = 6
    ) -> pd.DataFrame:
        """
        预计算所有多周期状态

        Args:
            data_ids: 股票代码列表
            end_date: 截止日期
            lookback_months: 回溯月数

        Returns:
            包含所有多周期状态的 DataFrame
        """
        # Step 1: 加载所有日线数据
        all_data = []
        for data_id in data_ids:
            df = self._load_daily_data(data_id, end_date, lookback_months)
            if not df.empty:
                df['data_id'] = data_id
                all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        daily_df = pd.concat(all_data, ignore_index=True)
        daily_df = daily_df.sort_values(['data_id', 'date']).reset_index(drop=True)

        # Step 2: 转换为月线和周线
        daily_df = self._compute_all_cycle_data(daily_df)

        # Step 3: 向量化检测各周期趋势
        daily_df = self._compute_trends_vectorized(daily_df)

        # Step 4: 计算共振等级
        daily_df = self._compute_resonance_vectorized(daily_df)

        # 返回需要的列
        result_cols = [
            'data_id', 'date',
            'monthly_trend', 'monthly_conf',
            'weekly_trend', 'weekly_conf',
            'daily_trend', 'daily_conf',
            'resonance_level', 'position_limit', 'is_bullish'
        ]

        existing_cols = [c for c in result_cols if c in daily_df.columns]
        return daily_df[existing_cols].copy()

    def _load_daily_data(
        self,
        data_id: str,
        end_date: str,
        lookback_months: int
    ) -> pd.DataFrame:
        """加载单只股票的日线数据"""
        import datetime

        # 计算开始日期
        end_dt = pd.to_datetime(end_date)
        start_dt = end_dt - datetime.timedelta(days=lookback_months * 30 + 60)
        start_date = start_dt.strftime('%Y-%m-%d')

        df = self.loader.load_daily(data_id, start_date, end_date)

        if df.empty:
            return df

        # 确保日期格式
        if pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        return df

    def _compute_all_cycle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算月线、周线数据并合并到日线"""
        results = []

        for data_id, group in df.groupby('data_id', sort=False):
            group = group.sort_values('date').copy()

            # 月线转换
            monthly = self._to_monthly(group[['date', 'open', 'high', 'low', 'close', 'volume']])
            monthly = monthly.rename(columns={
                'open': 'monthly_open',
                'high': 'monthly_high',
                'low': 'monthly_low',
                'close': 'monthly_close',
                'volume': 'monthly_volume'
            })

            # 周线转换
            weekly = self._to_weekly(group[['date', 'open', 'high', 'low', 'close', 'volume']])
            weekly = weekly.rename(columns={
                'open': 'weekly_open',
                'high': 'weekly_high',
                'low': 'weekly_low',
                'close': 'weekly_close',
                'volume': 'weekly_volume'
            })

            # 合并到日线（前向填充缺失值）
            group = group.merge(monthly[['date', 'monthly_open', 'monthly_high',
                                         'monthly_low', 'monthly_close', 'monthly_volume']],
                               on='date', how='left')
            group = group.merge(weekly[['date', 'weekly_open', 'weekly_high',
                                        'weekly_low', 'weekly_close', 'weekly_volume']],
                               on='date', how='left')

            # 前向填充月线周线数据
            for col in ['monthly_open', 'monthly_high', 'monthly_low', 'monthly_close', 'monthly_volume',
                        'weekly_open', 'weekly_high', 'weekly_low', 'weekly_close', 'weekly_volume']:
                if col in group.columns:
                    group[col] = group[col].ffill()

            results.append(group)

        return pd.concat(results, ignore_index=True)

    def _to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """日线转换为月线"""
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        monthly = df.resample('ME').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        monthly = monthly.reset_index()
        monthly['date'] = monthly['date'].dt.strftime('%Y-%m-%d')

        return monthly

    def _to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """日线转换为周线"""
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        weekly = df.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        weekly = weekly.reset_index()
        weekly['date'] = weekly['date'].dt.strftime('%Y-%m-%d')

        return weekly

    def _compute_trends_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """向量化计算所有周期的趋势"""
        cfg = self.config

        # 日线趋势（使用日线自己的MA）
        if 'close' in df.columns:
            df['daily_ma_short'] = df.groupby('data_id')['close'].transform(
                lambda x: x.rolling(window=cfg.daily_ma_short, min_periods=1).mean()
            )
            df['daily_ma_long'] = df.groupby('data_id')['close'].transform(
                lambda x: x.rolling(window=cfg.daily_ma_long, min_periods=1).mean()
            )
            df['daily_trend'], df['daily_conf'] = self._calc_trend(
                df['daily_ma_short'], df['daily_ma_long'], cfg.trend_threshold_pct
            )

        # 月线趋势
        if 'monthly_close' in df.columns:
            df['monthly_ma_short'] = df.groupby('data_id')['monthly_close'].transform(
                lambda x: x.rolling(window=cfg.monthly_ma_short, min_periods=1).mean()
            )
            df['monthly_ma_long'] = df.groupby('data_id')['monthly_close'].transform(
                lambda x: x.rolling(window=cfg.monthly_ma_long, min_periods=1).mean()
            )
            df['monthly_trend'], df['monthly_conf'] = self._calc_trend(
                df['monthly_ma_short'], df['monthly_ma_long'], cfg.trend_threshold_pct
            )

        # 周线趋势
        if 'weekly_close' in df.columns:
            df['weekly_ma_short'] = df.groupby('data_id')['weekly_close'].transform(
                lambda x: x.rolling(window=cfg.weekly_ma_short, min_periods=1).mean()
            )
            df['weekly_ma_long'] = df.groupby('data_id')['weekly_close'].transform(
                lambda x: x.rolling(window=cfg.weekly_ma_long, min_periods=1).mean()
            )
            df['weekly_trend'], df['weekly_conf'] = self._calc_trend(
                df['weekly_ma_short'], df['weekly_ma_long'], cfg.trend_threshold_pct
            )

        return df

    def _calc_trend(
        self,
        ma_short: pd.Series,
        ma_long: pd.Series,
        threshold_pct: float
    ) -> Tuple[pd.Series, pd.Series]:
        """
        向量化计算趋势方向和置信度

        Args:
            ma_short: 短期均线
            ma_long: 长期均线
            threshold_pct: 阈值百分比

        Returns:
            (trend_series, confidence_series)
        """
        # 计算差值百分比
        diff_pct = (ma_short - ma_long) / ma_long * 100

        # 初始化为 sideways
        trend = pd.Series('sideways', index=ma_short.index)
        confidence = pd.Series(0.5, index=ma_short.index)

        # 上涨：short > long
        up_mask = ma_short > ma_long
        trend[up_mask] = 'up'
        confidence[up_mask] = np.minimum(1.0, diff_pct[up_mask].abs() / threshold_pct)

        # 下跌：short < long
        down_mask = ma_short < ma_long
        trend[down_mask] = 'down'
        confidence[down_mask] = np.minimum(1.0, diff_pct[down_mask].abs() / threshold_pct)

        # 处理 NaN
        nan_mask = ma_short.isna() | ma_long.isna()
        trend[nan_mask] = 'sideways'
        confidence[nan_mask] = 0.0

        return trend, confidence

    def _compute_resonance_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """向量化计算共振等级"""
        # 统计向上趋势数量
        up_count = pd.Series(0, index=df.index)

        if 'monthly_trend' in df.columns:
            up_count += (df['monthly_trend'] == 'up').astype(int)

        if 'weekly_trend' in df.columns:
            up_count += (df['weekly_trend'] == 'up').astype(int)

        if 'daily_trend' in df.columns:
            up_count += (df['daily_trend'] == 'up').astype(int)

        # 统计向下趋势数量
        down_count = pd.Series(0, index=df.index)

        if 'monthly_trend' in df.columns:
            down_count += (df['monthly_trend'] == 'down').astype(int)

        if 'weekly_trend' in df.columns:
            down_count += (df['weekly_trend'] == 'down').astype(int)

        if 'daily_trend' in df.columns:
            down_count += (df['daily_trend'] == 'down').astype(int)

        # 计算共振等级
        resonance_level = pd.Series(0, index=df.index)

        # 三周期全部向上 -> 5
        resonance_level[(up_count == 3)] = 5

        # 月周共振，日线待确认 -> 4
        resonance_level[
            (df.get('monthly_trend', pd.Series('sideways', index=df.index)) == 'up') &
            (df.get('weekly_trend', pd.Series('sideways', index=df.index)) == 'up') &
            (df.get('daily_trend', pd.Series('sideways', index=df.index)) != 'up') &
            (up_count < 3)
        ] = 4

        # 两周期向上 -> 3
        resonance_level[(up_count == 2)] = 3

        # 只有日线向上 -> 3
        resonance_level[
            (up_count == 1) &
            (df.get('daily_trend', pd.Series('sideways', index=df.index)) == 'up')
        ] = 3

        # 两周期以上向下 -> 0
        resonance_level[(down_count >= 2)] = 0

        df['resonance_level'] = resonance_level

        # 仓位上限
        position_limit = pd.Series(0.0, index=df.index)
        position_limit[resonance_level == 5] = 0.8
        position_limit[resonance_level == 4] = 0.6
        position_limit[resonance_level == 3] = 0.2
        position_limit[resonance_level == 0] = 0.0
        df['position_limit'] = position_limit

        # 是否看多（至少2个周期向上）
        df['is_bullish'] = up_count >= 2

        return df

    def get_state(self, df: pd.DataFrame, data_id: str, date: str) -> Optional[Dict]:
        """获取指定股票在指定日期的多周期状态"""
        row = df[
            (df['data_id'] == data_id) &
            (df['date'] == date)
        ]

        if row.empty:
            return None

        row = row.iloc[0]
        return {
            'data_id': row.get('data_id'),
            'date': row.get('date'),
            'monthly_trend': row.get('monthly_trend', 'sideways'),
            'monthly_conf': row.get('monthly_conf', 0.0),
            'weekly_trend': row.get('weekly_trend', 'sideways'),
            'weekly_conf': row.get('weekly_conf', 0.0),
            'daily_trend': row.get('daily_trend', 'sideways'),
            'daily_conf': row.get('daily_conf', 0.0),
            'resonance_level': row.get('resonance_level', 0),
            'position_limit': row.get('position_limit', 0.0),
            'is_bullish': row.get('is_bullish', False)
        }
