"""多周期共振检测

月/周/日三层周期共振系统：
- 月线定方向：判断大盘牛熊，设定仓位上限
- 周线定趋势：确认日线信号真伪
- 日线定入场：精确入场点位

共振等级：
- 5: 三周期共振买入（月周周日全部向上）
- 4: 强信号（月周共振，日线待确认）
- 3: 中信号（只有日线信号）
- 0: 禁止操作（三层逆势）
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import IntEnum
import datetime

import pandas as pd

from ..data.loader import StockDataLoader


class MultiCycleLevel(IntEnum):
    """多周期共振等级"""
    FORBIDDEN = 0      # 三层逆势
    DAILY_ONLY = 3     # 只有日线信号
    MONTHLY_WEEKLY = 4 # 月周共振，日线待确认
    THREE_CYCLE = 5    # 三周期共振

    @property
    def label(self) -> str:
        labels = {
            0: "禁止操作",
            3: "中信号",
            4: "强信号",
            5: "三周期共振"
        }
        return labels.get(self.value, "未知")

    @property
    def position_limit(self) -> float:
        """仓位上限"""
        limits = {
            0: 0.0,
            3: 0.2,
            4: 0.6,
            5: 0.8
        }
        return limits.get(self.value, 0.0)


@dataclass
class MultiCycleResult:
    """多周期共振检测结果"""
    stock_code: str
    date: str

    # 月线指标
    monthly_trend: str = "sideways"  # up/down/sideways
    monthly_conf: float = 0.0

    # 周线指标
    weekly_trend: str = "sideways"
    weekly_conf: float = 0.0

    # 日线指标
    daily_trend: str = "sideways"
    daily_conf: float = 0.0

    # 共振评分（0-5）
    resonance_level: int = 0

    # 仓位建议
    position_limit: float = 0.0

    # 是否看多（至少2个周期看多）
    is_bullish: bool = False

    # 共振原因
    reasons: List[str] = field(default_factory=list)

    @property
    def level_label(self) -> str:
        """共振等级标签"""
        return MultiCycleLevel(self.resonance_level).label

    @property
    def monthly_up(self) -> bool:
        return self.monthly_trend == "up"

    @property
    def weekly_up(self) -> bool:
        return self.weekly_trend == "up"

    @property
    def daily_up(self) -> bool:
        return self.daily_trend == "up"


class MultiCycleResonance:
    """
    多周期共振检测器

    通过聚合日线数据生成月线、周线数据，分析三层周期的趋势一致性。

    使用示例：
        resonance = MultiCycleResonance(
            stockdata_root="/Users/bruce/workspace/trade/StockData"
        )

        result = resonance.check_resonance("600519", "2026-03-28")
        print(f"共振等级: {result.level_label}")
        print(f"仓位上限: {result.position_limit}")
    """

    # 各周期最低数据量要求
    MIN_DAILY_BARS = 20
    MIN_WEEKLY_BARS = 10
    MIN_MONTHLY_BARS = 6

    def __init__(self, stockdata_root: str = "/Users/bruce/workspace/trade/StockData"):
        """
        初始化多周期共振检测器

        Args:
            stockdata_root: StockData 根目录
        """
        self.stock_loader = StockDataLoader(stockdata_root)

    def check_resonance(
        self,
        stock_code: str,
        end_date: str,
        lookback_months: int = 3
    ) -> MultiCycleResult:
        """
        检测月/周/日三层共振

        Args:
            stock_code: 股票代码
            end_date: 截止日期
            lookback_months: 回溯月数（确保各周期有足够数据）

        Returns:
            MultiCycleResult
        """
        # 计算需要的回溯日期（确保月/周/日都有足够数据）
        lookback_days = lookback_months * 30 + 60  # 额外60天确保周线月线数据

        # 加载日线数据
        daily_df = self._load_daily_data(stock_code, end_date, lookback_days)
        if daily_df.empty:
            return self._create_empty_result(stock_code, end_date, "日线数据为空")

        if len(daily_df) < self.MIN_DAILY_BARS:
            return self._create_empty_result(stock_code, end_date, f"日线数据不足，仅有 {len(daily_df)} 条")

        # 过滤到 end_date
        daily_df = daily_df[daily_df['date'] <= end_date].copy()
        if daily_df.empty:
            return self._create_empty_result(stock_code, end_date, "截止日期后无数据")

        # 生成月线和周线数据
        monthly_df = self._to_monthly(daily_df)
        weekly_df = self._to_weekly(daily_df)

        # 检测各周期趋势
        monthly_trend, monthly_conf = self._detect_trend(monthly_df, 'monthly')
        weekly_trend, weekly_conf = self._detect_trend(weekly_df, 'weekly')
        daily_trend, daily_conf = self._detect_trend(daily_df, 'daily')

        # 计算共振等级和仓位
        resonance_level, reasons = self._calc_resonance_level(
            monthly_trend, weekly_trend, daily_trend
        )
        position_limit = MultiCycleLevel(resonance_level).position_limit

        # 判断是否看多（至少2个周期向上）
        bullish_count = sum([monthly_trend == "up", weekly_trend == "up", daily_trend == "up"])
        is_bullish = bullish_count >= 2

        return MultiCycleResult(
            stock_code=stock_code,
            date=end_date,
            monthly_trend=monthly_trend,
            monthly_conf=monthly_conf,
            weekly_trend=weekly_trend,
            weekly_conf=weekly_conf,
            daily_trend=daily_trend,
            daily_conf=daily_conf,
            resonance_level=resonance_level,
            position_limit=position_limit,
            is_bullish=is_bullish,
            reasons=reasons
        )

    def _load_daily_data(
        self,
        code: str,
        end_date: str,
        lookback_days: int
    ) -> pd.DataFrame:
        """
        加载日线数据

        Args:
            code: 股票代码
            end_date: 截止日期
            lookback_days: 回溯天数

        Returns:
            日线 DataFrame
        """
        # 计算开始日期
        end_dt = pd.to_datetime(end_date)
        start_dt = end_dt - datetime.timedelta(days=lookback_days)
        start_date = start_dt.strftime('%Y-%m-%d')

        df = self.stock_loader.load_daily(code, start_date, end_date)

        if df.empty:
            return pd.DataFrame()

        # 确保日期列是字符串格式
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        return df

    def _to_monthly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        日线转换为月线

        取每月最后一根K线

        Args:
            daily_df: 日线 DataFrame

        Returns:
            月线 DataFrame
        """
        if daily_df.empty:
            return pd.DataFrame()

        df = daily_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # 按月重采样，取每月最后一根K线
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

    def _to_weekly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        日线转换为周线

        取每周最后一根K线

        Args:
            daily_df: 日线 DataFrame

        Returns:
            周线 DataFrame
        """
        if daily_df.empty:
            return pd.DataFrame()

        df = daily_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # 按周重采样，取每周最后一根K线
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

    def _detect_trend(self, df: pd.DataFrame, cycle: str) -> Tuple[str, float]:
        """
        检测趋势方向

        Args:
            df: K线数据
            cycle: 周期类型 ('daily', 'weekly', 'monthly')

        Returns:
            (趋势方向, 置信度)
            - ("up", 0.0~1.0): 上涨趋势
            - ("down", 0.0~1.0): 下跌趋势
            - ("sideways", 0.0~1.0): 盘整
        """
        if df.empty:
            return ("sideways", 0.0)

        # 根据周期选择MA参数
        if cycle == 'monthly':
            ma_short, ma_long = 5, 10  # 月线用较短周期
        elif cycle == 'weekly':
            ma_short, ma_long = 10, 20  # 周线
        else:
            ma_short, ma_long = 20, 60  # 日线

        min_bars = {
            'monthly': self.MIN_MONTHLY_BARS,
            'weekly': self.MIN_WEEKLY_BARS,
            'daily': self.MIN_DAILY_BARS
        }.get(cycle, 20)

        if len(df) < min_bars:
            return ("sideways", 0.0)

        # 计算MA
        df = df.copy()
        df['ma_short'] = df['close'].rolling(window=ma_short, min_periods=1).mean()
        df['ma_long'] = df['close'].rolling(window=ma_long, min_periods=1).mean()

        # 趋势判断
        ma_short_vals = df['ma_short'].dropna()
        ma_long_vals = df['ma_long'].dropna()

        if len(ma_short_vals) < 2 or len(ma_long_vals) < 2:
            return ("sideways", 0.0)

        # 获取最新值
        last_short = ma_short_vals.iloc[-1]
        last_long = ma_long_vals.iloc[-1]

        # 上涨趋势：短期均线 > 长期均线
        if last_short > last_long:
            # 置信度基于差距
            diff_pct = (last_short - last_long) / last_long * 100
            confidence = min(1.0, diff_pct / 3)  # 3%差距 = 100%置信度
            return ("up", confidence)

        # 下跌趋势：短期均线 < 长期均线
        if last_short < last_long:
            diff_pct = (last_long - last_short) / last_long * 100
            confidence = min(1.0, diff_pct / 3)
            return ("down", confidence)

        return ("sideways", 0.5)

    def _calc_resonance_level(
        self,
        monthly_trend: str,
        weekly_trend: str,
        daily_trend: str
    ) -> Tuple[int, List[str]]:
        """
        计算共振等级

        Args:
            monthly_trend: 月线趋势
            weekly_trend: 周线趋势
            daily_trend: 日线趋势

        Returns:
            (共振等级, 原因列表)
        """
        reasons = []

        # 统计向上趋势的数量
        up_count = sum([
            monthly_trend == "up",
            weekly_trend == "up",
            daily_trend == "up"
        ])

        # 统计向下趋势的数量
        down_count = sum([
            monthly_trend == "down",
            weekly_trend == "down",
            daily_trend == "down"
        ])

        # 三周期全部向上
        if up_count == 3:
            reasons.append("月周周日三层共振向上")
            reasons.append(f"月线{'向上' if monthly_trend == 'up' else monthly_trend}")
            reasons.append(f"周线{'向上' if weekly_trend == 'up' else weekly_trend}")
            reasons.append(f"日线{'向上' if daily_trend == 'up' else daily_trend}")
            return (MultiCycleLevel.THREE_CYCLE.value, reasons)

        # 月周共振，日线待确认
        if monthly_trend == "up" and weekly_trend == "up" and daily_trend != "up":
            reasons.append("月周共振，日线待确认")
            reasons.append(f"月线{'向上' if monthly_trend == 'up' else monthly_trend}")
            reasons.append(f"周线{'向上' if weekly_trend == 'up' else weekly_trend}")
            reasons.append(f"日线{'向下' if daily_trend == 'down' else '盘整'}")
            return (MultiCycleLevel.MONTHLY_WEEKLY.value, reasons)

        # 只有日线信号（其余不共振）
        if down_count >= 2:
            reasons.append("三层逆势，禁止操作")
            if monthly_trend == "down":
                reasons.append("月线向下")
            if weekly_trend == "down":
                reasons.append("周线向下")
            if daily_trend == "down":
                reasons.append("日线向下")
            return (MultiCycleLevel.FORBIDDEN.value, reasons)

        # 只有日线向上
        if daily_trend == "up" and up_count == 1:
            reasons.append("仅日线信号，谨慎操作")
            reasons.append("日线向上")
            return (MultiCycleLevel.DAILY_ONLY.value, reasons)

        # 其他情况（部分共振但不够强）
        if up_count == 2:
            # 两周期向上
            reasons.append("双周期共振")
            if monthly_trend == "up":
                reasons.append("月线向上")
            if weekly_trend == "up":
                reasons.append("周线向上")
            if daily_trend == "up":
                reasons.append("日线向上")
            return (MultiCycleLevel.DAILY_ONLY.value, reasons)

        # 默认：盘整
        reasons.append("各周期方向不一，等待明确信号")
        return (MultiCycleLevel.FORBIDDEN.value, reasons)

    def _create_empty_result(
        self,
        stock_code: str,
        date: str,
        error_msg: str
    ) -> MultiCycleResult:
        """创建空/错误结果"""
        return MultiCycleResult(
            stock_code=stock_code,
            date=date,
            reasons=[error_msg]
        )

    def batch_check(
        self,
        stock_codes: List[str],
        end_date: str,
        lookback_months: int = 3
    ) -> List[MultiCycleResult]:
        """
        批量检测多只股票

        Args:
            stock_codes: 股票代码列表
            end_date: 截止日期
            lookback_months: 回溯月数

        Returns:
            MultiCycleResult 列表
        """
        results = []
        for code in stock_codes:
            result = self.check_resonance(code, end_date, lookback_months)
            results.append(result)
        return results

    def get_resonance_summary(
        self,
        results: List[MultiCycleResult]
    ) -> dict:
        """
        获取共振汇总统计

        Args:
            results: MultiCycleResult 列表

        Returns:
            汇总字典
        """
        if not results:
            return {
                "total": 0,
                "three_cycle": 0,
                "monthly_weekly": 0,
                "daily_only": 0,
                "forbidden": 0,
                "bullish_count": 0
            }

        return {
            "total": len(results),
            "three_cycle": sum(1 for r in results if r.resonance_level == MultiCycleLevel.THREE_CYCLE.value),
            "monthly_weekly": sum(1 for r in results if r.resonance_level == MultiCycleLevel.MONTHLY_WEEKLY.value),
            "daily_only": sum(1 for r in results if r.resonance_level == MultiCycleLevel.DAILY_ONLY.value),
            "forbidden": sum(1 for r in results if r.resonance_level == MultiCycleLevel.FORBIDDEN.value),
            "bullish_count": sum(1 for r in results if r.is_bullish)
        }
