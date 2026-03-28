"""
StockData 测试数据构造
提供各种场景的测试数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple


class TestDataFactory:
    """测试数据工厂"""

    @staticmethod
    def create_daily(
        code: str = '000001',
        dates: List[str] = None,
        base_price: float = 10.0
    ) -> pd.DataFrame:
        """
        创建标准日线数据

        Args:
            code: 股票代码
            dates: 日期列表，默认最近3天
            base_price: 基础价格
        """
        if dates is None:
            dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                     for i in range(3, 0, -1)]

        n = len(dates)
        prices = [base_price + i * 0.1 for i in range(n)]

        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'code': code,
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

    @staticmethod
    def create_perfect_data() -> Tuple[pd.DataFrame, list]:
        """创建完美数据，无异常"""
        df = TestDataFactory.create_daily()
        return df, []

    @staticmethod
    def create_price_out_of_range() -> Tuple[pd.DataFrame, list]:
        """价格超出合理范围"""
        df = TestDataFactory.create_daily()
        df.loc[0, 'close'] = 50000  # 超出 0.01-10000
        anomalies = [{'reason': 'price_out_of_range', 'count': 1}]
        return df, anomalies

    @staticmethod
    def create_price_zero() -> Tuple[pd.DataFrame, list]:
        """价格为零"""
        df = TestDataFactory.create_daily()
        df.loc[0, 'close'] = 0
        anomalies = [{'reason': 'price_out_of_range', 'count': 1}]
        return df, anomalies

    @staticmethod
    def create_ohlc_invalid() -> Tuple[pd.DataFrame, list]:
        """OHLC 关系不合法: close > high"""
        df = TestDataFactory.create_daily()
        # 设置 close > high (close=10.0, high=9.5)
        df.loc[0, 'close'] = df.loc[0, 'high'] + 0.5
        anomalies = [{'reason': 'ohlc_close_out', 'count': 1}]
        return df, anomalies

    @staticmethod
    def create_ohlc_low_gt_high() -> Tuple[pd.DataFrame, list]:
        """OHLC 关系不合法: low > high"""
        df = TestDataFactory.create_daily()
        df.loc[0, 'low'] = df.loc[0, 'high'] + 0.5  # low > high
        anomalies = [{'reason': 'ohlc_invalid', 'count': 1}]
        return df, anomalies

    @staticmethod
    def create_adj_continuity_break() -> Tuple[pd.DataFrame, list]:
        """复权连续性断裂 - 因子突变但价格未补偿（异常）"""
        df = TestDataFactory.create_daily()

        # 第2天的因子突然变为0.5（分红导致）但价格没有补偿
        # 这会导致后复权价格不连续
        df.loc[1, 'adj_factor'] = 0.5
        # 注意：这里不调整close，这样后复权价格 = close * adj_factor 会断裂

        anomalies = [{'reason': 'adj_continuity_break', 'count': 1}]
        return df, anomalies

    @staticmethod
    def create_adj_factor_negative() -> Tuple[pd.DataFrame, list]:
        """复权因子为负"""
        df = TestDataFactory.create_daily()
        df.loc[0, 'adj_factor'] = -1.0
        anomalies = [{'reason': 'adj_factor_invalid', 'count': 1}]
        return df, anomalies

    @staticmethod
    def create_volume_negative() -> Tuple[pd.DataFrame, list]:
        """成交量为负 - 归类为完整性问题"""
        df = TestDataFactory.create_daily()
        df.loc[0, 'volume'] = -1000
        anomalies = [{'reason': 'volume_invalid', 'count': 1}]
        return df, anomalies

    @staticmethod
    def create_pct_chg_exceed() -> Tuple[pd.DataFrame, list]:
        """涨跌幅超出范围"""
        df = TestDataFactory.create_daily()
        df.loc[0, 'pct_chg'] = 0.5  # 超出 [-0.2, 0.2]
        anomalies = [{'reason': 'pct_chg_exceed', 'count': 1}]
        return df, anomalies

    @staticmethod
    def create_limit_up() -> Tuple[pd.DataFrame, list]:
        """涨停数据（正常）"""
        df = TestDataFactory.create_daily()
        yesterday_close = df.loc[0, 'close']
        df.loc[0, 'close'] = yesterday_close * 1.1  # 涨停价
        df.loc[0, 'high'] = df.loc[0, 'close']
        df.loc[0, 'pct_chg'] = 0.1
        # 涨停是正常的，不应该触发异常
        return df, []

    @staticmethod
    def create_limit_down() -> Tuple[pd.DataFrame, list]:
        """跌停数据（正常）"""
        df = TestDataFactory.create_daily()
        yesterday_close = df.loc[0, 'close']
        df.loc[0, 'close'] = yesterday_close * 0.9  # 跌停价
        df.loc[0, 'low'] = df.loc[0, 'close']
        df.loc[0, 'pct_chg'] = -0.1
        # 跌停是正常的，不应该触发异常
        return df, []

    @staticmethod
    def create_single_row() -> Tuple[pd.DataFrame, list]:
        """单行数据（新股首日）"""
        today = datetime.now().strftime('%Y-%m-%d')
        df = TestDataFactory.create_daily(dates=[today])
        # 单行数据是正常的
        return df, []

    @staticmethod
    def create_empty() -> Tuple[pd.DataFrame, list]:
        """空数据"""
        df = pd.DataFrame()
        anomalies = [{'reason': 'empty_data', 'count': 0}]
        return df, anomalies

    @staticmethod
    def create_missing_fields() -> Tuple[pd.DataFrame, list]:
        """缺少必填字段"""
        df = TestDataFactory.create_daily()
        df = df.drop(columns=['volume'])  # 删除必填字段
        anomalies = [{'reason': 'missing_required_field', 'count': 1}]
        return df, anomalies

    @staticmethod
    def create_multi_anomalies() -> Tuple[pd.DataFrame, list]:
        """多种异常同时存在"""
        df = TestDataFactory.create_daily()
        df.loc[0, 'close'] = 50000  # 价格异常
        df.loc[1, 'adj_factor'] = 0.5  # 因子断裂
        anomalies = [
            {'reason': 'price_out_of_range', 'count': 1},
            {'reason': 'adj_continuity_break', 'count': 1}
        ]
        return df, anomalies

    @staticmethod
    def create_suspended_stock() -> Tuple[pd.DataFrame, list]:
        """长期停牌股票"""
        df = TestDataFactory.create_daily()

        # 标记最后几天停牌
        df.loc[1, 'is_halt'] = True
        df.loc[1, 'volume'] = 0
        df.loc[1, 'pct_chg'] = 0

        # 连续停牌不超过60天是正常的
        return df, []

    @staticmethod
    def create_resume_trading() -> Tuple[pd.DataFrame, list]:
        """复牌首日（停牌后第一天）"""
        df = TestDataFactory.create_daily()

        # 前一天停牌
        df.loc[0, 'is_halt'] = True
        df.loc[0, 'volume'] = 0

        # 今天复牌
        df.loc[1, 'is_halt'] = False
        df.loc[1, 'volume'] = 2000000  # 放量

        # 复牌首日因子可能跳跃，这是正常的
        return df, []


# 便于调用的字典
ANOMALY_TEST_CASES = {
    'perfect': TestDataFactory.create_perfect_data,
    'price_out_of_range': TestDataFactory.create_price_out_of_range,
    'price_zero': TestDataFactory.create_price_zero,
    'ohlc_invalid': TestDataFactory.create_ohlc_invalid,
    'ohlc_low_gt_high': TestDataFactory.create_ohlc_low_gt_high,
    'adj_continuity_break': TestDataFactory.create_adj_continuity_break,
    'adj_factor_negative': TestDataFactory.create_adj_factor_negative,
    'volume_negative': TestDataFactory.create_volume_negative,
    'pct_chg_exceed': TestDataFactory.create_pct_chg_exceed,
    'limit_up': TestDataFactory.create_limit_up,
    'limit_down': TestDataFactory.create_limit_down,
    'single_row': TestDataFactory.create_single_row,
    'empty': TestDataFactory.create_empty,
    'missing_fields': TestDataFactory.create_missing_fields,
    'multi_anomalies': TestDataFactory.create_multi_anomalies,
    'suspended': TestDataFactory.create_suspended_stock,
    'resume_trading': TestDataFactory.create_resume_trading,
}
