"""东方财富数据源适配器

专注于资金流数据：
- 个股资金流（主力/超大单/大单/中单/小单）
- 行业资金流排名
- 北向资金（沪深港通）
"""

import time
import logging
from typing import Optional
import pandas as pd

from .base import DataSource
from ..exceptions import NetworkError, SourceError, ConfigurationError

logger = logging.getLogger(__name__)


class EastMoneySource(DataSource):
    """东方财富数据源（资金流专项数据源）

    数据接口：
    - 个股资金流：东方财富网-数据中心-资金流向
    - 行业资金流：东方财富网-数据中心-资金流向-板块资金流
    - 北向资金：东方财富网-数据中心-沪深港通
    """

    name = "eastmoney"

    # 东方财富接口通常没有严格的限流，但建议控制频率
    RATE_LIMIT_PER_SECOND = 5

    def __init__(self):
        """初始化东方财富数据源"""
        try:
            import akshare as ak
            self.ak = ak
        except ImportError:
            raise ConfigurationError(
                "akshare package is not installed",
                date=None
            )

        self._last_call_time = 0

    def _rate_limit_sleep(self):
        """简单的速率限制"""
        now = time.time()
        elapsed = now - self._last_call_time
        min_interval = 1.0 / self.RATE_LIMIT_PER_SECOND

        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self._last_call_time = time.time()

    def fetch_daily(
        self,
        code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        东方财富数据源不提供日线数据

        请使用 TushareSource.fetch_daily() 获取日线数据
        本接口仅保留以满足 DataSource 接口

        Returns:
            空 DataFrame
        """
        logger.warning("EastMoneySource does not provide daily data, use TushareSource instead")
        return pd.DataFrame()

    def fetch_stock_list(self) -> pd.DataFrame:
        """
        东方财富数据源不提供股票列表

        请使用 TushareSource.fetch_stock_list() 或 AkShareSource.fetch_stock_list()
        本接口仅保留以满足 DataSource 接口

        Returns:
            空 DataFrame
        """
        logger.warning("EastMoneySource does not provide stock list, use TushareSource instead")
        return pd.DataFrame()

    def fetch_individual_fund_flow(
        self,
        code: str,
        market: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取个股资金流数据

        Args:
            code: 股票代码，如 "600519"
            market: 市场标识，"sh" 或 "sz"，自动推断如果未提供

        Returns:
            DataFrame，包含列：
            - date: 日期
            - close: 收盘价
            - pct_chg: 涨跌幅
            - main_net_inflow: 主力净流入-净额
            - main_net_inflow_ratio: 主力净流入-净占比
            - super_net_inflow: 超大单净流入-净额
            - super_net_inflow_ratio: 超大单净流入-净占比
            - big_net_inflow: 大单净流入-净额
            - big_net_inflow_ratio: 大单净流入-净占比
            - medium_net_inflow: 中单净流入-净额
            - medium_net_inflow_ratio: 中单净流入-净占比
            - small_net_inflow: 小单净流入-净额
            - small_net_inflow_ratio: 小单净流入-净占比
        """
        self._rate_limit_sleep()

        # 自动推断市场
        if market is None:
            market = self._to_em_market(code)

        try:
            df = self.ak.stock_individual_fund_flow(
                stock=code,
                market=market
            )
        except Exception as e:
            raise NetworkError(
                f"Failed to fetch individual fund flow: {e}",
                code=code,
                date=None
            )

        if df is None or len(df) == 0:
            return pd.DataFrame()

        # 列名映射
        df = df.rename(columns={
            "日期": "date",
            "收盘价": "close",
            "涨跌幅": "pct_chg",
            "主力净流入-净额": "main_net_inflow",
            "主力净流入-净占比": "main_net_inflow_ratio",
            "超大单净流入-净额": "super_net_inflow",
            "超大单净流入-净占比": "super_net_inflow_ratio",
            "大单净流入-净额": "big_net_inflow",
            "大单净流入-净占比": "big_net_inflow_ratio",
            "中单净流入-净额": "medium_net_inflow",
            "中单净流入-净占比": "medium_net_inflow_ratio",
            "小单净流入-净额": "small_net_inflow",
            "小单净流入-净占比": "small_net_inflow_ratio"
        })

        df["code"] = code

        return df

    def fetch_industry_fund_flow(
        self,
        indicator: str = "今日"
    ) -> pd.DataFrame:
        """
        获取行业资金流排名

        Args:
            indicator: 时间窗口，"今日"、"5日" 或 "10日"

        Returns:
            DataFrame，包含列：
            - rank: 序号
            - sector: 行业名称
            - sector_index: 行业指数
            - pct_chg: 行业涨跌幅
            - inflow: 流入资金
            - outflow: 流出资金
            - net_inflow: 净额
            - stock_count: 公司家数
            - top_stock: 领涨股
            - top_stock_pct_chg: 领涨股涨跌幅
        """
        self._rate_limit_sleep()

        try:
            df = self.ak.stock_sector_fund_flow_rank(
                indicator=indicator,
                sector_type="行业资金流"
            )
        except Exception as e:
            raise NetworkError(
                f"Failed to fetch industry fund flow: {e}",
                date=None
            )

        if df is None or len(df) == 0:
            return pd.DataFrame()

        # 列名映射
        df = df.rename(columns={
            "序号": "rank",
            "名称": "sector",
            "行业指数": "sector_index",
            "行业-涨跌幅": "pct_chg",
            "流入资金": "inflow",
            "流出资金": "outflow",
            "净额": "net_inflow",
            "公司家数": "stock_count",
            "领涨股": "top_stock",
            "领涨股-涨跌幅": "top_stock_pct_chg"
        })

        return df

    def fetch_concept_fund_flow(
        self,
        indicator: str = "今日"
    ) -> pd.DataFrame:
        """
        获取概念资金流排名

        Args:
            indicator: 时间窗口，"今日"、"5日" 或 "10日"

        Returns:
            DataFrame，包含列：
            - rank: 序号
            - concept: 概念名称
            - pct_chg: 涨跌幅
            - inflow: 流入资金
            - outflow: 流出资金
            - net_inflow: 净额
            - stock_count: 相关股票数
            - top_stock: 领涨股
            - top_stock_pct_chg: 领涨股涨跌幅
        """
        self._rate_limit_sleep()

        try:
            df = self.ak.stock_sector_fund_flow_rank(
                indicator=indicator,
                sector_type="概念资金流"
            )
        except Exception as e:
            raise NetworkError(
                f"Failed to fetch concept fund flow: {e}",
                date=None
            )

        if df is None or len(df) == 0:
            return pd.DataFrame()

        # 列名映射
        df = df.rename(columns={
            "序号": "rank",
            "名称": "concept",
            "今日涨跌幅" if indicator == "今日" else f"{indicator}涨跌幅": "pct_chg",
            f"今日主力净流入-净额" if indicator == "今日" else f"{indicator}主力净流入-净额": "main_net_inflow",
            f"今日主力净流入-净占比" if indicator == "今日" else f"{indicator}主力净流入-净占比": "main_net_inflow_ratio",
            "领涨股": "top_stock",
            "领涨股-涨跌幅": "top_stock_pct_chg"
        })

        return df

    def fetch_hsgt_north_flow(self) -> pd.DataFrame:
        """
        获取沪深港通北向资金数据

        Returns:
            DataFrame，包含列：
            - trade_date: 交易日
            - type: 类型（沪港通/深港通）
            - direction: 方向（北向/南向）
            - net_inflow: 资金净流入
            - net_buy: 成交净买额
            - balance: 当日资金余额
            - rising_count: 上涨数
            - flat_count: 持平数
            - falling_count: 下跌数
            - index_name: 相关指数
            - index_pct_chg: 指数涨跌幅
        """
        self._rate_limit_sleep()

        try:
            df = self.ak.stock_hsgt_fund_flow_summary_em()
        except Exception as e:
            raise NetworkError(
                f"Failed to fetch HSGT north flow: {e}",
                date=None
            )

        if df is None or len(df) == 0:
            return pd.DataFrame()

        # 列名映射
        df = df.rename(columns={
            "交易日": "trade_date",
            "类型": "type",
            "板块": "board",
            "资金方向": "direction",
            "交易状态": "status",
            "成交净买额": "net_buy",
            "资金净流入": "net_inflow",
            "当日资金余额": "balance",
            "上涨数": "rising_count",
            "持平数": "flat_count",
            "下跌数": "falling_count",
            "相关指数": "index_name",
            "指数涨跌幅": "index_pct_chg"
        })

        return df

    def fetch_hsgt_north_flow_min(self) -> pd.DataFrame:
        """
        获取北向资金分钟级数据

        Returns:
            DataFrame，包含列：
            - date: 日期
            - time: 时间
            - shanghai: 沪股通
            - shenzhen: 深股通
            - total: 北向资金合计
        """
        self._rate_limit_sleep()

        try:
            df = self.ak.stock_hsgt_fund_min_em(symbol="北向资金")
        except Exception as e:
            raise NetworkError(
                f"Failed to fetch HSGT north flow minute data: {e}",
                date=None
            )

        if df is None or len(df) == 0:
            return pd.DataFrame()

        # 列名映射
        df = df.rename(columns={
            "日期": "date",
            "时间": "time",
            "沪股通": "shanghai",
            "深股通": "shenzhen",
            "北向资金": "total"
        })

        return df

    def is_available(self) -> bool:
        """检查东方财富数据源是否可用"""
        try:
            # 尝试获取行业资金流来测试连接
            self.ak.stock_sector_fund_flow_rank(
                indicator="今日",
                sector_type="行业资金流"
            )
            return True
        except Exception as e:
            logger.warning(f"EastMoney health check failed: {e}")
            return False

    def _to_em_market(self, code: str) -> str:
        """将股票代码转换为东方财富格式

        东方财富使用 sh/sz/bj
        """
        if code.startswith(("6", "5")):
            return "sh"
        elif code.startswith(("0", "1", "2", "3")):
            return "sz"
        elif code.startswith(("4", "8", "9")):
            return "bj"
        else:
            return "sh"
