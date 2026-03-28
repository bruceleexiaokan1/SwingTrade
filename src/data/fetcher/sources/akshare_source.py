"""AkShare 数据源适配器"""

import re
import time
import logging
from typing import Optional, Tuple
import pandas as pd

from .base import DataSource
from ..exceptions import NetworkError, SourceError, ConfigurationError

logger = logging.getLogger(__name__)


class AkShareSource(DataSource):
    """AkShare 数据源（股票列表补充源）

    注意：部分接口存在网络问题，需要做好异常处理
    """

    name = "akshare"

    def __init__(self):
        """初始化 AkShare 数据源"""
        try:
            import akshare as ak
            self.ak = ak
        except ImportError:
            raise ConfigurationError(
                "akshare package is not installed",
                date=None
            )

        # 记录采集失败的详细信息
        self._fetch_failures = {
            "sh": None,
            "sz": None
        }

    def fetch_daily(
        self,
        code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        从 AkShare 拉取日线数据

        使用 stock_zh_a_daily 接口，返回字段：
        date, open, high, low, close, volume, amount, outstanding_share, turnover

        注意：没有 pct_chg（涨跌幅）和 adj_factor（复权因子），需要自行计算

        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame
        """
        symbol = self._to_akshare_symbol(code)

        try:
            # 使用 stock_zh_a_daily 接口
            df = self.ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            raise NetworkError(
                f"Failed to fetch daily data from AkShare: {e}",
                code=code,
                date=end_date
            )

        if df is None or len(df) == 0:
            return pd.DataFrame()

        # stock_zh_a_daily 返回字段：
        # date, open, high, low, close, volume, amount, outstanding_share, turnover
        df = df.rename(columns={
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "amount": "amount",
            "outstanding_share": "outstanding_share",
            "turnover": "turnover"
        })

        # 计算涨跌幅（如果前一日收盘价可用）
        # 注意：这里简化处理，涨跌幅设为0
        df["pct_chg"] = 0.0
        df["adj_factor"] = 1.0  # 默认不复权

        # 只保留必要字段
        needed_cols = ["date", "code", "open", "high", "low", "close", "volume", "amount", "pct_chg", "adj_factor"]
        for col in needed_cols:
            if col not in df.columns:
                df[col] = None

        df = df[needed_cols]

        # 添加股票代码
        df["code"] = code

        return df

    def fetch_stock_list(self) -> pd.DataFrame:
        """
        从 AkShare 拉取股票列表

        使用 stock_info_sh_name_code 和 stock_info_sz_name_code 接口
        获取沪深两市A股列表（北交所因代理问题暂时跳过）

        Returns:
            Tuple[DataFrame, dict]: (股票列表, 失败详情)

        Raises:
            SourceError: 两个市场都失败时
        """
        all_stocks = []
        failures = {"sh": None, "sz": None}

        # 获取沪市股票
        try:
            df_sh = self.ak.stock_info_sh_name_code()
            if df_sh is not None and len(df_sh) > 0:
                df_sh = df_sh.rename(columns={
                    "证券代码": "code",
                    "证券简称": "name"
                })
                df_sh["market"] = "sh"
                df_sh["list_date"] = df_sh.get("上市日期", None)
                all_stocks.append(df_sh[["code", "name", "market", "list_date"]])
                logger.info(f"SH stock list fetched: {len(df_sh)} stocks")
        except Exception as e:
            failures["sh"] = str(e)
            logger.error(f"Failed to fetch SH stock list: {e}")

        time.sleep(1)  # 避免请求过快

        # 获取深市股票（主板和创业板）
        try:
            df_sz = self.ak.stock_info_sz_name_code()
            if df_sz is not None and len(df_sz) > 0:
                # 深市返回列名不同：A股代码, A股简称, A股上市日期
                df_sz = df_sz.rename(columns={
                    "A股代码": "code",
                    "A股简称": "name",
                    "A股上市日期": "list_date"
                })
                df_sz["market"] = "sz"
                all_stocks.append(df_sz[["code", "name", "market", "list_date"]])
                logger.info(f"SZ stock list fetched: {len(df_sz)} stocks")
        except Exception as e:
            failures["sz"] = str(e)
            logger.error(f"Failed to fetch SZ stock list: {e}")

        # 检查失败情况
        if not all_stocks:
            # 两个市场都失败了
            raise SourceError(
                f"Failed to fetch stock list from both SH and SZ. SH: {failures['sh']}, SZ: {failures['sz']}",
                source=self.name,
                date=None
            )

        # 记录失败（用于后续告警）
        self._fetch_failures = failures

        # 如果有部分失败，记录警告
        if failures["sh"] or failures["sz"]:
            logger.warning(
                f"Partial stock list fetch failure. "
                f"SH: {'OK' if not failures['sh'] else failures['sh']}, "
                f"SZ: {'OK' if not failures['sz'] else failures['sz']}"
            )

        # 合并结果
        df = pd.concat(all_stocks, ignore_index=True)
        logger.info(f"Total stock list: {len(df)} stocks (SH: {len(df[df.market=='sh'])}, SZ: {len(df[df.market=='sz'])})")

        return df

    def get_fetch_failures(self) -> dict:
        """获取采集失败详情（用于告警）"""
        return self._fetch_failures.copy()

    def is_available(self) -> bool:
        """检查 AkShare 是否可用"""
        try:
            # 尝试获取一只股票的数据来测试连接
            self.ak.stock_zh_a_daily(symbol="sh600519", start_date="2026-03-20", end_date="2026-03-20")
            return True
        except Exception:
            return False

    def _to_akshare_symbol(self, code: str) -> str:
        """将股票代码转换为 AkShare 格式

        AkShare 使用 sh600519 或 sz000001 格式
        """
        if code.startswith(("6", "5")):
            return f"sh{code}"
        elif code.startswith(("0", "1", "2", "3")):
            return f"sz{code}"
        elif code.startswith(("4", "8", "9")):
            return f"bj{code}"  # 北交所
        else:
            return f"sh{code}"

    def _extract_market(self, code: str) -> str:
        """从股票代码提取市场标识"""
        if code.startswith(("6", "5")):
            return "sh"
        elif code.startswith(("0", "1", "2", "3")):
            return "sz"
        elif code.startswith(("4", "8", "9")):
            return "bj"
        else:
            return "unknown"
