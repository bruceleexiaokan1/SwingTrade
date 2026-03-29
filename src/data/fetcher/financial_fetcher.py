"""财务数据获取器

从同花顺获取财务数据:
- 利润表指标 (EPS等)
- 资产负债表 (净资产等)
- 现金流数据

支持增量更新和本地缓存
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

# 字段映射: THS字段 -> 标准字段
FIELD_MAPPING = {
    # 估值因子
    '基本每股收益': 'eps_basic',           # 基本EPS
    '稀释每股收益': 'eps_diluted',         # 稀释EPS
    '每股净资产': 'bps',                    # Book Value Per Share
    '市盈率(动态)': 'pe_dynamic',          # 动态PE (可能没有)
    '市净率': 'pb',                        # PB (可能没有)

    # 质量因子
    '净资产收益率': 'roe',                  # ROE (摊薄前)
    '净资产收益率-摊薄': 'roe_diluted',     # ROE (摊薄)
    '扣非净利润': 'non_recurring_profit',  # 扣非净利润
    '净利润': 'net_profit',                 # 净利润
    '营业总收入': 'revenue',                # 营业收入

    # 杠杆因子
    '资产负债率': 'debt_ratio',             # 资产负债率
    '流动比率': 'current_ratio',            # 流动比率
    '速动比率': 'quick_ratio',              # 速动比率
}


def calculate_disclosure_date(report_date: pd.Timestamp) -> pd.Timestamp:
    """
    根据财报期间计算披露截止日期

    A股财报披露规则:
    - Q1(3月报): 4月30日前披露
    - Q2(6月报): 8月31日前披露
    - Q3(9月报): 10月31日前披露
    - Q4(年报): 次年4月30日前披露

    Args:
        report_date: 财报期间截止日

    Returns:
        披露截止日期
    """
    year = report_date.year
    month = report_date.month

    if month == 3:  # Q1
        return pd.Timestamp(year=year, month=4, day=30)
    elif month == 6:  # Q2
        return pd.Timestamp(year=year, month=8, day=31)
    elif month == 9:  # Q3
        return pd.Timestamp(year=year, month=10, day=31)
    elif month == 12:  # Annual
        return pd.Timestamp(year=year+1, month=4, day=30)
    else:
        # 其他期间（如IPO前的财务数据），假设T+90天披露
        return report_date + pd.Timedelta(days=90)


def get_available_fields() -> List[str]:
    """获取当前可用的标准字段列表"""
    return list(FIELD_MAPPING.values())


class FinancialFetcher:
    """
    财务数据获取器

    从同花顺获取财务数据，支持增量更新
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化财务数据获取器

        Args:
            cache_dir: 缓存目录，默认使用 StockData/financial
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent.parent / "StockData" / "financial"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 接口映射
        self.interfaces = {
            'benefit': self._fetch_benefit,
            'debt': self._fetch_debt,
            'cash': self._fetch_cash,
            'abstract': self._fetch_abstract,
        }

    def fetch_stock(
        self,
        code: str,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        获取单只股票的财务数据

        Args:
            code: 股票代码 (如 '600519')
            max_retries: 最大重试次数

        Returns:
            DataFrame: 包含报告期和所有财务字段
        """
        logger.info(f"Fetching financial data for {code}")

        all_data = []

        for name, fetch_func in self.interfaces.items():
            for retry in range(max_retries):
                try:
                    df = fetch_func(code)
                    if df is not None and len(df) > 0:
                        # 重命名字段
                        df = self._map_fields(df)
                        all_data.append(df)
                        logger.info(f"  {name}: {len(df)} rows fetched")
                        break
                except Exception as e:
                    logger.warning(f"  {name} failed (retry {retry+1}/{max_retries}): {e}")
                    if retry < max_retries - 1:
                        time.sleep(1)

        if not all_data:
            logger.warning(f"No financial data fetched for {code}")
            return pd.DataFrame()

        # 合并所有数据
        result = self._merge_data(all_data)
        logger.info(f"  Total: {len(result)} rows, {len(result.columns)} columns")

        return result

    def _fetch_benefit(self, code: str) -> pd.DataFrame:
        """获取收益数据"""
        import akshare as ak

        df = ak.stock_financial_benefit_ths(symbol=code)

        if df is None or len(df) == 0:
            return pd.DataFrame()

        # 选择关键字段 (注意：THS接口有些字段带*前缀，每股收益有括号)
        cols_to_keep = ['报告期']
        key_fields = [
            '（一）基本每股收益', '（二）稀释每股收益',  # 每股收益的正确列名
            '*净利润', '*营业总收入', '*营业总成本',  # 带*前缀
            '扣除非经常性损益后的净利润',  # 这个不带*
        ]

        for col in key_fields:
            if col in df.columns:
                cols_to_keep.append(col)

        df = df[cols_to_keep].copy()

        # 清理列名中的*前缀
        rename_map = {c: c.replace('*', '') for c in df.columns if c.startswith('*')}
        df = df.rename(columns=rename_map)

        return df

    def _parse_number(self, series: pd.Series) -> pd.Series:
        """解析带亿/万单位的数字"""
        def parse_val(val):
            if pd.isna(val):
                return 0
            if isinstance(val, (int, float)):
                return val
            s = str(val)
            try:
                if '亿' in s:
                    return float(s.replace('亿', '')) * 1e8
                elif '万' in s:
                    return float(s.replace('万', '')) * 1e4
                elif '元' in s:
                    return float(s.replace('元', ''))
                else:
                    return float(s)
            except:
                return 0
        return series.apply(parse_val)

    def _fetch_debt(self, code: str) -> pd.DataFrame:
        """获取债务数据"""
        import akshare as ak

        df = ak.stock_financial_debt_ths(symbol=code)

        if df is None or len(df) == 0:
            return pd.DataFrame()

        # 选择关键字段
        cols_to_keep = ['报告期']
        key_fields = [
            '每股净资产',
            '资产负债率', '流动比率', '速动比率',
        ]

        for col in key_fields:
            if col in df.columns:
                cols_to_keep.append(col)

        df = df[cols_to_keep].copy()
        return df

    def _fetch_cash(self, code: str) -> pd.DataFrame:
        """获取现金流数据"""
        import akshare as ak

        df = ak.stock_financial_cash_ths(symbol=code)

        if df is None or len(df) == 0:
            return pd.DataFrame()

        # 现金流数据暂不映射到因子，先只保留报告期
        return df[['报告期']].copy() if '报告期' in df.columns else pd.DataFrame()

    def _fetch_abstract(self, code: str) -> pd.DataFrame:
        """获取财务摘要"""
        import akshare as ak

        df = ak.stock_financial_abstract_ths(symbol=code)

        if df is None or len(df) == 0:
            return pd.DataFrame()

        # 解析百分号字段
        def parse_percent(val):
            if pd.isna(val):
                return None
            if isinstance(val, (int, float)):
                return val
            s = str(val).replace('%', '')
            try:
                return float(s)
            except:
                return None

        # 处理百分号字段
        pct_fields = ['销售毛利率', '销售净利率', '净资产收益率', '净资产收益率-摊薄', '资产负债率']
        for field in pct_fields:
            if field in df.columns:
                df[field] = df[field].apply(parse_percent)

        # 重命名字段
        rename_map = {
            '销售毛利率': 'gross_margin',
            '销售净利率': 'net_margin',
            '净资产收益率': 'roe',
            '净资产收益率-摊薄': 'roe_diluted',
            '资产负债率': 'debt_ratio',
            '每股净资产': 'bps',  # 市净率需要
            '基本每股收益': 'eps_basic',  # 市盈率需要
        }
        df = df.rename(columns=rename_map)

        # 只保留唯一列，避免与benefit重复
        unique_cols = ['报告期', 'gross_margin', 'net_margin', 'roe', 'roe_diluted', 'debt_ratio', 'bps', 'eps_basic']
        cols_to_keep = [c for c in unique_cols if c in df.columns]
        df = df[cols_to_keep].copy()

        # 计算每条数据的最早可用日期 (披露截止日)
        # 这是穿越风险防护的核心：只有在这个日期之后才应该使用该数据
        df['report_date'] = pd.to_datetime(df['报告期'])
        df['available_date'] = df['report_date'].apply(calculate_disclosure_date)
        # 转换为字符串存储，避免parquet时间戳问题
        df['available_date'] = df['available_date'].dt.strftime('%Y-%m-%d')
        df['report_date'] = df['report_date'].dt.strftime('%Y-%m-%d')

        return df

    def _map_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """映射THS字段到标准字段名"""
        rename_map = {}
        for ths_col, std_col in FIELD_MAPPING.items():
            if ths_col in df.columns:
                rename_map[ths_col] = std_col

        df = df.rename(columns=rename_map)
        return df

    def _merge_data(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """合并多个财务数据DataFrame"""
        if not dataframes:
            return pd.DataFrame()

        if len(dataframes) == 1:
            return dataframes[0]

        # 按报告期合并
        result = dataframes[0]
        for df in dataframes[1:]:
            if '报告期' not in df.columns:
                continue
            # 左连接合并
            cols_to_add = [c for c in df.columns if c != '报告期']
            result = result.merge(
                df[['报告期'] + cols_to_add],
                on='报告期',
                how='left'
            )

        # 清理数据类型 - 统一转为数值 (跳过日期列)
        date_cols = ['报告期', 'report_date', 'available_date']
        for col in result.columns:
            if col in date_cols:
                continue
            # 尝试转换为数值类型
            try:
                result[col] = pd.to_numeric(result[col], errors='coerce')
            except Exception:
                pass

        return result

    def save_to_cache(self, code: str, df: pd.DataFrame) -> None:
        """保存到本地缓存"""
        if df.empty:
            return

        # 确保数据类型一致 (跳过日期列)
        date_cols = ['报告期', 'report_date', 'available_date']
        for col in df.columns:
            if col in date_cols:
                continue
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass

        cache_file = self.cache_dir / f"{code}.parquet"
        df.to_parquet(cache_file, index=False)
        logger.info(f"Saved {code} financial data to {cache_file}")

    def load_from_cache(self, code: str) -> Optional[pd.DataFrame]:
        """从本地缓存加载"""
        cache_file = self.cache_dir / f"{code}.parquet"
        if not cache_file.exists():
            return None

        try:
            df = pd.read_parquet(cache_file)
            logger.info(f"Loaded {code} financial data from cache: {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache for {code}: {e}")
            return None

    def fetch_with_cache(
        self,
        code: str,
        force_update: bool = False
    ) -> pd.DataFrame:
        """
        带缓存的获取

        Args:
            code: 股票代码
            force_update: 是否强制更新缓存

        Returns:
            DataFrame: 财务数据
        """
        if not force_update:
            cached = self.load_from_cache(code)
            if cached is not None and len(cached) > 0:
                return cached

        df = self.fetch_stock(code)
        if not df.empty:
            self.save_to_cache(code, df)

        return df

    def batch_fetch(
        self,
        codes: List[str],
        delay: float = 1.0
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票的财务数据

        Args:
            codes: 股票代码列表
            delay: 请求间隔(秒)，避免频率限制

        Returns:
            Dict: code -> DataFrame
        """
        results = {}

        for i, code in enumerate(codes):
            logger.info(f"[{i+1}/{len(codes)}] Fetching {code}")
            df = self.fetch_with_cache(code)
            results[code] = df
            time.sleep(delay)  # 避免频率限制

        return results


def fetch_financial_for_factor(
    code: str,
    field: str,
    cache_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    便捷函数：获取单只股票的单项财务数据

    Args:
        code: 股票代码
        field: 标准字段名 (如 'eps_ttm', 'roe')
        cache_dir: 缓存目录

    Returns:
        DataFrame: date, code, field (标准化格式)
    """
    fetcher = FinancialFetcher(cache_dir)
    df = fetcher.fetch_with_cache(code)

    if df.empty:
        return pd.DataFrame(columns=['date', 'code', field])

    # 转换报告期为日期格式
    df['date'] = pd.to_datetime(df['报告期'])

    # 只保留需要的字段
    result = df[['date', '报告期']].copy()
    result['code'] = code

    # 如果目标字段存在，映射到标准名
    if field in df.columns:
        result[field] = df[field]
    elif field == 'eps_ttm' and 'eps_basic' in df.columns:
        # EPS_TTM 使用基本EPS (简化处理)
        result[field] = df['eps_basic']
    else:
        result[field] = None

    return result[['date', 'code', field]]


# 导出常量
AVAILABLE_FIELDS = get_available_fields()

__all__ = [
    "FinancialFetcher",
    "fetch_financial_for_factor",
    "FIELD_MAPPING",
    "AVAILABLE_FIELDS",
    "get_available_fields",
]
