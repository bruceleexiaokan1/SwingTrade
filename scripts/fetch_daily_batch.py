#!/usr/bin/env python3
"""批量获取日线数据 - 用于因子库验证

使用AkShare获取日线数据:
- 50只代表性股票
- 1年历史数据 (2025-01-01 ~ 2026-03-28)
- 保存到 StockData/raw/daily/
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import akshare as ak

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fetch_daily_batch")

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
STOCKDATA_ROOT = PROJECT_ROOT / "StockData"
DAILY_DIR = STOCKDATA_ROOT / "raw" / "daily"

# 目标股票列表 (50只代表性股票，涵盖不同行业和市值)
TARGET_STOCKS = [
    # 白酒/消费
    '600519', '000858', '000568', '600809', '002304',
    # 银行
    '600036', '601328', '600000', '601166', '000001',
    # 保险
    '601318', '601628', '601601',
    # 券商
    '600030', '601211', '000776',
    # 科技/互联网
    '600036', '601127', '002475', '300750', '688981',
    # 医药
    '600276', '000538', '603259', '301573',
    # 新能源
    '300274', '002459', '601012', '600438',
    # 地产链
    '000002', '600048', '001979', '600383',
    # 家电
    '000333', '600690', '000651',
    # 基建/制造
    '601390', '601668', '601186', '600585',
    # 汽车
    '600104', '002594', '000625',
    # 化工
    '600309', '601233', '002601',
    # 通信
    '601728', '600050', '000063',
    # 传媒
    '002027', '600037', '603444',
]

# 去重
TARGET_STOCKS = list(set(TARGET_STOCKS))
logger.info(f"目标股票数量: {len(TARGET_STOCKS)}")


def fetch_single_stock(code: str, start_date: str, end_date: str, retries: int = 3) -> pd.DataFrame:
    """
    获取单只股票日线数据

    Args:
        code: 股票代码 (如 '600519')
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        retries: 重试次数

    Returns:
        DataFrame with standardized columns
    """
    for attempt in range(retries):
        try:
            # AkShare接口
            df = ak.stock_zh_a_hist(
                symbol=code,
                period='daily',
                start_date=start_date,
                end_date=end_date,
                adjust='qfq'  # 前复权
            )

            if df is None or len(df) == 0:
                logger.warning(f"{code}: 无数据")
                return pd.DataFrame()

            # 重命名列为标准格式
            rename_map = {
                '日期': 'date',
                '股票代码': 'code',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_chg',
                '涨跌额': 'chg',
                '换手率': 'turnover_rate',
            }
            df = df.rename(columns=rename_map)

            # 添加市场后缀
            if code.startswith('6'):
                df['code'] = df['code'] + '.SH'
            else:
                df['code'] = df['code'] + '.SZ'

            # 确保数据类型
            df['date'] = pd.to_datetime(df['date'])

            return df

        except Exception as e:
            logger.warning(f"{code} attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2)
            continue

    logger.error(f"{code}: 获取失败")
    return pd.DataFrame()


def save_daily_data(code: str, df: pd.DataFrame) -> bool:
    """保存日线数据到Parquet"""
    if df.empty:
        return False

    DAILY_DIR.mkdir(parents=True, exist_ok=True)

    # 转换code格式 (600519.SH -> 600519)
    code_raw = code.replace('.SH', '').replace('.SZ', '')
    target_file = DAILY_DIR / f"{code_raw}.parquet"

    try:
        # 如果已存在，合并
        if target_file.exists():
            existing = pd.read_parquet(target_file)
            combined = pd.concat([existing, df]).drop_duplicates(subset=['date'])
            combined = combined.sort_values('date')
        else:
            combined = df

        combined.to_parquet(target_file, engine='pyarrow', compression='snappy')
        return True
    except Exception as e:
        logger.error(f"保存失败 {code}: {e}")
        return False


def fetch_batch(stocks: list, start_date: str, end_date: str, delay: float = 0.3) -> dict:
    """
    批量获取日线数据

    Args:
        stocks: 股票代码列表
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        delay: 请求间隔(秒)

    Returns:
        dict: {code: success: bool, rows: int}
    """
    results = {}

    for i, code in enumerate(stocks):
        logger.info(f"[{i+1}/{len(stocks)}] 获取 {code}")

        df = fetch_single_stock(code, start_date, end_date)

        if not df.empty:
            success = save_daily_data(code, df)
            results[code] = {'success': success, 'rows': len(df)}
            logger.info(f"  -> 成功: {len(df)} 行")
        else:
            results[code] = {'success': False, 'rows': 0}
            logger.warning(f"  -> 失败")

        time.sleep(delay)  # 避免频率限制

    return results


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("日线数据批量获取")
    logger.info("=" * 60)

    # 日期范围
    start_date = '20250101'
    end_date = '20260328'

    logger.info(f"日期范围: {start_date} ~ {end_date}")
    logger.info(f"目标股票: {len(TARGET_STOCKS)} 只")

    # 批量获取
    results = fetch_batch(TARGET_STOCKS, start_date, end_date, delay=0.5)

    # 统计
    success_count = sum(1 for r in results.values() if r['success'])
    total_rows = sum(r['rows'] for r in results.values())

    logger.info("=" * 60)
    logger.info(f"完成: {success_count}/{len(TARGET_STOCKS)} 成功")
    logger.info(f"总记录数: {total_rows}")

    # 列出已保存的文件
    saved_files = list(DAILY_DIR.glob('*.parquet'))
    logger.info(f"已保存文件: {len(saved_files)}")

    # 样本验证
    if saved_files:
        sample = pd.read_parquet(saved_files[0])
        logger.info(f"样本文件 {saved_files[0].name}: {len(sample)} 行")
        logger.info(f"列名: {list(sample.columns)}")
        logger.info(f"日期范围: {sample['date'].min()} ~ {sample['date'].max()}")


if __name__ == '__main__':
    main()
