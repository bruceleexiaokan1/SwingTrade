#!/usr/bin/env python3
"""批量获取沪深300日线数据

获取沪深300成分股日线数据:
- 批量获取 (分批执行避免超时)
- 断点续传
- 进度保存
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import akshare as ak

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fetch_hs300")

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
STOCKDATA_ROOT = PROJECT_ROOT / "StockData"
DAILY_DIR = STOCKDATA_ROOT / "raw" / "daily"

# 沪深300成分股代码 (手动维护，来源于中证指数官网)
# 这里使用一个大市值股票列表作为替代
HS300_STOCKS = [
    # 银行
    '600000', '600015', '600016', '600036', '601009', '601166', '601169', '601229', '601288', '601328', '601398', '601818', '601988', '601998',
    # 保险
    '601318', '601601', '601628', '601628',
    # 券商
    '600030', '600837', '600958', '600999', '601066', '601211', '601688', '601788',
    # 白酒/消费
    '600519', '000858', '000568', '600809', '002304', '000895', '600887',
    # 家电
    '000333', '000651', '600690', '600104',
    # 汽车
    '600104', '600741', '000625', '002594', '601238',
    # 地产
    '000002', '600048', '600066', '600383', '600395', '001979',
    # 医药
    '600276', '000538', '603259', '600196', '601607', '000423',
    # 科技/通信
    '600036', '000063', '600050', '601728', '601166',
    # 新能源
    '600900', '600585', '601012', '002594', '300750', '600438',
    # 基建
    '601186', '601390', '601668', '601618', '600028',
    # 化工
    '600309', '601233', '002601', '600596',
    # 钢铁
    '600019', '000898', '600010',
    # 有色
    '600111', '603799', '601600',
    # 煤炭
    '601088', '600188', '601699',
    # 电力
    '600900', '600795', '600023',
    # 传媒
    '600037', '002027', '603444',
    # 交通运输
    '601021', '600115', '600221', '600026',
    # 电子
    '000725', '002475', '600703',
    # 食品
    '603288', '603605', '600197',
]

# 去重
HS300_STOCKS = list(set(HS300_STOCKS))
logger.info(f"目标股票数量: {len(HS300_STOCKS)}")


def fetch_single_stock(code: str, start_date: str, end_date: str, retries: int = 3) -> pd.DataFrame:
    """获取单只股票日线数据"""
    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period='daily',
                start_date=start_date,
                end_date=end_date,
                adjust='qfq'
            )
            if df is None or len(df) == 0:
                return pd.DataFrame()

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
            df['date'] = pd.to_datetime(df['date'])

            if code.startswith('6'):
                df['code'] = df['code'] + '.SH'
            else:
                df['code'] = df['code'] + '.SZ'

            return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            continue
    return pd.DataFrame()


def save_daily_data(code: str, df: pd.DataFrame) -> bool:
    """保存日线数据"""
    if df.empty:
        return False

    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    code_raw = code.replace('.SH', '').replace('.SZ', '')
    target_file = DAILY_DIR / f"{code_raw}.parquet"

    try:
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


def load_progress() -> set:
    """加载进度"""
    progress_file = DAILY_DIR / "fetch_progress.json"
    if not progress_file.exists():
        return set()
    try:
        with open(progress_file) as f:
            data = json.load(f)
        return set(data.get('completed', []))
    except:
        return set()


def save_progress(completed: set):
    """保存进度"""
    progress_file = DAILY_DIR / "fetch_progress.json"
    with open(progress_file, 'w') as f:
        json.dump({'completed': list(completed)}, f)


def fetch_batch(stocks: List[str], start_date: str, end_date: str, batch_size: int = 20, delay: float = 0.3) -> dict:
    """批量获取"""
    # 加载进度
    completed = load_progress()
    stocks_to_fetch = [s for s in stocks if s not in completed]

    results = {}
    success_count = 0

    for i, code in enumerate(stocks_to_fetch):
        logger.info(f"[{i+1}/{len(stocks_to_fetch)}] 获取 {code}")

        df = fetch_single_stock(code, start_date, end_date)

        if not df.empty:
            if save_daily_data(code, df):
                results[code] = {'success': True, 'rows': len(df)}
                completed.add(code)
                success_count += 1
            else:
                results[code] = {'success': False, 'rows': 0}
        else:
            results[code] = {'success': False, 'rows': 0}

        # 每5个保存一次进度
        if (i + 1) % 5 == 0:
            save_progress(completed)

        time.sleep(delay)

    # 最终保存进度
    save_progress(completed)

    return results


def main():
    logger.info("=" * 60)
    logger.info("沪深300日线数据批量获取")
    logger.info("=" * 60)

    start_date = '20250101'
    end_date = '20260328'

    logger.info(f"日期范围: {start_date} ~ {end_date}")
    logger.info(f"目标股票: {len(HS300_STOCKS)} 只")

    results = fetch_batch(HS300_STOCKS, start_date, end_date, delay=0.5)

    success = sum(1 for r in results.values() if r['success'])
    logger.info(f"完成: {success}/{len(results)} 成功")

    # 统计
    files = list(DAILY_DIR.glob('*.parquet'))
    logger.info(f"已保存文件: {len(files)}")


if __name__ == '__main__':
    main()
