#!/usr/bin/env python3
"""全A股日线数据批量获取

获取全A股日线数据:
- 增量获取 (只获取尚未保存的股票)
- 分批执行 (避免超时)
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
logger = logging.getLogger("fetch_all_a")

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
STOCKDATA_ROOT = PROJECT_ROOT / "StockData"
DAILY_DIR = STOCKDATA_ROOT / "raw" / "daily"

# 获取进度
def load_progress() -> dict:
    """加载进度"""
    progress_file = DAILY_DIR / "fetch_progress_all_a.json"
    if not progress_file.exists():
        return {'completed': [], 'failed': [], 'last_update': None}
    try:
        with open(progress_file) as f:
            return json.load(f)
    except:
        return {'completed': [], 'failed': [], 'last_update': None}


def save_progress(progress: dict):
    """保存进度"""
    progress_file = DAILY_DIR / "fetch_progress_all_a.json"
    progress['last_update'] = datetime.now().isoformat()
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def get_all_a_codes() -> List[str]:
    """获取全A股股票代码列表"""
    df = ak.stock_info_a_code_name()
    # 过滤掉ST、退市等
    codes = df['code'].tolist()
    logger.info(f"全A股候选股票: {len(codes)} 只")
    return codes


def get_existing_codes() -> set:
    """获取已保存的股票代码"""
    existing = set()
    for f in DAILY_DIR.glob('*.parquet'):
        code = f.stem
        existing.add(code)
    logger.info(f"已保存股票: {len(existing)} 只")
    return existing


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

            # 添加市场后缀
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


def fetch_batch(
    stocks: List[str],
    start_date: str,
    end_date: str,
    batch_size: int = 50,
    delay: float = 0.3
) -> dict:
    """批量获取"""
    progress = load_progress()
    completed = set(progress.get('completed', []))
    failed = set(progress.get('failed', []))
    stocks_to_fetch = [s for s in stocks if s not in completed and s not in failed]

    results = {}
    success_count = 0

    total = len(stocks_to_fetch)
    logger.info(f"待获取股票: {total} 只")

    for i, code in enumerate(stocks_to_fetch):
        logger.info(f"[{i+1}/{total}] 获取 {code}")

        df = fetch_single_stock(code, start_date, end_date)

        if not df.empty:
            if save_daily_data(code, df):
                results[code] = {'success': True, 'rows': len(df)}
                completed.add(code)
                success_count += 1
            else:
                results[code] = {'success': False, 'rows': 0}
                failed.add(code)
        else:
            results[code] = {'success': False, 'rows': 0}
            failed.add(code)

        # 每20个保存一次进度
        if (i + 1) % 20 == 0:
            progress['completed'] = list(completed)
            progress['failed'] = list(failed)
            save_progress(progress)
            logger.info(f"进度: {len(completed)}/{total} 完成")

        time.sleep(delay)

    # 最终保存进度
    progress['completed'] = list(completed)
    progress['failed'] = list(failed)
    save_progress(progress)

    return results


def main():
    logger.info("=" * 60)
    logger.info("全A股日线数据批量获取")
    logger.info("=" * 60)

    start_date = '20250101'
    end_date = '20260328'

    logger.info(f"日期范围: {start_date} ~ {end_date}")

    # 获取全A股列表
    all_codes = get_all_a_codes()
    existing = get_existing_codes()

    # 找出需要获取的
    to_fetch = [c for c in all_codes if c not in existing]
    logger.info(f"需要获取: {len(to_fetch)} 只")

    if not to_fetch:
        logger.info("全部股票已获取完成")
        files = list(DAILY_DIR.glob('*.parquet'))
        logger.info(f"已保存文件: {len(files)}")
        return

    # 限制单次获取数量 (避免时间过长)
    max_fetch = 500
    if len(to_fetch) > max_fetch:
        logger.info(f"限制获取前 {max_fetch} 只 (按代码顺序)")
        to_fetch = to_fetch[:max_fetch]

    results = fetch_batch(to_fetch, start_date, end_date, delay=0.4)

    success = sum(1 for r in results.values() if r['success'])
    logger.info(f"完成: {success}/{len(results)} 成功")

    # 统计
    files = list(DAILY_DIR.glob('*.parquet'))
    logger.info(f"已保存文件: {len(files)}")

    progress = load_progress()
    logger.info(f"总完成: {len(progress.get('completed', []))} 只")
    logger.info(f"总失败: {len(progress.get('failed', []))} 只")


if __name__ == '__main__':
    main()
