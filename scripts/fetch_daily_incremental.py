#!/usr/bin/env python3
"""日线数据增量更新

功能:
- 只获取自上次更新以来的新数据
- 支持断点续传
- 记录更新进度
- 失败重试机制

使用方式:
    # 增量更新所有股票
    python3 scripts/fetch_daily_incremental.py

    # 更新指定股票
    python3 scripts/fetch_daily_incremental.py --codes 600519,000001

    # 强制更新(忽略缓存)
    python3 scripts/fetch_daily_incremental.py --force

    # 后台定时执行 (crontab)
    0 16 * * 1-5 cd /path/to/SwingTrade && python3 scripts/fetch_daily_incremental.py >> logs/daily_fetch.log 2>&1
"""

import argparse
import logging
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd
import akshare as ak

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/daily_fetch.log', mode='a')
    ]
)
logger = logging.getLogger("daily_fetch")

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
STOCKDATA_ROOT = PROJECT_ROOT / "StockData"
DAILY_DIR = STOCKDATA_ROOT / "raw" / "daily"
DAILY_DIR.mkdir(parents=True, exist_ok=True)


def get_last_trade_date() -> str:
    """获取最近交易日(排除周末)"""
    today = datetime.now()
    if today.weekday() == 5:  # Saturday
        today -= timedelta(days=1)
    elif today.weekday() == 6:  # Sunday
        today -= timedelta(days=2)
    return today.strftime('%Y%m%d')


def load_progress() -> dict:
    """加载更新进度"""
    progress_file = DAILY_DIR / "daily_fetch_progress.json"
    if not progress_file.exists():
        return {'last_update': None, 'completed': {}, 'failed': {}}
    try:
        with open(progress_file) as f:
            return json.load(f)
    except:
        return {'last_update': None, 'completed': {}, 'failed': {}}


def save_progress(progress: dict):
    """保存更新进度"""
    progress_file = DAILY_DIR / "daily_fetch_progress.json"
    progress['last_update'] = datetime.now().isoformat()
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def get_existing_codes() -> Set[str]:
    """获取已保存的股票代码"""
    existing = set()
    for f in DAILY_DIR.glob('*.parquet'):
        code = f.stem
        existing.add(code)
    return existing


def get_last_date_for_stock(code: str) -> Optional[str]:
    """获取某股票最后更新日期"""
    parquet_file = DAILY_DIR / f"{code}.parquet"
    if not parquet_file.exists():
        return None
    try:
        df = pd.read_parquet(parquet_file)
        if 'date' not in df.columns:
            return None
        last_date = pd.to_datetime(df['date']).max()
        return last_date.strftime('%Y%m%d')
    except:
        return None


def fetch_single_stock(
    code: str,
    start_date: str,
    end_date: str,
    retries: int = 3
) -> pd.DataFrame:
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


def fetch_incremental(
    codes: Optional[List[str]] = None,
    force: bool = False,
    delay: float = 0.3
) -> dict:
    """增量获取日线数据"""
    progress = load_progress()
    existing = get_existing_codes()

    # 确定要更新的股票
    if codes:
        codes_to_fetch = [c for c in codes if c in existing]
        logger.info(f"指定更新 {len(codes_to_fetch)} 只股票")
    else:
        codes_to_fetch = list(existing)

    if not codes_to_fetch:
        logger.warning("没有股票需要更新")
        return {}

    # 获取日期范围
    end_date = get_last_trade_date()
    logger.info(f"更新截止日期: {end_date}")

    results = {}
    success = 0
    failed = 0

    for i, code in enumerate(codes_to_fetch):
        # 检查上次更新日期
        last_date = get_last_date_for_stock(code) if not force else None
        start_date = last_date if last_date else '20250101'

        # 如果上次更新已是最新日期，跳过
        if start_date >= end_date:
            logger.debug(f"{code}: 已是最新")
            continue

        logger.info(f"[{i+1}/{len(codes_to_fetch)}] 更新 {code} ({start_date} -> {end_date})")

        df = fetch_single_stock(code, start_date, end_date)

        if not df.empty:
            if save_daily_data(code, df):
                results[code] = {'success': True, 'rows': len(df)}
                progress['completed'][code] = {
                    'last_update': datetime.now().isoformat(),
                    'rows': len(df)
                }
                success += 1
            else:
                results[code] = {'success': False, 'rows': 0}
                progress['failed'][code] = datetime.now().isoformat()
                failed += 1
        else:
            results[code] = {'success': False, 'rows': 0}
            failed += 1

        save_progress(progress)
        time.sleep(delay)

    logger.info(f"更新完成: {success} 成功, {failed} 失败")
    return results


def fetch_all_a_incremental(
    max_new: int = 50,
    delay: float = 0.4
) -> dict:
    """增量获取全A股(每天获取新增的)"""
    # 获取全A股列表
    try:
        df = ak.stock_info_a_code_name()
        all_codes = set(df['code'].tolist())
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        return {}

    existing = get_existing_codes()
    new_codes = all_codes - existing

    if not new_codes:
        logger.info("没有新增股票")
        return {}

    logger.info(f"发现 {len(new_codes)} 只新股票")

    # 获取前max_new只
    new_codes = list(new_codes)[:max_new]

    # 日期范围
    end_date = get_last_trade_date()
    start_date = '20250101'

    progress = load_progress()
    results = {}
    success = 0

    for i, code in enumerate(new_codes):
        logger.info(f"[{i+1}/{len(new_codes)}] 获取新股票 {code}")

        df = fetch_single_stock(code, start_date, end_date)

        if not df.empty:
            if save_daily_data(code, df):
                results[code] = {'success': True, 'rows': len(df)}
                progress['completed'][code] = {
                    'last_update': datetime.now().isoformat(),
                    'rows': len(df)
                }
                success += 1
            else:
                results[code] = {'success': False, 'rows': 0}
        else:
            results[code] = {'success': False, 'rows': 0}

        save_progress(progress)
        time.sleep(delay)

    logger.info(f"新增股票获取完成: {success}/{len(new_codes)}")
    return results


def main():
    parser = argparse.ArgumentParser(description='日线数据增量更新')
    parser.add_argument('--codes', type=str, help='指定股票代码(逗号分隔)')
    parser.add_argument('--force', action='store_true', help='强制更新(忽略缓存)')
    parser.add_argument('--new', action='store_true', help='获取新增股票')
    parser.add_argument('--max-new', type=int, default=50, help='每次最多获取新增股票数')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("日线数据增量更新")
    logger.info("=" * 60)

    if args.new:
        # 获取新增股票
        results = fetch_all_a_incremental(max_new=args.max_new)
    elif args.codes:
        # 更新指定股票
        codes = args.codes.split(',')
        results = fetch_incremental(codes=codes, force=args.force)
    else:
        # 增量更新所有股票
        results = fetch_incremental(force=args.force)

    # 统计
    total = len(results)
    success = sum(1 for r in results.values() if r.get('success'))
    logger.info(f"总计: {total}, 成功: {success}, 失败: {total - success}")

    # 打印进度
    progress = load_progress()
    logger.info(f"最后更新: {progress.get('last_update')}")


if __name__ == '__main__':
    main()
