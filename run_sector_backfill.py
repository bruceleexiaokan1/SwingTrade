#!/usr/bin/env python3
"""板块数据回填脚本 - EastMoney 全市场板块"""

import sys
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

from src.data.fetcher.sector_fetcher import SectorDataFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger('sector_backfill')


def main():
    # 创建板块数据获取器
    fetcher = SectorDataFetcher(
        cache_dir='/Users/bruce/workspace/trade/StockData/sector_cache',
        cache_ttl_hours=24 * 365
    )

    # 获取全市场板块列表
    logger.info("获取全市场板块列表...")
    all_sectors = fetcher.get_all_sectors()

    if not all_sectors:
        logger.error("无法获取板块列表")
        return

    logger.info(f"全市场共有 {len(all_sectors)} 个概念板块")

    # 回填参数
    start_date = '2025-03-01'
    end_date = '2026-03-28'

    success = 0
    failed = 0
    skipped = 0

    # 检查已缓存的板块
    cache_info = fetcher.get_cache_info()
    cached_files = set()
    for f in Path(fetcher.cache_dir).glob("*.parquet"):
        if f.name.endswith("_constituents.parquet"):
            continue
        cached_files.add(f.stem)

    logger.info(f"已有缓存: {len(cached_files)} 个板块")

    for i, sector in enumerate(all_sectors):
        name = sector['name']
        code = sector['code']

        # 跳过已有缓存
        if name in cached_files:
            logger.debug(f"[{i+1}/{len(all_sectors)}] 跳过(已缓存) {name}")
            skipped += 1
            continue

        try:
            logger.info(f"[{i+1}/{len(all_sectors)}] 回填 {name} ({code})")

            df = fetcher.backfill_sector_data(
                sector_name=name,
                start_date=start_date,
                end_date=end_date,
                force_update=True
            )

            if not df.empty:
                logger.info(f"  -> {len(df)} 条记录, {df['date'].min()} ~ {df['date'].max()}")
                success += 1
            else:
                logger.warning(f"  -> 无数据")
                failed += 1

        except Exception as e:
            logger.error(f"  -> 失败: {e}")
            failed += 1

        time.sleep(0.3)

    print(f"\n=== 全市场板块回填报告 ===")
    print(f"板块总数: {len(all_sectors)}")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    print(f"跳过(已缓存): {skipped}")
    print(f"日期范围: {start_date} ~ {end_date}")

    info = fetcher.get_cache_info()
    print(f"\n缓存状态:")
    print(f"  文件数: {info['n_files']}")
    print(f"  总大小: {info['total_size_mb']} MB")


if __name__ == '__main__':
    main()
