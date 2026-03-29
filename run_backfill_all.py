#!/usr/bin/env python3
"""全A股回填脚本"""

import os
import sys
import json
import time

sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

from src.data.fetcher.backfill import BackfillFetcher

def main():
    # 加载股票代码
    with open('/tmp/all_a_codes.json', 'r') as f:
        codes = json.load(f)

    print(f"加载了 {len(codes)} 只股票")

    # 创建回填器
    fetcher = BackfillFetcher(
        stockdata_root='/Users/bruce/workspace/trade/StockData',
        start_date='2021-03-29',
        end_date='2026-03-28',
        codes=codes,
    )

    print(f"开始回填...")
    print(f"速率限制: {fetcher.rate_limit} calls/min")

    result = fetcher.fetch_all()

    print(f"\n=== 回填报告 ===")
    print(f"日期范围: {result.start_date} ~ {result.end_date}")
    print(f"总股票数: {result.total_stocks}")
    print(f"成功: {result.success_count}")
    print(f"隔离: {result.quarantined_count}")
    print(f"失败: {result.failed_count}")
    print(f"总记录: {result.total_records}")
    print(f"耗时: {result.duration_minutes:.1f} 分钟")

if __name__ == "__main__":
    main()
