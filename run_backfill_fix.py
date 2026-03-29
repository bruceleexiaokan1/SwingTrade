#!/usr/bin/env python3
"""缺失文件重新回填脚本 - 绕过损坏的status文件"""

import sys
import os
import json
import logging
import time

sys.path.insert(0, '/Users/bruce/workspace/trade/SwingTrade')

from src.data.fetcher.backfill import BackfillFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger('backfill_fix')


def main():
    stockdata_root = '/Users/bruce/workspace/trade/StockData'
    status_file = os.path.join(stockdata_root, 'status', 'backfill_2021-03-29_2026-03-28.json')

    # 读取当前进度
    with open(status_file) as f:
        data = json.load(f)

    completed = set(data['completed_codes'])
    raw_dir = os.path.join(stockdata_root, 'raw', 'daily')
    existing_files = {f.replace('.parquet', '') for f in os.listdir(raw_dir) if f.endswith('.parquet')}

    # 找出缺失的
    missing = [c for c in completed if c not in existing_files]
    print(f"Status completed: {len(completed)}")
    print(f"Existing files: {len(existing_files)}")
    print(f"Missing files: {len(missing)}")

    if not missing:
        print("No missing files!")
        return

    # 只回填缺失的文件
    print(f"\n重新回填 {len(missing)} 个缺失文件...")

    # 直接调用 fetcher 的方法，绕过进度检查
    fetcher = BackfillFetcher(
        stockdata_root=stockdata_root,
        start_date='2021-03-29',
        end_date='2026-03-28',
        codes=missing,
        rate_limit_buffer=0.8
    )

    # 手动处理，不依赖进度文件
    from src.data.fetcher.backfill import BackfillReport

    report = BackfillReport(
        start_date='2021-03-29',
        end_date='2026-03-28',
        total_stocks=len(missing)
    )

    for i, code in enumerate(missing):
        logger.info(f"[{i+1}/{len(missing)}] 处理 {code}")

        result = fetcher._fetch_stock(code)

        report.results.append(result)
        if result.status == "success":
            report.success_count += 1
            report.total_records += result.records_count
        elif result.status == "quarantined":
            report.quarantined_count += 1
            report.total_records += result.records_count
        else:
            report.failed_count += 1

        # 跳过 _save_progress，直接写文件验证
        # 验证文件是否写入
        expected_file = os.path.join(raw_dir, f"{code}.parquet")
        if result.status == "success" and not os.path.exists(expected_file):
            logger.error(f"文件未写入: {code}")
        elif result.status == "success":
            logger.info(f"写入成功 {code}: {result.records_count} 条")

        # 速率限制
        fetcher._rate_limit_sleep()

        # 每100个报告一次进度
        if (i + 1) % 100 == 0:
            print(f"进度: {i+1}/{len(missing)} - 成功:{report.success_count} 失败:{report.failed_count}")

    print(f"\n=== 回填报告 ===")
    print(f"成功: {report.success_count}")
    print(f"隔离: {report.quarantined_count}")
    print(f"失败: {report.failed_count}")

    # 更新进度文件
    print("\n更新进度文件...")
    data['completed_codes'] = list(completed)  # Keep original
    # Add newly completed codes
    newly_completed = [r.code for r in report.results if r.status in ('success', 'quarantined')]
    for code in newly_completed:
        if code not in data['completed_codes']:
            data['completed_codes'].append(code)

    with open(status_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"进度文件已更新，共 {len(data['completed_codes'])} 个完成")


if __name__ == '__main__':
    main()
