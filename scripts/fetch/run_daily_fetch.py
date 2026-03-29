#!/usr/bin/env python3
"""
日线数据采集入口脚本

用于 crontab 定时执行：
30 16 * * 1-5 /Users/bruce/workspace/trade/SwingTrade/scripts/fetch/run_daily_fetch.py >> /Users/bruce/workspace/trade/StockData/logs/fetch_daily.log 2>&1

采集时间窗口：16:00 - 17:30
"""

import os
import sys
import json
from datetime import datetime

# 添加项目根目录到 path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# StockData 根目录
STOCKDATA_ROOT = "/Users/bruce/workspace/trade/StockData"

# 日志目录
LOG_DIR = os.path.join(STOCKDATA_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def main():
    """主入口"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting daily fetch...")

    try:
        from src.data.fetcher import DailyFetcher

        fetcher = DailyFetcher(
            stockdata_root=STOCKDATA_ROOT,
            target_date=None,  # 自动获取上一个交易日
            use_akshare_verify=True
        )

        report = fetcher.fetch_all()

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetch completed")
        print(f"Success: {report.summary['success_count']}, Failed: {report.summary['network_failed_count']}")

        # 保存日报
        report_path = os.path.join(STOCKDATA_ROOT, "status", f"daily_report_{report.date.replace('-', '')}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        return 0 if report.summary['network_failed_count'] == 0 else 1

    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
