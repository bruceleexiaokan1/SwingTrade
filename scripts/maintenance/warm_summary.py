"""
StockData 温数据汇总生成

每日采集完成后执行，从 raw/daily/*.parquet 聚合生成当日全市场汇总
用于快速选股扫描
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def get_stockdata_root() -> str:
    """获取 StockData 根目录"""
    import os
    return os.getenv('STOCKDATA_ROOT', '/Users/bruce/workspace/trade/StockData')


def generate_daily_summary(date: str) -> bool:
    """
    生成指定日期的全市场汇总

    Args:
        date: 日期 (YYYY-MM-DD)

    Returns:
        bool: 是否成功
    """
    stockdata_root = Path(get_stockdata_root())
    raw_dir = stockdata_root / "raw" / "daily"
    warm_dir = stockdata_root / "warm" / "daily_summary"

    # 确保目录存在
    warm_dir.mkdir(parents=True, exist_ok=True)

    date_str = date.replace('-', '')
    output_path = warm_dir / f"{date_str}.parquet"

    logger.info(f"生成 {date} 温数据汇总...")

    # 扫描所有 parquet 文件
    parquet_files = list(raw_dir.glob("*.parquet"))

    if not parquet_files:
        logger.warning(f"没有找到日线数据: {raw_dir}")
        return False

    # 读取所有股票数据
    all_data = []

    for pf in parquet_files:
        try:
            df = pd.read_parquet(str(pf))

            # 过滤指定日期
            df['date'] = pd.to_datetime(df['date'])
            day_data = df[df['date'].dt.strftime('%Y-%m-%d') == date]

            if not day_data.empty:
                all_data.append(day_data)
        except Exception as e:
            logger.warning(f"读取失败 {pf}: {e}")
            continue

    if not all_data:
        logger.warning(f"没有找到 {date} 的数据")
        return False

    # 合并所有数据
    combined = pd.concat(all_data, ignore_index=True)

    # 选择汇总所需的列
    summary_cols = ['code', 'close', 'pct_chg', 'volume', 'turnover']
    available_cols = [c for c in summary_cols if c in combined.columns]

    summary = combined[available_cols].copy()

    # 如果有股票名称，添加
    if 'name' in combined.columns:
        # 取每只股票的第一条名称
        names = combined.groupby('code')['name'].first().reset_index()
        summary = summary.merge(names, on='code', how='left')

    # 按代码排序
    summary = summary.sort_values('code').reset_index(drop=True)

    # 写入
    summary.to_parquet(str(output_path), engine='pyarrow', compression='snappy')

    logger.info(f"汇总生成完成: {output_path} ({len(summary)} 只股票)")
    return True


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='生成温数据汇总')
    parser.add_argument('--date', help='日期 (YYYY-MM-DD)，默认今天', default=None)
    args = parser.parse_args()

    if args.date:
        date = args.date
    else:
        # 默认今天
        date = datetime.now().strftime('%Y-%m-%d')

    success = generate_daily_summary(date)
    return 0 if success else 1


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    exit(main())
