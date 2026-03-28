"""
warm_summary 测试
"""

import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, 'scripts')

from maintenance.warm_summary import generate_daily_summary


class TestWarmSummary:
    """温数据汇总测试"""

    def test_generate_daily_summary(self, temp_stockdata):
        """生成指定日期汇总"""
        raw_dir = temp_stockdata / "raw" / "daily"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # 创建两只股票的日线数据
        df1 = pd.DataFrame({
            'date': pd.to_datetime(['2026-03-29', '2026-03-30']),
            'code': ['000001', '000001'],
            'open': [10.0, 10.2],
            'high': [10.5, 10.7],
            'low': [9.8, 10.0],
            'close': [10.2, 10.4],
            'volume': [1000000, 1100000],
            'pct_chg': [0.02, 0.02],
            'turnover': [0.05, 0.055],
        })

        df2 = pd.DataFrame({
            'date': pd.to_datetime(['2026-03-29']),
            'code': ['000002'],
            'open': [20.0],
            'high': [20.5],
            'low': [19.8],
            'close': [20.2],
            'volume': [2000000],
            'pct_chg': [0.01],
            'turnover': [0.04],
        })

        df1.to_parquet(str(raw_dir / "000001.parquet"), engine='pyarrow')
        df2.to_parquet(str(raw_dir / "000002.parquet"), engine='pyarrow')

        # 生成汇总
        result = generate_daily_summary('2026-03-29')

        assert result is True

        # 验证汇总文件
        warm_dir = temp_stockdata / "warm" / "daily_summary"
        summary_path = warm_dir / "20260329.parquet"
        assert summary_path.exists()

        summary = pd.read_parquet(str(summary_path))
        assert len(summary) == 2
        assert '000001' in summary['code'].values
        assert '000002' in summary['code'].values

    def test_no_data_for_date(self, temp_stockdata):
        """指定日期无数据"""
        raw_dir = temp_stockdata / "raw" / "daily"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # 创建一只股票但日期不匹配
        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-03-28']),
            'code': ['000001'],
            'close': [10.2],
            'volume': [1000000],
            'pct_chg': [0.02],
            'turnover': [0.05],
        })
        df.to_parquet(str(raw_dir / "000001.parquet"), engine='pyarrow')

        # 生成汇总（日期不匹配）
        result = generate_daily_summary('2026-03-29')

        assert result is False
