#!/usr/bin/env python3
"""数据完整性验证脚本

验证：
1. 核心股票池数据完整性
2. 指数数据完整性
3. ATR 覆盖率
4. 技术指标可用性
"""

import os
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.fetcher.price_converter import convert_to_forward_adj


def load_core_stocks():
    """加载核心股票池配置"""
    config_path = Path(__file__).parent.parent.parent / "config" / "core_stocks.json"
    with open(config_path) as f:
        config = json.load(f)
    return [s["code"] for s in config["stocks"]], config["stocks"]


def verify_stocks(stockdata_root: str, expected_min_rows: int = 1000):
    """验证股票数据"""
    print("=== 股票数据验证 ===")

    codes, stocks = load_core_stocks()
    daily_dir = Path(stockdata_root) / "raw" / "daily"

    results = {"total": len(codes), "passed": 0, "failed": []}

    for stock in stocks:
        code = stock["code"]
        parquet_file = daily_dir / f"{code}.parquet"

        if not parquet_file.exists():
            results["failed"].append(f"{code}: 文件不存在")
            print(f"  {code} ({stock['name']}): 文件不存在 ✗")
            continue

        try:
            df = pd.read_parquet(parquet_file)

            # 检查行数
            if len(df) < expected_min_rows:
                results["failed"].append(f"{code}: 数据不足 ({len(df)} < {expected_min_rows})")
                print(f"  {code} ({stock['name']}): 数据不足 ({len(df)} 行) ✗")
                continue

            # 检查必要字段
            required_cols = ["date", "close", "volume", "adj_factor", "close_adj"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                results["failed"].append(f"{code}: 缺失字段 {missing}")
                print(f"  {code} ({stock['name']}): 缺失字段 {missing} ✗")
                continue

            # 检查空值
            null_counts = df[required_cols].isnull().sum()
            if null_counts.any():
                results["failed"].append(f"{code}: 存在空值 {null_counts[null_counts > 0].to_dict()}")
                print(f"  {code} ({stock['name']}): 存在空值 ✗")
                continue

            # 检查复权价格合理性
            if (df["close_adj"] <= 0).any():
                results["failed"].append(f"{code}: 复权价格异常")
                print(f"  {code} ({stock['name']}): 复权价格异常 ✗")
                continue

            results["passed"] += 1
            print(f"  {code} ({stock['name']}): {len(df)} 行 ✓")

        except Exception as e:
            results["failed"].append(f"{code}: {str(e)}")
            print(f"  {code} ({stock['name']}): {e} ✗")

    print(f"\n股票验证: {results['passed']}/{results['total']} 通过")
    return results


def verify_indices(stockdata_root: str, expected_min_rows: int = 1000):
    """验证指数数据"""
    print("\n=== 指数数据验证 ===")

    index_dir = Path(stockdata_root) / "raw" / "index"
    indices = [
        ("000001.SH", "上证指数"),
        ("000300.SH", "沪深300"),
        ("000016.SH", "上证50"),
        ("399001.SZ", "深证成指"),
        ("399006.SZ", "创业板指"),
        ("000852.SH", "中证1000"),
    ]

    results = {"total": len(indices), "passed": 0, "failed": []}

    for code, name in indices:
        parquet_file = index_dir / f"{code}.parquet"

        if not parquet_file.exists():
            results["failed"].append(f"{code}: 文件不存在")
            print(f"  {name} ({code}): 文件不存在 ✗")
            continue

        try:
            df = pd.read_parquet(parquet_file)

            if len(df) < expected_min_rows:
                results["failed"].append(f"{code}: 数据不足 ({len(df)} < {expected_min_rows})")
                print(f"  {name} ({code}): 数据不足 ({len(df)} 行) ✗")
                continue

            # 检查必要字段
            required_cols = ["date", "close", "open", "high", "low", "volume"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                results["failed"].append(f"{code}: 缺失字段 {missing}")
                print(f"  {name} ({code}): 缺失字段 {missing} ✗")
                continue

            # 检查点位合理性 (0 < close < 20000)
            if not ((df["close"] > 0) & (df["close"] < 20000)).all():
                results["failed"].append(f"{code}: 点位异常")
                print(f"  {name} ({code}): 点位异常 ✗")
                continue

            results["passed"] += 1
            print(f"  {name} ({code}): {len(df)} 行, 范围 {df['close'].min():.2f}~{df['close'].max():.2f} ✓")

        except Exception as e:
            results["failed"].append(f"{code}: {str(e)}")
            print(f"  {name} ({code}): {e} ✗")

    print(f"\n指数验证: {results['passed']}/{results['total']} 通过")
    return results


def verify_atr_coverage(stockdata_root: str):
    """验证 ATR 覆盖率"""
    print("\n=== ATR 覆盖率验证 ===")

    codes, _ = load_core_stocks()
    daily_dir = Path(stockdata_root) / "raw" / "daily"

    total_records = 0
    valid_atr_records = 0

    for code in codes:
        parquet_file = daily_dir / f"{code}.parquet"
        if not parquet_file.exists():
            continue

        df = pd.read_parquet(parquet_file)
        if len(df) < 14:  # 至少需要14天计算ATR
            continue

        total_records += len(df)

        # 计算 ATR (14日)
        df = df.sort_values("date")
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean()

        valid_count = atr.notna().sum()
        valid_atr_records += valid_count

    coverage = (valid_atr_records / total_records * 100) if total_records > 0 else 0
    # ATR 需要 14 天窗口，前 13 天无法计算是正常的
    # 最低覆盖率 ≈ (1211-13)/1211 = 98.9%，98% 是合理的下界
    print(f"ATR 覆盖率: {coverage:.2f}% ({valid_atr_records}/{total_records})")

    return {"coverage": coverage, "valid": coverage >= 98}


def verify_forward_adj(stockdata_root: str):
    """验证前复权转换"""
    print("\n=== 前复权转换验证 ===")

    codes, _ = load_core_stocks()  # 验证所有
    daily_dir = Path(stockdata_root) / "raw" / "daily"

    all_passed = True

    for code in codes:
        parquet_file = daily_dir / f"{code}.parquet"
        if not parquet_file.exists():
            continue

        df = pd.read_parquet(parquet_file)

        try:
            result = convert_to_forward_adj(df.copy())

            # 验证前复权列存在
            if "forward_close" not in result.columns:
                print(f"  {code}: 前复权列不存在 ✗")
                all_passed = False
                continue

            # 验证前复权价格合理性
            if (result["forward_close"] <= 0).any():
                print(f"  {code}: 前复权价格异常 ✗")
                all_passed = False
                continue

            # 验证转换公式正确性（前复权价 = 后复权价 * (最新因子 / 历史因子)）
            latest_adj = result["adj_factor"].iloc[-1]
            expected_forward = result["close_adj"] * (latest_adj / result["adj_factor"])
            diff = (result["forward_close"] - expected_forward).abs()
            if diff.max() > 0.01:  # 允许小数误差
                print(f"  {code}: 转换公式错误 ✗")
                all_passed = False
                continue

            print(f"  {code}: 前复权转换正常 ✓")

        except Exception as e:
            print(f"  {code}: 转换失败 {e} ✗")
            all_passed = False

    return {"passed": all_passed}


def main():
    stockdata_root = os.environ.get("STOCKDATA_ROOT", "/Users/bruce/workspace/trade/StockData")

    print(f"StockData Root: {stockdata_root}")
    print("=" * 50)

    stock_results = verify_stocks(stockdata_root)
    index_results = verify_indices(stockdata_root)
    atr_results = verify_atr_coverage(stockdata_root)
    fwd_results = verify_forward_adj(stockdata_root)

    print("\n" + "=" * 50)
    print("=== 验证总结 ===")

    all_passed = (
        len(stock_results["failed"]) == 0 and
        len(index_results["failed"]) == 0 and
        atr_results["valid"] and
        fwd_results["passed"]
    )

    if all_passed:
        print("✓ 所有验证通过，置信度 100%")
        return 0
    else:
        print("✗ 部分验证失败")
        if stock_results["failed"]:
            print(f"  股票失败: {len(stock_results['failed'])} 只")
        if index_results["failed"]:
            print(f"  指数失败: {len(index_results['failed'])} 只")
        if not atr_results["valid"]:
            print(f"  ATR覆盖率不足: {atr_results['coverage']:.2f}%")
        if not fwd_results["passed"]:
            print(f"  前复权转换失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
