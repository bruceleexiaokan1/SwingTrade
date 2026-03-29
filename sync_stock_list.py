#!/usr/bin/env python3
"""同步全市场股票列表到SQLite（使用AkShare）"""

import os
import sys
import sqlite3

def sync_stock_list():
    """从AkShare同步全市场股票列表"""
    import akshare as ak

    print("从AkShare获取股票列表...")

    all_stocks = []

    # 获取沪市
    try:
        df_sh = ak.stock_info_sh_name_code()
        df_sh = df_sh.rename(columns={
            "证券代码": "code",
            "证券简称": "name"
        })
        df_sh["market"] = "sh"
        all_stocks.append(df_sh[["code", "name", "market"]])
        print(f"沪市: {len(df_sh)} 只")
    except Exception as e:
        print(f"沪市获取失败: {e}")

    # 获取深市
    try:
        df_sz = ak.stock_info_sz_name_code()
        df_sz = df_sz.rename(columns={
            "A股代码": "code",
            "A股简称": "name"
        })
        df_sz["market"] = "sz"
        all_stocks.append(df_sz[["code", "name", "market"]])
        print(f"深市: {len(df_sz)} 只")
    except Exception as e:
        print(f"深市获取失败: {e}")

    # 获取北交所
    try:
        df_bj = ak.stock_info_bj_name_code()
        df_bj = df_bj.rename(columns={
            "证券代码": "code",
            "证券简称": "name"
        })
        df_bj["market"] = "bj"
        all_stocks.append(df_bj[["code", "name", "market"]])
        print(f"北交所: {len(df_bj)} 只")
    except Exception as e:
        print(f"北交所获取失败: {e}")

    if not all_stocks:
        print("获取失败")
        return

    df = pd.concat(all_stocks, ignore_index=True)
    print(f"总计: {len(df)} 只")

    # 连接SQLite
    db_path = "/Users/bruce/workspace/trade/StockData/sqlite/market.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode=WAL')

    # 获取现有股票
    existing = set(row[0] for row in conn.execute("SELECT code FROM stocks").fetchall())
    print(f"现有股票: {len(existing)} 只")

    # 插入或更新
    inserted = 0
    updated = 0
    for _, row in df.iterrows():
        code = row['code']
        name = row['name']
        market = row['market']

        if code in existing:
            conn.execute("""
                UPDATE stocks SET name=?, market=?, updated_at=datetime('now')
                WHERE code=?
            """, (name, market, code))
            updated += 1
        else:
            conn.execute("""
                INSERT INTO stocks (code, name, market, is_active)
                VALUES (?, ?, ?, 1)
            """, (code, name, market))
            inserted += 1

    conn.commit()

    # 验证
    total = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
    active = conn.execute("SELECT COUNT(*) FROM stocks WHERE is_active=1").fetchone()[0]

    print(f"\n同步完成:")
    print(f"  新增: {inserted}")
    print(f"  更新: {updated}")
    print(f"  总数: {total} (活跃: {active})")

    conn.close()

if __name__ == "__main__":
    import pandas as pd
    sync_stock_list()
