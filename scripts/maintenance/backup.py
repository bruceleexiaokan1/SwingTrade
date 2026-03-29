"""
StockData 备份脚本

备份关键数据到外部存储
"""

import os
import sqlite3
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# 备份根目录
BACKUP_ROOT = "/Users/bruce/backup/StockData"

# 需要备份的目录
BACKUP_TARGETS = [
    ("raw/daily", "raw_daily"),
    ("sqlite/market.db", "sqlite"),
]

# 保留策略（天）
RETENTION_DAYS = {
    "daily": 7,      # 每日增量：7天
    "weekly": 28,     # 每周：4周
    "monthly": 365,   # 每月：12个月
}


def get_stockdata_root() -> str:
    """获取 StockData 根目录"""
    return os.getenv('STOCKDATA_ROOT', '/Users/bruce/workspace/trade/StockData')


def get_backup_root() -> str:
    """获取备份根目录"""
    return os.getenv('BACKUP_ROOT', BACKUP_ROOT)


def create_backup_dirs() -> None:
    """创建备份目录结构"""
    backup_root = Path(get_backup_root())

    # 创建日期子目录
    today = datetime.now().strftime('%Y%m%d')
    daily_dir = backup_root / "daily" / today
    daily_dir.mkdir(parents=True, exist_ok=True)

    # 创建周/月备份目录
    (backup_root / "weekly").mkdir(parents=True, exist_ok=True)
    (backup_root / "monthly").mkdir(parents=True, exist_ok=True)


def backup_raw_daily() -> bool:
    """备份 raw/daily/ 目录"""
    stockdata_root = Path(get_stockdata_root())
    backup_root = Path(get_backup_root())

    source = stockdata_root / "raw" / "daily"
    if not source.exists():
        logger.warning(f"源目录不存在: {source}")
        return False

    today = datetime.now().strftime('%Y%m%d')
    target = backup_root / "daily" / today / "raw_daily"

    try:
        # 使用 rsync 风格复制
        if target.exists():
            shutil.rmtree(target)

        shutil.copytree(source, target)
        logger.info(f"备份完成: {source} -> {target}")
        return True
    except Exception as e:
        logger.error(f"备份失败: {e}")
        return False


def backup_sqlite() -> bool:
    """备份 SQLite 数据库（使用 sqlite3.backup API 获取一致快照）"""
    stockdata_root = Path(get_stockdata_root())
    backup_root = Path(get_backup_root())

    source = stockdata_root / "sqlite" / "market.db"
    if not source.exists():
        logger.warning(f"SQLite 文件不存在: {source}")
        return False

    today = datetime.now().strftime('%Y%m%d')
    target_dir = backup_root / "daily" / today / "sqlite"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "market.db"

    try:
        # 使用 sqlite3.backup API 获取一致快照（WAL 模式安全）
        with sqlite3.connect(str(source), timeout=30) as src_conn:
            with sqlite3.connect(str(target), timeout=30) as dst_conn:
                src_conn.backup(dst_conn)
        logger.info(f"SQLite 备份完成: {source} -> {target}")
        return True
    except Exception as e:
        logger.error(f"SQLite 备份失败: {e}")
        return False


def cleanup_old_backups() -> int:
    """清理过期的备份文件"""
    backup_root = Path(get_backup_root())
    cleaned = 0

    today = datetime.now()

    for backup_type, retention_days in RETENTION_DAYS.items():
        backup_dir = backup_root / backup_type
        if not backup_dir.exists():
            continue

        cutoff = today - timedelta(days=retention_days)

        for item in backup_dir.iterdir():
            if not item.is_dir():
                continue

            # 检查是否是日期目录
            try:
                item_date = datetime.strptime(item.name, '%Y%m%d')
                if item_date < cutoff:
                    shutil.rmtree(item)
                    logger.info(f"清理过期备份: {item}")
                    cleaned += 1
            except ValueError:
                # 不是日期目录，跳过
                continue

    return cleaned


def run_backup() -> dict:
    """执行备份"""
    results = {
        "date": datetime.now().strftime('%Y-%m-%d'),
        "success": True,
        "raw_daily": False,
        "sqlite": False,
        "cleaned": 0,
    }

    logger.info(f"开始备份: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 创建目录
    create_backup_dirs()

    # 备份 raw/daily
    results["raw_daily"] = backup_raw_daily()

    # 备份 SQLite
    results["sqlite"] = backup_sqlite()

    # 清理过期备份
    results["cleaned"] = cleanup_old_backups()

    results["success"] = results["raw_daily"] and results["sqlite"]

    if results["success"]:
        logger.info(f"备份完成，清理 {results['cleaned']} 个过期备份")
    else:
        logger.error("备份部分失败")

    return results


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='StockData 备份')
    parser.add_argument('--dry-run', action='store_true', help='仅预览，不执行')
    args = parser.parse_args()

    if args.dry_run:
        print("预览模式:")
        print(f"  备份目录: {get_backup_root()}")
        print(f"  数据目录: {get_stockdata_root()}")
        return 0

    results = run_backup()
    print(f"备份结果: {'成功' if results['success'] else '失败'}")
    print(f"  raw_daily: {'✅' if results['raw_daily'] else '❌'}")
    print(f"  sqlite: {'✅' if results['sqlite'] else '❌'}")
    print(f"  清理过期: {results['cleaned']} 个")

    return 0 if results['success'] else 1


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    exit(main())
