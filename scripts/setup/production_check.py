#!/usr/bin/env python3
"""
StockData 生产环境检查

检查生产部署所需的环境配置是否正确
"""

import os
import sys
import shutil
from pathlib import Path


def check_env_vars():
    """检查环境变量"""
    print("=== 环境变量检查 ===")

    required = {
        'STOCKDATA_ROOT': '/Users/bruce/workspace/trade/StockData',
        'EMAIL_PASSWORD': None,  # 只要设置了就行
    }

    optional = {
        'TUSHARE_TOKEN': None,
        'BACKUP_ROOT': '/Users/bruce/backup/StockData',
        'EMAIL_SMTP_HOST': 'smtp.qq.com',
        'EMAIL_SMTP_PORT': '587',
    }

    all_ok = True

    for var, expected in required.items():
        value = os.getenv(var)
        if value:
            if expected and value != expected:
                print(f"  ⚠️  {var}: {value} (建议: {expected})")
            else:
                print(f"  ✅ {var}: {value}")
        else:
            print(f"  ❌ {var}: 未设置")
            all_ok = False

    for var, default in optional.items():
        value = os.getenv(var) or default
        print(f"  ✅ {var}: {value}")

    return all_ok


def check_directories():
    """检查目录权限"""
    print("\n=== 目录检查 ===")

    dirs = {
        'STOCKDATA_ROOT': Path(os.getenv('STOCKDATA_ROOT', '/Users/bruce/workspace/trade/StockData')),
        'BACKUP_ROOT': Path(os.getenv('BACKUP_ROOT', '/Users/bruce/backup/StockData')),
    }

    all_ok = True

    for name, path in dirs.items():
        if path.exists():
            # 检查是否可写
            test_file = path / '.write_test'
            try:
                test_file.touch()
                test_file.unlink()
                print(f"  ✅ {name}: {path} (可写)")
            except:
                print(f"  ❌ {name}: {path} (不可写)")
                all_ok = False
        else:
            print(f"  ⚠️  {name}: {path} (不存在，将自动创建)")

    return all_ok


def check_sqlite():
    """检查 SQLite 数据库"""
    print("\n=== SQLite 检查 ===")

    db_path = Path(os.getenv('STOCKDATA_ROOT', '/Users/bruce/workspace/trade/StockData')) / 'sqlite' / 'market.db'

    if not db_path.exists():
        print(f"  ⚠️  数据库不存在: {db_path}")
        print(f"     首次采集时会自动创建")
        return True

    print(f"  ✅ 数据库存在: {db_path}")

    size = db_path.stat().st_size
    print(f"     大小: {size / 1024:.1f} KB")

    # 检查 WAL 文件
    wal_path = db_path.with_suffix('.db-wal')
    shm_path = db_path.with_suffix('.db-shm')

    if wal_path.exists():
        print(f"  ✅ WAL 文件: {wal_path.stat().st_size / 1024:.1f} KB")

    return True


def check_scripts():
    """检查关键脚本"""
    print("\n=== 脚本检查 ===")

    root = Path(os.getenv('STOCKDATA_ROOT', '/Users/bruce/workspace/trade/StockData'))
    scripts_dir = root.parent / 'SwingTrade' / 'scripts'

    required_scripts = [
        'fetch/run_daily_fetch.py',
        'monitor/health_check.py',
        'maintenance/backup.py',
        'maintenance/warm_summary.py',
    ]

    all_ok = True

    for script in required_scripts:
        path = scripts_dir / script
        if path.exists():
            print(f"  ✅ {script}")
        else:
            print(f"  ❌ {script} (不存在)")
            all_ok = False

    return all_ok


def check_alert():
    """测试告警配置"""
    print("\n=== 告警配置检查 ===")

    password = os.getenv('EMAIL_PASSWORD')
    if password:
        print(f"  ✅ EMAIL_PASSWORD 已设置")
    else:
        print(f"  ⚠️  EMAIL_PASSWORD 未设置 (将跳过邮件告警)")
        return False

    # 尝试导入 alert 模块检查配置
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.utils.alert import send_alert

        print(f"  ✅ 告警模块可导入")

        # 简单测试
        # send_alert("INFO", "StockData 生产环境检查", {"test": True})

        return True
    except Exception as e:
        print(f"  ⚠️  告警模块导入失败: {e}")
        return False


def main():
    print("StockData 生产环境检查")
    print("=" * 50)

    results = []

    results.append(("环境变量", check_env_vars()))
    results.append(("目录权限", check_directories()))
    results.append(("SQLite", check_sqlite()))
    results.append(("脚本", check_scripts()))
    results.append(("告警", check_alert()))

    print("\n" + "=" * 50)
    print("检查结果汇总:")
    print("=" * 50)

    for name, ok in results:
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")

    all_ok = all(ok for _, ok in results)

    if all_ok:
        print("\n✅ 所有检查通过，可以启动采集")
        return 0
    else:
        print("\n⚠️  部分检查未通过，请修复后重试")
        return 1


if __name__ == '__main__':
    sys.exit(main())
