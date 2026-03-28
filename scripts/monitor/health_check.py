"""
StockData 健康检查

定时执行，检查数据采集状态，发送每日报告
"""

import os
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from scripts.utils.alert import send_alert, ALERT_CONFIG

logger = logging.getLogger(__name__)

# 配置
DEFAULT_THRESHOLDS = {
    'success_rate_error': 0.95,      # < 95% ERROR
    'success_rate_warning': 0.99,     # 95-99% WARNING
}

CHECKPOINT_FILE = "status/health_checkpoint.json"
MAX_CODES_IN_EMAIL = 20


@dataclass
class StorageStats:
    """存储统计"""
    raw_daily_mb: float = 0
    warm_mb: float = 0
    sqlite_mb: float = 0
    total_stocks: int = 0
    quarantine_mb: float = 0


@dataclass
class HealthAlert:
    """健康告警"""
    date: str
    level: str
    success_count: int
    total_count: int
    success_rate: float
    network_failed: list
    quality_rejected: list
    retry_failed: list
    other_failed: list
    start_time: str = ""
    end_time: str = ""

    @property
    def total_failed(self) -> int:
        return len(self.network_failed) + len(self.quality_rejected) + len(self.retry_failed) + len(self.other_failed)


def get_smtp_password() -> Optional[str]:
    """从环境变量获取 SMTP 密码"""
    return os.getenv('EMAIL_PASSWORD')


def get_stockdata_root() -> str:
    """获取 StockData 根目录"""
    return os.getenv('STOCKDATA_ROOT', '/Users/bruce/workspace/trade/StockData')


def get_storage_stats() -> StorageStats:
    """获取存储统计"""
    stockdata_root = Path(get_stockdata_root())
    stats = StorageStats()

    # raw/daily
    raw_dir = stockdata_root / "raw" / "daily"
    if raw_dir.exists():
        total_bytes = sum(f.stat().st_size for f in raw_dir.glob("*.parquet"))
        stats.raw_daily_mb = total_bytes / (1024 * 1024)
        stats.total_stocks = len(list(raw_dir.glob("*.parquet")))

    # warm
    warm_dir = stockdata_root / "warm" / "daily_summary"
    if warm_dir.exists():
        total_bytes = sum(f.stat().st_size for f in warm_dir.glob("*.parquet"))
        stats.warm_mb = total_bytes / (1024 * 1024)

    # sqlite
    sqlite_path = stockdata_root / "sqlite" / "market.db"
    if sqlite_path.exists():
        stats.sqlite_mb = sqlite_path.stat().st_size / (1024 * 1024)

    # quarantine
    q_dir = stockdata_root / "quarantine"
    if q_dir.exists():
        total_bytes = sum(f.stat().st_size for f in q_dir.rglob("*.parquet"))
        stats.quarantine_mb = total_bytes / (1024 * 1024)

    return stats


def get_check_date() -> str:
    """根据执行时间确定检查哪天的日报

    - 09:00-12:00: 检查昨天的数据（采集已在17:30完成）
    - 其他时间: 检查今天的数据
    """
    now = datetime.now()
    if 9 <= now.hour < 12:
        return (now - timedelta(days=1)).strftime('%Y-%m-%d')
    return now.strftime('%Y-%m-%d')


def get_report_path(date: str) -> Path:
    """获取日报路径"""
    stockdata_root = get_stockdata_root()
    date_str = date.replace('-', '')
    return Path(stockdata_root) / "status" / f"daily_report_{date_str}.json"


def load_report(date: str) -> Optional[dict]:
    """加载日报"""
    report_path = get_report_path(date)
    if not report_path.exists():
        logger.warning(f"日报不存在: {report_path}")
        return None

    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"日报加载失败: {e}")
        return None


def parse_report(report: dict) -> HealthAlert:
    """解析日报为 HealthAlert"""
    summary = report.get('summary', {})

    # 提取失败股票
    network_failed = [f['code'] for f in report.get('network_failed', [])]
    quality_rejected = [f['code'] for f in report.get('quality_rejected', [])]
    retry_failed = [f['code'] for f in report.get('retry_failed', [])]
    other_failed = [f['code'] for f in report.get('errors', [])]

    return HealthAlert(
        date=report.get('date', ''),
        level="",
        success_count=summary.get('success_count', 0),
        total_count=summary.get('total_count', 0),
        success_rate=summary.get('success_rate', 0),
        network_failed=network_failed,
        quality_rejected=quality_rejected,
        retry_failed=retry_failed,
        other_failed=other_failed,
        start_time=report.get('start_time', ''),
        end_time=report.get('end_time', ''),
    )


def calculate_level(alert: HealthAlert, thresholds: dict = None) -> str:
    """计算告警级别"""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    if alert.success_rate < thresholds['success_rate_error']:
        return "ERROR"
    elif alert.success_rate < thresholds['success_rate_warning']:
        return "WARNING"
    return "INFO"


def format_codes(codes: list, max_show: int = MAX_CODES_IN_EMAIL) -> str:
    """格式化股票代码列表"""
    if not codes:
        return "无"
    if len(codes) <= max_show:
        return ", ".join(codes)
    shown = ", ".join(codes[:max_show])
    return f"{shown}... (+{len(codes) - max_show}只)"


def build_alert_email(alert: HealthAlert, storage_stats: StorageStats = None) -> tuple[str, str]:
    """构建告警邮件"""
    stockdata_root = get_stockdata_root()

    # Subject
    if alert.level == "ERROR":
        status_emoji = "❌"
        status_text = "异常"
    elif alert.level == "WARNING":
        status_emoji = "⚠️"
        status_text = "警告"
    else:
        status_emoji = "✅"
        status_text = "正常"

    subject = f"[StockData {alert.level}] {alert.date} 采集{status_text} | 成功率 {alert.success_rate:.1%}"

    # Body
    body_lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📊 采集日报 - {alert.date}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        f"状态: {status_emoji} {status_text}",
        f"成功率: {alert.success_rate:.1%} ({alert.success_count}/{alert.total_count})",
        f"采集时间: {alert.start_time} - {alert.end_time}",
        f"失败总数: {alert.total_failed}",
    ]

    # 存储统计
    if storage_stats:
        body_lines.extend([
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "💾 存储统计",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"日线数据: {storage_stats.raw_daily_mb:.1f} MB ({storage_stats.total_stocks} 只股票)",
            f"温数据汇总: {storage_stats.warm_mb:.1f} MB",
            f"SQlite索引: {storage_stats.sqlite_mb:.1f} MB",
            f"隔离数据: {storage_stats.quarantine_mb:.1f} MB",
        ])

    # 失败详情（如果有）
    if alert.total_failed > 0:
        body_lines.extend([
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "❌ 失败详情",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ])

        # 网络失败
        if alert.network_failed:
            body_lines.extend([
                "",
                f"【网络失败】{len(alert.network_failed)}只",
                f"  影响: {format_codes(alert.network_failed)}",
                f"  原因: 数据源响应超时",
                f"  建议: 检查网络或稍后重试",
            ])

        # 质量拒绝
        if alert.quality_rejected:
            body_lines.extend([
                "",
                f"【质量拒绝】{len(alert.quality_rejected)}只",
                f"  影响: {format_codes(alert.quality_rejected)}",
                f"  原因: 价格/OHLC异常被拒绝",
                f"  建议: 检查数据源数据质量",
            ])

        # 重试失败
        if alert.retry_failed:
            body_lines.extend([
                "",
                f"【重试失败】{len(alert.retry_failed)}只",
                f"  影响: {format_codes(alert.retry_failed)}",
                f"  原因: 多次重试后仍失败",
                f"  建议: 检查数据源状态",
            ])

        # 其他错误
        if alert.other_failed:
            body_lines.extend([
                "",
                f"【其他错误】{len(alert.other_failed)}只",
                f"  影响: {format_codes(alert.other_failed)}",
            ])

    # 后续动作
    body_lines.extend([
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "📋 后续动作",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"□ 查看完整日报: {stockdata_root}/status/daily_report_{alert.date.replace('-', '')}.json",
        "□ 如需手动重试失败数据，请执行采集脚本",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"Sent by StockData Health Checker | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ])

    body = "\n".join(body_lines)
    return subject, body


def should_send_alert(date: str, level: str) -> bool:
    """检查是否需要发送告警（去重）"""
    if level == "INFO":
        return False  # 正常不发送

    stockdata_root = get_stockdata_root()
    checkpoint_path = Path(stockdata_root) / CHECKPOINT_FILE

    # 读取检查点
    checkpoint = {}
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
        except Exception:
            pass

    # 检查是否在冷却期内
    key = f"{level}_{date}"
    last_alert = checkpoint.get('last_alerts', {}).get(key)

    if last_alert:
        last_time = datetime.fromisoformat(last_alert)
        if datetime.now() - last_time < timedelta(hours=DEFAULT_THRESHOLDS['alert_cooldown_hours']):
            logger.info(f"告警在冷却期内，跳过: {key}")
            return False

    return True


def update_checkpoint(date: str, level: str):
    """更新检查点"""
    stockdata_root = get_stockdata_root()
    checkpoint_path = Path(stockdata_root) / CHECKPOINT_FILE

    checkpoint = {}
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
        except Exception:
            pass

    if 'last_alerts' not in checkpoint:
        checkpoint['last_alerts'] = {}

    key = f"{level}_{date}"
    checkpoint['last_alerts'][key] = datetime.now().isoformat()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def verify_smtp_config() -> bool:
    """验证 SMTP 配置"""
    password = get_smtp_password()
    if not password:
        logger.error("EMAIL_PASSWORD 环境变量未设置，跳过邮件告警")
        return False

    host = ALERT_CONFIG.get('smtp_host')
    if not host:
        logger.error("EMAIL_SMTP_HOST 环境变量未设置")
        return False

    return True


def run_health_check(date: str = None) -> bool:
    """执行健康检查

    Args:
        date: 检查日期，默认自动判断

    Returns:
        bool: 是否发送了告警
    """
    # 1. 确定检查日期
    if date is None:
        date = get_check_date()

    logger.info(f"执行健康检查: {date}")

    # 2. 验证 SMTP 配置
    if not verify_smtp_config():
        logger.warning("SMTP 配置不完整，跳过告警发送")
        return False

    # 3. 加载日报
    report = load_report(date)
    storage_stats = get_storage_stats()
    if report is None:
        # 日报不存在，发送 ERROR 告警
        subject = f"[StockData ERROR] {date} 采集未执行"
        body_lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📊 StockData 采集日报 - {date}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            "❌ 采集日报不存在",
            "",
            "可能原因:",
            "1. 采集脚本未执行",
            "2. 采集脚本执行失败",
            "3. 日报文件路径错误",
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "💾 存储统计",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"日线数据: {storage_stats.raw_daily_mb:.1f} MB ({storage_stats.total_stocks} 只股票)",
            f"温数据汇总: {storage_stats.warm_mb:.1f} MB",
            f"SQlite索引: {storage_stats.sqlite_mb:.1f} MB",
            f"隔离数据: {storage_stats.quarantine_mb:.1f} MB",
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Sent by StockData Health Checker | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        body = "\n".join(body_lines)
        if should_send_alert(date, "ERROR"):
            send_alert("ERROR", f"{date} 采集未执行", {"date": date})
            update_checkpoint(date, "ERROR")
            logger.info(f"发送告警: {subject}")
        return True

    # 4. 解析日报
    alert = parse_report(report)

    # 5. 计算告警级别
    alert.level = calculate_level(alert)

    # 6. 获取存储统计
    storage_stats = get_storage_stats()

    # 7. 构建邮件
    subject, body = build_alert_email(alert, storage_stats)

    # 8. 发送告警（ERROR/WARNING 级别走冷却机制，INFO 级别每日必发）
    if alert.level == "INFO":
        # INFO 级别：每日报告，直接发送
        send_alert(alert.level, f"{date} 采集日报", {
            "success_rate": f"{alert.success_rate:.1%}",
            "storage_raw_mb": f"{storage_stats.raw_daily_mb:.1f}",
            "storage_warm_mb": f"{storage_stats.warm_mb:.1f}",
            "storage_sqlite_mb": f"{storage_stats.sqlite_mb:.1f}",
        })
        logger.info(f"发送每日报告: {subject}")
    else:
        # ERROR/WARNING 级别：检查冷却期
        if not should_send_alert(date, alert.level):
            logger.info(f"告警在冷却期内，跳过: {date} {alert.level}")
            return False
        send_alert(alert.level, f"{date} 采集{alert.level}", {
            "success_rate": f"{alert.success_rate:.1%}",
            "failed_count": alert.total_failed,
        })
        update_checkpoint(date, alert.level)
        logger.info(f"发送告警: {subject}")

    return True


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='StockData 健康检查')
    parser.add_argument('--date', help='检查日期 (YYYY-MM-DD)', default=None)
    args = parser.parse_args()

    run_health_check(args.date)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
