"""
StockData 告警模块
"""

import os
import json
import smtplib
import logging
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ALERT_FALLBACK_DIR = Path("logs/alerts")
MAX_FALLBACK_SIZE = 100 * 1024 * 1024  # 100MB
MAX_ROTATED_FILES = 5                   # 保留最近5个
ROTATION_RETENTION_DAYS = 7             # 保留7天


ALERT_CONFIG = {
    'smtp_host': os.getenv('EMAIL_SMTP_HOST', 'smtp.qq.com'),
    'smtp_port': int(os.getenv('EMAIL_SMTP_PORT', 587)),
    'smtp_user': os.getenv('EMAIL_USERNAME', 'bruceleexiaokan@qq.com'),
    'from_addr': os.getenv('EMAIL_USERNAME', 'bruceleexiaokan@qq.com'),
    'to_addrs': [os.getenv('EMAIL_TO', 'bruceleexiaokan@qq.com')],
}


def get_smtp_password() -> Optional[str]:
    """从环境变量获取 SMTP 密码"""
    return os.getenv('EMAIL_PASSWORD')


def send_alert(level: str, message: str, details: dict = None):
    """
    发送告警

    Args:
        level: 告警级别 (INFO, WARNING, ERROR, CRITICAL)
        message: 告警消息
        details: 详细信息
    """
    # 验证告警级别
    valid_levels = ['INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if level not in valid_levels:
        level = 'INFO'

    # 构建邮件内容
    subject = f"[StockData {level}] {message}"

    body_lines = [
        f"时间: {datetime.now().isoformat()}",
        f"级别: {level}",
        f"消息: {message}",
    ]

    if details:
        body_lines.append(f"详情: {details}")

    body = "\n".join(body_lines)

    # 获取 SMTP 密码
    password = get_smtp_password()
    if not password:
        logger.warning(f"SMTP_PASSWORD 环境变量未设置，跳过邮件告警: {message}")
        return False

    try:
        msg = MIMEText(body, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = ALERT_CONFIG['from_addr']
        msg['To'] = ','.join(ALERT_CONFIG['to_addrs'])

        with smtplib.SMTP(ALERT_CONFIG['smtp_host'], ALERT_CONFIG['smtp_port'], timeout=10) as server:
            server.starttls()
            server.login(ALERT_CONFIG['smtp_user'], password)
            server.send_message(msg)

        logger.info(f"告警已发送: {level} - {message}")
        return True

    except Exception as e:
        logger.error(f"告警发送失败: {e}")
        return _write_alert_fallback(level, message, details, str(e))


def _rotate_fallback_file_if_needed():
    """检查文件大小，必要时轮转"""
    fallback_file = ALERT_FALLBACK_DIR / "failed_alerts.jsonl"
    if not fallback_file.exists():
        return
    if fallback_file.stat().st_size < MAX_FALLBACK_SIZE:
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    rotated_name = f"failed_alerts_{timestamp}.jsonl"
    rotated_file = ALERT_FALLBACK_DIR / rotated_name
    fallback_file.rename(rotated_file)
    logger.info(f"告警文件已轮转: {rotated_file}")
    _cleanup_rotated_files()


def _cleanup_rotated_files():
    """清理过期和超量轮转文件"""
    if not ALERT_FALLBACK_DIR.exists():
        return
    cutoff = datetime.now() - timedelta(days=ROTATION_RETENTION_DAYS)
    rotated = sorted(
        ALERT_FALLBACK_DIR.glob("failed_alerts_[0-9]*.jsonl"),
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )
    for f in rotated:
        if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
            f.unlink()
            logger.info(f"删除过期文件: {f}")
    for f in rotated[MAX_ROTATED_FILES:]:
        f.unlink()
        logger.info(f"删除超量文件: {f}")


def _write_alert_fallback(level: str, message: str, details: dict, error: str) -> bool:
    """告警失败时写入本地文件（带轮转）"""
    ALERT_FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
    _rotate_fallback_file_if_needed()

    record = {
        "time": datetime.now().isoformat(),
        "level": level,
        "message": message,
        "details": details,
        "error": error
    }
    fallback_file = ALERT_FALLBACK_DIR / "failed_alerts.jsonl"
    try:
        with open(fallback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.warning(f"告警已写入回退文件: {fallback_file}")
    except Exception as e:
        logger.error(f"告警回退文件写入失败: {e}")
    return False


def send_test_alert():
    """发送测试告警"""
    return send_alert(
        "INFO",
        "StockData 告警测试",
        {"test": True, "timestamp": datetime.now().isoformat()}
    )
