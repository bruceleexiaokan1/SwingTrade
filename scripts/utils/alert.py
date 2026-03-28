"""
StockData 告警模块
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


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

        with smtplib.SMTP(ALERT_CONFIG['smtp_host'], ALERT_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(ALERT_CONFIG['smtp_user'], password)
            server.send_message(msg)

        logger.info(f"告警已发送: {level} - {message}")
        return True

    except Exception as e:
        logger.error(f"告警发送失败: {e}")
        return False


def send_test_alert():
    """发送测试告警"""
    return send_alert(
        "INFO",
        "StockData 告警测试",
        {"test": True, "timestamp": datetime.now().isoformat()}
    )
