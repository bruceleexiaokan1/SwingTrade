"""
StockData 健康检查测试
"""

import pytest
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.insert(0, 'scripts')

from monitor.health_check import (
    HealthAlert,
    calculate_level,
    format_codes,
    parse_report,
    should_send_alert,
    get_check_date,
    get_report_path,
)


class TestHealthAlert:
    """HealthAlert 测试"""

    def test_total_failed(self):
        """失败总数计算"""
        alert = HealthAlert(
            date="2026-03-29",
            level="ERROR",
            success_count=4732,
            total_count=4805,
            success_rate=0.985,
            network_failed=["000001", "000002"],
            quality_rejected=["000010", "000011"],
            retry_failed=[],
            other_failed=["000020"],
        )
        assert alert.total_failed == 5


class TestCalculateLevel:
    """告警级别计算测试"""

    def test_error_level(self):
        """ERROR 级别: 成功率 < 95%"""
        alert = HealthAlert(
            date="2026-03-29",
            level="",
            success_count=4500,
            total_count=4800,
            success_rate=0.9375,
            network_failed=[],
            quality_rejected=[],
            retry_failed=[],
            other_failed=[],
        )
        assert calculate_level(alert) == "ERROR"

    def test_warning_level(self):
        """WARNING 级别: 95% <= 成功率 < 99%"""
        alert = HealthAlert(
            date="2026-03-29",
            level="",
            success_count=4700,
            total_count=4800,
            success_rate=0.979,
            network_failed=[],
            quality_rejected=[],
            retry_failed=[],
            other_failed=[],
        )
        assert calculate_level(alert) == "WARNING"

    def test_info_level(self):
        """INFO 级别: 成功率 >= 99%"""
        alert = HealthAlert(
            date="2026-03-29",
            level="",
            success_count=4760,
            total_count=4800,
            success_rate=0.992,
            network_failed=[],
            quality_rejected=[],
            retry_failed=[],
            other_failed=[],
        )
        assert calculate_level(alert) == "INFO"


class TestFormatCodes:
    """代码列表格式化测试"""

    def test_empty(self):
        assert format_codes([]) == "无"

    def test_short_list(self):
        codes = ["000001", "000002", "000003"]
        result = format_codes(codes)
        assert result == "000001, 000002, 000003"

    def test_long_list_truncate(self):
        codes = [f"{i:06d}" for i in range(30)]
        result = format_codes(codes, max_show=5)
        # 应该显示前5个: 000000, 000001, 000002, 000003, 000004
        assert "000000" in result
        assert "000001" in result
        assert "000004" in result
        # 后面被截断
        assert "000005" not in result
        # 显示总数量
        assert "(+25只)" in result

    def test_exact_max(self):
        codes = ["000001", "000002"]
        result = format_codes(codes, max_show=5)
        assert result == "000001, 000002"


class TestParseReport:
    """日报解析测试"""

    def test_parse_full_report(self):
        """完整日报解析"""
        report = {
            "date": "2026-03-29",
            "start_time": "17:30:00",
            "end_time": "17:58:00",
            "summary": {
                "success_count": 4732,
                "total_count": 4805,
                "success_rate": 0.985,
            },
            "network_failed": [
                {"code": "000001", "reason": "timeout"},
                {"code": "000002", "reason": "timeout"},
            ],
            "quality_rejected": [
                {"code": "000010", "reason": "price_out_of_range"},
            ],
            "retry_failed": [],
            "errors": [],
        }

        alert = parse_report(report)

        assert alert.date == "2026-03-29"
        assert alert.success_count == 4732
        assert alert.total_count == 4805
        assert alert.success_rate == 0.985
        assert alert.network_failed == ["000001", "000002"]
        assert alert.quality_rejected == ["000010"]
        assert alert.start_time == "17:30:00"
        assert alert.end_time == "17:58:00"


class TestGetCheckDate:
    """检查日期判断测试"""

    def test_morning_check(self):
        """上午检查返回昨天"""
        # 模拟 09:00 的情况
        # 这个测试需要 mock datetime.now()
        # 这里只做基本验证
        date = get_check_date()
        assert date is not None
        assert len(date) == 10  # YYYY-MM-DD format


class TestGetReportPath:
    """日报路径测试"""

    def test_report_path(self):
        """日报路径格式"""
        path = get_report_path("2026-03-29")
        assert "20260329" in str(path)
        assert "daily_report_" in str(path)


class TestShouldSendAlert:
    """告警去重测试"""

    def test_info_never_sends(self, tmp_path, monkeypatch):
        """INFO 级别不发送"""
        # 设置临时 StockData 路径
        stockdata_root = tmp_path / "stockdata"
        stockdata_root.mkdir()
        monkeypatch.setenv("STOCKDATA_ROOT", str(stockdata_root))

        assert should_send_alert("2026-03-29", "INFO") is False
