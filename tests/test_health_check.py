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

    def test_error_sends_first_time(self, tmp_path, monkeypatch):
        """ERROR 级别首次发送"""
        stockdata_root = tmp_path / "stockdata"
        stockdata_root.mkdir()
        monkeypatch.setenv("STOCKDATA_ROOT", str(stockdata_root))

        assert should_send_alert("2026-03-29", "ERROR") is True

    def test_error_skips_within_cooldown(self, tmp_path, monkeypatch):
        """ERROR 级别在冷却期内跳过"""
        stockdata_root = tmp_path / "stockdata"
        stockdata_root.mkdir()
        monkeypatch.setenv("STOCKDATA_ROOT", str(stockdata_root))

        # 第一次发送
        assert should_send_alert("2026-03-29", "ERROR") is True

        # 模拟最近发送过（修改 checkpoint）
        checkpoint_file = stockdata_root / "status" / "health_checkpoint.json"
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'last_alerts': {
                    'ERROR_2026-03-29': datetime.now().isoformat()
                }
            }, f)

        # 冷却期内，应该跳过
        assert should_send_alert("2026-03-29", "ERROR") is False

    def test_error_sends_after_cooldown(self, tmp_path, monkeypatch):
        """ERROR 级别冷却期后重新发送"""
        stockdata_root = tmp_path / "stockdata"
        stockdata_root.mkdir()
        monkeypatch.setenv("STOCKDATA_ROOT", str(stockdata_root))

        # 模拟冷却期已过（25小时前）
        checkpoint_file = stockdata_root / "status" / "health_checkpoint.json"
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        old_time = (datetime.now() - timedelta(hours=25)).isoformat()
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'last_alerts': {
                    'ERROR_2026-03-29': old_time
                }
            }, f)

        # 冷却期已过，应该发送
        assert should_send_alert("2026-03-29", "ERROR") is True


class TestLoadReport:
    """日报加载测试"""

    def test_missing_report_returns_none(self, tmp_path, monkeypatch):
        """日报不存在返回 None"""
        stockdata_root = tmp_path / "stockdata"
        stockdata_root.mkdir()
        monkeypatch.setenv("STOCKDATA_ROOT", str(stockdata_root))

        from monitor.health_check import load_report
        result = load_report("2026-03-29")
        assert result is None

    def test_corrupted_report_returns_none(self, tmp_path, monkeypatch):
        """日报损坏返回 None"""
        stockdata_root = tmp_path / "stockdata"
        stockdata_root.mkdir()
        monkeypatch.setenv("STOCKDATA_ROOT", str(stockdata_root))

        from monitor.health_check import load_report

        # 创建损坏的日报
        report_file = stockdata_root / "status" / "daily_report_20260329.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w') as f:
            f.write('{"invalid json')

        result = load_report("2026-03-29")
        assert result is None

    def test_valid_report_loaded(self, tmp_path, monkeypatch):
        """有效日报正确加载"""
        stockdata_root = tmp_path / "stockdata"
        stockdata_root.mkdir()
        monkeypatch.setenv("STOCKDATA_ROOT", str(stockdata_root))

        from monitor.health_check import load_report

        # 创建有效的日报
        report_file = stockdata_root / "status" / "daily_report_20260329.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump({
                "date": "2026-03-29",
                "summary": {"success_count": 100, "total_count": 100}
            }, f)

        result = load_report("2026-03-29")
        assert result is not None
        assert result["date"] == "2026-03-29"


class TestUpdateCheckpoint:
    """检查点更新测试"""

    def test_checkpoint_atomic_write(self, tmp_path, monkeypatch):
        """检查点原子写入"""
        stockdata_root = tmp_path / "stockdata"
        stockdata_root.mkdir()
        monkeypatch.setenv("STOCKDATA_ROOT", str(stockdata_root))

        from monitor.health_check import update_checkpoint

        update_checkpoint("2026-03-29", "ERROR")

        checkpoint_file = stockdata_root / "status" / "health_checkpoint.json"
        assert checkpoint_file.exists()

        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        assert "last_alerts" in data
        assert "ERROR_2026-03-29" in data["last_alerts"]


class TestStorageStats:
    """存储统计测试"""

    def test_storage_stats_empty(self, tmp_path, monkeypatch):
        """空目录的存储统计"""
        stockdata_root = tmp_path / "stockdata"
        stockdata_root.mkdir()
        monkeypatch.setenv("STOCKDATA_ROOT", str(stockdata_root))

        from monitor.health_check import get_storage_stats

        stats = get_storage_stats()
        assert stats.raw_daily_mb == 0
        assert stats.warm_mb == 0
        assert stats.sqlite_mb == 0
        assert stats.total_stocks == 0
