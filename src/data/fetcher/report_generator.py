"""日报生成器"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from pathlib import Path


@dataclass
class DailyReport:
    """日报数据"""
    date: str
    start_time: str
    end_time: str
    duration_seconds: float
    summary: dict
    successes: list[dict]
    quality_rejected: list[dict]
    network_failed: list[dict]
    retry_failed: list[dict]
    warnings: list[dict]
    errors: list[dict]

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "date": self.date,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "summary": self.summary,
            "successes": self.successes,
            "quality_rejected": self.quality_rejected,
            "network_failed": self.network_failed,
            "retry_failed": self.retry_failed,
            "warnings": self.warnings,
            "errors": self.errors
        }


class DailyReportGenerator:
    """
    日报生成器

    生成格式：
    {
        "date": "2026-03-28",
        "start_time": "16:00:00",
        "end_time": "16:45:23",
        "duration_seconds": 2723,
        "summary": {
            "total_stocks": 4823,
            "success_count": 4815,
            "quality_rejected_count": 3,
            "network_failed_count": 5,
            "success_rate": 0.9983
        },
        "successes": [...],
        "quality_rejected": [...],
        "network_failed": [...],
        "retry_failed": [...],
        "warnings": [...],
        "errors": [...]
    }
    """

    # 告警阈值
    SUCCESS_RATE_ERROR = 0.95
    SUCCESS_RATE_WARNING = 0.99

    def __init__(self, date: str):
        """
        Args:
            date: 采集日期，格式 YYYY-MM-DD
        """
        self.date = date
        self.start_time = datetime.now().strftime("%H:%M:%S")
        self.results: list = []
        self.write_results: list = []

    def add_result(self, result):
        """添加采集结果"""
        self.results.append(result)

    def add_write_result(self, result):
        """添加写入结果"""
        self.write_results.append(result)

    def generate(self, output_path: Optional[str] = None) -> DailyReport:
        """
        生成日报

        Args:
            output_path: 日报输出路径，若不提供则不保存

        Returns:
            DailyReport
        """
        end_time = datetime.now().strftime("%H:%M:%S")
        duration = (datetime.now() - datetime.strptime(self.start_time, "%H:%M:%S")).total_seconds()

        summary = self._compute_summary()
        report = DailyReport(
            date=self.date,
            start_time=self.start_time,
            end_time=end_time,
            duration_seconds=duration,
            summary=summary,
            successes=self._get_successes(),
            quality_rejected=self._get_quality_rejected(),
            network_failed=self._get_network_failed(),
            retry_failed=self._get_retry_failed(),
            warnings=self._get_warnings(summary),
            errors=self._get_errors()
        )

        if output_path:
            self._save_report(report, output_path)

        return report

    def _compute_summary(self) -> dict:
        """计算汇总统计"""
        total = len(self.results)
        if total == 0:
            return {
                "total_stocks": 0,
                "success_count": 0,
                "write_rejected_count": 0,
                "quality_rejected_count": 0,
                "network_failed_count": 0,
                "success_rate": 0.0
            }

        success_count = sum(1 for r in self.results if r.fetch_status == "success")
        network_failed = sum(1 for r in self.results
                            if r.fetch_status == "failed" and r.fail_type == "network")
        quality_rejected = sum(1 for r in self.results
                             if r.fetch_status == "success" and r.write_status == "rejected")

        return {
            "total_stocks": total,
            "success_count": success_count,
            "write_rejected_count": len(self.write_results) - success_count if self.write_results else 0,
            "quality_rejected_count": quality_rejected,
            "network_failed_count": network_failed,
            "success_rate": round(success_count / total, 4)
        }

    def _get_successes(self) -> list:
        """获取成功采集的列表"""
        successes = []
        for r in self.results:
            if r.fetch_status == "success" and r.write_status != "rejected":
                successes.append({
                    "code": r.code,
                    "source": r.source,
                    "quality_score": r.quality_score,
                    "quality_dims": r.quality_dims
                })
        return successes

    def _get_quality_rejected(self) -> list:
        """获取因质量被拒绝的列表"""
        rejected = []
        for r in self.results:
            if r.fetch_status == "success" and r.write_status == "rejected":
                rejected.append({
                    "code": r.code,
                    "fail_type": r.fail_type,
                    "fail_reason": r.fail_reason,
                    "quality_score": r.quality_score
                })
        return rejected

    def _get_network_failed(self) -> list:
        """获取因网络问题失败的列表"""
        failed = []
        for r in self.results:
            if r.fetch_status == "failed" and r.fail_type == "network":
                failed.append({
                    "code": r.code,
                    "reason": r.fail_reason,
                    "attempts": r.attempts,
                    "fail_type": r.fail_type
                })
        return failed

    def _get_retry_failed(self) -> list:
        """获取重试后仍然失败的列表"""
        failed = []
        for r in self.results:
            if r.fetch_status == "failed" and r.attempts >= 2:
                failed.append({
                    "code": r.code,
                    "reason": r.fail_reason,
                    "attempts": r.attempts,
                    "fail_type": r.fail_type
                })
        return failed

    def _get_warnings(self, summary: dict) -> list:
        """获取告警列表"""
        warnings = []
        rate = summary["success_rate"]

        if rate < self.SUCCESS_RATE_ERROR:
            warnings.append({
                "type": "critical",
                "message": f"成功率 {rate:.1%} 低于 95% 阈值"
            })
        elif rate < self.SUCCESS_RATE_WARNING:
            warnings.append({
                "type": "warning",
                "message": f"成功率 {rate:.1%} 低于 99% 阈值"
            })

        return warnings

    def _get_errors(self) -> list:
        """获取所有错误"""
        errors = []
        for r in self.results:
            if r.fetch_status == "failed":
                errors.append({
                    "code": r.code,
                    "fail_type": r.fail_type,
                    "fail_reason": r.fail_reason,
                    "attempts": r.attempts
                })
        return errors

    def _save_report(self, report: DailyReport, output_path: str):
        """保存日报到文件"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)


def load_daily_report(date: str, report_dir: str) -> Optional[dict]:
    """
    加载指定日期的日报

    Args:
        date: 日期，格式 YYYY-MM-DD
        report_dir: 日报目录

    Returns:
        日报字典，若不存在返回 None
    """
    report_file = Path(report_dir) / f"daily_report_{date.replace('-', '')}.json"
    if not report_file.exists():
        return None

    with open(report_file, encoding="utf-8") as f:
        return json.load(f)
