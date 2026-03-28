"""重试处理器"""

import time
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from .exceptions import NetworkError, SourceError
from .sources.base import DataSource


@dataclass
class FetchResult:
    """采集结果"""
    code: str
    date: str
    data: Optional["pd.DataFrame"] = None
    source: str = ""
    fetch_status: str = "pending"      # pending / success / failed
    write_status: str = "pending"      # pending / success / rejected
    fail_type: Optional[str] = None     # None / network / quality / source
    fail_reason: Optional[str] = None
    quality_score: Optional[float] = None
    quality_dims: dict = field(default_factory=dict)
    attempts: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    @property
    def duration_ms(self) -> float:
        """获取耗时（毫秒）"""
        if self.end_time is None:
            end = datetime.now()
        else:
            end = self.end_time
        return (end - self.start_time).total_seconds() * 1000

    def to_dict(self) -> dict:
        """转换为字典（用于日志/报告）"""
        return {
            "code": self.code,
            "date": self.date,
            "source": self.source,
            "fetch_status": self.fetch_status,
            "write_status": self.write_status,
            "fail_type": self.fail_type,
            "fail_reason": self.fail_reason,
            "quality_score": self.quality_score,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms
        }


class RetryHandler:
    """
    重试处理器

    重试策略：
    1. 最多重试 max_attempts 次
    2. 第1次失败后立即重试1次（处理偶发网络抖动）
    3. 第2次失败后等待30秒再重试1次（给数据源恢复时间）
    4. 超过 max_attempts 后标记为失败
    """

    def __init__(self, max_attempts: int = 2):
        """
        初始化重试处理器

        Args:
            max_attempts: 最大重试次数
        """
        self.max_attempts = max_attempts

    def fetch_with_retry(
        self,
        source: DataSource,
        code: str,
        start_date: str,
        end_date: str
    ) -> FetchResult:
        """
        带重试的采集

        Args:
            source: 数据源
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            FetchResult
        """
        result = FetchResult(
            code=code,
            date=end_date,
            source=source.name
        )

        last_error = None

        for attempt in range(1, self.max_attempts + 1):
            result.attempts = attempt
            result.start_time = datetime.now()

            try:
                df = source.fetch_daily(code, start_date, end_date)

                # 检查是否返回有效数据
                if df is not None and len(df) > 0:
                    result.data = df
                    result.fetch_status = "success"
                    result.end_time = datetime.now()
                    return result
                else:
                    # 数据为空（可能是停牌）
                    last_error = "no_data"
                    result.fetch_status = "failed"
                    result.fail_type = "source"
                    result.fail_reason = "no_data_returned"

            except NetworkError as e:
                last_error = str(e)
                result.fail_type = "network"
                result.fail_reason = last_error

            except SourceError as e:
                last_error = str(e)
                result.fail_type = "source"
                result.fail_reason = last_error

            except Exception as e:
                last_error = str(e)
                result.fail_type = "source"
                result.fail_reason = last_error

            result.end_time = datetime.now()

            # 重试延迟
            if attempt < self.max_attempts:
                if attempt == 1:
                    # 第1次失败后立即重试
                    pass
                else:
                    # 第2次失败后等待30秒
                    time.sleep(30)

        # 全部失败
        result.fetch_status = "failed"
        result.fail_reason = last_error
        return result


class BatchRetryHandler:
    """
    批量重试处理器

    用于处理一组股票的采集失败重试
    """

    def __init__(self, max_attempts: int = 2, max_concurrent: int = 10):
        """
        Args:
            max_attempts: 每个股票的最大重试次数
            max_concurrent: 最大并发采集数
        """
        self.max_attempts = max_attempts
        self.max_concurrent = max_concurrent
        self.retry_handler = RetryHandler(max_attempts=max_attempts)

    def fetch_batch(
        self,
        source: DataSource,
        tasks: list[tuple[str, str, str]]
    ) -> list[FetchResult]:
        """
        批量采集

        Args:
            source: 数据源
            tasks: [(code, start_date, end_date), ...]

        Returns:
            FetchResult 列表
        """
        results = []

        for code, start_date, end_date in tasks:
            result = self.retry_handler.fetch_with_retry(
                source, code, start_date, end_date
            )
            results.append(result)

        return results

    def get_failed_results(self, results: list[FetchResult]) -> list[FetchResult]:
        """获取失败的采集结果"""
        return [r for r in results if r.fetch_status == "failed"]

    def get_success_results(self, results: list[FetchResult]) -> list[FetchResult]:
        """获取成功的采集结果"""
        return [r for r in results if r.fetch_status == "success"]
