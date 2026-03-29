"""共享速率限制器

支持多线程并发使用的速率限制器：
- 线程安全
- 可设置每秒/每分钟限制
- 支持多个数据源共享
"""

import threading
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SharedRateLimiter:
    """
    共享速率限制器

    使用令牌桶算法 + 锁，实现线程安全的速率限制。

    使用方式：
    1. 创建限流器实例
    2. 在多线程中共享使用
    3. 每次API调用前调用 wait_and_acquire()

    示例：
        limiter = SharedRateLimiter(calls_per_second=0.75)  # 45次/分钟

        def worker():
            while True:
                limiter.wait_and_acquire()
                call_api()

        threads = [threading.Thread(target=worker) for _ in range(10)]
    """

    def __init__(
        self,
        calls_per_second: Optional[float] = None,
        calls_per_minute: Optional[float] = None,
        safety_margin: float = 0.9
    ):
        """
        初始化限流器

        Args:
            calls_per_second: 每秒允许调用次数（如 0.75 表示 45次/分钟）
            calls_per_minute: 每分钟允许调用次数（与 calls_per_second 二选一）
            safety_margin: 安全系数，默认 0.9（留 10% 余量）
        """
        if calls_per_minute:
            self.calls_per_second = calls_per_minute / 60 * safety_margin
        elif calls_per_second:
            self.calls_per_second = calls_per_second * safety_margin
        else:
            raise ValueError("必须设置 calls_per_second 或 calls_per_minute")

        self._lock = threading.Lock()
        self._last_call_time = 0.0
        self._min_interval = 1.0 / self.calls_per_second if self.calls_per_second > 0 else 0

        # 统计
        self._total_calls = 0
        self._total_wait_time = 0.0

    def wait_and_acquire(self) -> float:
        """
        等待并获取调用令牌

        Returns:
            等待时间（秒）
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call_time

            if elapsed < self._min_interval:
                sleep_time = self._min_interval - elapsed
                time.sleep(sleep_time)
                now = time.time()
                self._total_wait_time += sleep_time

            self._last_call_time = now
            self._total_calls += 1

            return self._total_wait_time

    def acquire(self) -> bool:
        """
        非阻塞获取调用令牌

        Returns:
            True 表示获取成功，False 表示需要等待
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call_time

            if elapsed >= self._min_interval:
                self._last_call_time = now
                self._total_calls += 1
                return True
            return False

    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            return {
                "total_calls": self._total_calls,
                "total_wait_time": round(self._total_wait_time, 2),
                "calls_per_second": self.calls_per_second,
                "min_interval": self._min_interval
            }


# 全局共享限流器
_tushare_limiter: Optional[SharedRateLimiter] = None
_limiter_lock = threading.Lock()


def get_tushare_limiter() -> SharedRateLimiter:
    """
    获取全局 Tushare 限流器（单例）

    45次/分钟 = 0.75次/秒
    """
    global _tushare_limiter

    if _tushare_limiter is None:
        with _limiter_lock:
            if _tushare_limiter is None:
                # 200积分用户限制 50次/分钟，使用 90% 安全系数 = 45次/分钟
                _tushare_limiter = SharedRateLimiter(
                    calls_per_minute=45,
                    safety_margin=0.9
                )
                logger.info(f"创建Tushare限流器: {45*0.9:.0f}次/分钟")

    return _tushare_limiter
