"""
重试与弹性工具模块

为 LLM API 调用、Embedding 请求等网络依赖操作提供统一的重试装饰器。
参考 RAG-Anything/raganything/resilience.py 设计，适配本项目的 DashScope/Qwen 环境。

可重试异常范围：仅限瞬时网络/限流错误，不重试本地编程错误（TypeError、ValueError 等）。
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
from typing import Any, Callable, Optional, Sequence, Type, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# 默认可重试异常：仅覆盖网络/上游瞬时故障
_DEFAULT_RETRYABLE: tuple[Type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
)

try:
    import httpx

    _DEFAULT_RETRYABLE = _DEFAULT_RETRYABLE + (
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
    )
except ImportError:
    pass

try:
    import openai

    _DEFAULT_RETRYABLE = _DEFAULT_RETRYABLE + (
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.InternalServerError,
    )
except ImportError:
    pass


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[Sequence[Type[BaseException]]] = None,
    on_retry: Optional[Callable[[BaseException, int, float], None]] = None,
) -> Callable[[F], F]:
    """同步函数重试装饰器，使用指数退避策略。

    Args:
        max_attempts: 最大尝试次数（含首次调用）。
        base_delay: 首次重试前等待秒数。
        max_delay: 重试等待上限秒数。
        exponential_base: 每次重试的延迟倍数。
        jitter: 是否在延迟上叠加随机抖动（0~50%），避免惊群效应。
        retryable_exceptions: 触发重试的异常类型，默认为网络/限流瞬时错误。
        on_retry: 每次重试前的回调 ``(exception, attempt, delay)``。

    Example::

        @retry(max_attempts=3, base_delay=1.0)
        def call_llm(prompt: str) -> str:
            ...
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if base_delay < 0 or max_delay < 0:
        raise ValueError("base_delay and max_delay must be >= 0")
    if exponential_base <= 0:
        raise ValueError("exponential_base must be > 0")

    exc_types = tuple(retryable_exceptions) if retryable_exceptions is not None else _DEFAULT_RETRYABLE

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exc_types as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.error(
                            "%s 在 %d 次尝试后仍失败: %s",
                            func.__qualname__,
                            max_attempts,
                            exc,
                        )
                        raise
                    delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)
                    if jitter:
                        import random
                        delay *= 1.0 + random.uniform(0, 0.5)
                    if on_retry is not None:
                        on_retry(exc, attempt, delay)
                    logger.warning(
                        "%s 第 %d/%d 次失败 (%s)，%.1fs 后重试…",
                        func.__qualname__,
                        attempt,
                        max_attempts,
                        type(exc).__name__,
                        delay,
                    )
                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[Sequence[Type[BaseException]]] = None,
    on_retry: Optional[Callable[[BaseException, int, float], Any]] = None,
) -> Callable[[F], F]:
    """异步函数重试装饰器，使用 asyncio.sleep 退避。

    用法与 :func:`retry` 相同，适用于 async def 函数。

    Example::

        @async_retry(max_attempts=3, base_delay=1.0)
        async def call_llm_async(prompt: str) -> str:
            ...
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if base_delay < 0 or max_delay < 0:
        raise ValueError("base_delay and max_delay must be >= 0")
    if exponential_base <= 0:
        raise ValueError("exponential_base must be > 0")

    exc_types = tuple(retryable_exceptions) if retryable_exceptions is not None else _DEFAULT_RETRYABLE

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exc_types as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.error(
                            "%s 在 %d 次尝试后仍失败: %s",
                            func.__qualname__,
                            max_attempts,
                            exc,
                        )
                        raise
                    delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)
                    if jitter:
                        import random
                        delay *= 1.0 + random.uniform(0, 0.5)
                    if on_retry is not None:
                        result = on_retry(exc, attempt, delay)
                        if asyncio.iscoroutine(result):
                            await result
                    logger.warning(
                        "%s 第 %d/%d 次失败 (%s)，%.1fs 后重试…",
                        func.__qualname__,
                        attempt,
                        max_attempts,
                        type(exc).__name__,
                        delay,
                    )
                    await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


class CircuitBreaker:
    """简单熔断器，防止级联故障。

    当失败次数在 ``reset_timeout`` 窗口内超过 ``failure_threshold`` 时，
    熔断器进入 *open* 状态，后续调用直接抛出 ``CircuitBreakerOpen``。
    经过 ``reset_timeout`` 秒后进入 *half-open* 状态，允许一次试探调用。

    Args:
        failure_threshold: 触发熔断的失败次数阈值。
        reset_timeout: 熔断后等待多少秒进入 half-open 状态。
        name: 用于日志的可读名称。
        failure_exceptions: 计入失败计数的异常类型，默认与重试装饰器一致。
    """

    class CircuitBreakerOpen(Exception):
        """熔断器处于 open 状态时抛出。"""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        name: str = "default",
        failure_exceptions: Optional[Sequence[Type[BaseException]]] = None,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.name = name
        self._failure_exceptions: tuple[Type[BaseException], ...] = tuple(
            failure_exceptions or _DEFAULT_RETRYABLE
        )
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._state: str = "closed"
        self._lock = threading.Lock()
        self._trial_in_flight: bool = False

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == "open":
                if time.time() - self._last_failure_time >= self.reset_timeout:
                    self._state = "half-open"
            return self._state

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = "closed"
            self._trial_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            now = time.time()
            if self._state == "half-open":
                self._failure_count = self.failure_threshold
            else:
                if self._last_failure_time and now - self._last_failure_time >= self.reset_timeout:
                    self._failure_count = 0
                self._failure_count += 1
            self._last_failure_time = now
            if self._failure_count >= self.failure_threshold:
                self._state = "open"
                self._trial_in_flight = False
                logger.warning("熔断器 '%s' 已打开，累计失败 %d 次", self.name, self._failure_count)

    def _acquire_permission(self) -> None:
        with self._lock:
            if self._state == "open":
                if time.time() - self._last_failure_time >= self.reset_timeout:
                    self._state = "half-open"
            if self._state == "open":
                raise self.CircuitBreakerOpen(f"熔断器 '{self.name}' 已打开，请求被拒绝")
            if self._state == "half-open":
                if self._trial_in_flight:
                    raise self.CircuitBreakerOpen(f"熔断器 '{self.name}' 处于 half-open，试探调用进行中")
                self._trial_in_flight = True

    def __call__(self, func: F) -> F:
        """作为同步函数装饰器使用。"""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self._acquire_permission()
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except tuple(self._failure_exceptions):
                self.record_failure()
                raise
            except Exception:
                with self._lock:
                    if self._state == "half-open":
                        self._trial_in_flight = False
                raise

        return wrapper  # type: ignore[return-value]

    def async_call(self, func: F) -> F:
        """作为异步函数装饰器使用。"""

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            self._acquire_permission()
            try:
                result = await func(*args, **kwargs)
                self.record_success()
                return result
            except tuple(self._failure_exceptions):
                self.record_failure()
                raise
            except Exception:
                with self._lock:
                    if self._state == "half-open":
                        self._trial_in_flight = False
                raise

        return wrapper  # type: ignore[return-value]
