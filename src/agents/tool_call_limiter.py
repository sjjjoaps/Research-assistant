"""
工具调用频率熔断器（P5-Step 1.5）。

职责：
- 基于滑动时间窗口，对同一 session_id + tool_name 的调用频率进行限制
- 超限时返回拒绝提示字符串，不抛出异常
- 不同 session / 不同 tool 的计数相互独立

配置项（.env）：
    TOOL_CALL_MAX_PER_WINDOW=5    # 窗口内最大调用次数（默认 5）
    TOOL_CALL_WINDOW_SECONDS=60   # 滑动窗口时长（秒，默认 60）

Public API：
    ToolCallLimiter(max_calls, window_seconds)
    .check(session_id, tool_name) -> bool
    .reject_message(tool_name) -> str
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class ToolCallLimiter:
    """
    基于滑动时间窗口的工具调用频率限制器。

    策略：同一 session_id + tool_name 在 window_seconds 内调用超过 max_calls 次，
    则拒绝本次调用并返回拒绝提示，要求 Agent 换方法或调整参数。
    """

    max_calls: int = 5
    window_seconds: float = 60.0

    _history: dict = field(default_factory=lambda: defaultdict(deque))

    def check(self, session_id: str, tool_name: str) -> bool:
        """
        检查是否允许本次调用。

        Returns:
            True  — 允许调用（并记录本次时间戳）
            False — 触发熔断，调用者应使用 reject_message() 作为工具返回值
        """
        key = (session_id, tool_name)
        now = time.monotonic()
        q = self._history[key]
        while q and now - q[0] > self.window_seconds:
            q.popleft()
        if len(q) >= self.max_calls:
            return False
        q.append(now)
        return True

    def reject_message(self, tool_name: str) -> str:
        """返回对 Agent 友好的拒绝提示，说明原因并引导换方法。"""
        return (
            f"[工具调用被暂时拒绝] `{tool_name}` 在短时间内调用次数过多（"
            f"{self.window_seconds:.0f}s 内超过 {self.max_calls} 次）。"
            "请采用另一种方法、调整参数，或等待片刻后重试。"
        )

    def reset(self, session_id: str | None = None, tool_name: str | None = None) -> None:
        """
        重置计数（测试/管理用途）。

        - 同时指定 session_id 和 tool_name → 重置该组合
        - 仅指定 session_id → 重置该 session 的所有工具
        - 均不指定 → 重置全部历史
        """
        if session_id is not None and tool_name is not None:
            self._history.pop((session_id, tool_name), None)
        elif session_id is not None:
            keys = [k for k in self._history if k[0] == session_id]
            for k in keys:
                del self._history[k]
        else:
            self._history.clear()
