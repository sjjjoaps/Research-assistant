"""
Agent 基类
定义所有 Agent 的通用骨架：会话状态管理 + 抽象接口约束
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Turn:
    """一轮对话记录"""
    role: str               # "user" 或 "assistant"
    content: str
    sources: list[str] = field(default_factory=list)


class BaseAgent(ABC):
    """
    所有具体 Agent 的基类，负责：
    1. 统一会话状态管理（内存态，按 thread_id 区分）
    2. 约束子类必须实现的两个方法
    """

    def __init__(self, max_history_turns: int = 5) -> None:
        """
        Args:
            max_history_turns: 传给 LLM 时最多携带的历史轮数，防止 prompt 过大
        """
        self.max_history_turns = max_history_turns
        # {thread_id: list[Turn]}
        self._threads: dict[str, list[Turn]] = {}

    # ── 会话状态管理 ──────────────────────────────────────────────────────────

    def start_thread(self, thread_id: str) -> None:
        """初始化一个会话线程（若已存在则不覆盖）"""
        if thread_id not in self._threads:
            self._threads[thread_id] = []

    def append_user_message(self, thread_id: str, text: str) -> None:
        """写入用户消息"""
        self.start_thread(thread_id)
        self._threads[thread_id].append(Turn(role="user", content=text))

    def append_assistant_message(self, thread_id: str, text: str, sources: list[str] | None = None) -> None:
        """写入 Assistant 消息"""
        self.start_thread(thread_id)
        self._threads[thread_id].append(
            Turn(role="assistant", content=text, sources=sources or [])
        )

    def get_history(self, thread_id: str, max_turns: int | None = None) -> list[Turn]:
        """
        获取会话历史，默认截取最近 max_history_turns 轮（一轮 = user + assistant 各一条）
        """
        self.start_thread(thread_id)
        history = self._threads[thread_id]
        limit = (max_turns or self.max_history_turns) * 2   # 每轮两条消息
        return history[-limit:] if len(history) > limit else list(history)

    def clear_thread(self, thread_id: str) -> None:
        """清空某个会话线程（调试用）"""
        self._threads.pop(thread_id, None)

    # ── 子类必须实现 ─────────────────────────────────────────────────────────

    @abstractmethod
    def run_turn(self, thread_id: str, user_input: str) -> dict:
        """
        执行一轮对话，子类在这里完成：检索 -> 组装 prompt -> 调用 LLM -> 回写历史

        返回格式（固定）：
        {
            "answer":    str,
            "sources":   list[str],
            "thread_id": str,
        }
        """
        raise NotImplementedError
