"""
SSE 事件模型定义。
MasterAgent 向前端推送的所有 SSE 事件类型。
"""
from dataclasses import dataclass, field, asdict
from typing import Literal
import json

EventType = Literal[
    "session_start", "thinking", "tool_start", "tool_end",
    "text_delta", "sources", "usage", "done", "error"
]


@dataclass
class BaseEvent:
    type: EventType

    def to_sse(self) -> str:
        """格式化为 SSE 协议字符串"""
        return f"event: {self.type}\ndata: {json.dumps(asdict(self), ensure_ascii=False)}\n\n"


@dataclass
class SessionStartEvent(BaseEvent):
    type: EventType = field(default="session_start", init=False)
    session_id: str = ""


@dataclass
class ThinkingEvent(BaseEvent):
    type: EventType = field(default="thinking", init=False)
    iteration: int = 0


@dataclass
class ToolStartEvent(BaseEvent):
    type: EventType = field(default="tool_start", init=False)
    tool_name: str = ""
    display_message: str = ""  # 例如"正在检索知识库..."


@dataclass
class ToolEndEvent(BaseEvent):
    type: EventType = field(default="tool_end", init=False)
    tool_name: str = ""
    result_summary: str = ""   # 例如"找到 5 个相关片段"
    elapsed_ms: int = 0


@dataclass
class TextDeltaEvent(BaseEvent):
    type: EventType = field(default="text_delta", init=False)
    delta: str = ""


@dataclass
class SourcesEvent(BaseEvent):
    type: EventType = field(default="sources", init=False)
    sources: list = field(default_factory=list)


@dataclass
class UsageEvent(BaseEvent):
    type: EventType = field(default="usage", init=False)
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_cny: float = 0.0


@dataclass
class DoneEvent(BaseEvent):
    type: EventType = field(default="done", init=False)
    session_id: str = ""


@dataclass
class ErrorEvent(BaseEvent):
    type: EventType = field(default="error", init=False)
    message: str = ""
    code: str = ""
