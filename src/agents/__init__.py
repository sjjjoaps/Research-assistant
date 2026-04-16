"""Agent 模块包。"""

from src.agents.events import (
    BaseEvent,
    SessionStartEvent,
    ThinkingEvent,
    ToolStartEvent,
    ToolEndEvent,
    TextDeltaEvent,
    SourcesEvent,
    UsageEvent,
    DoneEvent,
    ErrorEvent,
    EventType,
)
from src.agents.session_manager import SessionManager
from src.agents.tool_registry import build_tool_registry, TOOL_DISPLAY_NAMES

__all__ = [
    # SSE 事件
    "BaseEvent",
    "SessionStartEvent",
    "ThinkingEvent",
    "ToolStartEvent",
    "ToolEndEvent",
    "TextDeltaEvent",
    "SourcesEvent",
    "UsageEvent",
    "DoneEvent",
    "ErrorEvent",
    "EventType",
    # 会话管理
    "SessionManager",
    # 工具注册
    "build_tool_registry",
    "TOOL_DISPLAY_NAMES",
]
