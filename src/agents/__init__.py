"""
Agent 模块包。

导出策略（避免包级导入引入跨层耦合）：
- events、session_manager 属于基础设施层，可安全在包级导出
- tool_registry 属于工具层，在包级导出（被 MasterAgent 等上层依赖）
- prompt_loader 属于配置/基础层，不在此处包级导出，避免与 tool_registry 形成
  间接循环依赖；调用方应直接 from src.agents.prompt_loader import ... 使用

如需在外部访问 prompt_loader，请直接导入：
    from src.agents.prompt_loader import load_master_agent_system_prompt
"""

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
    # prompt_loader 不在此处导出，调用方请直接从 src.agents.prompt_loader 导入
]
