"""
MasterAgent SSE 流式 API 路由（Phase 8-6，Review 修订版）。

接口列表：
    POST   /agent/chat                         — SSE 流式对话
    GET    /agent/sessions                     — 列出所有会话（含元数据）
    DELETE /agent/sessions/{session_id}        — 删除指定会话
    GET    /agent/sessions/{session_id}/history — 获取指定会话的对话历史

Review 修订要点：
    [Fix-1] session_id 路径穿越防御：
        - 路由层用 _validate_session_id() 校验格式，非法值直接 HTTP 400
        - 正则：^[A-Za-z0-9_-]{1,64}$
        - 空值由服务端生成 UUID4（不触发校验）
        - SessionManager 内部也有防御式校验（双重防线）

    [Fix-2] MasterAgent 初始化失败时推送 SSE 错误而非裸 HTTP 500：
        - get_master_agent() 调用点移入 event_generator() 内部
        - 初始化异常被捕获，推送 error + done SSE 事件
        - 保留 get_master_agent() 函数以便测试 patch

    [P5-Step4] get_session_history 改用原生 dict 格式：
        - 移除 LangChain message 类依赖，messages 已是原生 dict 列表
        - 直接按 role 字段（user/assistant/tool/system）规范化后返回

SSE 事件序列（正常流）：
    session_start → (thinking → [tool_start → tool_end]*)* → text_delta* →
    sources? → usage → done

协议说明（v2.1 修正）：
    采用 POST + text/event-stream，而非浏览器原生 GET EventSource。
    前端（Streamlit）通过 requests.post(..., stream=True) 消费；
    curl 测试时需加 -H "Accept: text/event-stream"。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.schemas import AgentChatRequest, SessionMeta
from src.agents.master_agent import MasterAgent
from src.agents.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])

# ── session_id 安全校验 ──────────────────────────────────────────────────────

# [Fix-1] 只允许字母、数字、下划线、连字符，长度 1-64
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _validate_session_id(session_id: str) -> None:
    """
    [Fix-1] 校验 session_id 格式，防止路径穿越攻击。

    合法格式：^[A-Za-z0-9_-]{1,64}$
    非法时抛出 HTTP 400（包含 ../、路径分隔符等都会被拦截）。
    """
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(
            status_code=400,
            detail=(
                f"session_id 格式非法：{session_id!r}。"
                "只允许字母、数字、下划线、连字符，长度 1-64。"
            ),
        )


# ── 进程内单例 ──────────────────────────────────────────────────────────────

_master_agent: MasterAgent | None = None
_session_manager: SessionManager | None = None


def get_master_agent() -> MasterAgent:
    """
    获取 MasterAgent 单例（延迟初始化）。

    [Fix-2] 此函数保留供测试 patch；调用点已移入 event_generator() 内部，
    保证初始化失败时异常能被生成器捕获并推送 SSE error 事件。
    """
    global _master_agent
    if _master_agent is None:
        logger.info("初始化 MasterAgent 单例...")
        _master_agent = MasterAgent()
        logger.info("MasterAgent 单例初始化完成。")
    return _master_agent


def get_session_manager() -> SessionManager:
    """
    获取 SessionManager 单例（独立于 MasterAgent，供会话管理接口使用）。
    不依赖 MasterAgent 初始化，即使 LLM 配置缺失会话管理接口依然可用。
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


# ════════════════════════════════════════════════════════════════════════════
# POST /agent/chat — SSE 流式对话
# ════════════════════════════════════════════════════════════════════════════

@router.post("/chat", summary="SSE 流式对话")
async def agent_chat(request: AgentChatRequest):
    """
    流式对话接口（POST text/event-stream）。

    请求体：
        user_input  str   — 用户输入文本（不可为空）
        session_id  str   — 会话 ID；留空时服务端自动生成（UUID4）

    session_id 安全规则（[Fix-1]）：
        - 留空 → 服务端生成 UUID4（合法，跳过格式校验）
        - 非空 → 必须匹配 ^[A-Za-z0-9_-]{1,64}$，否则 HTTP 400

    响应：
        Content-Type: text/event-stream
        每行格式：event: <type>\\ndata: <json>\\n\\n

    事件类型（顺序）：
        session_start → thinking → tool_start → tool_end → text_delta →
        sources → usage → done | error

    前端消费方式（Streamlit / requests）：
        with requests.post(url, json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line.startswith(b"data:"):
                    event = json.loads(line[5:])
    """
    # 1. user_input 非空校验
    if not request.user_input or not request.user_input.strip():
        raise HTTPException(status_code=400, detail="user_input 不能为空")

    # 2. session_id 处理：空 → 生成 UUID4；非空 → 格式校验 [Fix-1]
    raw_sid = request.session_id.strip()
    if raw_sid:
        _validate_session_id(raw_sid)
        session_id = raw_sid
    else:
        session_id = str(uuid4())

    user_input = request.user_input

    async def event_generator():
        """
        将 MasterAgent.run_stream() 的 BaseEvent 转换为 SSE dict 格式。

        [Fix-2] get_master_agent() 调用点在此处（生成器内部）：
            - MasterAgent 初始化失败时异常在此被捕获；
            - 前端始终能接收到 error + done 事件，而不是裸 HTTP 500。
        """
        # [Fix-2] 在 generator 内获取 agent，捕获初始化异常
        try:
            agent = get_master_agent()
        except Exception as exc:
            logger.error("MasterAgent 初始化失败: %s", exc, exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps(
                    {
                        "type": "error",
                        "message": f"服务初始化失败：{exc}",
                        "code": "AGENT_INIT_ERROR",
                    },
                    ensure_ascii=False,
                ),
            }
            yield {
                "event": "done",
                "data": json.dumps(
                    {"type": "done", "session_id": session_id},
                    ensure_ascii=False,
                ),
            }
            return

        # 正常推理流
        try:
            async for event in agent.run_stream(session_id, user_input):
                yield {
                    "event": event.type,
                    "data": json.dumps(asdict(event), ensure_ascii=False),
                }
        except Exception as exc:
            # 最外层兜底：正常情况下 run_stream 内部已处理异常
            logger.error(
                "agent_chat event_generator 未预期异常 [session=%s]: %s",
                session_id, exc, exc_info=True,
            )
            yield {
                "event": "error",
                "data": json.dumps(
                    {"type": "error", "message": f"服务异常：{exc}", "code": "ROUTER_ERROR"},
                    ensure_ascii=False,
                ),
            }
            yield {
                "event": "done",
                "data": json.dumps(
                    {"type": "done", "session_id": session_id},
                    ensure_ascii=False,
                ),
            }

    return EventSourceResponse(event_generator())


# ════════════════════════════════════════════════════════════════════════════
# GET /agent/sessions — 列出所有会话
# ════════════════════════════════════════════════════════════════════════════

@router.get(
    "/sessions",
    response_model=list[SessionMeta],
    summary="列出所有会话",
)
async def list_sessions():
    """
    返回所有会话的元数据列表，按最近更新时间倒序排列。

    响应字段：
        session_id      str   — 会话唯一标识
        title           str   — 会话标题（第一轮 user_input 前 30 字）
        created_at      str   — 会话创建时间（ISO 8601）
        updated_at      str   — 最近更新时间（ISO 8601）
        turn_count      int   — 已完成对话轮数
        total_tokens    int   — 累计 token 用量
        total_cost_cny  float — 累计估算人民币费用（Phase 9-3）
    """
    sm = get_session_manager()
    sessions = sm.list_sessions()   # list[dict]

    result: list[SessionMeta] = []
    for s in sessions:
        result.append(SessionMeta(
            session_id=s.get("session_id", ""),
            title=s.get("title", "未命名会话"),
            created_at=s.get("created_at", ""),
            updated_at=s.get("updated_at", ""),
            turn_count=s.get("turn_count", 0),
            total_tokens=s.get("total_tokens", 0),
            total_cost_cny=s.get("total_cost_cny", 0.0),
        ))

    # 按 updated_at 倒序（最近在前）
    result.sort(key=lambda x: x.updated_at, reverse=True)
    return result


# ════════════════════════════════════════════════════════════════════════════
# DELETE /agent/sessions/{session_id} — 删除指定会话
# ════════════════════════════════════════════════════════════════════════════

@router.delete(
    "/sessions/{session_id}",
    summary="删除指定会话",
)
async def delete_session(session_id: str):
    """
    删除指定会话的 JSONL 文件与 meta 文件。

    [Fix-1] 校验 session_id 格式，防止路径穿越。

    返回：
        {"deleted": true,  "session_id": "<id>"}  — 删除成功
        {"deleted": false, "session_id": "<id>"}  — 会话不存在（不报错，幂等）
    """
    _validate_session_id(session_id)    # [Fix-1]
    sm = get_session_manager()
    deleted = sm.delete_session(session_id)
    return {"deleted": deleted, "session_id": session_id}


# ════════════════════════════════════════════════════════════════════════════
# GET /agent/sessions/{session_id}/history — 获取会话对话历史
# ════════════════════════════════════════════════════════════════════════════

@router.get(
    "/sessions/{session_id}/history",
    summary="获取会话对话历史",
)
async def get_session_history(session_id: str):
    """
    返回指定会话的完整对话历史（原生 dict 列表）。

    [Fix-1] 校验 session_id 格式，防止路径穿越。
    P5-Step4：messages 已是原生 dict，直接规范化 role 后返回。

    响应格式：
        {
            "session_id": "xxx",
            "messages": [
                {"role": "human",     "content": "用户问题"},
                {"role": "assistant", "content": "AI 回复"},
                {"role": "tool",      "tool_call_id": "...", "content": "工具结果"},
                ...
            ]
        }

    若会话不存在，返回空列表（不报 404，保持前端兼容）。
    """
    _validate_session_id(session_id)    # [Fix-1]
    sm = get_session_manager()

    try:
        messages = sm.load(session_id)   # list[BaseMessage]
    except Exception as exc:
        logger.warning("加载会话历史失败 [session=%s]: %s", session_id, exc)
        messages = []

    # P5-Step4：messages 已是原生 dict 列表，直接透传（规范化 role 显示）
    serialized = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "user":
            serialized.append({"role": "human", "content": content})
        elif role == "tool":
            serialized.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", ""),
                "content": content,
            })
        elif role == "assistant":
            entry: dict = {"role": "assistant", "content": content}
            if msg.get("tool_calls"):
                entry["tool_calls"] = msg["tool_calls"]
            serialized.append(entry)
        elif role == "system":
            serialized.append({"role": "system", "content": content})
        else:
            serialized.append({"role": role, "content": str(content)})

    return {"session_id": session_id, "messages": serialized}
