"""
MasterAgent 统一对话页（Phase 8-7，Review 修订版）。

替代旧版 page_chat.py + page_research.py，提供单一对话框入口。
MasterAgent 在后端自动决策调用哪些工具（retrieve_knowledge / deep_research /
generate_research_ideas / list_documents / get_document_metadata /
search_by_entity / get_knowledge_graph_stats），前端只需展示流式事件。

Review 修订要点（Phase 8-7 Review）：
    [Fix-1+3] 点击历史会话时调用 GET /agent/sessions/{id}/history 恢复前端消息列表；
              新增 _load_session_history()，切换会话时填充 agent_messages。
    [Fix-2]   错误流结束后不再追加空 assistant 消息：
              - 记录 error_occurred + error_message；
              - 无文本且有错误时把错误信息作为 final_content 持久化到历史。
    [Fix-4]   _STREAM_TIMEOUT 提高至 300 s；注释说明后端应对长工具发送 heartbeat。
    [Fix-5]   _render_message() 支持 system / tool / human 等角色归一化；
              system → st.info()，tool → 跳过，human → "user"。
    [Fix-6]   工具配对改为按 tool_name FIFO 配对（_pair_tool_events()），
              不再依赖相邻下标，支持并发工具与同名工具多次调用。

页面布局（两列）：
┌─────────────────┬─────────────────────────────────────────┐
│ 侧边栏           │ 主区域                                   │
│ ─────────       │ ─────────────────────────────────────   │
│ 📚 会话列表      │ 对话历史（气泡）                          │
│ > 今天 - 会话1   │   [user] 你好                            │
│   昨天 - 会话2   │   🔍 正在检索知识库... ✅(342ms)          │
│ ─────────       │   [assistant] 根据检索...                 │
│ [+ 新建会话]     │                                          │
│                 │ [输入框] [发送]                           │
└─────────────────┴─────────────────────────────────────────┘

SSE 消费策略：
    requests.post(..., stream=True) → iter_lines() → 按行解析 SSE
    支持事件：session_start / thinking / tool_start / tool_end /
              text_delta / sources / usage / done / error
"""
from __future__ import annotations

import json
from typing import Generator

import requests
import streamlit as st


# ── 常量 ────────────────────────────────────────────────────────────────────

_STREAM_TIMEOUT = 300   # seconds
# [Fix-4] 对 deep_research 等长工具，300 s 仍可能不足。
# 根本解法是后端在工具执行期间定期发送 SSE heartbeat（如 ": heartbeat\n\n"），
# 防止因两端之间无字节流而触发 read timeout。


# ════════════════════════════════════════════════════════════════════════════
# SSE 消费
# ════════════════════════════════════════════════════════════════════════════

def _stream_agent(
    api_base: str,
    session_id: str,
    user_input: str,
) -> Generator[dict, None, None]:
    """
    POST /agent/chat 并逐事件 yield SSE data dict。

    每个 SSE 消息格式：
        event: <type>
        data: <json>
        （空行分隔）

    解析策略：逐行扫描，遇到 "data:" 前缀即解析 JSON；
    event 字段覆盖 data["type"]，保持前端一致性。
    """
    url = f"{api_base}/agent/chat"
    payload = {"user_input": user_input, "session_id": session_id}

    try:
        with requests.post(url, json=payload, stream=True, timeout=_STREAM_TIMEOUT) as resp:
            if not resp.ok:
                yield {"type": "error", "message": f"HTTP {resp.status_code}: {resp.text}",
                       "code": "HTTP_ERROR"}
                yield {"type": "done", "session_id": session_id}
                return

            current_event_type: str = ""
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    current_event_type = ""
                    continue
                if raw_line.startswith("event:"):
                    current_event_type = raw_line[len("event:"):].strip()
                elif raw_line.startswith("data:"):
                    raw_data = raw_line[len("data:"):].strip()
                    try:
                        data = json.loads(raw_data)
                        if current_event_type:
                            data["type"] = current_event_type
                        yield data
                    except (json.JSONDecodeError, ValueError):
                        pass   # 忽略非 JSON 行

    except requests.exceptions.Timeout:
        yield {"type": "error", "message": "请求超时，MasterAgent 可能仍在处理中",
               "code": "TIMEOUT"}
        yield {"type": "done", "session_id": session_id}
    except Exception as exc:
        yield {"type": "error", "message": f"连接 API 失败：{exc}", "code": "CONNECTION_ERROR"}
        yield {"type": "done", "session_id": session_id}


# ════════════════════════════════════════════════════════════════════════════
# 会话管理辅助
# ════════════════════════════════════════════════════════════════════════════

def _load_session_list(api_base: str) -> list[dict]:
    """从 GET /agent/sessions 加载会话列表，失败时返回空列表。"""
    try:
        resp = requests.get(f"{api_base}/agent/sessions", timeout=5)
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return []


def _load_session_history(api_base: str, session_id: str) -> list[dict]:
    """
    [Fix-1+3] 从 GET /agent/sessions/{id}/history 加载历史消息，
    转换为 agent_messages 格式（含角色归一化）。

    角色映射：
        human / user  → {"role": "user",      "content": ...}
        assistant     → {"role": "assistant",  "content": ..., "tool_events": [], ...}
        system        → 作为 assistant 消息展示，标注"对话历史摘要"
        tool          → 跳过（工具调用结果是内部细节，不直接作为对话气泡）
    """
    messages: list[dict] = []
    try:
        resp = requests.get(
            f"{api_base}/agent/sessions/{session_id}/history",
            timeout=5,
        )
        if resp.ok:
            data = resp.json()
            for msg in data.get("messages", []):
                role    = msg.get("role", "")
                content = msg.get("content", "")

                if role in ("human", "user"):
                    messages.append({"role": "user", "content": content})

                elif role == "assistant":
                    messages.append({
                        "role":        "assistant",
                        "content":     content,
                        "tool_events": [],
                        "sources":     [],
                        "usage":       {},
                    })

                elif role == "system":
                    # 历史摘要由 SessionManager 以 SystemMessage 注入，
                    # 呈现为带标识的 assistant 消息，方便用户了解上下文
                    messages.append({
                        "role":        "assistant",
                        "content":     f"📋 **对话历史摘要**\n\n{content}",
                        "tool_events": [],
                        "sources":     [],
                        "usage":       {},
                    })

                # tool 消息是工具调用的原始结果，跳过直接渲染
                # （后续可放入对应 assistant 消息的 tool_events 中）

    except Exception:
        pass
    return messages


def _delete_session(api_base: str, session_id: str) -> bool:
    """删除指定会话，成功返回 True。"""
    try:
        resp = requests.delete(f"{api_base}/agent/sessions/{session_id}", timeout=5)
        return resp.ok and resp.json().get("deleted", False)
    except Exception:
        return False


# ════════════════════════════════════════════════════════════════════════════
# 工具事件配对
# ════════════════════════════════════════════════════════════════════════════

def _pair_tool_events(
    tool_events: list[dict],
) -> list[tuple[dict, dict | None]]:
    """
    [Fix-6] 按 tool_name 将 tool_start / tool_end 配对为 (start, end?) 元组列表。

    策略：同名工具按 FIFO 顺序配对，支持：
        - 顺序执行同名工具（多次调用）
        - 并发不同工具（不同名）
        - 中间插入 thinking / text_delta / error 等事件（不影响配对）
        - 尚未收到 tool_end 的工具（end=None，表示仍在运行）
    """
    pending: dict[str, list[dict]] = {}  # tool_name → [start_event, ...]
    pairs:   list[tuple[dict, dict | None]] = []

    for evt in tool_events:
        evt_type = evt.get("type")
        name     = evt.get("tool_name", "")

        if evt_type == "tool_start":
            pending.setdefault(name, []).append(evt)

        elif evt_type == "tool_end":
            if name in pending and pending[name]:
                start = pending[name].pop(0)   # FIFO
                pairs.append((start, evt))
            # 孤立的 tool_end：忽略

    # 仍在运行（未收到 tool_end）的工具
    for name_starts in pending.values():
        for start in name_starts:
            pairs.append((start, None))

    return pairs


# ════════════════════════════════════════════════════════════════════════════
# 消息渲染辅助
# ════════════════════════════════════════════════════════════════════════════

def _render_tool_events(tool_events: list[dict]) -> None:
    """
    [Fix-6] 在可折叠区块中渲染工具调用过程（按 tool_name FIFO 配对）。
    """
    if not tool_events:
        return
    pairs = _pair_tool_events(tool_events)
    if not pairs:
        return

    with st.expander("🔧 工具调用过程", expanded=False):
        for start, end in pairs:
            display = start.get("display_message", start.get("tool_name", ""))
            if end is None:
                st.write(f"⏳ {display}")
            else:
                st.write(
                    f"✅ {display}  ·  {end.get('result_summary', '')}  "
                    f"`{end.get('elapsed_ms', 0)} ms`"
                )


def _render_message(msg: dict) -> None:
    """
    [Fix-5] 渲染单条历史消息，支持多种角色归一化。

    角色处理规则：
        user / human  → st.chat_message("user")
        assistant     → st.chat_message("assistant")
        system        → st.info()（历史摘要，不用对话气泡）
        tool          → 跳过（工具调用结果应在 tool_events 折叠区展示）
    """
    role = msg["role"]

    # system 摘要：用 info 块替代 chat_message，避免 Streamlit 不支持 system 角色
    if role == "system":
        st.info(f"📋 {msg['content']}")
        return

    # tool 消息：内部细节，不直接渲染为对话气泡
    if role == "tool":
        return

    # human → user（角色归一化）
    display_role = "user" if role in ("user", "human") else "assistant"

    with st.chat_message(display_role):
        st.markdown(msg["content"])

        # 工具调用过程
        if msg.get("tool_events"):
            _render_tool_events(msg["tool_events"])

        # 来源
        if msg.get("sources"):
            with st.expander("📎 参考来源", expanded=False):
                for src in msg["sources"]:
                    st.caption(f"· {src}")

        # token 用量
        if msg.get("usage"):
            u = msg["usage"]
            st.caption(
                f"📊 tokens: {u.get('total_tokens', 0)}  "
                f"(prompt {u.get('prompt_tokens', 0)} / "
                f"completion {u.get('completion_tokens', 0)})  "
                f"· 费用≈¥{u.get('estimated_cost_cny', 0):.5f}"
            )


# ════════════════════════════════════════════════════════════════════════════
# 主渲染函数
# ════════════════════════════════════════════════════════════════════════════

def render(api_base: str) -> None:
    """
    MasterAgent 统一对话页主入口。

    Session State 键：
        agent_session_id   str        — 当前会话 ID（由后端维护）
        agent_messages     list[dict] — 对话历史，每条包含 role/content/tool_events/sources/usage
        agent_sessions     list[dict] — 从 /agent/sessions 拉取的会话列表
    """
    st.header("🤖 智能对话助手")
    st.caption("MasterAgent · 自动决策工具调用 · 流式响应")

    # ── 初始化 Session State ───────────────────────────────────────────────
    if "agent_session_id" not in st.session_state:
        st.session_state["agent_session_id"] = ""   # 空 = 下次发送时由后端生成
    if "agent_messages" not in st.session_state:
        st.session_state["agent_messages"] = []
    if "agent_sessions" not in st.session_state:
        st.session_state["agent_sessions"] = []

    # ── 侧边栏：会话列表 ───────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("💬 会话列表")

        # 刷新按钮
        if st.button("🔄 刷新列表", key="agent_refresh_sessions", use_container_width=True):
            st.session_state["agent_sessions"] = _load_session_list(api_base)

        # 新建会话
        if st.button("➕ 新建会话", key="agent_new_session", use_container_width=True):
            st.session_state["agent_session_id"] = ""
            st.session_state["agent_messages"] = []
            st.rerun()

        st.divider()

        # 会话条目列表
        sessions: list[dict] = st.session_state.get("agent_sessions", [])
        if not sessions:
            st.caption("暂无历史会话，刷新列表或发送第一条消息。")
        else:
            current_sid = st.session_state.get("agent_session_id", "")
            for sess in sessions:
                sid   = sess.get("session_id", "")
                title = sess.get("title", "未命名会话")[:28]
                turns = sess.get("turn_count", 0)
                is_current = (sid == current_sid)

                col_title, col_del = st.columns([5, 1])
                with col_title:
                    label = f"**{title}**" if is_current else title
                    if st.button(
                        f"{label}  `{turns}轮`",
                        key=f"agent_sess_{sid}",
                        use_container_width=True,
                    ):
                        # [Fix-1+3] 切换会话：调用 history 接口恢复前端消息列表
                        st.session_state["agent_session_id"] = sid
                        st.session_state["agent_messages"] = _load_session_history(
                            api_base, sid
                        )
                        st.rerun()

                with col_del:
                    if st.button("🗑", key=f"agent_del_{sid}", help="删除此会话"):
                        if _delete_session(api_base, sid):
                            st.session_state["agent_sessions"] = _load_session_list(api_base)
                            if st.session_state.get("agent_session_id") == sid:
                                st.session_state["agent_session_id"] = ""
                                st.session_state["agent_messages"] = []
                            st.rerun()

    # ── 主区域：对话历史 + 输入框 ─────────────────────────────────────────

    # 当前会话 ID 标注
    current_sid = st.session_state.get("agent_session_id", "")
    if current_sid:
        st.caption(f"会话 ID：`{current_sid}`")
    else:
        st.caption("新会话（发送第一条消息后自动分配 ID）")

    # 渲染历史消息（[Fix-5] _render_message 已支持角色归一化）
    for msg in st.session_state["agent_messages"]:
        _render_message(msg)

    # ── 输入框 ────────────────────────────────────────────────────────────
    user_input = st.chat_input("向 GraphAssistant 提问…")
    if not user_input:
        return

    # 显示用户消息
    st.session_state["agent_messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ── 流式请求 MasterAgent ─────────────────────────────────────────────
    with st.chat_message("assistant"):
        # 各类事件累积容器
        tool_events:      list[dict] = []
        sources:          list[str]  = []
        usage:            dict       = {}
        final_session_id: str        = current_sid

        # [Fix-2] 错误状态追踪
        error_occurred: bool = False
        error_message:  str  = ""

        # 占位：工具调用区块（实时更新）
        tool_placeholder = st.empty()
        # 占位：流式文本
        text_placeholder = st.empty()
        accumulated_text = ""

        def _refresh_tool_display() -> None:
            """
            [Fix-6] 实时刷新工具调用状态（按 tool_name FIFO 配对）。
            """
            if not tool_events:
                return
            pairs = _pair_tool_events(tool_events)
            lines: list[str] = []
            for start, end in pairs:
                display = start.get("display_message", start.get("tool_name", ""))
                if end is None:
                    lines.append(f"⏳ {display}")
                else:
                    lines.append(
                        f"✅ {display}  ·  {end.get('result_summary', '')}  "
                        f"`{end.get('elapsed_ms', 0)} ms`"
                    )
            tool_placeholder.info("\n\n".join(lines) if lines else "")

        # 消费 SSE 流
        for evt_data in _stream_agent(api_base, current_sid, user_input):
            evt_type = evt_data.get("type", "")

            if evt_type == "session_start":
                final_session_id = evt_data.get("session_id", current_sid)

            elif evt_type == "thinking":
                iteration = evt_data.get("iteration", 0)
                if iteration == 0:
                    tool_placeholder.caption("🤔 正在思考…")

            elif evt_type == "tool_start":
                tool_events.append(evt_data)
                _refresh_tool_display()

            elif evt_type == "tool_end":
                tool_events.append(evt_data)
                _refresh_tool_display()

            elif evt_type == "text_delta":
                accumulated_text += evt_data.get("delta", "")
                text_placeholder.markdown(accumulated_text + "▌")

            elif evt_type == "sources":
                sources = evt_data.get("sources", [])

            elif evt_type == "usage":
                usage = evt_data

            elif evt_type == "error":
                # [Fix-2] 记录错误状态，供流结束后持久化
                error_occurred = True
                error_message  = evt_data.get("message", "未知错误")
                st.error(
                    f"❌ {error_message}  "
                    f"（code: {evt_data.get('code', '')}）"
                )

            elif evt_type == "done":
                final_session_id = evt_data.get("session_id", final_session_id)

        # ── 流结束处理 ────────────────────────────────────────────────────

        # [Fix-2] 确定最终展示内容：
        #   - 有文本 → 使用文本（即使同时有错误，文本优先）
        #   - 无文本且有错误 → 把错误信息作为 assistant 内容持久化，刷新后不丢失
        #   - 无文本且无错误 → 默认占位
        if accumulated_text:
            final_content = accumulated_text
        elif error_occurred:
            final_content = f"❌ 请求失败：{error_message}"
        else:
            final_content = "（无文本回复）"

        text_placeholder.markdown(final_content)

        # 工具调用过程：折叠展示（替换 info 占位）
        tool_placeholder.empty()
        if tool_events:
            _render_tool_events(tool_events)

        # 来源
        if sources:
            with st.expander("📎 参考来源", expanded=False):
                for src in sources:
                    st.caption(f"· {src}")

        # token 用量
        if usage:
            st.caption(
                f"📊 tokens: {usage.get('total_tokens', 0)}  "
                f"(prompt {usage.get('prompt_tokens', 0)} / "
                f"completion {usage.get('completion_tokens', 0)})  "
                f"· 费用≈¥{usage.get('estimated_cost_cny', 0):.5f}"
            )

    # ── 持久化本轮消息 ────────────────────────────────────────────────────
    st.session_state["agent_session_id"] = final_session_id
    # [Fix-2] 使用 final_content 而非 accumulated_text，确保错误信息被持久化
    st.session_state["agent_messages"].append({
        "role":        "assistant",
        "content":     final_content,
        "tool_events": tool_events,
        "sources":     sources,
        "usage":       usage,
    })

    # 刷新会话列表（拉取最新 title/turn_count）
    st.session_state["agent_sessions"] = _load_session_list(api_base)
