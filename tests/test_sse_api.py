"""
Phase 8-6 SSE API 接口测试（tests/test_sse_api.py，Review 修订版）。

测试策略：
    - 使用 FastAPI TestClient（同步 HTTPX）测试路由注册与基本响应格式
    - MasterAgent 整体 mock，不触发真实 LLM 调用
    - SessionManager 整体 mock，不读写磁盘
    - SSE 内容解析：按行解析 "event:..." / "data:..." 格式，验证事件类型与字段

Review 修订新增覆盖（TC-11 ~ TC-16）：
    [TC-11][Fix-1] 非法 session_id（含 ../）→ HTTP 400
    [TC-12][Fix-1] 超长 session_id（>64 字符）→ HTTP 400
    [TC-13][Fix-1] DELETE 非法 session_id → HTTP 400
    [TC-14][Fix-2] MasterAgent 初始化失败 → 推送 error + done SSE（不裸 500）
    [TC-15][Fix-5] history 端到端：对话后 history 包含 human + assistant 消息
    [TC-16][Fix-6] main.py 集成：/agent/* 路由已注册，/health 版本一致

原有测试（TC-1 ~ TC-10）保持兼容，无修改。
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.agent import router
from src.agents.events import (
    DoneEvent,
    ErrorEvent,
    SessionStartEvent,
    TextDeltaEvent,
    ThinkingEvent,
    ToolEndEvent,
    ToolStartEvent,
    UsageEvent,
)


# ════════════════════════════════════════════════════════════════════════════
# 测试夹具
# ════════════════════════════════════════════════════════════════════════════

def _make_app() -> FastAPI:
    """每个测试用例独立创建 FastAPI 实例，避免单例状态污染。"""
    app = FastAPI()
    app.include_router(router)
    return app


def _parse_sse(body: bytes) -> list[dict]:
    """
    解析 SSE 响应体，返回 event dict 列表。
    每条 SSE 消息由若干 "field: value" 行组成，以空行分隔。
    """
    events: list[dict] = []
    current: dict = {}
    for raw_line in body.decode("utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                events.append(current)
                current = {}
        elif line.startswith("event:"):
            current["_event_type"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            raw_data = line[len("data:"):].strip()
            try:
                current["_data"] = json.loads(raw_data)
            except (json.JSONDecodeError, ValueError):
                current["_data"] = raw_data
    if current:
        events.append(current)
    return events


def _make_mock_agent(event_sequence: list):
    """
    构造 mock MasterAgent，其 run_stream() 按顺序 yield event_sequence。
    """
    mock_agent = MagicMock()

    async def _fake_run_stream(*_args, **_kwargs):
        for evt in event_sequence:
            yield evt

    mock_agent.run_stream = _fake_run_stream
    return mock_agent


def _make_mock_session_manager(
    sessions: list[dict] | None = None,
    delete_return: bool = True,
    history: list | None = None,
):
    """构造 mock SessionManager。"""
    mock_sm = MagicMock()
    mock_sm.list_sessions.return_value = sessions or []
    mock_sm.delete_session.return_value = delete_return
    mock_sm.load.return_value = history or []
    return mock_sm


# ════════════════════════════════════════════════════════════════════════════
# [TC-1] POST /agent/chat 正常流
# ════════════════════════════════════════════════════════════════════════════

def test_agent_chat_normal_stream():
    """正常流应包含 session_start 和 done 事件，data 字段可解析为 JSON。"""
    events_seq = [
        SessionStartEvent(session_id="sess-001"),
        ThinkingEvent(iteration=0),
        TextDeltaEvent(delta="你好"),
        UsageEvent(total_tokens=100, prompt_tokens=80, completion_tokens=20, estimated_cost_cny=0.01),
        DoneEvent(session_id="sess-001"),
    ]
    mock_agent = _make_mock_agent(events_seq)

    with patch("api.routers.agent.get_master_agent", return_value=mock_agent):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/agent/chat",
            json={"user_input": "你好", "session_id": "sess-001"},
            headers={"Accept": "text/event-stream"},
        )

    assert resp.status_code == 200
    parsed = _parse_sse(resp.content)
    event_types = [e.get("_event_type") for e in parsed]

    assert "session_start" in event_types, f"缺少 session_start，实际事件: {event_types}"
    assert "done" in event_types, f"缺少 done，实际事件: {event_types}"

    ss_evt = next(e for e in parsed if e.get("_event_type") == "session_start")
    assert ss_evt["_data"].get("session_id") == "sess-001"


# ════════════════════════════════════════════════════════════════════════════
# [TC-2] POST /agent/chat 空 user_input — 期望 HTTP 400
# ════════════════════════════════════════════════════════════════════════════

def test_agent_chat_empty_input_returns_400():
    """user_input 为空字符串时，接口应返回 HTTP 400。"""
    mock_agent = _make_mock_agent([])
    with patch("api.routers.agent.get_master_agent", return_value=mock_agent):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/agent/chat", json={"user_input": "", "session_id": ""})
    assert resp.status_code == 400


# ════════════════════════════════════════════════════════════════════════════
# [TC-3] 自动生成 session_id
# ════════════════════════════════════════════════════════════════════════════

def test_agent_chat_auto_session_id():
    """
    session_id 为空时，服务端应自动生成非空 UUID4 格式字符串。
    """
    captured_ids: list[str] = []

    async def _fake_run_stream(session_id: str, user_input: str):
        captured_ids.append(session_id)
        yield SessionStartEvent(session_id=session_id)
        yield DoneEvent(session_id=session_id)

    mock_agent = MagicMock()
    mock_agent.run_stream = _fake_run_stream

    with patch("api.routers.agent.get_master_agent", return_value=mock_agent):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/agent/chat", json={"user_input": "hello", "session_id": ""})

    assert resp.status_code == 200
    assert len(captured_ids) == 1
    assert len(captured_ids[0]) == 36       # UUID4 长度
    assert captured_ids[0].count("-") == 4  # UUID4 格式


# ════════════════════════════════════════════════════════════════════════════
# [TC-4] GET /agent/sessions — 返回 list[SessionMeta]
# ════════════════════════════════════════════════════════════════════════════

def test_list_sessions():
    """GET /agent/sessions 应返回 200 和 SessionMeta 格式的 JSON 数组。"""
    mock_sessions = [
        {
            "session_id": "abc",
            "title": "RAG 问题",
            "created_at": "2026-04-16T10:00:00Z",
            "updated_at": "2026-04-16T10:01:00Z",
            "turn_count": 2,
            "total_tokens": 1500,
        }
    ]
    mock_sm = _make_mock_session_manager(sessions=mock_sessions)

    with patch("api.routers.agent.get_session_manager", return_value=mock_sm):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/agent/sessions")

    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["session_id"] == "abc"
    assert data[0]["title"] == "RAG 问题"
    assert data[0]["turn_count"] == 2


# ════════════════════════════════════════════════════════════════════════════
# [TC-5] DELETE /agent/sessions/{id} — 会话存在
# ════════════════════════════════════════════════════════════════════════════

def test_delete_session_existing():
    """删除存在的会话，应返回 deleted=true。"""
    mock_sm = _make_mock_session_manager(delete_return=True)
    with patch("api.routers.agent.get_session_manager", return_value=mock_sm):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.delete("/agent/sessions/abc")

    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted"] is True
    assert data["session_id"] == "abc"


# ════════════════════════════════════════════════════════════════════════════
# [TC-6] DELETE /agent/sessions/{id} 不存在 — 幂等返回 deleted=false
# ════════════════════════════════════════════════════════════════════════════

def test_delete_session_nonexistent():
    """删除不存在的会话，应返回 deleted=false（幂等，不报 404）。"""
    mock_sm = _make_mock_session_manager(delete_return=False)
    with patch("api.routers.agent.get_session_manager", return_value=mock_sm):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.delete("/agent/sessions/nonexistent")

    assert resp.status_code == 200
    assert resp.json()["deleted"] is False


# ════════════════════════════════════════════════════════════════════════════
# [TC-7] GET /agent/sessions/{id}/history — 返回 messages 列表
# ════════════════════════════════════════════════════════════════════════════

def test_get_session_history():
    """GET history 应返回 messages 列表，包含角色和内容字段。"""
    from langchain_core.messages import AIMessage, HumanMessage

    mock_history = [
        HumanMessage(content="RAG 是什么？"),
        AIMessage(content="RAG 是检索增强生成..."),
    ]
    mock_sm = _make_mock_session_manager(history=mock_history)

    with patch("api.routers.agent.get_session_manager", return_value=mock_sm):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/agent/sessions/abc/history")

    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "abc"
    msgs = data["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "human"
    assert msgs[0]["content"] == "RAG 是什么？"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "RAG 是检索增强生成..."


# ════════════════════════════════════════════════════════════════════════════
# [TC-8] SSE 事件顺序验证
# ════════════════════════════════════════════════════════════════════════════

def test_agent_chat_event_order():
    """session_start 必须是第一个事件，done 必须是最后一个事件。"""
    events_seq = [
        SessionStartEvent(session_id="s1"),
        ThinkingEvent(iteration=0),
        ToolStartEvent(tool_name="retrieve_knowledge", display_message="检索中..."),
        ToolEndEvent(tool_name="retrieve_knowledge", result_summary="找到 3 条", elapsed_ms=200),
        TextDeltaEvent(delta="结论如下..."),
        UsageEvent(total_tokens=50, prompt_tokens=30, completion_tokens=20, estimated_cost_cny=0.002),
        DoneEvent(session_id="s1"),
    ]
    mock_agent = _make_mock_agent(events_seq)

    with patch("api.routers.agent.get_master_agent", return_value=mock_agent):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/agent/chat", json={"user_input": "有哪些论文？", "session_id": "s1"})

    assert resp.status_code == 200
    parsed = _parse_sse(resp.content)
    types = [e.get("_event_type") for e in parsed]

    assert types[0] == "session_start", f"第一个事件应是 session_start，实际: {types[0]}"
    assert types[-1] == "done", f"最后一个事件应是 done，实际: {types[-1]}"


# ════════════════════════════════════════════════════════════════════════════
# [TC-9] tool_start / tool_end 事件字段验证
# ════════════════════════════════════════════════════════════════════════════

def test_tool_events_fields():
    """
    tool_start 事件应包含 tool_name 和 display_message；
    tool_end 事件应包含 tool_name、result_summary 和 elapsed_ms。
    """
    events_seq = [
        SessionStartEvent(session_id="s2"),
        ToolStartEvent(tool_name="list_documents", display_message="正在查询文献列表..."),
        ToolEndEvent(tool_name="list_documents", result_summary="共 5 篇文献", elapsed_ms=150),
        DoneEvent(session_id="s2"),
    ]
    mock_agent = _make_mock_agent(events_seq)

    with patch("api.routers.agent.get_master_agent", return_value=mock_agent):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/agent/chat", json={"user_input": "列出文献", "session_id": "s2"})

    parsed = _parse_sse(resp.content)
    ts_evt = next((e for e in parsed if e.get("_event_type") == "tool_start"), None)
    te_evt = next((e for e in parsed if e.get("_event_type") == "tool_end"), None)

    assert ts_evt is not None, "缺少 tool_start 事件"
    assert ts_evt["_data"]["tool_name"] == "list_documents"
    assert ts_evt["_data"]["display_message"] == "正在查询文献列表..."

    assert te_evt is not None, "缺少 tool_end 事件"
    assert te_evt["_data"]["result_summary"] == "共 5 篇文献"
    assert te_evt["_data"]["elapsed_ms"] == 150


# ════════════════════════════════════════════════════════════════════════════
# [TC-10] agent 内部异常时推送 error + done
# ════════════════════════════════════════════════════════════════════════════

def test_agent_chat_error_event():
    """
    MasterAgent.run_stream() 推送 ErrorEvent 时，SSE 流中应包含 error 事件，
    且 error 事件后跟 done 事件。
    """
    events_seq = [
        SessionStartEvent(session_id="s3"),
        ErrorEvent(message="LLM 推理失败：connection timeout", code="LLM_STREAM_ERROR"),
        DoneEvent(session_id="s3"),
    ]
    mock_agent = _make_mock_agent(events_seq)

    with patch("api.routers.agent.get_master_agent", return_value=mock_agent):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/agent/chat", json={"user_input": "测试错误", "session_id": "s3"})

    parsed = _parse_sse(resp.content)
    types = [e.get("_event_type") for e in parsed]

    assert "error" in types, f"缺少 error 事件，实际事件序列: {types}"
    assert "done" in types, f"缺少 done 事件，实际事件序列: {types}"
    assert types.index("error") < types.index("done"), "error 事件应在 done 之前"

    err_evt = next(e for e in parsed if e.get("_event_type") == "error")
    assert "LLM 推理失败" in err_evt["_data"]["message"]
    assert err_evt["_data"]["code"] == "LLM_STREAM_ERROR"


# ════════════════════════════════════════════════════════════════════════════
# [TC-11][Fix-1] 非法 session_id（含路径分隔符）→ HTTP 400
# ════════════════════════════════════════════════════════════════════════════

def test_invalid_session_id_path_traversal_returns_400():
    """
    [Fix-1] session_id 含 ../ 或路径分隔符时，POST /agent/chat 应返回 HTTP 400，
    不触发 MasterAgent 初始化，防止路径穿越。
    """
    app = _make_app()
    client = TestClient(app, raise_server_exceptions=False)

    for bad_sid in ["../evil", "..\\outside", "foo/bar", "foo bar", "a" * 65]:
        resp = client.post("/agent/chat", json={"user_input": "hello", "session_id": bad_sid})
        assert resp.status_code == 400, (
            f"session_id={bad_sid!r} 应被拒绝，实际返回 {resp.status_code}"
        )


# ════════════════════════════════════════════════════════════════════════════
# [TC-12][Fix-1] 合法 session_id 格式通过校验
# ════════════════════════════════════════════════════════════════════════════

def test_valid_session_id_formats_accepted():
    """
    [Fix-1] 合法格式的 session_id 不应被拒绝（字母、数字、_ 、-，长度 1-64）。
    """
    mock_agent = _make_mock_agent([
        SessionStartEvent(session_id="ok"),
        DoneEvent(session_id="ok"),
    ])

    with patch("api.routers.agent.get_master_agent", return_value=mock_agent):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)

        for good_sid in ["abc", "abc-123", "abc_123", "A" * 64]:
            mock_agent.run_stream = _make_mock_agent([
                SessionStartEvent(session_id=good_sid),
                DoneEvent(session_id=good_sid),
            ]).run_stream
            resp = client.post(
                "/agent/chat",
                json={"user_input": "hello", "session_id": good_sid},
            )
            assert resp.status_code == 200, (
                f"session_id={good_sid!r} 应被接受，实际返回 {resp.status_code}"
            )


# ════════════════════════════════════════════════════════════════════════════
# [TC-13][Fix-1] DELETE 非法 session_id → HTTP 400
# ════════════════════════════════════════════════════════════════════════════

def test_delete_invalid_session_id_returns_400():
    """
    [Fix-1] DELETE /agent/sessions/{id} 传入非法 session_id 时应返回 HTTP 400。

    注意：含 '/' 的路径穿越（如 ../evil）在 HTTP 层就被 FastAPI 路由规范化处理，
    永远无法到达我们的处理函数（返回 404）。真正需要防御的是不含 '/' 的变体：
    - "..evil"    — 含 '.'，不在 [A-Za-z0-9_-] 字符集内
    - "abc!def"   — 含 '!'，特殊字符
    - 超长 ID     — 超出 64 字符限制
    这些变体能到达路由处理函数，由 _validate_session_id() 拦截并返回 400。
    """
    app = _make_app()
    client = TestClient(app, raise_server_exceptions=False)

    for bad_sid in ["..evil", "abc!def", "a b", "a" * 65]:
        resp = client.delete(f"/agent/sessions/{bad_sid}")
        assert resp.status_code == 400, (
            f"session_id={bad_sid!r} 应被拒绝（400），实际返回 {resp.status_code}"
        )


# ════════════════════════════════════════════════════════════════════════════
# [TC-14][Fix-2] MasterAgent 初始化失败 → 推送 SSE error + done（不裸 500）
# ════════════════════════════════════════════════════════════════════════════

def test_agent_init_failure_yields_sse_error():
    """
    [Fix-2] get_master_agent() 抛出异常时（如 API_KEY 缺失），
    接口应返回 HTTP 200 SSE 流，并在流中推送 error + done 事件，
    而不是裸 HTTP 500。
    """
    def _raise_on_init():
        raise RuntimeError("API_KEY 未配置，无法初始化 LLM")

    with patch("api.routers.agent.get_master_agent", side_effect=_raise_on_init):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/agent/chat",
            json={"user_input": "hello", "session_id": "s4"},
        )

    # HTTP 状态码应为 200（SSE 流已建立）
    assert resp.status_code == 200, (
        f"初始化失败应返回 200 SSE 流，实际返回 {resp.status_code}"
    )

    parsed = _parse_sse(resp.content)
    types = [e.get("_event_type") for e in parsed]

    assert "error" in types, f"缺少 error 事件，实际: {types}"
    assert "done" in types, f"缺少 done 事件，实际: {types}"
    assert types.index("error") < types.index("done"), "error 应在 done 之前"

    err_evt = next(e for e in parsed if e.get("_event_type") == "error")
    assert err_evt["_data"]["code"] == "AGENT_INIT_ERROR"
    assert "API_KEY" in err_evt["_data"]["message"]


# ════════════════════════════════════════════════════════════════════════════
# [TC-15][Fix-5] history 端到端：save_turn 后 load 能恢复 human + assistant
# ════════════════════════════════════════════════════════════════════════════

def test_history_contains_human_and_assistant_after_turn():
    """
    [Fix-5] 发送一轮对话后，GET /history 必须同时包含 human 和 assistant 消息。

    使用真实 SessionManager（写临时目录）验证完整的 save → load → history 链路，
    不 mock SessionManager，以捕获 Phase 8-5 base_length 修复的回归。
    """
    import tempfile
    from pathlib import Path
    from unittest.mock import patch as _patch

    from langchain_core.messages import AIMessage, HumanMessage

    from src.agents.session_manager import SessionManager

    # 使用临时目录隔离磁盘操作
    with tempfile.TemporaryDirectory() as tmpdir:
        real_sm = SessionManager.__new__(SessionManager)
        real_sm.SESSION_DIR = Path(tmpdir) / "sessions"
        real_sm.SESSION_DIR.mkdir(parents=True, exist_ok=True)
        real_sm.COMPACT_TOKEN_THRESHOLD = 8000
        real_sm.KEEP_RECENT_TURNS = 3

        sid = "test-history-e2e"

        # 模拟 MasterAgent 调用 save_turn 时的 new_messages（含 HumanMessage）
        new_msgs = [
            HumanMessage(content="RAG 是什么？"),
            AIMessage(content="RAG 是检索增强生成技术。"),
        ]
        real_sm.save_turn(
            session_id=sid,
            user_input="RAG 是什么？",
            new_messages=new_msgs,
            sources=[],
            token_usage={"prompt": 100, "completion": 50, "total": 150},
        )

        # 通过 history 接口查看（注入真实 SM）
        with _patch("api.routers.agent.get_session_manager", return_value=real_sm):
            app = _make_app()
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get(f"/agent/sessions/{sid}/history")

    assert resp.status_code == 200
    msgs = resp.json()["messages"]
    roles = [m["role"] for m in msgs]

    assert "human" in roles, f"history 缺少 human 消息，实际 roles: {roles}"
    assert "assistant" in roles, f"history 缺少 assistant 消息，实际 roles: {roles}"

    human_msg = next(m for m in msgs if m["role"] == "human")
    assert human_msg["content"] == "RAG 是什么？"

    ai_msg = next(m for m in msgs if m["role"] == "assistant")
    assert ai_msg["content"] == "RAG 是检索增强生成技术。"


# ════════════════════════════════════════════════════════════════════════════
# [TC-16][Fix-6] main.py 集成：/agent/* 已注册，/health 版本与 app 一致
# ════════════════════════════════════════════════════════════════════════════

def test_main_app_agent_routes_registered():
    """
    [Fix-6] 验证 main.app 已注册 /agent/* 路由，且 /health 版本与 app 版本一致。
    """
    import main  # 导入实际的 main.py

    paths = {r.path for r in main.app.routes if hasattr(r, "path")}

    # 验证 4 条 agent 路由均已注册
    assert "/agent/chat" in paths, "/agent/chat 未注册"
    assert "/agent/sessions" in paths, "/agent/sessions 未注册"
    assert "/agent/sessions/{session_id}" in paths, "/agent/sessions/{session_id} 未注册"
    assert "/agent/sessions/{session_id}/history" in paths, (
        "/agent/sessions/{session_id}/history 未注册"
    )

    # 验证 /health 与 FastAPI app 版本一致
    client = TestClient(main.app, raise_server_exceptions=False)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["version"] == main.app.version, (
        f"/health 版本 {data['version']!r} 与 app.version {main.app.version!r} 不一致"
    )


# ════════════════════════════════════════════════════════════════════════════
# 入口（可直接 python tests/test_sse_api.py 运行，需在项目根目录执行）
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    test_fns = [
        test_agent_chat_normal_stream,
        test_agent_chat_empty_input_returns_400,
        test_agent_chat_auto_session_id,
        test_list_sessions,
        test_delete_session_existing,
        test_delete_session_nonexistent,
        test_get_session_history,
        test_agent_chat_event_order,
        test_tool_events_fields,
        test_agent_chat_error_event,
        test_invalid_session_id_path_traversal_returns_400,
        test_valid_session_id_formats_accepted,
        test_delete_invalid_session_id_returns_400,
        test_agent_init_failure_yields_sse_error,
        test_history_contains_human_and_assistant_after_turn,
        test_main_app_agent_routes_registered,
    ]

    passed = failed = 0
    for fn in test_fns:
        try:
            fn()
            print(f"  \u2705 {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  \u274c {fn.__name__}: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"结果：{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
