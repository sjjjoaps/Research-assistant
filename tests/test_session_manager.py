"""
测试 SessionManager（Phase 8-2）

测试内容：
  1. 新建会话 → 写入 → 读取 → 验证消息一致
  2. 进程重启恢复（重新实例化 SessionManager 模拟）
  3. 多轮写入 → 消息顺序与数量正确
  4. delete_session → 文件被删除，list_sessions 不再出现
  5. list_sessions → 返回正确条数，按更新时间倒序
  6. 消息类型完整性：HumanMessage / AIMessage / ToolMessage / AIMessage+tool_calls
  7. 压缩游标（covers_turn_ids 集合判定）：load() 仅返回 SystemMessage + 新轮消息
  8. 多次压缩 → 所有 summary 按时序合并，旧轮全部跳过
  9. token 累计正确
  10. turn_id 纳秒精度，高频写入无碰撞
"""
import sys
import os
import uuid
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agents.session_manager import SessionManager, _serialize_message, _deserialize_message


# ── 测试工具 ──────────────────────────────────────────────────────────────────

def fresh_session_id() -> str:
    return f"test_{uuid.uuid4().hex[:8]}"


def make_turn_messages(text: str = "Hello"):
    """构造一组典型的 turn 消息（Human + AI）"""
    return [
        HumanMessage(content=text),
        AIMessage(content=f"回答：{text}"),
    ]


def make_tool_turn_messages():
    """构造包含 tool_calls 和 ToolMessage 的 turn 消息"""
    return [
        AIMessage(
            content="",
            tool_calls=[{"id": "tc_001", "name": "retrieve_knowledge", "args": {"query": "test"}}],
        ),
        ToolMessage(content="检索到 3 个结果", tool_call_id="tc_001"),
        AIMessage(content="根据检索结果，答案是..."),
    ]


# ── Case 1：基础写入与读取 ─────────────────────────────────────────────────────

def test_basic_save_and_load():
    sm = SessionManager()
    sid = fresh_session_id()

    msgs = make_turn_messages("RAG 是什么？")
    sm.save_turn(
        session_id=sid,
        user_input="RAG 是什么？",
        new_messages=msgs,
        sources=["paper.pdf#chunk-1"],
        token_usage={"prompt": 100, "completion": 50, "total": 150},
    )

    loaded = sm.load(sid)
    assert len(loaded) == 2, f"期望 2 条消息，实际 {len(loaded)}"
    assert loaded[0].content == "RAG 是什么？"
    assert loaded[1].content == "回答：RAG 是什么？"

    sm.delete_session(sid)
    print("[PASS] test_basic_save_and_load")


# ── Case 2：进程重启后恢复 ─────────────────────────────────────────────────────

def test_restart_recovery():
    sm1 = SessionManager()
    sid = fresh_session_id()

    msgs = make_turn_messages("GraphRAG 原理")
    sm1.save_turn(
        session_id=sid,
        user_input="GraphRAG 原理",
        new_messages=msgs,
        sources=[],
        token_usage={"prompt": 200, "completion": 80, "total": 280},
    )

    # 模拟进程重启：新建 SessionManager 实例
    sm2 = SessionManager()
    loaded = sm2.load(sid)

    assert len(loaded) == 2, f"期望 2 条消息，实际 {len(loaded)}"
    assert loaded[0].content == "GraphRAG 原理"

    sm2.delete_session(sid)
    print("[PASS] test_restart_recovery")


# ── Case 3：多轮写入消息顺序正确 ───────────────────────────────────────────────

def test_multi_turn_order():
    sm = SessionManager()
    sid = fresh_session_id()

    # 3 轮对话
    for i in range(1, 4):
        sm.save_turn(
            session_id=sid,
            user_input=f"问题 {i}",
            new_messages=[
                HumanMessage(content=f"问题 {i}"),
                AIMessage(content=f"答案 {i}"),
            ],
            sources=[],
            token_usage={"prompt": 50, "completion": 20, "total": 70},
        )

    loaded = sm.load(sid)
    # 3 轮 × 2 条消息 = 6 条
    assert len(loaded) == 6, f"期望 6 条消息，实际 {len(loaded)}"
    # 顺序检查
    assert loaded[0].content == "问题 1"
    assert loaded[2].content == "问题 2"
    assert loaded[4].content == "问题 3"

    # meta 中 turn_count 应为 3
    meta = sm.get_session_meta(sid)
    assert meta["turn_count"] == 3, f"期望 turn_count=3，实际 {meta['turn_count']}"

    sm.delete_session(sid)
    print("[PASS] test_multi_turn_order")


# ── Case 4：delete_session 清理文件 ─────────────────────────────────────────────

def test_delete_session():
    sm = SessionManager()
    sid = fresh_session_id()

    sm.save_turn(sid, "测试删除", make_turn_messages("测试删除"), [], {"prompt": 10, "completion": 5, "total": 15})

    # 删除前应存在
    assert sm._jsonl_path(sid).exists()
    assert sm._meta_path(sid).exists()

    result = sm.delete_session(sid)
    assert result is True

    # 删除后不存在
    assert not sm._jsonl_path(sid).exists()
    assert not sm._meta_path(sid).exists()

    # 再次删除应返回 False
    assert sm.delete_session(sid) is False

    print("[PASS] test_delete_session")


# ── Case 5：list_sessions 返回正确条数并倒序 ──────────────────────────────────

def test_list_sessions():
    sm = SessionManager()
    sids = [fresh_session_id() for _ in range(3)]

    for i, sid in enumerate(sids):
        sm.save_turn(
            session_id=sid,
            user_input=f"会话 {i}",
            new_messages=make_turn_messages(f"会话 {i}"),
            sources=[],
            token_usage={"prompt": 30, "completion": 10, "total": 40},
        )
        time.sleep(0.05)  # 确保 updated_at 不同

    sessions = sm.list_sessions()
    # 新建的 3 个 session 都应在列表中
    listed_ids = {s["session_id"] for s in sessions}
    for sid in sids:
        assert sid in listed_ids, f"{sid} 不在 list_sessions 结果中"

    # 按 updated_at 倒序：最新的在最前
    my_sessions = [s for s in sessions if s["session_id"] in sids]
    dates = [s["updated_at"] for s in my_sessions]
    assert dates == sorted(dates, reverse=True), "list_sessions 未按 updated_at 倒序"

    # 清理
    for sid in sids:
        sm.delete_session(sid)

    print("[PASS] test_list_sessions")


# ── Case 6：工具调用消息序列化 / 反序列化 ────────────────────────────────────

def test_tool_message_roundtrip():
    sm = SessionManager()
    sid = fresh_session_id()

    tool_msgs = make_tool_turn_messages()
    sm.save_turn(
        session_id=sid,
        user_input="检索相关文献",
        new_messages=tool_msgs,
        sources=["paper.pdf#chunk-2"],
        token_usage={"prompt": 300, "completion": 100, "total": 400},
    )

    loaded = sm.load(sid)
    assert len(loaded) == 3, f"期望 3 条消息，实际 {len(loaded)}"

    ai_with_tool = loaded[0]
    assert isinstance(ai_with_tool, AIMessage)
    assert len(ai_with_tool.tool_calls) == 1
    assert ai_with_tool.tool_calls[0]["name"] == "retrieve_knowledge"

    tool_result = loaded[1]
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.tool_call_id == "tc_001"
    assert tool_result.content == "检索到 3 个结果"

    sm.delete_session(sid)
    print("[PASS] test_tool_message_roundtrip")


# ── Case 7：covers_turn_ids 集合判定 —— summary 注入与旧轮跳过 ────────────────

def test_compact_cursor_set_based():
    """
    验证 load() 通过 covers_turn_ids 集合跳过旧轮（不依赖字符串顺序比较）：
    - 手动写入 summary 记录并指定 covers_turn_ids
    - load() 应跳过被覆盖的旧轮，返回 1 条 SystemMessage + 新轮消息
    """
    sm = SessionManager()
    sid = fresh_session_id()

    # 第 1 轮（将被压缩）
    sm.save_turn(
        session_id=sid,
        user_input="旧问题",
        new_messages=make_turn_messages("旧问题"),
        sources=[],
        token_usage={"prompt": 100, "completion": 50, "total": 150},
    )

    # 读出旧轮 turn_id
    records = sm._read_all_records(sid)
    old_turn_id = records[0]["turn_id"]

    # 手动追加 summary 记录（模拟 _compact_if_needed 行为）
    sm._append_jsonl(sid, {
        "type":            "summary",
        "covers_turn_ids": [old_turn_id],
        "summary_text":    "用户询问了旧问题，助手作了简短回答。",
        "compressed_at":   "2026-01-01T00:00:00Z",
    })

    # 第 2 轮（新轮）
    sm.save_turn(
        session_id=sid,
        user_input="新问题",
        new_messages=make_turn_messages("新问题"),
        sources=[],
        token_usage={"prompt": 100, "completion": 50, "total": 150},
    )

    loaded = sm.load(sid)
    # 期望：1 条 SystemMessage（summary）+ 新轮 2 条 = 3 条；旧轮 2 条被跳过
    assert len(loaded) == 3, f"期望 3 条消息，实际 {len(loaded)}"

    # [4] 第 1 条是 SystemMessage（不是伪造的 HumanMessage）
    assert isinstance(loaded[0], SystemMessage), f"期望 SystemMessage，实际 {type(loaded[0])}"
    assert "历史摘要" in loaded[0].content
    assert "旧问题" in loaded[0].content

    # 第 2 条是新轮的 HumanMessage
    assert isinstance(loaded[1], HumanMessage)
    assert loaded[1].content == "新问题"

    sm.delete_session(sid)
    print("[PASS] test_compact_cursor_set_based")


# ── Case 8：多次压缩 → 所有 summary 按时序合并 ────────────────────────────────

def test_multiple_compactions():
    """
    验证两次压缩后 load() 保留两条 summary 的内容（不静默丢失）：
    - 第 1 次压缩：覆盖 turn A
    - 第 2 次压缩：覆盖 turn B
    - 新轮 C 未被压缩
    - load() 应返回 1 条 SystemMessage（含两段摘要）+ 新轮 C 消息
    """
    sm = SessionManager()
    sid = fresh_session_id()

    # 写入 3 轮对话
    for label in ("A", "B", "C"):
        sm.save_turn(
            session_id=sid,
            user_input=f"问题 {label}",
            new_messages=make_turn_messages(f"问题 {label}"),
            sources=[],
            token_usage={"prompt": 50, "completion": 20, "total": 70},
        )

    all_recs = sm._read_all_records(sid)
    msg_recs = [r for r in all_recs if r.get("type") == "messages"]
    assert len(msg_recs) == 3

    turn_a = msg_recs[0]["turn_id"]
    turn_b = msg_recs[1]["turn_id"]

    # 第 1 次压缩：覆盖 turn A
    sm._append_jsonl(sid, {
        "type":            "summary",
        "covers_turn_ids": [turn_a],
        "summary_text":    "第一段摘要：用户问了问题 A。",
        "compressed_at":   "2026-01-01T10:00:00Z",
    })

    # 第 2 次压缩：覆盖 turn B
    sm._append_jsonl(sid, {
        "type":            "summary",
        "covers_turn_ids": [turn_b],
        "summary_text":    "第二段摘要：用户问了问题 B。",
        "compressed_at":   "2026-01-01T11:00:00Z",
    })

    loaded = sm.load(sid)
    # 期望：1 条 SystemMessage（2 段摘要合并）+ 轮 C 的 2 条消息 = 3 条
    assert len(loaded) == 3, f"期望 3 条消息，实际 {len(loaded)}"

    assert isinstance(loaded[0], SystemMessage), f"期望 SystemMessage，实际 {type(loaded[0])}"
    # 两段摘要都应出现在 SystemMessage 中
    assert "第一段摘要" in loaded[0].content, "第一段摘要丢失"
    assert "第二段摘要" in loaded[0].content, "第二段摘要丢失"

    # 轮 C 的消息正常出现
    assert isinstance(loaded[1], HumanMessage)
    assert loaded[1].content == "问题 C"

    sm.delete_session(sid)
    print("[PASS] test_multiple_compactions")


# ── Case 9：token 累计正确 ────────────────────────────────────────────────────

def test_token_accumulation():
    sm = SessionManager()
    sid = fresh_session_id()

    for _ in range(3):
        sm.save_turn(
            session_id=sid,
            user_input="x",
            new_messages=make_turn_messages("x"),
            sources=[],
            token_usage={"prompt": 100, "completion": 50, "total": 150},
        )

    meta = sm.get_session_meta(sid)
    assert meta["total_tokens"] == 450, f"期望 450 tokens，实际 {meta['total_tokens']}"

    sm.delete_session(sid)
    print("[PASS] test_token_accumulation")


# ── Case 10：turn_id 纳秒精度，高频写入无碰撞 ─────────────────────────────────

def test_turn_id_uniqueness():
    """验证高频写入时 turn_id 不碰撞。"""
    sm = SessionManager()
    sid = fresh_session_id()

    # 连续写 20 轮，不加 sleep
    for i in range(20):
        sm.save_turn(
            session_id=sid,
            user_input=f"q{i}",
            new_messages=make_turn_messages(f"q{i}"),
            sources=[],
            token_usage={"prompt": 10, "completion": 5, "total": 15},
        )

    records = sm._read_all_records(sid)
    msg_records = [r for r in records if r.get("type") == "messages"]
    turn_ids = [r["turn_id"] for r in msg_records]

    assert len(turn_ids) == 20, f"期望 20 条，实际 {len(turn_ids)}"
    assert len(set(turn_ids)) == 20, f"turn_id 有碰撞：{len(set(turn_ids))} 个唯一值"

    sm.delete_session(sid)
    print("[PASS] test_turn_id_uniqueness")


# ── 序列化单元测试 ────────────────────────────────────────────────────────────

def test_serialize_deserialize():
    msgs = [
        HumanMessage(content="你好"),
        AIMessage(content="你好！"),
        AIMessage(
            content="",
            tool_calls=[{"id": "t1", "name": "list_documents", "args": {}}],
        ),
        ToolMessage(content="找到 5 篇文献", tool_call_id="t1"),
    ]
    for msg in msgs:
        data   = _serialize_message(msg)
        result = _deserialize_message(data)
        assert type(result) == type(msg), f"类型不一致: {type(result)} vs {type(msg)}"
        assert result.content == msg.content, f"内容不一致: {result.content!r} vs {msg.content!r}"

    print("[PASS] test_serialize_deserialize")


# ── 主入口 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 8-2  SessionManager 测试（Review 修复版）")
    print("=" * 60)

    tests = [
        test_serialize_deserialize,
        test_basic_save_and_load,
        test_restart_recovery,
        test_multi_turn_order,
        test_delete_session,
        test_list_sessions,
        test_tool_message_roundtrip,
        test_compact_cursor_set_based,
        test_multiple_compactions,
        test_token_accumulation,
        test_turn_id_uniqueness,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"结果：{passed} 通过，{failed} 失败")
    print("=" * 60)
