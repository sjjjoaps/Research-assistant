"""
测试 MasterAgent（P5-Step4 原生化版本）。

测试内容（Mock LLM，不需要真实 API Key 和外部服务）：

辅助函数单元测试：
  1. test_extract_sources               — [来源: xxx] 提取正确
  2. test_summarize_tool_result         — 工具结果摘要逻辑
  3. test_estimate_cost                 — 费用估算计算
  4. test_accumulate_tool_calls_string  — 字符串 args 拼接（真实流式场景）
  5. test_accumulate_tool_calls_dict    — dict args 覆盖（整块 dict 场景）
  6. test_accumulate_tool_calls_multi   — 多工具调用按 index 路由
  7. test_finalize_tool_calls           — 字符串 args 解析为 dict
  8. test_accumulate_tokens_native      — 原生 OpenAI chunk.usage 读取
  9. test_collect_new_messages_includes_user — new_messages 包含 user 消息

Agent 集成测试（Mock LLM）：
  10. test_simple_greeting              — 无工具调用，事件序列正确
  11. test_single_tool_call             — 一次工具调用，text_delta 仅在最终轮推送
  12. test_multi_tool_calls             — 两轮工具调用
  13. test_tool_failure_graceful        — 工具异常降级，Agent 不崩溃
  14. test_save_turn_includes_user      — 验证 save_turn() 收到 user 消息
  15. test_no_text_delta_in_tool_round  — 工具调用轮不推送 text_delta

运行方式（需在项目根目录）：
    F:/Anaconda/envs/llm_universe/python.exe tests/test_master_agent.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.events import (
    TextDeltaEvent,
    ToolEndEvent,
    ToolStartEvent,
    SourcesEvent,
)
from src.agents.master_agent import (
    MasterAgent,
    _accumulate_tool_calls,
    _accumulate_tokens,
    _collect_new_messages,
    _estimate_cost,
    _extract_sources,
    _finalize_tool_calls,
    _summarize_tool_result,
)


# ════════════════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════════════════

def make_session_id() -> str:
    return f"test_{uuid.uuid4().hex[:8]}"


def _make_chunk(content: str = "", tool_calls=None, usage=None):
    """构造模拟原生 OpenAI ChatCompletionChunk 对象。"""
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(delta=delta)
    chunk = SimpleNamespace(choices=[choice], usage=usage)
    return chunk


def _make_tool_call_chunk(index: int, id: str = "", name: str = "", arguments: str = ""):
    """构造模拟原生 ChoiceDeltaToolCall 对象（流式工具调用）。"""
    func = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, function=func)


def _build_agent_with_mocks(astream_side_effect, tools=None, history=None):
    """
    构造带 Mock LLM 的 MasterAgent，返回 (agent, mock_sm)。
    astream_side_effect: async generator function。
    """
    with (
        patch("src.agents.master_agent.get_native_llm") as mock_get_llm,
        patch("src.agents.master_agent.build_native_tool_registry") as mock_registry,
        patch("src.agents.master_agent.SessionManager") as mock_sm_cls,
        patch("src.agents.master_agent.load_master_agent_system_prompt"),
        patch("src.agents.master_agent.ToolCallLimiter") as mock_limiter_cls,
    ):
        mock_llm = MagicMock()
        mock_llm.astream = astream_side_effect
        mock_llm.model_name = "test-model"
        mock_get_llm.return_value = mock_llm

        mock_registry.return_value = tools or []

        mock_sm = MagicMock()
        mock_sm.load.return_value = history or []
        mock_sm_cls.return_value = mock_sm

        mock_limiter = MagicMock()
        mock_limiter.check.return_value = True
        mock_limiter_cls.return_value = mock_limiter

        agent = MasterAgent()
        agent._system_prompt = "你是 GraphAssistant。"
        agent._tool_schemas = []
        return agent, mock_sm


def _run(coro):
    return asyncio.run(coro)


# ════════════════════════════════════════════════════════════════════════════
# Case 1-3：基础辅助函数
# ════════════════════════════════════════════════════════════════════════════

def test_extract_sources():
    content = (
        "根据文献 [来源: paper1.pdf#chunk-3] 可知...\n"
        "另见 [来源: paper2.pdf#chunk-7]。\n"
        "普通文字不应被提取。"
    )
    sources = _extract_sources(content)
    assert "paper1.pdf#chunk-3" in sources
    assert "paper2.pdf#chunk-7" in sources
    assert len(sources) == 2
    print("[PASS] test_extract_sources")


def test_summarize_tool_result():
    assert _summarize_tool_result("找到 5 个相关片段") == "找到 5 个相关片段"
    assert _summarize_tool_result("**检索到 3 个结果：**\n\n[来源: x]") == "**检索到 3 个结果：**"
    assert _summarize_tool_result("") == "执行完成"
    assert _summarize_tool_result("   ") == "执行完成"
    long_line = "A" * 100
    result = _summarize_tool_result(long_line)
    assert len(result) == 81  # 80 字 + "…"
    assert result.endswith("…")
    print("[PASS] test_summarize_tool_result")


def test_estimate_cost():
    tokens = {"prompt": 1000, "completion": 500}
    cost = _estimate_cost(tokens)
    expected = 1000 / 1000 * 0.04 + 500 / 1000 * 0.12
    assert abs(cost - expected) < 1e-9, f"费用计算错误: {cost}"
    assert _estimate_cost({}) == 0.0
    print("[PASS] test_estimate_cost")


# ════════════════════════════════════════════════════════════════════════════
# Case 4-7：_accumulate_tool_calls / _finalize_tool_calls
# ════════════════════════════════════════════════════════════════════════════

def test_accumulate_tool_calls_string():
    """字符串 args 拼接——模拟真实原生流式场景。"""
    pending: list[dict] = []

    _accumulate_tool_calls(pending, [{"index": 0, "id": "tc_001", "name": "retrieve_knowledge", "args": ""}])
    assert len(pending) == 1
    assert pending[0]["name"] == "retrieve_knowledge"
    assert pending[0]["args"] == ""

    _accumulate_tool_calls(pending, [{"index": 0, "id": None, "name": None, "args": '{"query": "'}])
    assert pending[0]["args"] == '{"query": "'

    _accumulate_tool_calls(pending, [{"index": 0, "id": None, "name": None, "args": 'RAG 原理"}'}])
    assert pending[0]["args"] == '{"query": "RAG 原理"}'

    final = _finalize_tool_calls(pending)
    assert isinstance(final[0]["args"], dict)
    assert final[0]["args"] == {"query": "RAG 原理"}

    print("[PASS] test_accumulate_tool_calls_string")


def test_accumulate_tool_calls_dict():
    """dict args 覆盖——Mock 测试中常见的整块 dict 格式。"""
    pending: list[dict] = []
    _accumulate_tool_calls(pending, [{"index": 0, "id": "tc_002", "name": "list_documents", "args": {"keyword": "RAG"}}])
    assert len(pending) == 1
    final = _finalize_tool_calls(pending)
    assert final[0]["args"] == {"keyword": "RAG"}
    print("[PASS] test_accumulate_tool_calls_dict")


def test_accumulate_tool_calls_multi():
    """多工具调用：不同 index 分别累积，互不干扰。"""
    pending: list[dict] = []

    _accumulate_tool_calls(pending, [{"index": 0, "id": "tc_a", "name": "list_documents", "args": ""}])
    _accumulate_tool_calls(pending, [{"index": 1, "id": "tc_b", "name": "retrieve_knowledge", "args": ""}])
    _accumulate_tool_calls(pending, [{"index": 0, "args": '{"keyword": "RAG"}'}])
    _accumulate_tool_calls(pending, [{"index": 1, "args": '{"query": "RAG 方法"}'}])

    final = _finalize_tool_calls(pending)
    assert len(final) == 2
    assert final[0]["name"] == "list_documents"
    assert final[0]["args"] == {"keyword": "RAG"}
    assert final[1]["name"] == "retrieve_knowledge"
    assert final[1]["args"] == {"query": "RAG 方法"}
    print("[PASS] test_accumulate_tool_calls_multi")


def test_finalize_tool_calls():
    """字符串 args 解析：合法 JSON → dict；空字符串 → {}；损坏 JSON → {}（+WARNING）。"""
    cases = [
        ({"args": '{"q": "test"}'}, {"q": "test"}),
        ({"args": ""},               {}),
        ({"args": "{bad json}"},     {}),
        ({"args": {"q": "test"}},    {"q": "test"}),
    ]
    for raw, expected in cases:
        result = _finalize_tool_calls([{"id": "t", "name": "tool", **raw}])
        assert result[0]["args"] == expected, f"args={raw!r} 期望 {expected} 实际 {result[0]['args']}"
    print("[PASS] test_finalize_tool_calls")


# ════════════════════════════════════════════════════════════════════════════
# Case 8：_accumulate_tokens 原生 chunk.usage 读取
# ════════════════════════════════════════════════════════════════════════════

def test_accumulate_tokens_native():
    """验证原生 OpenAI chunk.usage 字段被正确读取。"""

    totals: dict = {}

    # 原生 OpenAI chunk with usage
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
    chunk = SimpleNamespace(choices=[], usage=usage)
    _accumulate_tokens(totals, chunk)
    assert totals["prompt"] == 100
    assert totals["completion"] == 50

    # 第二个 chunk（流式末尾）
    usage2 = SimpleNamespace(prompt_tokens=200, completion_tokens=80)
    chunk2 = SimpleNamespace(choices=[], usage=usage2)
    _accumulate_tokens(totals, chunk2)
    assert totals["prompt"] == 300
    assert totals["completion"] == 130

    # 无 usage 的中间 chunk：不影响已有统计
    chunk3 = SimpleNamespace(choices=[], usage=None)
    _accumulate_tokens(totals, chunk3)
    assert totals["prompt"] == 300

    print("[PASS] test_accumulate_tokens_native")


# ════════════════════════════════════════════════════════════════════════════
# Case 9：_collect_new_messages 包含 user 消息
# ════════════════════════════════════════════════════════════════════════════

def test_collect_new_messages_includes_user():
    """
    验证 base_length = 1 + history_length 时，
    user 消息被包含在 new_messages 中（不再丢失用户消息）。
    """
    history = [
        {"role": "user", "content": "旧问题"},
        {"role": "assistant", "content": "旧答案"},
    ]
    history_length = len(history)

    sys_msg  = {"role": "system",    "content": "系统提示"}
    new_user = {"role": "user",      "content": "新问题"}
    new_ai   = {"role": "assistant", "content": "新答案"}
    tool_msg = {"role": "tool",      "content": "工具结果", "tool_call_id": "tc_001"}

    base_length = 1 + history_length
    messages = [sys_msg, *history, new_user, new_ai, tool_msg]

    new_msgs = _collect_new_messages(messages, base_length)

    assert len(new_msgs) == 3, f"期望 3 条，实际 {len(new_msgs)}"
    assert new_msgs[0]["role"] == "user",      f"第 1 条应为 user，实际 {new_msgs[0]['role']}"
    assert new_msgs[0]["content"] == "新问题"
    assert new_msgs[1]["content"] == "新答案"
    assert new_msgs[2]["role"] == "tool"

    for m in new_msgs:
        assert m.get("role") != "system", "system 消息不应出现在 new_messages 中"

    print("[PASS] test_collect_new_messages_includes_user")


# ════════════════════════════════════════════════════════════════════════════
# Case 10：简单问候（无工具调用）
# ════════════════════════════════════════════════════════════════════════════

def test_simple_greeting():
    """LLM 直接回答，不调用任何工具，事件序列正确，文本完整。"""
    answer = "你好！有什么可以帮你的？"

    async def mock_astream(*_, **_kw):
        for char in answer:
            yield _make_chunk(content=char)

    agent, _ = _build_agent_with_mocks(mock_astream)

    async def run():
        events = []
        async for ev in agent.run_stream(make_session_id(), "你好"):
            events.append(ev)
        return events

    events = _run(run())
    types = [type(e).__name__ for e in events]

    assert "SessionStartEvent" in types
    assert "ThinkingEvent" in types
    assert "TextDeltaEvent" in types
    assert "UsageEvent" in types
    assert "DoneEvent" in types
    assert "ToolStartEvent" not in types, "不应出现 ToolStartEvent"
    assert "ToolEndEvent" not in types, "不应出现 ToolEndEvent"

    text = "".join(e.delta for e in events if isinstance(e, TextDeltaEvent))
    assert text == answer, f"回答文本不一致: {text!r}"
    print("[PASS] test_simple_greeting")


# ════════════════════════════════════════════════════════════════════════════
# Case 11：文献检索（一次工具调用）+ 工具调用轮无 text_delta
# ════════════════════════════════════════════════════════════════════════════

def test_single_tool_call():
    """
    LLM 调用一次工具后生成回答：
    - 工具调用轮（含 tool_calls）：不推送 TextDeltaEvent
    - 最终回答轮（无 tool_calls）：推送 TextDeltaEvent
    """
    tool_result = "**检索到 2 个相关片段：**\n\n**[1]** [来源: paper.pdf#chunk-1]\n内容摘要..."
    answer = "根据检索结果，答案是..."
    iteration_counter = {"n": 0}

    async def mock_astream(*_, **_kw):
        n = iteration_counter["n"]
        iteration_counter["n"] += 1
        if n == 0:
            yield _make_chunk(content="我来检索一下")
            tc = _make_tool_call_chunk(
                index=0, id="tc_001", name="retrieve_knowledge",
                arguments='{"query": "RAG 和 GraphRAG 的区别"}',
            )
            yield _make_chunk(tool_calls=[tc])
        else:
            for char in answer:
                yield _make_chunk(content=char)

    mock_tool = MagicMock()
    mock_tool.name = "retrieve_knowledge"
    mock_tool.invoke = MagicMock(return_value=tool_result)

    agent, _ = _build_agent_with_mocks(mock_astream, tools=[mock_tool])

    async def run():
        events = []
        async for ev in agent.run_stream(make_session_id(), "RAG 和 GraphRAG 的区别"):
            events.append(ev)
        return events

    events = _run(run())
    types = [type(e).__name__ for e in events]

    assert "ToolStartEvent" in types
    assert "ToolEndEvent" in types
    assert "TextDeltaEvent" in types
    assert "DoneEvent" in types
    assert any(isinstance(e, SourcesEvent) for e in events), "缺少 SourcesEvent（工具结果含来源引用）"

    text = "".join(e.delta for e in events if isinstance(e, TextDeltaEvent))
    assert text == answer, f"期望最终回答文本，实际: {text!r}"
    assert "我来检索一下" not in text, "工具调用轮的中间文本不应透传"

    print("[PASS] test_single_tool_call")


# ════════════════════════════════════════════════════════════════════════════
# Case 12：多工具顺序调用
# ════════════════════════════════════════════════════════════════════════════

def test_multi_tool_calls():
    """LLM 先调用 list_documents，再调用 retrieve_knowledge，最后生成回答。"""
    iteration_counter = {"n": 0}
    answer = "综合两次检索，结论如下..."

    async def mock_astream(*_, **_kw):
        n = iteration_counter["n"]
        iteration_counter["n"] += 1
        if n == 0:
            tc = _make_tool_call_chunk(0, "tc_001", "list_documents", '{"keyword": "RAG"}')
            yield _make_chunk(tool_calls=[tc])
        elif n == 1:
            tc = _make_tool_call_chunk(0, "tc_002", "retrieve_knowledge", '{"query": "RAG 方法"}')
            yield _make_chunk(tool_calls=[tc])
        else:
            for char in answer:
                yield _make_chunk(content=char)

    mock_list = MagicMock()
    mock_list.name = "list_documents"
    mock_list.invoke = MagicMock(return_value="找到 3 篇文献")

    mock_retrieve = MagicMock()
    mock_retrieve.name = "retrieve_knowledge"
    mock_retrieve.invoke = MagicMock(
        return_value="**检索到 2 个相关片段：**\n\n[来源: doc1.pdf#chunk-2]\n内容..."
    )

    agent, _ = _build_agent_with_mocks(mock_astream, tools=[mock_list, mock_retrieve])

    async def run():
        events = []
        async for ev in agent.run_stream(make_session_id(), "知识库有什么 RAG 相关文献？"):
            events.append(ev)
        return events

    events = _run(run())
    tool_starts = [e for e in events if isinstance(e, ToolStartEvent)]
    tool_ends   = [e for e in events if isinstance(e, ToolEndEvent)]

    assert len(tool_starts) == 2, f"期望 2 次工具调用，实际 {len(tool_starts)}"
    assert len(tool_ends)   == 2
    assert tool_starts[0].tool_name == "list_documents"
    assert tool_starts[1].tool_name == "retrieve_knowledge"

    text = "".join(e.delta for e in events if isinstance(e, TextDeltaEvent))
    assert text == answer
    print("[PASS] test_multi_tool_calls")


# ════════════════════════════════════════════════════════════════════════════
# Case 13：工具执行失败的降级处理
# ════════════════════════════════════════════════════════════════════════════

def test_tool_failure_graceful():
    """工具抛异常时，_execute_tool_safe 返回结构化错误，Agent 不崩溃继续完成。"""
    iteration_counter = {"n": 0}
    fallback_answer = "很抱歉，检索失败，我将基于已有知识回答..."

    async def mock_astream(*_, **_kw):
        n = iteration_counter["n"]
        iteration_counter["n"] += 1
        if n == 0:
            tc = _make_tool_call_chunk(0, "tc_err", "retrieve_knowledge", '{"query": "某查询"}')
            yield _make_chunk(tool_calls=[tc])
        else:
            for char in fallback_answer:
                yield _make_chunk(content=char)

    mock_tool = MagicMock()
    mock_tool.name = "retrieve_knowledge"
    mock_tool.invoke = MagicMock(side_effect=RuntimeError("FAISS 索引未初始化"))

    agent, _ = _build_agent_with_mocks(mock_astream, tools=[mock_tool])

    async def run():
        events = []
        async for ev in agent.run_stream(make_session_id(), "检索某内容"):
            events.append(ev)
        return events

    events = _run(run())
    types = [type(e).__name__ for e in events]

    assert "DoneEvent" in types, "工具失败后仍应到达 DoneEvent"
    assert "ToolEndEvent" in types, "工具失败仍应有 ToolEndEvent"

    tool_end = next(e for e in events if isinstance(e, ToolEndEvent))
    assert "工具调用失败" in tool_end.result_summary or len(tool_end.result_summary) > 0

    text = "".join(e.delta for e in events if isinstance(e, TextDeltaEvent))
    assert len(text) > 0, "工具失败后 LLM 仍应生成回答"
    print("[PASS] test_tool_failure_graceful")


# ════════════════════════════════════════════════════════════════════════════
# Case 14：验证 save_turn() 收到的 new_messages 含 user 消息
# ════════════════════════════════════════════════════════════════════════════

def test_save_turn_includes_user():
    """
    核心集成验证：
    save_turn() 的 new_messages 参数必须包含本轮 user 消息，
    否则跨会话恢复时历史会缺失用户提问。
    """
    answer = "这是回答"

    async def mock_astream(*_, **_kw):
        for char in answer:
            yield _make_chunk(content=char)

    agent, mock_sm = _build_agent_with_mocks(mock_astream)

    async def run():
        events = []
        async for ev in agent.run_stream("sid_fix1", "用户的问题"):
            events.append(ev)
        return events

    _run(run())

    assert mock_sm.save_turn.called, "save_turn() 未被调用"

    call_kwargs = mock_sm.save_turn.call_args[1]
    new_messages = call_kwargs.get("new_messages", [])

    user_msgs = [m for m in new_messages if isinstance(m, dict) and m.get("role") == "user"]
    assert len(user_msgs) >= 1, (
        f"new_messages 中缺少 user 消息！\n"
        f"实际 new_messages: {[m.get('role') for m in new_messages]}"
    )
    assert user_msgs[0]["content"] == "用户的问题", (
        f"user 消息内容错误: {user_msgs[0]['content']!r}"
    )

    ai_msgs = [m for m in new_messages if isinstance(m, dict) and m.get("role") == "assistant"]
    assert len(ai_msgs) >= 1, "new_messages 中缺少 assistant 消息"

    sys_msgs = [m for m in new_messages if isinstance(m, dict) and m.get("role") == "system"]
    assert len(sys_msgs) == 0, "new_messages 中不应包含 system 消息"

    print("[PASS] test_save_turn_includes_user")


# ════════════════════════════════════════════════════════════════════════════
# Case 15：工具调用轮不推送 text_delta
# ════════════════════════════════════════════════════════════════════════════

def test_no_text_delta_in_tool_round():
    """
    含 tool_calls 的轮次即使有文本输出也不推送给前端。
    TextDeltaEvent 仅出现在最终回答轮（无 tool_calls 的轮次）。
    """
    iteration_counter = {"n": 0}
    intermediate_text = "让我先检索一下相关内容，请稍候"
    final_text = "基于检索结果，回答如下"

    async def mock_astream(*_, **_kw):
        n = iteration_counter["n"]
        iteration_counter["n"] += 1
        if n == 0:
            yield _make_chunk(content=intermediate_text)
            tc = _make_tool_call_chunk(0, "tc_x", "list_documents", "{}")
            yield _make_chunk(tool_calls=[tc])
        else:
            yield _make_chunk(content=final_text)

    mock_tool = MagicMock()
    mock_tool.name = "list_documents"
    mock_tool.invoke = MagicMock(return_value="找到 2 篇文献")

    agent, _ = _build_agent_with_mocks(mock_astream, tools=[mock_tool])

    async def run():
        events = []
        async for ev in agent.run_stream(make_session_id(), "有哪些文献？"):
            events.append(ev)
        return events

    events = _run(run())
    text = "".join(e.delta for e in events if isinstance(e, TextDeltaEvent))

    assert intermediate_text not in text, (
        f"工具调用轮的中间文本不应透传给前端: {text!r}"
    )
    assert final_text in text, f"最终回答文本应出现: {text!r}"
    print("[PASS] test_no_text_delta_in_tool_round")


# ════════════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("P5-Step4  MasterAgent 测试（原生化版本）")
    print("=" * 65)

    tests = [
        test_extract_sources,
        test_summarize_tool_result,
        test_estimate_cost,
        test_accumulate_tool_calls_string,
        test_accumulate_tool_calls_dict,
        test_accumulate_tool_calls_multi,
        test_finalize_tool_calls,
        test_accumulate_tokens_native,
        test_collect_new_messages_includes_user,
        test_simple_greeting,
        test_single_tool_call,
        test_multi_tool_calls,
        test_tool_failure_graceful,
        test_save_turn_includes_user,
        test_no_text_delta_in_tool_round,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 65)
    print(f"结果：{passed} 通过，{failed} 失败")
    print("=" * 65)
