"""
测试 P5-Step 2：原生 LLM 客户端（LLMClient / get_native_llm）

测试策略：
  - Mock AsyncOpenAI，不发起真实网络请求
  - 验证 ainvoke 返回格式（role / content / tool_calls）
  - 验证 astream chunk 迭代行为
  - 验证工具参数正确透传给 AsyncOpenAI
  - 验证 lru_cache 单例语义
  - 验证 api_key 未配置时抛出 ValueError
  - 验证 get_llm()（旧路径）仍可正常导入，不受影响

覆盖用例：
  1. ainvoke — 返回 dict，含 role/content/tool_calls
  2. ainvoke — 传入 tools 参数时正确透传
  3. ainvoke — content 为 None（纯工具调用场景）
  4. astream — 能迭代 yield chunk
  5. astream — 传入 tools 时正确透传
  6. model_name 属性返回 settings.model_name
  7. get_native_llm() — 相同 temperature 返回同一实例（lru_cache）
  8. get_native_llm() — 不同 temperature 返回不同实例
  9. get_native_llm() — api_key 为空时抛出 ValueError
 10. get_llm()（旧路径）— 仍可正常导入调用，返回 ChatOpenAI
"""
from __future__ import annotations

import sys
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── 辅助：构造 mock ChatCompletionChunk ───────────────────────────────────

def _make_chunk(content: str | None = None, tool_calls=None, finish_reason: str | None = None):
    """构造最小的 mock ChatCompletionChunk，结构与 openai 返回一致。"""
    delta = MagicMock()
    delta.content    = content
    delta.tool_calls = tool_calls

    choice = MagicMock()
    choice.delta         = delta
    choice.finish_reason = finish_reason

    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


def _make_completion_response(role="assistant", content="hello", tool_calls=None):
    """构造 mock non-stream ChatCompletion response。"""
    msg = MagicMock()
    msg.model_dump.return_value = {
        "role": role,
        "content": content,
        "tool_calls": tool_calls,
    }

    choice = MagicMock()
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _run(coro):
    """在测试中同步执行协程。"""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── mock AsyncOpenAI 工厂 ─────────────────────────────────────────────────

def _patch_async_openai(ainvoke_resp=None, stream_chunks=None):
    """
    返回 patch("openai.AsyncOpenAI") 的上下文管理器。
    - ainvoke_resp: non-stream create() 的返回值
    - stream_chunks: stream() 迭代的 chunk 列表
    """
    mock_client    = MagicMock()
    mock_completions = MagicMock()
    mock_client.chat.completions = mock_completions

    # non-stream create()
    if ainvoke_resp is not None:
        mock_completions.create = AsyncMock(return_value=ainvoke_resp)

    # stream context manager
    if stream_chunks is not None:
        async def _aiter():
            for c in stream_chunks:
                yield c

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=_aiter())
        mock_stream_ctx.__aexit__  = AsyncMock(return_value=False)
        mock_completions.stream = MagicMock(return_value=mock_stream_ctx)

    return patch("openai.AsyncOpenAI", return_value=mock_client)


# ════════════════════════════════════════════════════════════════════════════
# Case 1：ainvoke 返回格式
# ════════════════════════════════════════════════════════════════════════════

def test_ainvoke_returns_dict():
    """ainvoke 应返回含 role/content/tool_calls 的 dict。"""
    resp = _make_completion_response(role="assistant", content="你好", tool_calls=None)

    with patch("src.infrastructure.config.settings") as mock_settings, \
         _patch_async_openai(ainvoke_resp=resp):
        mock_settings.api_key             = "test-key"
        mock_settings.model_name          = "test-model"
        mock_settings.base_url            = "http://test"
        mock_settings.llm_timeout_seconds = 30
        mock_settings.llm_max_retries     = 3

        from src.infrastructure.llm_client import LLMClient
        client = LLMClient.__new__(LLMClient)
        client.model       = "test-model"
        client.temperature = 0.0

        from openai import AsyncOpenAI
        client._client = AsyncOpenAI()

        result = _run(client.ainvoke([{"role": "user", "content": "你好"}]))

    assert isinstance(result, dict),        f"应返回 dict，实际: {type(result)}"
    assert result["role"] == "assistant",   f"role 不正确: {result}"
    assert result["content"] == "你好",     f"content 不正确: {result}"
    assert result["tool_calls"] is None,    f"tool_calls 应为 None: {result}"
    print("[PASS] test_ainvoke_returns_dict")


# ════════════════════════════════════════════════════════════════════════════
# Case 2：ainvoke 透传 tools 参数
# ════════════════════════════════════════════════════════════════════════════

def test_ainvoke_passes_tools():
    """ainvoke 传入 tools 时，应将 tools 透传给 AsyncOpenAI.create()。"""
    resp = _make_completion_response()
    fake_tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

    mock_create = AsyncMock(return_value=resp)

    with patch("src.infrastructure.config.settings") as mock_settings:
        mock_settings.api_key             = "test-key"
        mock_settings.model_name          = "test-model"
        mock_settings.base_url            = "http://test"
        mock_settings.llm_timeout_seconds = 30
        mock_settings.llm_max_retries     = 3

        from src.infrastructure.llm_client import LLMClient
        client = LLMClient.__new__(LLMClient)
        client.model       = "test-model"
        client.temperature = 0.0

        mock_client_obj = MagicMock()
        mock_client_obj.chat.completions.create = mock_create
        client._client = mock_client_obj

        _run(client.ainvoke([{"role": "user", "content": "test"}], tools=fake_tools))

    call_kwargs = mock_create.call_args.kwargs
    assert "tools" in call_kwargs,               "tools 未传给 create()"
    assert call_kwargs["tools"] == fake_tools,   f"tools 值不匹配: {call_kwargs['tools']}"
    print("[PASS] test_ainvoke_passes_tools")


# ════════════════════════════════════════════════════════════════════════════
# Case 3：ainvoke content 为 None（纯工具调用场景）
# ════════════════════════════════════════════════════════════════════════════

def test_ainvoke_content_none_with_tool_calls():
    """纯工具调用时 content 为 None，tool_calls 应有值。"""
    fake_tool_calls = [{"id": "call_1", "type": "function",
                        "function": {"name": "search", "arguments": "{}"}}]
    resp = _make_completion_response(content=None, tool_calls=fake_tool_calls)

    mock_create = AsyncMock(return_value=resp)

    from src.infrastructure.llm_client import LLMClient
    client = LLMClient.__new__(LLMClient)
    client.model       = "test-model"
    client.temperature = 0.0

    mock_client_obj = MagicMock()
    mock_client_obj.chat.completions.create = mock_create
    client._client = mock_client_obj

    result = _run(client.ainvoke([{"role": "user", "content": "call tool"}]))

    assert result["content"] is None,              f"content 应为 None: {result}"
    assert result["tool_calls"] == fake_tool_calls, f"tool_calls 不匹配: {result}"
    print("[PASS] test_ainvoke_content_none_with_tool_calls")


# ════════════════════════════════════════════════════════════════════════════
# Case 4：astream 能迭代 yield chunk
# ════════════════════════════════════════════════════════════════════════════

def test_astream_yields_chunks():
    """astream 应依次 yield 传入的 chunk，且数量正确。"""
    chunks = [
        _make_chunk(content="Hello"),
        _make_chunk(content=" world"),
        _make_chunk(finish_reason="stop"),
    ]

    async def _aiter():
        for c in chunks:
            yield c

    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=_aiter())
    mock_stream_ctx.__aexit__  = AsyncMock(return_value=False)

    from src.infrastructure.llm_client import LLMClient
    client = LLMClient.__new__(LLMClient)
    client.model       = "test-model"
    client.temperature = 0.0

    mock_client_obj = MagicMock()
    mock_client_obj.chat.completions.stream = MagicMock(return_value=mock_stream_ctx)
    client._client = mock_client_obj

    async def _collect():
        collected = []
        async for chunk in client.astream([{"role": "user", "content": "hi"}]):
            collected.append(chunk)
        return collected

    result = _run(_collect())
    assert len(result) == 3, f"期望 3 个 chunk，实际 {len(result)}"
    assert result[0].choices[0].delta.content == "Hello"
    assert result[1].choices[0].delta.content == " world"
    print("[PASS] test_astream_yields_chunks")


# ════════════════════════════════════════════════════════════════════════════
# Case 5：astream 透传 tools 参数
# ════════════════════════════════════════════════════════════════════════════

def test_astream_passes_tools():
    """astream 传入 tools 时，应将 tools 透传给 stream()。"""
    fake_tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

    async def _aiter():
        return
        yield  # make it an async generator

    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=_aiter())
    mock_stream_ctx.__aexit__  = AsyncMock(return_value=False)

    mock_stream_fn = MagicMock(return_value=mock_stream_ctx)

    from src.infrastructure.llm_client import LLMClient
    client = LLMClient.__new__(LLMClient)
    client.model       = "test-model"
    client.temperature = 0.0

    mock_client_obj = MagicMock()
    mock_client_obj.chat.completions.stream = mock_stream_fn
    client._client = mock_client_obj

    async def _consume():
        async for _ in client.astream([{"role": "user", "content": "hi"}], tools=fake_tools):
            pass

    _run(_consume())

    call_kwargs = mock_stream_fn.call_args.kwargs
    assert "tools" in call_kwargs,              "tools 未传给 stream()"
    assert call_kwargs["tools"] == fake_tools,  f"tools 值不匹配: {call_kwargs['tools']}"
    print("[PASS] test_astream_passes_tools")


# ════════════════════════════════════════════════════════════════════════════
# Case 6：model_name 属性
# ════════════════════════════════════════════════════════════════════════════

def test_model_name_property():
    """model_name 属性应返回构造时传入的 model。"""
    from src.infrastructure.llm_client import LLMClient
    client = LLMClient.__new__(LLMClient)
    client.model       = "qwen-max"
    client.temperature = 0.0

    assert client.model_name == "qwen-max"
    print("[PASS] test_model_name_property")


# ════════════════════════════════════════════════════════════════════════════
# Case 7：get_native_llm lru_cache — 相同 temperature 返回同一实例
# ════════════════════════════════════════════════════════════════════════════

def test_get_native_llm_singleton():
    """相同 temperature 的两次调用应返回同一 LLMClient 实例。"""
    with patch("src.infrastructure.llm_client.settings") as mock_settings, \
         patch("openai.AsyncOpenAI"):
        mock_settings.api_key             = "test-key"
        mock_settings.model_name          = "test-model"
        mock_settings.base_url            = "http://test"
        mock_settings.llm_timeout_seconds = 30
        mock_settings.llm_max_retries     = 3

        from src.infrastructure.llm_client import get_native_llm
        get_native_llm.cache_clear()

        a = get_native_llm(temperature=0.0)
        b = get_native_llm(temperature=0.0)
        assert a is b, "相同 temperature 应返回同一实例（lru_cache 失效）"

    get_native_llm.cache_clear()
    print("[PASS] test_get_native_llm_singleton")


# ════════════════════════════════════════════════════════════════════════════
# Case 8：get_native_llm — 不同 temperature 返回不同实例
# ════════════════════════════════════════════════════════════════════════════

def test_get_native_llm_different_temperature():
    """不同 temperature 应返回不同的 LLMClient 实例。"""
    with patch("src.infrastructure.llm_client.settings") as mock_settings, \
         patch("openai.AsyncOpenAI"):
        mock_settings.api_key             = "test-key"
        mock_settings.model_name          = "test-model"
        mock_settings.base_url            = "http://test"
        mock_settings.llm_timeout_seconds = 30
        mock_settings.llm_max_retries     = 3

        from src.infrastructure.llm_client import get_native_llm
        get_native_llm.cache_clear()

        a = get_native_llm(temperature=0.0)
        b = get_native_llm(temperature=0.7)
        assert a is not b, "不同 temperature 应返回不同实例"

    get_native_llm.cache_clear()
    print("[PASS] test_get_native_llm_different_temperature")


# ════════════════════════════════════════════════════════════════════════════
# Case 9：api_key 为空时抛出 ValueError
# ════════════════════════════════════════════════════════════════════════════

def test_get_native_llm_no_api_key():
    """api_key 未配置时，get_native_llm() 应抛出 ValueError。"""
    with patch("src.infrastructure.llm_client.settings") as mock_settings:
        mock_settings.api_key = ""

        from src.infrastructure.llm_client import get_native_llm
        get_native_llm.cache_clear()

        try:
            get_native_llm()
            assert False, "应抛出 ValueError"
        except ValueError as e:
            assert "API_KEY" in str(e), f"ValueError 消息不含 API_KEY: {e}"

    get_native_llm.cache_clear()
    print("[PASS] test_get_native_llm_no_api_key")


# ════════════════════════════════════════════════════════════════════════════
# Case 10：get_llm()（旧路径）仍可正常导入
# ════════════════════════════════════════════════════════════════════════════

def test_get_llm_still_importable():
    """旧 get_llm() 应仍可正常导入，不受 P5-Step 2 改动影响。"""
    from src.infrastructure.llm_client import get_llm
    assert callable(get_llm), "get_llm 应为可调用函数"
    print("[PASS] test_get_llm_still_importable")


# ── 主入口 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("P5-Step 2  原生 LLMClient 测试")
    print("=" * 60)

    tests = [
        test_ainvoke_returns_dict,
        test_ainvoke_passes_tools,
        test_ainvoke_content_none_with_tool_calls,
        test_astream_yields_chunks,
        test_astream_passes_tools,
        test_model_name_property,
        test_get_native_llm_singleton,
        test_get_native_llm_different_temperature,
        test_get_native_llm_no_api_key,
        test_get_llm_still_importable,
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

    print("=" * 60)
    print(f"结果：{passed} 通过，{failed} 失败")
    print("=" * 60)
