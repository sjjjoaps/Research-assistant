"""
MasterAgent：GraphAssistant 的核心 Agent（Phase 8-5，Review 修订版）。

架构参考 claw-code src/runtime.py 的 run_turn_loop() 的 for-break 安全阀结构：
- for-break 循环（MAX_ITERATIONS=10）：终止条件由 LangChain tool_calls 是否为空决定
- 工具执行结果回注 messages，供 LLM 下一轮决策
- 每轮结束后通过 SessionManager 持久化到 JSONL

Review 修订要点（v2）：
  [Fix-1] base_length 修正：HumanMessage 纳入 new_messages 持久化，
          避免跨会话恢复时用户消息全部丢失。
  [Fix-2] 工具调用累积重构：拆分为 _accumulate_tool_calls / _finalize_tool_calls，
          使用独立 list[dict] 避免 Pydantic 对未完整 args 的校验，
          同时支持 tool_call_chunks（真实 LangChain 流式格式）和整块 dict 两种来源。
  [Fix-3] 中间文本屏蔽：含 tool_calls 的推理轮文本不透传给前端（缓冲后丢弃），
          仅最终回答轮（tool_calls 为空）的文本流式发送给前端。
  [Fix-4] Token 统计多字段兼容：同时识别 input_tokens/output_tokens
          和 prompt_tokens/completion_tokens，适配不同 provider/版本差异。
  [Fix-6] asyncio.get_running_loop() 替换 asyncio.get_event_loop()。

辅助函数说明：
    _accumulate_tool_calls(pending, chunk_calls) — 增量累积 tool_calls 为 list[dict]
    _finalize_tool_calls(pending)                — 将字符串 args 解析为 dict
    _accumulate_tokens(totals, chunk)            — 多字段兼容的 token 统计
    _summarize_tool_result(content)              — 提取工具结果一行摘要
    _extract_sources(content)                    — 提取 [来源: xxx] 列表
    _estimate_cost(tokens)                       — 估算人民币费用
    _get_tool_func(name, tools)                  — 按名称查找工具函数
    _collect_new_messages(messages, base_length) — 提取本轮新增消息（含 HumanMessage）

用法示例：
    agent = MasterAgent()
    async for event in agent.run_stream(session_id, user_input):
        print(event.to_sse())
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
import re
import time
from typing import AsyncGenerator

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from src.agents.events import (
    BaseEvent,
    DoneEvent,
    ErrorEvent,
    SessionStartEvent,
    SourcesEvent,
    TextDeltaEvent,
    ThinkingEvent,
    ToolEndEvent,
    ToolStartEvent,
    UsageEvent,
)
from src.agents.prompt_loader import load_master_agent_system_prompt
from src.agents.session_manager import SessionManager
from src.agents.tool_registry import TOOL_DISPLAY_NAMES, build_tool_registry
from src.llm_client import get_llm

logger = logging.getLogger(__name__)

# ── 费用估算（DashScope 参考定价，可从 config 覆盖）──────────────────────────
_PRICE_PER_1K_PROMPT: float = 0.04      # 元/千 token（输入）
_PRICE_PER_1K_COMPLETION: float = 0.12  # 元/千 token（输出）


# ════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ════════════════════════════════════════════════════════════════════════════

def _get_tool_func(name: str, tools: list):
    """从工具列表中按名称查找工具函数；未找到返回 None。"""
    for t in tools:
        if t.name == name:
            return t
    return None


def _accumulate_tool_calls(pending: list[dict], chunk_calls) -> None:
    """
    将流式 chunk 中的工具调用信息增量累积到 pending list[dict] 中。

    [Fix-2] 使用独立的 list[dict] 而非 AIMessage.tool_calls，
    避免 Pydantic 对未完整 args（JSON 字符串分片）的校验失败。

    支持：
    - tool_call_chunks：真实 LangChain 流式格式，args 为字符串，按 index 路由
    - tool_calls dict：整块格式（非流式或 mock），args 已是 dict，直接覆盖
    - ToolCallChunk 对象：LangChain 内部类型，支持 .id/.name/.args/.index 属性
    """
    for tc in chunk_calls:
        # 统一转为 dict（兼容 ToolCallChunk 对象和普通 dict）
        if not isinstance(tc, dict):
            tc = {
                "id":    getattr(tc, "id", None),
                "name":  getattr(tc, "name", None),
                "args":  getattr(tc, "args", ""),
                "index": getattr(tc, "index", None),
                "type":  getattr(tc, "type", None),
            }

        index = tc.get("index")

        if index is not None and isinstance(index, int) and index < len(pending):
            # 已存在同 index 的 tool_call → 增量合并
            existing = pending[index]
            old_args = existing.get("args", "")
            new_args = tc.get("args", "")

            if isinstance(old_args, str) and isinstance(new_args, str):
                existing["args"] = old_args + new_args   # JSON 字符串拼接
            elif isinstance(new_args, dict) and new_args:
                existing["args"] = new_args              # dict 直接覆盖
            # elif new_args 为空：保持 old_args 不变

            # 补全可能延迟到来的 id / name
            if tc.get("id") and not existing.get("id"):
                existing["id"] = tc["id"]
            if tc.get("name") and not existing.get("name"):
                existing["name"] = tc["name"]
        else:
            # 新的 tool_call：追加
            pending.append({
                "id":   tc.get("id", "") or "",
                "name": tc.get("name", "") or "",
                "args": tc.get("args", ""),
            })


def _finalize_tool_calls(pending: list[dict]) -> list[dict]:
    """
    [Fix-2] 流结束后将字符串 args 解析为 dict。

    若解析失败（JSON 损坏），args 置为空 dict 并记录 WARNING，
    不让解析错误阻断主循环。
    """
    result = []
    for tc in pending:
        tc = dict(tc)
        args = tc.get("args", "")
        if isinstance(args, str):
            if args.strip():
                try:
                    tc["args"] = _json.loads(args)
                except (ValueError, TypeError):
                    logger.warning(
                        "tool_call args 解析失败（JSON 不完整或损坏），已置为空 dict: "
                        "tool=%r  args=%r",
                        tc.get("name"), args[:200],
                    )
                    tc["args"] = {}
            else:
                tc["args"] = {}
        result.append(tc)
    return result


def _accumulate_tokens(totals: dict, chunk) -> None:
    """
    [Fix-4] 从 chunk.usage_metadata 累计 token 数。

    兼容多种字段名约定：
    - input_tokens / output_tokens（LangChain 新版 / Anthropic 风格）
    - prompt_tokens / completion_tokens（OpenAI / DashScope 风格）

    usage_metadata 通常只在流式最后一个 chunk 中出现，因此多轮累加不会重复计数。
    """
    meta = getattr(chunk, "usage_metadata", None)
    if not meta:
        return
    totals["prompt"] = totals.get("prompt", 0) + (
        meta.get("input_tokens") or meta.get("prompt_tokens", 0)
    )
    totals["completion"] = totals.get("completion", 0) + (
        meta.get("output_tokens") or meta.get("completion_tokens", 0)
    )


def _summarize_tool_result(content: str) -> str:
    """
    从工具结果中提取一行摘要（用于 ToolEndEvent.result_summary）。

    策略：
    - 若第一行非空且 ≤ 80 字，直接用作摘要
    - 否则截取前 80 字并加省略号
    - 若内容为空，返回 "执行完成"
    """
    if not content or not content.strip():
        return "执行完成"
    first_line = content.strip().split("\n")[0].strip()
    if len(first_line) <= 80:
        return first_line
    return first_line[:80] + "…"


def _extract_sources(content: str) -> list[str]:
    """
    从工具结果文本中提取 [来源: xxx] 格式的引用列表。

    匹配格式：[来源: 文件名#chunk-n]
    """
    return re.findall(r"\[来源: ([^\]]+)\]", content)


def _estimate_cost(tokens: dict) -> float:
    """根据 token 数估算人民币费用。"""
    return (
        tokens.get("prompt", 0) / 1000 * _PRICE_PER_1K_PROMPT
        + tokens.get("completion", 0) / 1000 * _PRICE_PER_1K_COMPLETION
    )


def _collect_new_messages(
    messages: list[BaseMessage],
    base_length: int,
) -> list[BaseMessage]:
    """
    [Fix-1] 从完整消息列表中提取本轮新增的消息。

    base_length 设计为：len(SystemMessage) + len(history) = 1 + history_length
    即 messages[:base_length] 包含：SystemMessage + 全部历史消息
    messages[base_length:] 包含：HumanMessage（本轮用户输入）+ 本轮 AI/Tool 消息

    SystemMessage 通过 isinstance 过滤去除（不应存入 JSONL），
    HumanMessage 会被包含在内 —— 这正是修复 Issue 1 的关键：
    保证跨会话恢复时用户消息完整，历史不会只剩助手/工具消息。

    Args:
        messages:    完整消息列表（SystemMessage + history + HumanMessage + loop新增）
        base_length: 从此 index 开始切片（= 1 + history_length，不含 HumanMessage）
    """
    return [
        m for m in messages[base_length:]
        if not isinstance(m, SystemMessage)
    ]


# ════════════════════════════════════════════════════════════════════════════
# MasterAgent
# ════════════════════════════════════════════════════════════════════════════

class MasterAgent:
    """
    GraphAssistant 核心 Agent（Phase 8-5，Review 修订版）。

    for-break 安全阀循环（参考 claw-code run_turn_loop）：
    - LLM 流式推理，按轮次缓冲文本和 tool_calls
    - 含 tool_calls 的轮次文本不推送给前端（中间推理屏蔽）
    - 最终回答轮（tool_calls 为空）文本流式推送给前端
    - 工具执行失败降级，不抛异常
    - 每轮 SessionManager 持久化到 JSONL（含本轮 HumanMessage）

    用法：
        agent = MasterAgent()
        async for event in agent.run_stream(session_id, user_input):
            yield event.to_sse()  # 推送给前端
    """

    MAX_ITERATIONS: int = 10   # 安全阀，防止无限循环（参考 claw-code max_turns=8）

    def __init__(self) -> None:
        self.tools           = build_tool_registry()         # @lru_cache，全局单例
        self.llm             = get_llm()
        self.llm_with_tools  = self.llm.bind_tools(self.tools)
        self.session_manager = SessionManager()
        self._system_prompt  = load_master_agent_system_prompt()

    # ── 公开接口 ──────────────────────────────────────────────────────────

    async def run_stream(
        self,
        session_id: str,
        user_input: str,
    ) -> AsyncGenerator[BaseEvent, None]:
        """
        流式执行 Agent 主循环。

        通过 yield 向调用方推送 SSE 事件；调用方负责写入 HTTP 响应流。

        事件顺序：
            session_start
            → (thinking → [tool_start → tool_end]*)* 工具调用轮（无 text_delta）
            → thinking → text_delta* 最终回答轮
            → sources? → usage → done

        Args:
            session_id: 会话 ID（不存在时自动新建）
            user_input: 本轮用户输入文本
        """
        yield SessionStartEvent(session_id=session_id)

        try:
            async for event in self._run_loop(session_id, user_input):
                yield event
        except Exception as exc:
            logger.error(
                "MasterAgent.run_stream 顶层异常 [session=%s]: %s",
                session_id, exc, exc_info=True,
            )
            yield ErrorEvent(
                message=f"服务内部错误：{exc}",
                code="MASTER_AGENT_ERROR",
            )
            yield DoneEvent(session_id=session_id)

    # ── 私有主循环 ────────────────────────────────────────────────────────

    async def _run_loop(
        self,
        session_id: str,
        user_input: str,
    ) -> AsyncGenerator[BaseEvent, None]:
        """
        真正的 for-break Agent 循环，从 run_stream() 分离以便顶层异常统一捕获。
        """
        # 1. 加载历史消息
        history = self.session_manager.load(session_id)
        history_length = len(history)

        # [Fix-1] base_length = 1(SystemMessage) + history_length
        # messages[base_length:] 从 HumanMessage 开始，保证持久化时不丢用户消息
        base_length = 1 + history_length

        messages: list[BaseMessage] = [
            SystemMessage(content=self._system_prompt),
            *history,
            HumanMessage(content=user_input),
        ]

        sources_collected: list[str] = []
        total_tokens: dict = {"prompt": 0, "completion": 0}

        # 2. Agent 主循环（for-break 安全阀）
        for iteration in range(self.MAX_ITERATIONS):
            yield ThinkingEvent(iteration=iteration)

            # 3. LLM 流式推理（按轮次缓冲文本和 tool_calls）
            text_buffer: list[str] = []
            # [Fix-2] 独立 list[dict] 累积 tool_calls，避免 Pydantic 校验未完整 args
            pending_tool_calls: list[dict] = []

            try:
                async for chunk in self.llm_with_tools.astream(messages):
                    # 缓冲文本（[Fix-3] 先缓冲，轮次结束后判断是否推送）
                    if chunk.content:
                        text_buffer.append(chunk.content)

                    # [Fix-2] 优先取 tool_call_chunks（真实流式原始格式），
                    # 回退到 tool_calls（整块 dict 格式）
                    raw_calls = (
                        getattr(chunk, "tool_call_chunks", None)
                        or getattr(chunk, "tool_calls", None)
                    )
                    if raw_calls:
                        _accumulate_tool_calls(pending_tool_calls, raw_calls)

                    # [Fix-4] 多字段兼容的 token 统计
                    _accumulate_tokens(total_tokens, chunk)

            except Exception as exc:
                logger.error("LLM 流式推理异常 [iter=%d]: %s", iteration, exc, exc_info=True)
                yield ErrorEvent(message=f"LLM 推理失败：{exc}", code="LLM_STREAM_ERROR")
                break

            # [Fix-2] 流结束后解析字符串 args → dict
            final_tool_calls = _finalize_tool_calls(pending_tool_calls)
            full_response = AIMessage(
                content="".join(text_buffer),
                tool_calls=final_tool_calls,
            )
            messages.append(full_response)

            # 4. 核心终止逻辑：tool_calls 为空 → LLM 完成，发送文本
            if not full_response.tool_calls:
                # [Fix-3] 仅最终回答轮推送文本 delta
                for delta in text_buffer:
                    yield TextDeltaEvent(delta=delta)
                break

            # [Fix-3] 含 tool_calls 的轮次：text_buffer 内容属于中间推理，丢弃不推送
            if text_buffer:
                logger.debug(
                    "轮次 %d 含 tool_calls，屏蔽中间文本（%d 字）",
                    iteration, sum(len(t) for t in text_buffer),
                )

            # 5. 执行所有工具调用（顺序执行）
            for tool_call in full_response.tool_calls:
                start_time = time.time()
                tool_name  = tool_call.get("name", "")

                yield ToolStartEvent(
                    tool_name=tool_name,
                    display_message=TOOL_DISPLAY_NAMES.get(tool_name, "正在处理..."),
                )

                result_content = await self._execute_tool_safe(tool_call)

                elapsed_ms = int((time.time() - start_time) * 1000)
                yield ToolEndEvent(
                    tool_name=tool_name,
                    result_summary=_summarize_tool_result(result_content),
                    elapsed_ms=elapsed_ms,
                )

                messages.append(ToolMessage(
                    content=result_content,
                    tool_call_id=tool_call.get("id", ""),
                ))
                sources_collected.extend(_extract_sources(result_content))

        # 6. 推送来源与用量事件
        if sources_collected:
            yield SourcesEvent(sources=list(dict.fromkeys(sources_collected)))

        yield UsageEvent(
            total_tokens=total_tokens.get("prompt", 0) + total_tokens.get("completion", 0),
            prompt_tokens=total_tokens.get("prompt", 0),
            completion_tokens=total_tokens.get("completion", 0),
            estimated_cost_cny=round(_estimate_cost(total_tokens), 6),
        )

        # 7. [Fix-1] 持久化本轮新增消息（含 HumanMessage）
        new_messages = _collect_new_messages(messages, base_length)
        try:
            self.session_manager.save_turn(
                session_id=session_id,
                user_input=user_input,
                new_messages=new_messages,
                sources=sources_collected,
                token_usage={
                    "prompt":     total_tokens.get("prompt", 0),
                    "completion": total_tokens.get("completion", 0),
                    "total": (
                        total_tokens.get("prompt", 0)
                        + total_tokens.get("completion", 0)
                    ),
                },
            )
        except Exception as exc:
            logger.warning("会话持久化失败 [session=%s]: %s", session_id, exc)

        yield DoneEvent(session_id=session_id)

    # ── 工具执行（失败降级）──────────────────────────────────────────────

    async def _execute_tool_safe(self, tool_call: dict) -> str:
        """
        工具执行 + 失败降级。

        - 失败时返回结构化错误信息（不抛异常），让 LLM 自行决策
        - 同步工具通过线程池执行，不阻塞事件循环
        - [Fix-6] 使用 asyncio.get_running_loop() 替代 get_event_loop()
        """
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})

        tool_func = _get_tool_func(tool_name, self.tools)
        if tool_func is None:
            return f"[错误] 未知工具：{tool_name}。请使用已注册的工具之一。"

        try:
            # [Fix-6] get_running_loop() 更明确，适用于当前 async 上下文
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: tool_func.invoke(tool_args),
            )
            return str(result)

        except Exception as exc:
            logger.warning(
                "工具执行失败 [tool=%s, args=%s]: %s",
                tool_name, tool_args, exc, exc_info=True,
            )
            return (
                f"[工具调用失败] {tool_name}: {exc}\n"
                f"请尝试换一种方式或直接基于已有信息回答用户。"
            )
