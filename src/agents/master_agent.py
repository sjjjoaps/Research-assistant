"""
MasterAgent：GraphAssistant 的核心 Agent（P5-Step4 原生化版本）。

架构参考 claw-code src/runtime.py 的 run_turn_loop() 的 for-break 安全阀结构：
- for-break 循环（MAX_ITERATIONS=10）：终止条件由工具调用是否为空决定
- 工具执行结果回注 messages，供 LLM 下一轮决策
- 每轮结束后通过 SessionManager 持久化到 JSONL

P5-Step4 变更：
  - 完全移除 langchain_core.messages 依赖
  - 使用原生 dict 消息格式（role/content/tool_calls/tool_call_id）
  - 使用 build_native_tool_registry() 替代 build_tool_registry()
  - 使用 LLMClient.astream() 替代 llm.bind_tools().astream()
  - 接入 ToolCallLimiter 频率熔断器
  - _accumulate_tokens 从原生 chunk.usage 读取（替代 LangChain usage_metadata）
  - _collect_new_messages 使用 m.get("role") == "system" 过滤（替代 isinstance）
  - _fire_memory_extractor 使用 msg.get("role") == "assistant" 查找 AI 最终回答

辅助函数说明：
    _accumulate_tool_calls(pending, chunk_calls) — 增量累积 tool_calls 为 list[dict]
    _finalize_tool_calls(pending)                — 将字符串 args 解析为 dict
    _accumulate_tokens(totals, chunk)            — 从原生 OpenAI chunk 累计 token 统计
    _summarize_tool_result(content)              — 提取工具结果一行摘要
    _extract_sources(content)                    — 提取 [来源: xxx] 列表
    _estimate_cost(tokens)                       — 估算人民币费用（从 settings 读取定价）
    _get_tool_func(name, tools)                  — 按名称查找工具函数
    _collect_new_messages(messages, base_length) — 提取本轮新增消息（含 user 消息）

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
from src.agents.tool_registry import TOOL_DISPLAY_NAMES, build_native_tool_registry
from src.agents.tool_call_limiter import ToolCallLimiter
from src.infrastructure.config import settings
from src.infrastructure.llm_client import get_native_llm

logger = logging.getLogger(__name__)


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

    支持原生 OpenAI ChoiceDeltaToolCall 对象（.index/.id/.function.name/.function.arguments）
    以及整块 dict 格式（Mock 测试常见）。
    """
    for tc in chunk_calls:
        if not isinstance(tc, dict):
            func = getattr(tc, "function", None)
            tc = {
                "id":    getattr(tc, "id", None),
                "name":  getattr(func, "name", None) if func else None,
                "args":  getattr(func, "arguments", "") if func else "",
                "index": getattr(tc, "index", None),
            }

        index = tc.get("index")

        if index is not None and isinstance(index, int) and index < len(pending):
            existing = pending[index]
            old_args = existing.get("args", "")
            new_args = tc.get("args", "")

            if isinstance(old_args, str) and isinstance(new_args, str):
                existing["args"] = old_args + new_args
            elif isinstance(new_args, dict) and new_args:
                existing["args"] = new_args

            if tc.get("id") and not existing.get("id"):
                existing["id"] = tc["id"]
            if tc.get("name") and not existing.get("name"):
                existing["name"] = tc["name"]
        else:
            pending.append({
                "id":   tc.get("id", "") or "",
                "name": tc.get("name", "") or "",
                "args": tc.get("args", ""),
            })


def _finalize_tool_calls(pending: list[dict]) -> list[dict]:
    """
    流结束后将字符串 args 解析为 dict。

    若解析失败（JSON 损坏），args 置为空 dict 并记录 WARNING。
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
    从原生 OpenAI chunk 累计 token 数。

    原生 OpenAI streaming 在最后一个 chunk 的 usage 字段提供 token 统计
    （stream_options={"include_usage": True} 时或使用 SDK stream 上下文管理器）。

    兼容：
    - chunk.usage（原生 OpenAI ChatCompletionChunk，最后一个 chunk）
    - chunk.usage_metadata（LangChain 兼容路径，保留向后兼容）
    """
    usage = getattr(chunk, "usage", None)
    if usage:
        totals["prompt"] = totals.get("prompt", 0) + (
            getattr(usage, "prompt_tokens", 0) or 0
        )
        totals["completion"] = totals.get("completion", 0) + (
            getattr(usage, "completion_tokens", 0) or 0
        )
        return

    meta = getattr(chunk, "usage_metadata", None)
    if meta:
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
    """根据 token 数估算人民币费用（定价从 settings 读取，支持 .env 覆盖）。"""
    return (
        tokens.get("prompt", 0) / 1000 * settings.price_per_1k_prompt
        + tokens.get("completion", 0) / 1000 * settings.price_per_1k_completion
    )


def _collect_new_messages(
    messages: list[dict],
    base_length: int,
) -> list[dict]:
    """
    从完整消息列表中提取本轮新增的消息。

    base_length 设计为：1(system) + history_length
    messages[base_length:] 包含：用户消息（本轮输入）+ 本轮 AI/Tool 消息

    system 消息通过 role=="system" 过滤去除（不应存入 JSONL），
    user 消息会被包含在内，保证跨会话恢复时不丢失用户消息。
    """
    return [
        m for m in messages[base_length:]
        if m.get("role") != "system"
    ]


# ════════════════════════════════════════════════════════════════════════════
# MasterAgent
# ════════════════════════════════════════════════════════════════════════════

class MasterAgent:
    """
    GraphAssistant 核心 Agent（P5-Step4 原生化版本）。

    for-break 安全阀循环：
    - LLM 流式推理，按轮次缓冲文本和 tool_calls
    - 含 tool_calls 的轮次文本不推送给前端（中间推理屏蔽）
    - 最终回答轮（tool_calls 为空）文本流式推送给前端
    - 工具执行失败降级，不抛异常
    - ToolCallLimiter 频率熔断，超限返回拒绝提示
    - 每轮 SessionManager 持久化到 JSONL（含本轮 user 消息）

    用法：
        agent = MasterAgent()
        async for event in agent.run_stream(session_id, user_input):
            yield event.to_sse()
    """

    MAX_ITERATIONS: int = 10

    def __init__(self) -> None:
        self.tools           = build_native_tool_registry()
        self._llm            = get_native_llm(temperature=0.2)
        self._tool_schemas   = [t.to_openai_schema() for t in self.tools]
        self.session_manager = SessionManager()
        self._system_prompt  = load_master_agent_system_prompt()
        self._limiter        = ToolCallLimiter(
            max_calls=settings.tool_call_max_per_window,
            window_seconds=settings.tool_call_window_seconds,
        )

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
        history = self.session_manager.load(session_id)
        history_length = len(history)

        base_length = 1 + history_length

        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
            *history,
            {"role": "user", "content": user_input},
        ]

        sources_collected: list[str] = []
        total_tokens: dict = {"prompt": 0, "completion": 0}
        memory_written_this_turn: bool = False

        for iteration in range(self.MAX_ITERATIONS):
            yield ThinkingEvent(iteration=iteration)

            text_buffer: list[str] = []
            pending_tool_calls: list[dict] = []
            # Track which tool names we've already sent tool_start for this iteration
            announced_tool_names: set[str] = set()

            try:
                async for chunk in self._llm.astream(messages, tools=self._tool_schemas):
                    choice = chunk.choices[0] if chunk.choices else None
                    if choice is None:
                        _accumulate_tokens(total_tokens, chunk)
                        continue

                    delta = choice.delta

                    if delta.content:
                        text_buffer.append(delta.content)

                    raw_calls = getattr(delta, "tool_calls", None)
                    if raw_calls:
                        _accumulate_tool_calls(pending_tool_calls, raw_calls)
                        # Eagerly announce tool_start as soon as we know the tool name
                        for tc in pending_tool_calls:
                            name = tc.get("name", "")
                            if name and name not in announced_tool_names:
                                announced_tool_names.add(name)
                                yield ToolStartEvent(
                                    tool_name=name,
                                    display_message=TOOL_DISPLAY_NAMES.get(name, "正在处理..."),
                                )

                    _accumulate_tokens(total_tokens, chunk)

            except Exception as exc:
                logger.error("LLM 流式推理异常 [iter=%d]: %s", iteration, exc, exc_info=True)
                yield ErrorEvent(message=f"LLM 推理失败：{exc}", code="LLM_STREAM_ERROR")
                break

            final_tool_calls = _finalize_tool_calls(pending_tool_calls)

            ai_message: dict = {"role": "assistant", "content": "".join(text_buffer)}
            if final_tool_calls:
                ai_message["tool_calls"] = [
                    {
                        "id":       tc.get("id", ""),
                        "type":     "function",
                        "function": {
                            "name":      tc.get("name", ""),
                            "arguments": _json.dumps(tc.get("args", {}), ensure_ascii=False),
                        },
                    }
                    for tc in final_tool_calls
                ]
            messages.append(ai_message)

            if not final_tool_calls:
                for delta in text_buffer:
                    yield TextDeltaEvent(delta=delta)
                break

            if text_buffer:
                logger.debug(
                    "轮次 %d 含 tool_calls，屏蔽中间文本（%d 字）",
                    iteration, sum(len(t) for t in text_buffer),
                )

            # First pass: announce all tool_starts for tools not caught during streaming
            for tool_call in final_tool_calls:
                tool_name = tool_call.get("name", "")
                if tool_name and tool_name not in announced_tool_names:
                    announced_tool_names.add(tool_name)
                    yield ToolStartEvent(
                        tool_name=tool_name,
                        display_message=TOOL_DISPLAY_NAMES.get(tool_name, "正在处理..."),
                    )

            # Second pass: execute each tool and emit tool_end
            for tool_call in final_tool_calls:
                start_time = time.time()
                tool_name  = tool_call.get("name", "")

                result_content = await self._execute_tool_safe(
                    session_id=session_id,
                    tool_call=tool_call,
                )

                if tool_name == "save_user_memory" and result_content.startswith("[记忆已保存]"):
                    memory_written_this_turn = True

                elapsed_ms = int((time.time() - start_time) * 1000)
                yield ToolEndEvent(
                    tool_name=tool_name,
                    result_summary=_summarize_tool_result(result_content),
                    elapsed_ms=elapsed_ms,
                )

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "content":      result_content,
                })
                sources_collected.extend(_extract_sources(result_content))

        if sources_collected:
            yield SourcesEvent(sources=list(dict.fromkeys(sources_collected)))

        estimated_cost_cny = round(_estimate_cost(total_tokens), 6)
        yield UsageEvent(
            total_tokens=total_tokens.get("prompt", 0) + total_tokens.get("completion", 0),
            prompt_tokens=total_tokens.get("prompt", 0),
            completion_tokens=total_tokens.get("completion", 0),
            estimated_cost_cny=estimated_cost_cny,
        )

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
                cost_cny=estimated_cost_cny,
            )
        except Exception as exc:
            logger.warning("会话持久化失败 [session=%s]: %s", session_id, exc)

        self._fire_memory_extractor(
            session_id=session_id,
            user_input=user_input,
            messages=messages,
            memory_written=memory_written_this_turn,
        )

        yield DoneEvent(session_id=session_id)

    # ── 工具执行（失败降级 + ToolCallLimiter）────────────────────────────

    async def _execute_tool_safe(self, session_id: str, tool_call: dict) -> str:
        """
        工具执行 + 失败降级 + 频率熔断。

        - ToolCallLimiter 超限时返回拒绝提示（不执行工具）
        - 失败时返回结构化错误信息（不抛异常），让 LLM 自行决策
        - 同步工具通过线程池执行，不阻塞事件循环
        """
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})

        if not self._limiter.check(session_id, tool_name):
            return self._limiter.reject_message(tool_name)

        tool_func = _get_tool_func(tool_name, self.tools)
        if tool_func is None:
            return f"[错误] 未知工具：{tool_name}。请使用已注册的工具之一。"

        try:
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

    # ── 后台记忆提取（Phase 9-5）──────────────────────────────────────────

    def _fire_memory_extractor(
        self,
        session_id:     str,
        user_input:     str,
        messages:       list,
        memory_written: bool,
    ) -> None:
        """
        turn end 后触发后台记忆提取器（Phase 9-5-3）。

        若主模型本轮已调用 save_user_memory 写入记忆（memory_written=True），
        则跳过，避免后台 extractor 重复写入。
        """
        if memory_written:
            logger.debug(
                "本轮主模型已写入记忆，跳过后台 extractor [session=%s]",
                session_id,
            )
            return

        ai_response = ""
        for msg in reversed(messages):
            if (
                isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and not msg.get("tool_calls")
                and msg.get("content")
                and str(msg.get("content", "")).strip()
            ):
                ai_response = str(msg["content"]).strip()
                break

        if not ai_response or not user_input.strip():
            return

        try:
            import threading
            from src.agents.memory_extractor import extract_and_save_memory
            t = threading.Thread(
                target=extract_and_save_memory,
                args=(user_input, ai_response, session_id),
                daemon=True,
                name=f"mem-extract-{session_id[:8]}",
            )
            t.start()
        except Exception as exc:
            logger.warning("后台记忆提取器启动失败 [session=%s]: %s", session_id, exc)
