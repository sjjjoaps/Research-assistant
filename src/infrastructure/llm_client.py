"""
LLM 客户端封装（基础设施层）

提供两套实现，并行存在：
  - get_llm()        → LangChain ChatOpenAI（旧路径，P5-Step 4 前保持不变）
  - get_native_llm() → LLMClient（P5-Step 2 新增，原生 AsyncOpenAI 封装）
"""
from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from src.infrastructure.config import settings


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """创建统一的聊天模型实例（LangChain 路径，旧接口保持不变）。"""
    if not settings.api_key:
        raise ValueError("API_KEY 未配置，请检查 .env 文件")

    return ChatOpenAI(
        model=settings.model_name,
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        temperature=temperature,
    )


# ════════════════════════════════════════════════════════════════════════════
# P5-Step 2：原生 LLM 客户端（并行实现，不修改 get_llm()）
# ════════════════════════════════════════════════════════════════════════════

class LLMClient:
    """
    原生 OpenAI 客户端封装，替代 LangChain ChatOpenAI。

    使用 AsyncOpenAI 直接调用，不依赖 LangChain。
    消息格式为原生 dict（{"role": ..., "content": ...}），
    工具格式为 OpenAI function calling JSON Schema。
    """

    def __init__(self, temperature: float = 0.0) -> None:
        from openai import AsyncOpenAI

        self.model       = settings.model_name
        self.temperature = temperature
        self._client     = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    async def astream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ):
        """
        流式调用，yield 原生 OpenAI chunk。

        每个 chunk 为 ChatCompletionChunk 对象，含：
          chunk.choices[0].delta.content          — 文本增量
          chunk.choices[0].delta.tool_calls       — 工具调用增量
          chunk.choices[0].finish_reason          — 结束原因
        """
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )
        if tools:
            kwargs["tools"] = tools

        async with self._client.chat.completions.stream(**kwargs) as stream:
            async for chunk in stream:
                yield chunk

    async def ainvoke(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """
        非流式调用，返回第一个 choice 的 message dict。

        返回格式：
          {
            "role": "assistant",
            "content": "...",          # 文本回复（可为 None）
            "tool_calls": [...] | None # 工具调用列表
          }
        """
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        if tools:
            kwargs["tools"] = tools

        resp = await self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.model_dump()

    @property
    def model_name(self) -> str:
        return self.model


@lru_cache(maxsize=8)
def get_native_llm(temperature: float = 0.0) -> LLMClient:
    """
    原生 LLMClient 工厂（P5-Step 2 新增，与 get_llm() 并行存在）。

    通过 lru_cache 保证相同 temperature 复用同一实例。
    """
    if not settings.api_key:
        raise ValueError("API_KEY 未配置，请检查 .env 文件")
    return LLMClient(temperature=temperature)
