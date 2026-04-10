"""
LLM 客户端封装
基于 LangChain 1.0 封装统一的聊天模型创建逻辑
"""
from langchain_openai import ChatOpenAI

from src.config import settings


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """创建统一的聊天模型实例"""
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
