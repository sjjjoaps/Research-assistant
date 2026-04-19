"""
LLM 客户端封装（兼容转发层）
实际实现已迁移至 src/infrastructure/llm_client.py（P2-Step 2）
"""
from langchain_openai import ChatOpenAI

from src.infrastructure.config import settings


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
