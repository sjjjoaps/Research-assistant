"""
问答路由
POST /chat — 多轮问答（进程内单例 Agent，按配置缓存）
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter

from api.schemas import ChatRequest, ChatResponse
from src.agents.qa_agent import QAAgent

router = APIRouter(tags=["chat"])

# 进程内单例缓存：(top_k, max_history, retriever_mode) -> QAAgent
_qa_agents: dict[tuple, QAAgent] = {}


def _get_agent(
    top_k: int,
    max_history: int,
    retriever_mode: Literal["semantic", "hybrid", "graph"],
) -> QAAgent:
    key = (top_k, max_history, retriever_mode)
    if key not in _qa_agents:
        _qa_agents[key] = QAAgent(
            top_k=top_k,
            max_history_turns=max_history,
            retriever_mode=retriever_mode,
        )
    return _qa_agents[key]


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """多轮问答接口，同配置请求复用同一 Agent 实例以保持会话状态"""
    agent = _get_agent(req.top_k, req.max_history, req.retriever_mode)
    result = agent.run_turn(req.thread_id, req.message)

    token_usage = result.get("token_usage")
    token_dict: dict | None = None
    if token_usage is not None:
        token_dict = {
            "prompt_tokens": token_usage.prompt_tokens,
            "completion_tokens": token_usage.completion_tokens,
            "total_tokens": token_usage.total_tokens,
            "model_name": token_usage.model_name,
            "estimated_cost_cny": token_usage.estimated_cost_cny,
        }

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        thread_id=result["thread_id"],
        retriever_mode=result["retriever_mode"],
        token_usage=token_dict,
    )
