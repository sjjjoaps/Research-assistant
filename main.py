"""
FastAPI 应用入口（Phase 8-8 最终版）。

启动方式：
  F:/Anaconda/envs/llm_universe/python.exe c:/Users/Administrator/Desktop/GraphAssitant/main.py
或：
  uvicorn main:app --host 127.0.0.1 --port 8000 --reload

已注册路由（Phase 8 兼容层 + MasterAgent 新路由）：
  /documents/*   — 文献入库与管理
  /chat/*        — 旧 QAAgent 问答接口（保留，兼容层）
  /research/*    — 旧 DeepResearchAgent 接口（保留，兼容层）
  /community/*   — 社区检测
  /graph/*       — 知识图谱
  /agent/*       — MasterAgent SSE 流式对话（Phase 8-6 新增）
    POST /agent/chat                         — 流式对话（SSE）
    GET  /agent/sessions                     — 会话列表
    DELETE /agent/sessions/{id}              — 删除会话
    GET  /agent/sessions/{id}/history        — 会话历史

健康检查：
  GET /health → {"status": "ok", "version": "1.1.0"}
"""
import uvicorn
from fastapi import FastAPI

from api.routers import chat, community, documents, graph, research
from api.routers import agent as agent_router_module   # Phase 8-6: MasterAgent SSE
from api.schemas import HealthResponse
from src.infrastructure.config import settings

app = FastAPI(
    title="学术文献知识库助手 API",
    description="GraphAssistant — 文献入库、问答、深度研究、Idea 生成、MasterAgent 流式对话",
    version="1.1.0",
)

# 注册路由
# ── 兼容层（Phase 8 保留旧 Router，不删除）────────────────────────────────
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(research.router)
app.include_router(community.router)
app.include_router(graph.router)

# ── Phase 8-6：MasterAgent SSE 流式路由（/agent/*）───────────────────────
app.include_router(agent_router_module.router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    return HealthResponse()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )
