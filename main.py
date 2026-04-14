"""
FastAPI 应用入口
启动方式：
  F:/Anaconda/envs/llm_universe/python.exe c:/Users/Administrator/Desktop/GraphAssitant/main.py
或：
  uvicorn main:app --host 127.0.0.1 --port 8000 --reload
"""
import uvicorn
from fastapi import FastAPI

from api.routers import chat, community, documents, graph, research
from api.schemas import HealthResponse
from src.config import settings

app = FastAPI(
    title="学术文献知识库助手 API",
    description="GraphAssistant — 文献入库、问答、深度研究、Idea 生成",
    version="1.0.0",
)

# 注册路由
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(research.router)
app.include_router(community.router)
app.include_router(graph.router)


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
