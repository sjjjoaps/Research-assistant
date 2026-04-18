"""
图谱可视化路由

GET /graph/stats    — 图谱节点/关系统计
GET /graph/subgraph — 查询可视化子图
"""
from __future__ import annotations

from fastapi import APIRouter, Query

from api.schemas import GraphStatsResponse, GraphSubgraphResponse
from src.storage.graph_store import GraphStore

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/stats", response_model=GraphStatsResponse)
def get_graph_stats():
    graph_store = GraphStore()
    try:
        return GraphStatsResponse(**graph_store.get_graph_stats())
    finally:
        graph_store.close()


@router.get("/subgraph", response_model=GraphSubgraphResponse)
def get_subgraph(
    limit: int = Query(default=100, ge=1, le=500, description="最大节点数"),
    node_types: list[str] = Query(default=[], description="节点类型过滤，可重复传参"),
    search: str = Query(default="", description="按节点名称/标题/路径搜索"),
):
    graph_store = GraphStore()
    try:
        return GraphSubgraphResponse(
            **graph_store.get_subgraph(
                limit=limit,
                node_types=node_types,
                search=search,
            )
        )
    finally:
        graph_store.close()
