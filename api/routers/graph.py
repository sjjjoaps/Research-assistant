"""
图谱可视化路由

GET /graph/stats    — 图谱节点/关系统计
GET /graph/subgraph — 查询可视化子图
GET /graph/nodes    — 独立节点列表（subgraph 的节点部分）
GET /graph/edges    — 独立边列表（subgraph 的边部分）
"""
from __future__ import annotations

from fastapi import APIRouter, Query

from api.schemas import GraphStatsResponse, GraphSubgraphResponse, GraphNodesResponse, GraphEdgesResponse
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


@router.get("/nodes", response_model=GraphNodesResponse)
def get_nodes(
    limit: int = Query(default=100, ge=1, le=500),
    node_types: list[str] = Query(default=[]),
    search: str = Query(default=""),
):
    """返回节点列表（subgraph 的 nodes 部分）。"""
    graph_store = GraphStore()
    try:
        data = graph_store.get_subgraph(limit=limit, node_types=node_types, search=search)
        return GraphNodesResponse(nodes=data.get("nodes", []))
    finally:
        graph_store.close()


@router.get("/edges", response_model=GraphEdgesResponse)
def get_edges(
    limit: int = Query(default=200, ge=1, le=1000),
    node_types: list[str] = Query(default=[]),
    node_id: str = Query(default="", description="按节点 ID 过滤关联边；为空时返回全部"),
):
    """返回边列表。node_id 非空时只返回该节点的关联边。"""
    graph_store = GraphStore()
    try:
        data = graph_store.get_subgraph(limit=limit, node_types=node_types)
        edges = data.get("edges", [])
        if node_id:
            edges = [e for e in edges if e.get("source") == node_id or e.get("target") == node_id]
        return GraphEdgesResponse(edges=edges)
    finally:
        graph_store.close()

