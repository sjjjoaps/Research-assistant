"""
社区检测路由（Phase 15 补充）
POST /community/detect — 触发 Louvain 社区检测并写回 Neo4j
GET  /community/list   — 读取已有社区摘要列表
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.community_detector import CommunityDetector
from src.graph_store import GraphStore

router = APIRouter(prefix="/community", tags=["community"])


class DetectRequest(BaseModel):
    min_community_size: int = Field(default=3, ge=1, description="社区最小实体数")


class DetectResponse(BaseModel):
    total_entities: int
    detected_communities: int
    written_communities: int
    skipped_communities: int


class CommunityItem(BaseModel):
    id: str
    label: int
    summary: str
    entity_count: int


@router.post("/detect", response_model=DetectResponse)
def detect_communities(req: DetectRequest):
    """触发 Louvain 社区检测，将社区节点与 BELONGS_TO 关系写回 Neo4j"""
    graph_store = GraphStore()
    try:
        detector = CommunityDetector(graph_store)
        stats = detector.run(min_community_size=req.min_community_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        graph_store.close()
    return DetectResponse(
        total_entities=stats.total_entities,
        detected_communities=stats.detected_communities,
        written_communities=stats.written_communities,
        skipped_communities=stats.skipped_communities,
    )


@router.get("/list", response_model=list[CommunityItem])
def list_communities(limit: int = 20):
    """按实体数降序返回已检测的社区摘要列表"""
    graph_store = GraphStore()
    try:
        summaries = graph_store.get_community_summaries(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        graph_store.close()
    return [
        CommunityItem(
            id=s["id"],
            label=s["label"],
            summary=s["summary"] or "",
            entity_count=s["entity_count"],
        )
        for s in summaries
    ]
