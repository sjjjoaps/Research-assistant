"""
Local 检索器

定位具体实体、方法、数据集问题，优先融合：
- GraphRetriever：实体/关系邻域证据
- SemanticRetriever：chunk 语义证据

Reranker 可用时：合并两路候选池直接 rerank，跳过 RRF。
Reranker 不可用时：RRF 融合后截断。

Phase 10-2：retrieve() 新增 section_filter 参数，透传给 SemanticRetriever；
GraphRetriever 的结果在融合后做后过滤（图数据库不按 section_type 索引）。

Phase 10-2 Review 修复：section_filter 非空时，融合候选池扩大到 top_k * 3，
过滤后再裁至 top_k，避免"先截断再过滤"导致符合章节的候选被提前丢弃。
"""
from __future__ import annotations

from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.fusion import fuse_ranked_lists
from src.retrieval.reranker import get_reranker, rerank_or_truncate
from src.retrieval.retriever import RetrievedChunk, SemanticRetriever

_FILTER_EXPAND = 3


class LocalRetriever:
    def __init__(self, top_k: int = 5) -> None:
        self.top_k = int(top_k)
        self.semantic_retriever = SemanticRetriever(top_k=top_k * 2)
        self.graph_retriever = GraphRetriever(top_k=top_k * 2, expand_entities=True)

    def retrieve(self, query: str, section_filter: str = "") -> list[RetrievedChunk]:
        graph_results = self.graph_retriever.retrieve(query)
        sf = section_filter if section_filter else None
        semantic_results = self.semantic_retriever.retrieve(query, section_type=sf)

        if get_reranker():
            # reranker 可用：合并去重后直接精排，跳过 RRF
            seen: set[tuple] = set()
            candidates: list[RetrievedChunk] = []
            for c in graph_results + semantic_results:
                key = (c.file_path, c.chunk_index)
                if key not in seen:
                    seen.add(key)
                    candidates.append(c)
            if section_filter:
                candidates = [c for c in candidates if getattr(c, "section_type", "") == section_filter]
            return rerank_or_truncate(query, candidates, self.top_k)

        fuse_k = self.top_k * _FILTER_EXPAND if section_filter else self.top_k
        fused = fuse_ranked_lists([graph_results, semantic_results], top_k=fuse_k)
        if section_filter:
            fused = [c for c in fused if getattr(c, "section_type", "") == section_filter]
        return fused[: self.top_k]

    def close(self) -> None:
        self.graph_retriever.close()
