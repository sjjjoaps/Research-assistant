"""
Local 检索器

定位具体实体、方法、数据集问题，优先融合：
- GraphRetriever：实体/关系邻域证据
- SemanticRetriever：chunk 语义证据

Phase 10-2：retrieve() 新增 section_filter 参数，透传给 SemanticRetriever；
GraphRetriever 的结果在融合后做后过滤（图数据库不按 section_type 索引）。

Phase 10-2 Review 修复：section_filter 非空时，融合候选池扩大到 top_k * 3，
过滤后再裁至 top_k，避免"先截断再过滤"导致符合章节的候选被提前丢弃。
"""
from __future__ import annotations

from src.graph_retriever import GraphRetriever
from src.retrieval.fusion import fuse_ranked_lists
from src.retriever import RetrievedChunk, SemanticRetriever

_FILTER_EXPAND = 3   # section_filter 时扩大候选池的倍数


class LocalRetriever:
    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self.semantic_retriever = SemanticRetriever(top_k=top_k * 2)
        self.graph_retriever = GraphRetriever(top_k=top_k * 2, expand_entities=True)

    def retrieve(self, query: str, section_filter: str = "") -> list[RetrievedChunk]:
        """
        Args:
            query:          用户查询字符串。
            section_filter: 章节类型过滤（如 "method"、"abstract"），空字符串表示不过滤。
                            SemanticRetriever 在向量检索阶段原生支持；
                            GraphRetriever 结果在融合后做后过滤。
                            有 section_filter 时融合池扩大到 top_k * 3，过滤后再裁至 top_k。
        """
        graph_results = self.graph_retriever.retrieve(query)
        # SemanticRetriever 原生支持 section_type 过滤（内部扩大 k*5 再筛选）
        sf = section_filter if section_filter else None
        semantic_results = self.semantic_retriever.retrieve(query, section_type=sf)

        # section_filter 时扩大融合候选池，保证后过滤有足够候选再裁至 top_k
        fuse_k = self.top_k * _FILTER_EXPAND if section_filter else self.top_k
        fused = fuse_ranked_lists([graph_results, semantic_results], top_k=fuse_k)

        if section_filter:
            fused = [c for c in fused if getattr(c, "section_type", "") == section_filter]

        return fused[: self.top_k]

    def close(self) -> None:
        self.graph_retriever.close()
