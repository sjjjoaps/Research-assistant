"""
Mix 检索器

综合 semantic + local + global 三路结果。

Phase 10-2：retrieve() 新增 section_filter 参数，透传给 SemanticRetriever 和
LocalRetriever（两者内部均支持原生过滤）；GlobalRetriever 产生的关系/图结果
在融合后做后过滤（图数据库不按 section_type 索引）。

Phase 10-2 Review 修复：section_filter 非空时，融合候选池扩大到 top_k * 3，
过滤后再裁至 top_k，避免三路融合后早截断导致目标章节候选被丢弃。

P1-Step 3：RRF 融合后接入 Reranker 精排（RERANKER_ENABLED=true 时生效）。
"""
from __future__ import annotations

from src.retrieval.fusion import fuse_ranked_lists
from src.retrieval.global_retriever import GlobalRetriever
from src.retrieval.local_retriever import LocalRetriever
from src.retrieval.reranker import get_reranker
from src.retrieval.retriever import RetrievedChunk, SemanticRetriever

_FILTER_EXPAND = 3   # section_filter 时扩大候选池的倍数


class MixRetriever:
    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self.semantic_retriever = SemanticRetriever(top_k=top_k * 2)
        self.local_retriever = LocalRetriever(top_k=top_k * 2)
        self.global_retriever = GlobalRetriever(top_k=top_k * 2)

    def retrieve(self, query: str, section_filter: str = "") -> list[RetrievedChunk]:
        """
        Args:
            query:          用户查询字符串。
            section_filter: 章节类型过滤（如 "method"、"abstract"），空字符串表示不过滤。
                            SemanticRetriever 和 LocalRetriever 在向量检索阶段原生支持；
                            GlobalRetriever 结果在融合后做后过滤。
                            有 section_filter 时融合池扩大到 top_k * 3，过滤后再裁至 top_k。
        """
        sf = section_filter if section_filter else None
        # semantic 和 local 路径原生透传 section_filter
        semantic_results = self.semantic_retriever.retrieve(query, section_type=sf)
        local_results = self.local_retriever.retrieve(query, section_filter=section_filter)
        # global 路径（关系/图索引）不支持 section_type 过滤，融合后后过滤
        global_results = self.global_retriever.retrieve(query)

        # section_filter 时扩大融合候选池，保证后过滤有足够候选再裁至 top_k
        fuse_k = self.top_k * _FILTER_EXPAND if section_filter else self.top_k
        fused = fuse_ranked_lists(
            [semantic_results, local_results, global_results],
            top_k=fuse_k,
        )

        if section_filter:
            fused = [c for c in fused if getattr(c, "section_type", "") == section_filter]

        reranker = get_reranker()
        if reranker:
            return reranker.rerank(query, fused)[: self.top_k]
        return fused[: self.top_k]

    def close(self) -> None:
        self.local_retriever.close()
        self.global_retriever.close()
