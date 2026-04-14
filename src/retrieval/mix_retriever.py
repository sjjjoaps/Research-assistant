"""
Mix 检索器

综合 semantic + local + global 三路结果。
"""
from __future__ import annotations

from src.retrieval.fusion import fuse_ranked_lists
from src.retrieval.global_retriever import GlobalRetriever
from src.retrieval.local_retriever import LocalRetriever
from src.retriever import RetrievedChunk, SemanticRetriever


class MixRetriever:
    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self.semantic_retriever = SemanticRetriever(top_k=top_k * 2)
        self.local_retriever = LocalRetriever(top_k=top_k * 2)
        self.global_retriever = GlobalRetriever(top_k=top_k * 2)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        semantic_results = self.semantic_retriever.retrieve(query)
        local_results = self.local_retriever.retrieve(query)
        global_results = self.global_retriever.retrieve(query)
        return fuse_ranked_lists(
            [semantic_results, local_results, global_results],
            top_k=self.top_k,
        )

    def close(self) -> None:
        self.local_retriever.close()
        self.global_retriever.close()
