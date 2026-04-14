"""
Local 检索器

定位具体实体、方法、数据集问题，优先融合：
- GraphRetriever：实体/关系邻域证据
- SemanticRetriever：chunk 语义证据
"""
from __future__ import annotations

from src.graph_retriever import GraphRetriever
from src.retrieval.fusion import fuse_ranked_lists
from src.retriever import RetrievedChunk, SemanticRetriever


class LocalRetriever:
    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self.semantic_retriever = SemanticRetriever(top_k=top_k * 2)
        self.graph_retriever = GraphRetriever(top_k=top_k * 2, expand_entities=True)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        graph_results = self.graph_retriever.retrieve(query)
        semantic_results = self.semantic_retriever.retrieve(query)
        return fuse_ranked_lists([graph_results, semantic_results], top_k=self.top_k)

    def close(self) -> None:
        self.graph_retriever.close()
