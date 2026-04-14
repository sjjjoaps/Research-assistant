"""
Global 检索器

定位宏观主题、趋势、关系结构问题，优先融合：
- RelationVectorStore：关系级向量召回
- GraphRetriever：图中实体关联补充
"""
from __future__ import annotations

from src.graph_retriever import GraphRetriever
from src.retrieval.fusion import fuse_ranked_lists
from src.retrieval.keyword_extractor import KeywordExtractor
from src.retriever import RetrievedChunk
from src.storage.relation_vector_store import RelationVectorStore


class GlobalRetriever:
    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self.relation_vector_store = RelationVectorStore()
        self.relation_vector_store.load()
        self.graph_retriever = GraphRetriever(top_k=top_k * 2, expand_entities=True)
        self.keyword_extractor = KeywordExtractor()

    def _relation_docs_to_chunks(self, query: str) -> list[RetrievedChunk]:
        keywords = self.keyword_extractor.extract(query)
        relation_query = " ".join(keywords.hl_keywords or keywords.ll_keywords or [query])
        docs = self.relation_vector_store.similarity_search(relation_query, k=self.top_k * 2)

        chunks: list[RetrievedChunk] = []
        for doc in docs:
            metadata = doc.metadata
            content = (
                f"[关系检索]\n"
                f"来源实体: {metadata.get('source_name', '')}\n"
                f"关系类型: {metadata.get('relation_type', '')}\n"
                f"目标实体: {metadata.get('target_name', '')}\n"
                f"关系描述: {metadata.get('description', '')}\n"
            ).strip()
            chunks.append(
                RetrievedChunk(
                    content=content,
                    file_path=str(metadata.get("file_path", "relation_index")),
                    chunk_index=-1,
                    section_type="relation",
                )
            )
        return chunks

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        relation_results = self._relation_docs_to_chunks(query)
        graph_results = self.graph_retriever.retrieve(query)
        return fuse_ranked_lists([relation_results, graph_results], top_k=self.top_k)

    def close(self) -> None:
        self.graph_retriever.close()
