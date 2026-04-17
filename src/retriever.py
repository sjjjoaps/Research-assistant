"""
检索模块
封装 FAISS 语义检索，返回带来源信息的结果

Phase 3.2 变更：
- retrieve() 新增可选 section_type 参数，支持按章节类型过滤检索结果
"""
from dataclasses import dataclass

from src.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    content: str
    file_path: str
    chunk_index: int
    section_type: str = "unknown"
    entity_id: str = ""   # Phase 9-1: LightRAG 双极检索 one-hop 扩展使用


class SemanticRetriever:
    def __init__(self, top_k: int = 3) -> None:
        self.top_k = top_k
        self.vector_store = VectorStore()
        self.vector_store.load()

    def retrieve(
        self,
        query: str,
        section_type: str | None = None,
    ) -> list[RetrievedChunk]:
        """语义检索。

        Args:
            query: 查询文本。
            section_type: 可选，按章节类型过滤（如 "method"、"experiment"）。

        Returns:
            RetrievedChunk 列表。
        """
        docs = self.vector_store.similarity_search(
            query=query,
            k=self.top_k,
            section_type=section_type,
        )

        results: list[RetrievedChunk] = []
        for doc in docs:
            results.append(
                RetrievedChunk(
                    content=doc.page_content,
                    file_path=str(doc.metadata.get("file_path", "unknown")),
                    chunk_index=int(doc.metadata.get("chunk_index", -1)),
                    section_type=str(doc.metadata.get("section_type", "unknown")),
                )
            )
        return results
