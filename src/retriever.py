"""
检索模块
封装 FAISS 语义检索，返回带来源信息的结果
"""
from dataclasses import dataclass

from src.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    content: str
    file_path: str
    chunk_index: int


class SemanticRetriever:
    def __init__(self, top_k: int = 3) -> None:
        self.top_k = top_k
        self.vector_store = VectorStore()
        self.vector_store.load()

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        docs = self.vector_store.similarity_search(query=query, k=self.top_k)

        results: list[RetrievedChunk] = []
        for doc in docs:
            results.append(
                RetrievedChunk(
                    content=doc.page_content,
                    file_path=str(doc.metadata.get("file_path", "unknown")),
                    chunk_index=int(doc.metadata.get("chunk_index", -1)),
                )
            )
        return results
