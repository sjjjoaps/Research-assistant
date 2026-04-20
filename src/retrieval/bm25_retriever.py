"""
BM25 检索模块
从 FAISS docstore 中加载全部 chunk 文本，构建稀疏检索器
"""
import re

from rank_bm25 import BM25Okapi

from src.retrieval.retriever import RetrievedChunk
from src.storage.vector_store import VectorStore


class BM25Retriever:
    def __init__(self, top_k: int = 3) -> None:
        self.top_k = int(top_k)
        self.vector_store = VectorStore()
        self.vector_store.load()

        self._chunks: list[RetrievedChunk] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

        self._build_index()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in re.split(r"\W+", text.lower()) if t]

    def _build_index(self) -> None:
        documents = self.vector_store.get_all_documents()
        self._chunks = [
            RetrievedChunk(
                content=doc.page_content,
                file_path=str(doc.metadata.get("file_path", "unknown")),
                chunk_index=int(doc.metadata.get("chunk_index", -1)),
            )
            for doc in documents
        ]

        self._tokenized_corpus = [self._tokenize(chunk.content) for chunk in self._chunks]
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        if self._bm25 is None or not self._chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results: list[RetrievedChunk] = []
        for index, score in ranked[: self.top_k]:
            if score <= 0:
                continue
            results.append(self._chunks[index])
        return results
