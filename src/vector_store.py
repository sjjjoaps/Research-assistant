"""
FAISS 向量存储模块
支持 chunk 入库、相似度检索、保存与加载
"""
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.chunker import TextChunk
from src.config import settings
from src.embedder import Embedder


class VectorStore:
    def __init__(self, index_dir: Path | None = None) -> None:
        self.index_dir = index_dir or settings.faiss_index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = Embedder()
        self._store: FAISS | None = None

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            return

        documents = [
            Document(page_content=chunk.content, metadata=chunk.metadata)
            for chunk in chunks
        ]

        if self._store is None:
            self._store = FAISS.from_documents(documents, self.embedder.langchain_embeddings)
        else:
            self._store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        if self._store is None:
            return []
        return self._store.similarity_search(query, k=k)

    def save(self) -> None:
        if self._store is None:
            return
        self._store.save_local(str(self.index_dir))

    def load(self) -> None:
        index_file = self.index_dir / "index.faiss"
        if not index_file.exists():
            self._store = None
            return

        self._store = FAISS.load_local(
            str(self.index_dir),
            self.embedder.langchain_embeddings,
            allow_dangerous_deserialization=True,
        )
