"""
FAISS 向量存储模块
支持 chunk 入库、相似度检索、保存与加载

Phase 2.3 变更：
- add_chunks 将 doc_id 写入 chunk metadata，支持按文档过滤
- 新增 delete_by_doc_id：保守实现，重建索引排除指定文档的向量
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

    def delete_by_doc_id(self, doc_id: str) -> int:
        """删除指定 doc_id 的所有向量，返回删除数量。

        保守实现：过滤出需要保留的文档后重建 FAISS 索引。
        仅在 doc_id 非空且索引存在时执行。
        """
        if not doc_id or self._store is None:
            return 0

        all_docs = self.get_all_documents()
        keep_docs = [d for d in all_docs if d.metadata.get("doc_id") != doc_id]
        deleted = len(all_docs) - len(keep_docs)

        if deleted == 0:
            return 0

        if keep_docs:
            self._store = FAISS.from_documents(keep_docs, self.embedder.langchain_embeddings)
        else:
            self._store = None

        return deleted

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        if self._store is None:
            return []
        return self._store.similarity_search(query, k=k)

    def get_all_documents(self) -> list[Document]:
        if self._store is None:
            return []

        documents: list[Document] = []
        for value in self._store.docstore._dict.values():
            if isinstance(value, Document):
                documents.append(value)
        return documents

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
