"""
FAISS 向量存储模块
支持 chunk 入库、相似度检索、保存与加载

Phase 2.3 变更：
- add_chunks 将 doc_id 写入 chunk metadata，支持按文档过滤
- 新增 delete_by_doc_id：保守实现，重建索引排除指定文档的向量

Phase 2.4 变更：
- save() 在 _store 为 None 时主动删除磁盘索引文件，防止重启后已删向量复现

Phase 2.5 变更：
- 新增内置 threading.Lock，add_chunks / save / delete_by_doc_id 均在锁内执行
  保证多线程并发入库时 FAISS 索引不被并发修改（FAISS 本身不是线程安全的）

Phase 3.2 变更：
- similarity_search 新增可选 section_type 参数，支持按章节类型过滤检索结果
"""
import threading
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.ingestion.chunker import TextChunk
from src.infrastructure.config import settings
from src.infrastructure.embedder import Embedder


class VectorStore:
    def __init__(self, index_dir: Path | None = None) -> None:
        self.index_dir = index_dir or settings.faiss_index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = Embedder()
        self._store: FAISS | None = None
        # 保护 _store 的进程内锁；FAISS 不是线程安全的，所有写操作均在此锁内执行
        self._lock = threading.Lock()

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            return

        documents = [
            Document(page_content=chunk.content, metadata=chunk.metadata)
            for chunk in chunks
        ]

        with self._lock:
            if self._store is None:
                self._store = FAISS.from_documents(documents, self.embedder.langchain_embeddings)
            else:
                self._store.add_documents(documents)

    def delete_by_doc_id(self, doc_id: str) -> int:
        """删除指定 doc_id 的所有向量，返回删除数量。

        保守实现：过滤出需要保留的文档后重建 FAISS 索引。
        仅在 doc_id 非空且索引存在时执行。
        """
        if not doc_id:
            return 0

        with self._lock:
            if self._store is None:
                return 0

            all_docs = self._get_all_documents_unsafe()
            keep_docs = [d for d in all_docs if d.metadata.get("doc_id") != doc_id]
            deleted = len(all_docs) - len(keep_docs)

            if deleted == 0:
                return 0

            if keep_docs:
                self._store = FAISS.from_documents(keep_docs, self.embedder.langchain_embeddings)
            else:
                self._store = None

            return deleted

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        section_type: str | None = None,
    ) -> list[Document]:
        """语义相似度检索。

        Args:
            query: 查询文本。
            k: 返回结果数量。
            section_type: 可选，按章节类型过滤（如 "method"、"experiment"）。
                若指定，先召回 k * 5 个候选，再按 section_type 过滤后取前 k 个。
                过滤后不足 k 个时返回实际数量。

        Returns:
            匹配的 Document 列表。
        """
        if self._store is None:
            return []
        if section_type is None:
            return self._store.similarity_search(query, k=k)
        # 过滤模式：多召回候选，再按 section_type 筛选
        candidates = self._store.similarity_search(query, k=k * 5)
        filtered = [d for d in candidates if d.metadata.get("section_type") == section_type]
        return filtered[:k]

    def get_all_documents(self) -> list[Document]:
        with self._lock:
            return self._get_all_documents_unsafe()

    def _get_all_documents_unsafe(self) -> list[Document]:
        """不加锁地读取所有文档，调用方必须已持有 self._lock。"""
        if self._store is None:
            return []
        documents: list[Document] = []
        for value in self._store.docstore._dict.values():
            if isinstance(value, Document):
                documents.append(value)
        return documents

    def save(self) -> None:
        with self._lock:
            if self._store is None:
                # 索引已清空：主动删除磁盘文件，防止重启后已删向量复现
                for fname in ("index.faiss", "index.pkl"):
                    fpath = self.index_dir / fname
                    if fpath.exists():
                        fpath.unlink()
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
