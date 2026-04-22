"""
实体向量索引

为 Entity 节点维护独立的 FAISS 索引，支持：
- 实体文本写入
- 按查询检索实体
- 按 doc_id 删除文档贡献的实体向量
- 独立保存 / 加载索引文件

对齐 LightRAG §3.2 Low-Level 检索：
- content 格式："{entity_name}\n{description}"（与 LightRAG entities_vdb 一致）
- 检索时以 ll_keywords 为查询，返回最相关实体，再通过 entity_id 扩展图邻域
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.infrastructure.config import settings
from src.infrastructure.embedder import Embedder


@dataclass
class EntityVectorRecord:
    """实体向量记录。"""

    entity_id: str
    entity_name: str
    entity_type: str
    description: str
    doc_id: str
    file_path: str

    @property
    def content(self) -> str:
        parts = [self.entity_name.strip()]
        desc = self.description.strip()
        if desc:
            parts.append(desc)
        return "\n".join(parts)

    def to_document(self) -> Document:
        return Document(
            page_content=self.content,
            metadata={
                "entity_id": self.entity_id,
                "entity_name": self.entity_name,
                "entity_type": self.entity_type,
                "description": self.description,
                "doc_id": self.doc_id,
                "file_path": self.file_path,
            },
        )


class EntityVectorStore:
    def __init__(self, index_dir: Path | None = None) -> None:
        self.index_dir = index_dir or settings.entity_faiss_index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = Embedder()
        self._store: FAISS | None = None
        self._lock = threading.Lock()

    def add_entities(self, entities: list[EntityVectorRecord]) -> None:
        if not entities:
            return

        documents = [record.to_document() for record in entities]
        with self._lock:
            if self._store is None:
                self._store = FAISS.from_documents(documents, self.embedder.langchain_embeddings)
            else:
                self._store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        if self._store is None:
            return []
        return self._store.similarity_search(query, k=k)

    def delete_by_doc_id(self, doc_id: str) -> int:
        if not doc_id:
            return 0

        with self._lock:
            if self._store is None:
                return 0

            all_docs = self._get_all_documents_unsafe()
            keep_docs = [doc for doc in all_docs if doc.metadata.get("doc_id") != doc_id]
            deleted = len(all_docs) - len(keep_docs)
            if deleted == 0:
                return 0

            if keep_docs:
                self._store = FAISS.from_documents(keep_docs, self.embedder.langchain_embeddings)
            else:
                self._store = None
            return deleted

    def get_all_documents(self) -> list[Document]:
        with self._lock:
            return self._get_all_documents_unsafe()

    def _get_all_documents_unsafe(self) -> list[Document]:
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
