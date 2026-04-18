"""
关系向量索引

为 Entity-RELATES_TO 关系维护独立的 FAISS 索引，支持：
- 关系文本写入
- 按查询检索关系
- 按 doc_id 删除文档贡献的关系向量
- 独立保存 / 加载索引文件

Phase 4.2 说明：
- 关系索引与 chunk 索引分离，磁盘目录独立保存
- 每条关系按文档贡献写入一份向量，删除文档时可按 doc_id 精确清理
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
class RelationVectorRecord:
    """关系向量记录。"""

    relation_key: str
    source_entity_id: str
    source_name: str
    relation_type: str
    target_entity_id: str
    target_name: str
    description: str
    doc_id: str
    file_path: str

    @property
    def content(self) -> str:
        parts = [
            self.source_name.strip(),
            self.relation_type.strip(),
            self.target_name.strip(),
        ]
        description = self.description.strip()
        if description:
            parts.append(description)
        return " ".join(part for part in parts if part)

    def to_document(self) -> Document:
        return Document(
            page_content=self.content,
            metadata={
                "relation_key": self.relation_key,
                "source_entity_id": self.source_entity_id,
                "source_name": self.source_name,
                "relation_type": self.relation_type,
                "target_entity_id": self.target_entity_id,
                "target_name": self.target_name,
                "description": self.description,
                "doc_id": self.doc_id,
                "file_path": self.file_path,
            },
        )


class RelationVectorStore:
    def __init__(self, index_dir: Path | None = None) -> None:
        self.index_dir = index_dir or settings.relation_faiss_index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = Embedder()
        self._store: FAISS | None = None
        self._lock = threading.Lock()

    def add_relations(self, relations: list[RelationVectorRecord]) -> None:
        if not relations:
            return

        documents = [record.to_document() for record in relations]
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
