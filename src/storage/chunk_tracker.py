"""
Chunk 追踪模块

持久化记录 doc_id -> chunk_ids 的映射关系，为增量更新和精确删除提供基础。

chunk_id 基于 doc_id + chunk 内容哈希 + 文档内出现序号生成：
- 同文档更新后内容未变且位置不变的 chunk → chunk_id 不变（来源追踪稳定）
- 内容变更的 chunk → 新 chunk_id（旧来源绑定自动失效）
- 同文档内相同内容出现在不同位置 → 不同 chunk_id（避免主键冲突）
- occurrence_index 参与主键，防止同文档内重复内容互相覆盖
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from sqlalchemy import String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.config import settings


def generate_chunk_id(doc_id: str, content_hash: str, occurrence_index: int = 0) -> str:
    """生成稳定的 chunk_id：基于 doc_id + chunk 内容哈希 + 文档内出现序号的 MD5。

    occurrence_index 是同一文档内该内容哈希第几次出现（0-based），
    用于区分文档内完全相同内容的多个 chunk，避免主键冲突。

    Args:
        doc_id: 文档唯一标识（即文件内容 MD5）。
        content_hash: 该 chunk 文本内容的 MD5，由调用方计算。
        occurrence_index: 同文档内该内容哈希的出现序号（默认 0）。

    Returns:
        32 位十六进制字符串，作为 chunk 的长期主键。
    """
    raw = f"{doc_id}::{content_hash}::{occurrence_index}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def compute_chunk_content_hash(text: str) -> str:
    """计算 chunk 文本内容的 MD5，用于生成 chunk_id。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ── ORM ──────────────────────────────────────────────────────────────────────

class _Base(DeclarativeBase):
    pass


class _ChunkMappingRecord(_Base):
    """doc_id -> chunk_ids 的持久化映射。"""

    __tablename__ = "chunk_mappings"

    doc_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    chunk_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")


class _ChunkMetaRecord(_Base):
    """单个 chunk 的元信息（chunk_id -> doc_id + 位置元数据）。"""

    __tablename__ = "chunk_meta"

    chunk_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(nullable=False)   # 位置元数据，不参与主键
    content_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    text_preview: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


# ── Tracker ───────────────────────────────────────────────────────────────────

class ChunkTracker:
    """文档到 chunk 的双向追踪存储，底层使用 SQLite。

    Example::

        tracker = ChunkTracker()
        tracker.init_db()

        # 入库时注册 chunk（传入每个 chunk 的内容哈希）
        hashes = [compute_chunk_content_hash(t) for t in chunk_texts]
        chunk_ids = tracker.register_chunks(doc_id="<file_md5>", content_hashes=hashes, texts=chunk_texts)

        # 查询某文档的所有 chunk
        ids = tracker.get_chunk_ids("<file_md5>")

        # 删除文档时清理 chunk 记录
        tracker.delete_by_doc("<file_md5>")
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        path = db_path or (settings.data_dir / "chunk_tracker.db")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{path}", future=True)
        self._session_factory = sessionmaker(bind=self._engine, future=True)

    def init_db(self) -> None:
        """建表（幂等）。"""
        _Base.metadata.create_all(self._engine)

    # ── 写 ────────────────────────────────────────────────────────────────────

    def register_chunks(
        self,
        doc_id: str,
        content_hashes: list[str],
        texts: Optional[list[str]] = None,
    ) -> list[str]:
        """为文档注册 chunk，返回生成的 chunk_id 列表。

        chunk_id 由 ``doc_id + content_hash + occurrence_index`` 决定。
        occurrence_index 是同一文档内该内容哈希第几次出现（0-based），
        确保文档内相同内容的多个 chunk 拥有不同主键。

        若该 doc_id 已有记录，会先清除旧记录再重新注册（用于文档更新场景）。

        Args:
            doc_id: 文档唯一标识。
            content_hashes: 每个 chunk 的内容哈希列表（由 ``compute_chunk_content_hash`` 生成）。
            texts: 可选，每个 chunk 的原始文本，用于生成预览（取前 100 字符）。
        """
        # 统计每个 content_hash 在本次列表中的出现次数，生成 occurrence_index
        occurrence_counter: dict[str, int] = {}
        chunk_ids: list[str] = []
        for ch in content_hashes:
            occ = occurrence_counter.get(ch, 0)
            chunk_ids.append(generate_chunk_id(doc_id, ch, occ))
            occurrence_counter[ch] = occ + 1

        previews = texts or [""] * len(content_hashes)

        with self._session_factory() as session:
            # 清除旧 chunk_meta 记录
            old_mapping = session.get(_ChunkMappingRecord, doc_id)
            if old_mapping is not None:
                old_ids: list[str] = json.loads(old_mapping.chunk_ids_json)
                for cid in old_ids:
                    old_meta = session.get(_ChunkMetaRecord, cid)
                    if old_meta is not None:
                        session.delete(old_meta)

            # 写入新 chunk_meta
            for i, (cid, ch, preview) in enumerate(zip(chunk_ids, content_hashes, previews)):
                meta = _ChunkMetaRecord(
                    chunk_id=cid,
                    doc_id=doc_id,
                    chunk_index=i,
                    content_hash=ch,
                    text_preview=preview[:100] if preview else None,
                )
                session.merge(meta)

            # 更新 mapping
            mapping = old_mapping or _ChunkMappingRecord(doc_id=doc_id)
            mapping.chunk_ids_json = json.dumps(chunk_ids, ensure_ascii=False)
            session.merge(mapping)
            session.commit()

        return chunk_ids

    def delete_by_doc(self, doc_id: str) -> int:
        """删除某文档的所有 chunk 记录，返回删除的 chunk 数量。"""
        with self._session_factory() as session:
            mapping = session.get(_ChunkMappingRecord, doc_id)
            if mapping is None:
                return 0
            chunk_ids: list[str] = json.loads(mapping.chunk_ids_json)
            for cid in chunk_ids:
                meta = session.get(_ChunkMetaRecord, cid)
                if meta is not None:
                    session.delete(meta)
            session.delete(mapping)
            session.commit()
            return len(chunk_ids)

    # ── 读 ────────────────────────────────────────────────────────────────────

    def get_chunk_ids(self, doc_id: str) -> list[str]:
        """返回文档对应的所有 chunk_id，不存在则返回空列表。"""
        with self._session_factory() as session:
            mapping = session.get(_ChunkMappingRecord, doc_id)
            if mapping is None:
                return []
            return json.loads(mapping.chunk_ids_json)

    def get_doc_id(self, chunk_id: str) -> Optional[str]:
        """根据 chunk_id 反查所属 doc_id。"""
        with self._session_factory() as session:
            meta = session.get(_ChunkMetaRecord, chunk_id)
            return meta.doc_id if meta else None

    def has_doc(self, doc_id: str) -> bool:
        """判断某文档是否已有 chunk 记录。"""
        with self._session_factory() as session:
            return session.get(_ChunkMappingRecord, doc_id) is not None

    def close(self) -> None:
        self._engine.dispose()
