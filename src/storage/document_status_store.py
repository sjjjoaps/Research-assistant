"""
文档状态存储模块

为每个入库文档维护一个状态机，记录当前处理阶段、关联的 chunk/实体/关系 ID，
以及失败时的错误信息。支持断点续传和失败定位。

状态流转：
  pending -> parsing -> chunking -> metadata -> indexing -> graph -> extracting -> processed
                                                                                 -> failed（任意阶段均可跳转）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.config import settings

# 合法状态集合
VALID_STATUSES = frozenset({
    "pending",
    "parsing",
    "chunking",
    "metadata",
    "indexing",
    "graph",
    "extracting",
    "processed",
    "failed",
})


def generate_doc_id(file_hash: str) -> str:
    """根据文件内容哈希生成稳定的 doc_id。

    doc_id 即文件内容的 MD5，与路径无关：
    - 同内容不同路径 → 同一 doc_id（去重）
    - 文件重命名/移动后 → doc_id 不变（身份延续）
    - 文件内容变更后 → doc_id 改变（触发增量更新）

    Args:
        file_hash: 由 ``extraction_cache.compute_file_hash()`` 计算的文件内容 MD5。
    """
    return file_hash  # file_hash 本身就是内容身份，直接复用


@dataclass
class DocumentStatus:
    """文档处理状态快照。"""

    file_path: str
    doc_id: str
    status: str                          # 当前状态
    current_step: str                    # 当前步骤描述
    chunk_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    relation_ids: list[str] = field(default_factory=list)
    error_message: Optional[str] = None
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── ORM ──────────────────────────────────────────────────────────────────────

class _Base(DeclarativeBase):
    pass


class _DocStatusRecord(_Base):
    __tablename__ = "doc_status"

    doc_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    file_path: Mapped[str] = mapped_column(String(1024), unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    current_step: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    chunk_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    entity_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    relation_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False, default="")


# ── Store ─────────────────────────────────────────────────────────────────────

class DocumentStatusStore:
    """文档状态的持久化存储，底层使用 SQLite。

    Example::

        store = DocumentStatusStore()
        store.init_db()
        store.upsert(DocumentStatus(file_path="paper.pdf", doc_id="abc123",
                                    status="pending", current_step="等待入库"))
        status = store.get_by_path("paper.pdf")
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        path = db_path or (settings.data_dir / "doc_status.db")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{path}", future=True)
        self._session_factory = sessionmaker(bind=self._engine, future=True)

    def init_db(self) -> None:
        """建表（幂等）。"""
        _Base.metadata.create_all(self._engine)

    # ── 写 ────────────────────────────────────────────────────────────────────

    def upsert(self, status: DocumentStatus) -> None:
        """插入或更新文档状态。"""
        if status.status not in VALID_STATUSES:
            raise ValueError(f"非法状态: {status.status!r}，合法值: {VALID_STATUSES}")
        status.updated_at = datetime.now(timezone.utc).isoformat()
        with self._session_factory() as session:
            record = session.get(_DocStatusRecord, status.doc_id)
            if record is None:
                record = _DocStatusRecord(doc_id=status.doc_id)
                session.add(record)
            record.file_path = status.file_path
            record.status = status.status
            record.current_step = status.current_step
            record.chunk_ids_json = json.dumps(status.chunk_ids, ensure_ascii=False)
            record.entity_ids_json = json.dumps(status.entity_ids, ensure_ascii=False)
            record.relation_ids_json = json.dumps(status.relation_ids, ensure_ascii=False)
            record.error_message = status.error_message
            record.updated_at = status.updated_at
            session.commit()

    def mark_failed(self, doc_id: str, step: str, error: str, file_path: str = "") -> None:
        """将文档标记为失败，记录失败步骤和错误信息。

        若状态记录尚未创建（如解析前置检查阶段就失败），会自动补建一条 failed 记录，
        确保断点续传和前端状态展示不会丢失早期失败信息。

        Args:
            doc_id: 文档唯一标识。
            step: 失败时所处的步骤名称。
            error: 错误信息。
            file_path: 文档路径，仅在记录不存在时用于补建，已有记录时忽略。
        """
        with self._session_factory() as session:
            record = session.get(_DocStatusRecord, doc_id)
            if record is None:
                record = _DocStatusRecord(
                    doc_id=doc_id,
                    file_path=file_path,
                    chunk_ids_json="[]",
                    entity_ids_json="[]",
                    relation_ids_json="[]",
                )
                session.add(record)
            record.status = "failed"
            record.current_step = step
            record.error_message = error
            record.updated_at = datetime.now(timezone.utc).isoformat()
            session.commit()

    def delete(self, doc_id: str) -> bool:
        """删除文档状态记录，返回是否存在并删除成功。"""
        with self._session_factory() as session:
            record = session.get(_DocStatusRecord, doc_id)
            if record is None:
                return False
            session.delete(record)
            session.commit()
            return True

    # ── 读 ────────────────────────────────────────────────────────────────────

    def get(self, doc_id: str) -> Optional[DocumentStatus]:
        """按 doc_id 查询状态。"""
        with self._session_factory() as session:
            record = session.get(_DocStatusRecord, doc_id)
            return self._to_dataclass(record) if record else None

    def get_by_path(self, file_path: str) -> Optional[DocumentStatus]:
        """按文件路径查询状态。"""
        with self._session_factory() as session:
            stmt = select(_DocStatusRecord).where(_DocStatusRecord.file_path == file_path)
            record = session.scalar(stmt)
            return self._to_dataclass(record) if record else None

    def list_all(self) -> list[DocumentStatus]:
        """返回所有文档状态列表。"""
        with self._session_factory() as session:
            records = list(session.scalars(select(_DocStatusRecord)).all())
            return [self._to_dataclass(r) for r in records]

    def list_by_status(self, status: str) -> list[DocumentStatus]:
        """按状态过滤文档列表。"""
        with self._session_factory() as session:
            stmt = select(_DocStatusRecord).where(_DocStatusRecord.status == status)
            records = list(session.scalars(stmt).all())
            return [self._to_dataclass(r) for r in records]

    # ── 内部 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_dataclass(record: _DocStatusRecord) -> DocumentStatus:
        return DocumentStatus(
            file_path=record.file_path,
            doc_id=record.doc_id,
            status=record.status,
            current_step=record.current_step,
            chunk_ids=json.loads(record.chunk_ids_json),
            entity_ids=json.loads(record.entity_ids_json),
            relation_ids=json.loads(record.relation_ids_json),
            error_message=record.error_message,
            updated_at=record.updated_at,
        )

    def close(self) -> None:
        self._engine.dispose()
