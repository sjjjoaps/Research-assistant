"""
解析结果缓存模块

以 file_hash + config_hash 为键缓存文档解析结果，避免对同一文件重复调用解析器。
文件内容或解析配置发生变化时缓存自动失效。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.config import settings


def compute_file_hash(file_path: str | Path) -> str:
    """计算文件内容的 MD5 哈希（用于缓存键）。"""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_config_hash(config: dict[str, Any]) -> str:
    """计算解析配置的 MD5 哈希（用于缓存键）。

    使用自定义 encoder 处理 Path、枚举等非标准 JSON 类型，
    避免真实 parser config 传入时抛序列化异常。
    """
    class _SafeEncoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if hasattr(obj, "value"):  # Enum-like
                return obj.value
            try:
                return super().default(obj)
            except TypeError:
                return repr(obj)

    serialized = json.dumps(config, sort_keys=True, ensure_ascii=False, cls=_SafeEncoder)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


# ── ORM ──────────────────────────────────────────────────────────────────────

class _Base(DeclarativeBase):
    pass


class _CacheRecord(_Base):
    __tablename__ = "extraction_cache"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)


# ── Cache ─────────────────────────────────────────────────────────────────────

class ExtractionCache:
    """文档解析结果缓存，底层使用 SQLite。

    缓存键 = MD5(file_hash + config_hash)，命中时直接返回上次解析结果，
    文件内容或配置变化时自动失效。

    Example::

        cache = ExtractionCache()
        cache.init_db()

        file_hash = compute_file_hash("paper.pdf")
        config_hash = compute_config_hash({"chunk_size": 512})

        result = cache.get("paper.pdf", file_hash, config_hash)
        if result is None:
            result = expensive_parse("paper.pdf")
            cache.set("paper.pdf", file_hash, config_hash, result)
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        path = db_path or (settings.data_dir / "extraction_cache.db")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{path}", future=True)
        self._session_factory = sessionmaker(bind=self._engine, future=True)

    def init_db(self) -> None:
        """建表（幂等）。"""
        _Base.metadata.create_all(self._engine)

    # ── 写 ────────────────────────────────────────────────────────────────────

    def set(
        self,
        file_path: str,
        file_hash: str,
        config_hash: str,
        result: Any,
    ) -> None:
        """写入缓存。``result`` 必须是 JSON 可序列化对象。"""
        cache_key = self._make_key(file_hash, config_hash)
        with self._session_factory() as session:
            record = session.get(_CacheRecord, cache_key)
            if record is None:
                record = _CacheRecord(cache_key=cache_key)
                session.add(record)
            record.file_path = file_path
            record.file_hash = file_hash
            record.config_hash = config_hash
            record.result_json = json.dumps(result, ensure_ascii=False)
            record.created_at = datetime.now(timezone.utc).isoformat()
            session.commit()

    def invalidate(self, file_path: str) -> int:
        """删除某文件的所有缓存条目，返回删除数量。"""
        with self._session_factory() as session:
            stmt = select(_CacheRecord).where(_CacheRecord.file_path == file_path)
            records = list(session.scalars(stmt).all())
            for r in records:
                session.delete(r)
            session.commit()
            return len(records)

    def clear_all(self) -> None:
        """清空全部缓存（慎用）。"""
        with self._session_factory() as session:
            for r in session.scalars(select(_CacheRecord)).all():
                session.delete(r)
            session.commit()

    # ── 读 ────────────────────────────────────────────────────────────────────

    def get(
        self,
        file_path: str,
        file_hash: str,
        config_hash: str,
    ) -> Optional[Any]:
        """查询缓存。命中返回反序列化结果，未命中返回 None。"""
        cache_key = self._make_key(file_hash, config_hash)
        with self._session_factory() as session:
            record = session.get(_CacheRecord, cache_key)
            if record is None:
                return None
            # 双重校验：防止哈希碰撞导致误命中
            if record.file_hash != file_hash or record.config_hash != config_hash:
                return None
            return json.loads(record.result_json)

    def has(self, file_path: str, file_hash: str, config_hash: str) -> bool:
        """判断缓存是否命中，不反序列化结果。"""
        return self.get(file_path, file_hash, config_hash) is not None

    # ── 内部 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_key(file_hash: str, config_hash: str) -> str:
        raw = f"{file_hash}::{config_hash}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def close(self) -> None:
        self._engine.dispose()
