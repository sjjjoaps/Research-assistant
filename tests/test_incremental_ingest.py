"""
Phase 2.3 增量入库测试

验收标准：
- 未变化文档重复入库不会新增重复 chunk
- 变更后重新入库不会残留旧 chunk/source 绑定
- 新增文档加入时历史文档相关实体和关系保持稳定
"""
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.storage.chunk_tracker import ChunkTracker, compute_chunk_content_hash, generate_chunk_id
from src.storage.document_status_store import DocumentStatus, DocumentStatusStore, generate_doc_id
from src.storage.extraction_cache import compute_file_hash


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _make_temp_txt(content: str) -> Path:
    """创建临时文本文件，返回路径。"""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


# ── ChunkTracker 单元测试 ─────────────────────────────────────────────────────

class TestChunkTracker:
    def setup_method(self):
        self.tmp_db = Path(tempfile.mktemp(suffix=".db"))
        self.tracker = ChunkTracker(db_path=self.tmp_db)
        self.tracker.init_db()

    def teardown_method(self):
        self.tracker.close()
        self.tmp_db.unlink(missing_ok=True)

    def test_register_and_query(self):
        hashes = [compute_chunk_content_hash(f"chunk {i}") for i in range(3)]
        ids = self.tracker.register_chunks("doc1", hashes)
        assert len(ids) == 3
        assert self.tracker.get_chunk_ids("doc1") == ids

    def test_stable_ids_same_content(self):
        """相同内容的 chunk 在同一文档中生成相同 ID。"""
        hashes = [compute_chunk_content_hash("hello world")]
        ids1 = self.tracker.register_chunks("doc1", hashes)
        # 重新注册（模拟重复入库）
        ids2 = self.tracker.register_chunks("doc1", hashes)
        assert ids1 == ids2

    def test_different_doc_same_content_different_ids(self):
        """不同文档中相同内容的 chunk 生成不同 ID（chunk_id 包含 doc_id）。"""
        hashes = [compute_chunk_content_hash("same content")]
        ids_a = self.tracker.register_chunks("docA", hashes)
        ids_b = self.tracker.register_chunks("docB", hashes)
        assert ids_a[0] != ids_b[0]

    def test_delete_by_doc(self):
        hashes = [compute_chunk_content_hash(f"c{i}") for i in range(2)]
        self.tracker.register_chunks("doc1", hashes)
        deleted = self.tracker.delete_by_doc("doc1")
        assert deleted == 2
        assert self.tracker.get_chunk_ids("doc1") == []

    def test_has_doc(self):
        hashes = [compute_chunk_content_hash("x")]
        self.tracker.register_chunks("doc1", hashes)
        assert self.tracker.has_doc("doc1") is True
        assert self.tracker.has_doc("doc_missing") is False


# ── DocumentStatusStore 增量检测测试 ─────────────────────────────────────────

class TestDocumentStatusStore:
    def setup_method(self):
        self.tmp_db = Path(tempfile.mktemp(suffix=".db"))
        self.store = DocumentStatusStore(db_path=self.tmp_db)
        self.store.init_db()

    def teardown_method(self):
        self.store.close()
        self.tmp_db.unlink(missing_ok=True)

    def test_upsert_and_get(self):
        status = DocumentStatus(
            file_path="/tmp/paper.pdf",
            doc_id="abc123",
            status="processed",
            current_step="入库完成",
        )
        self.store.upsert(status)
        result = self.store.get("abc123")
        assert result is not None
        assert result.status == "processed"

    def test_get_by_path(self):
        status = DocumentStatus(
            file_path="/tmp/paper.pdf",
            doc_id="abc123",
            status="processed",
            current_step="入库完成",
        )
        self.store.upsert(status)
        result = self.store.get_by_path("/tmp/paper.pdf")
        assert result is not None
        assert result.doc_id == "abc123"

    def test_detect_unchanged_document(self):
        """doc_id 相同且状态为 processed → 应跳过。"""
        status = DocumentStatus(
            file_path="/tmp/paper.pdf",
            doc_id="abc123",
            status="processed",
            current_step="入库完成",
        )
        self.store.upsert(status)
        existing = self.store.get("abc123")
        assert existing is not None and existing.status == "processed"

    def test_detect_changed_document(self):
        """同路径但 doc_id 不同 → 应触发清理。"""
        old_status = DocumentStatus(
            file_path="/tmp/paper.pdf",
            doc_id="old_hash",
            status="processed",
            current_step="入库完成",
        )
        self.store.upsert(old_status)

        # 模拟文件内容变化后的新 doc_id
        new_doc_id = "new_hash"
        by_path = self.store.get_by_path("/tmp/paper.pdf")
        assert by_path is not None
        assert by_path.doc_id != new_doc_id  # 检测到变化

    def test_delete_old_status(self):
        status = DocumentStatus(
            file_path="/tmp/paper.pdf",
            doc_id="abc123",
            status="processed",
            current_step="入库完成",
        )
        self.store.upsert(status)
        deleted = self.store.delete("abc123")
        assert deleted is True
        assert self.store.get("abc123") is None


# ── 增量入库集成测试（mock 外部依赖）────────────────────────────────────────

class TestIncrementalIngestPipeline:
    """使用 mock 隔离 LLM / Neo4j / FAISS，只测试增量逻辑。"""

    def _make_pipeline(self, tmp_path: Path):
        """构造一个使用临时目录的 IngestionPipeline，mock 掉外部依赖。"""
        from src.workflows.ingestion_pipeline import IngestionPipeline

        with (
            patch("src.workflows.ingestion_pipeline.DocumentParser"),
            patch("src.workflows.ingestion_pipeline.MetadataExtractor"),
            patch("src.workflows.ingestion_pipeline.VectorStore"),
            patch("src.workflows.ingestion_pipeline.GraphStore"),
            patch("src.workflows.ingestion_pipeline.EntityExtractor"),
            patch("src.workflows.ingestion_pipeline.settings") as mock_settings,
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.faiss_index_dir = tmp_path / "faiss"
            mock_settings.sqlite_path = tmp_path / "meta.db"
            mock_settings.enable_entity_extraction = False
            mock_settings.entity_extraction_max_chunks = 5
            mock_settings.neo4j_uri = "bolt://localhost:7687"
            mock_settings.neo4j_username = "neo4j"
            mock_settings.neo4j_password = "password"

            pipeline = IngestionPipeline.__new__(IngestionPipeline)
            pipeline._enable_entity_extraction = False
            pipeline._entity_extractor = None
            pipeline._enable_citation_extraction = False
            pipeline._citation_extractor = None

            # 真实的 status_store 和 chunk_tracker（使用临时 DB）
            pipeline.status_store = DocumentStatusStore(db_path=tmp_path / "doc_status.db")
            pipeline.status_store.init_db()
            pipeline.chunk_tracker = ChunkTracker(db_path=tmp_path / "chunk_tracker.db")
            pipeline.chunk_tracker.init_db()

            # mock 其他组件
            pipeline.document_parser = MagicMock()
            pipeline.chunker = MagicMock()
            pipeline.metadata_extractor = MagicMock()
            pipeline.database = MagicMock()
            pipeline.vector_store = MagicMock()
            pipeline.relation_vector_store = MagicMock()
            pipeline.graph_store = MagicMock()

            return pipeline

    def test_skip_unchanged_document(self, tmp_path):
        """未变化文档重复入库应直接跳过。"""
        pipeline = self._make_pipeline(tmp_path)

        # 预先写入 processed 状态
        txt = _make_temp_txt("hello world")
        file_hash = compute_file_hash(txt)
        doc_id = generate_doc_id(file_hash)
        pipeline.status_store.upsert(DocumentStatus(
            file_path=str(txt),
            doc_id=doc_id,
            status="processed",
            current_step="入库完成",
            chunk_ids=["c1", "c2"],
        ))

        result = pipeline.ingest_file(txt)
        assert result["skipped"] is True
        assert result["reason"] == "already_processed"
        # 不应调用解析器
        pipeline.document_parser.parse.assert_not_called()
        txt.unlink(missing_ok=True)

    def test_changed_document_triggers_cleanup(self, tmp_path):
        """文档内容变化时应触发 _cleanup_old_document。"""
        pipeline = self._make_pipeline(tmp_path)

        txt = _make_temp_txt("version 1 content")
        old_hash = "old_fake_hash"
        old_doc_id = generate_doc_id(old_hash)

        # 写入旧状态（路径相同，doc_id 不同）
        pipeline.status_store.upsert(DocumentStatus(
            file_path=str(txt),
            doc_id=old_doc_id,
            status="processed",
            current_step="入库完成",
        ))

        # mock 解析和切块返回值
        from src.ingestion.document_parser import ParsedDocument
        from src.ingestion.chunker import TextChunk
        from src.ingestion.metadata_extractor import DocumentMetadata

        parsed = ParsedDocument(file_path=str(txt), raw_text="version 1 content", pages=[])
        pipeline.document_parser.parse.return_value = parsed

        chunk = TextChunk(chunk_id="c1", content="version 1 content", chunk_index=0,
                          metadata={"file_path": str(txt), "chunk_index": 0, "doc_id": ""})
        pipeline.chunker.chunk.return_value = [chunk]

        meta = DocumentMetadata(title="Test", authors=[], institution=None, year=None,
                                abstract=None, keywords=[])
        pipeline.metadata_extractor.extract.return_value = meta
        pipeline.database.get_document.return_value = None
        pipeline.database.add_document.return_value = 1
        pipeline.graph_store.is_document_entity_extracted.return_value = False
        pipeline.vector_store.delete_by_doc_id.return_value = 0
        pipeline.relation_vector_store.delete_by_doc_id.return_value = 0
        pipeline.graph_store.delete_document_chunks.return_value = 0
        pipeline.graph_store.get_orphan_entity_ids.return_value = []

        result = pipeline.ingest_file(txt)

        # 应调用清理
        pipeline.vector_store.delete_by_doc_id.assert_called_once_with(old_doc_id)
        pipeline.relation_vector_store.delete_by_doc_id.assert_called_once_with(old_doc_id)
        pipeline.graph_store.delete_document_chunks.assert_called_once_with(str(txt))
        assert result["skipped"] is False
        txt.unlink(missing_ok=True)

    def test_new_document_registers_chunks(self, tmp_path):
        """新文档入库后 ChunkTracker 应有记录。"""
        pipeline = self._make_pipeline(tmp_path)

        txt = _make_temp_txt("brand new document content")
        file_hash = compute_file_hash(txt)
        doc_id = generate_doc_id(file_hash)

        from src.ingestion.document_parser import ParsedDocument
        from src.ingestion.chunker import TextChunk
        from src.ingestion.metadata_extractor import DocumentMetadata

        parsed = ParsedDocument(file_path=str(txt), raw_text="brand new document content", pages=[])
        pipeline.document_parser.parse.return_value = parsed

        content_hash = compute_chunk_content_hash("brand new document content")
        chunk_id = generate_chunk_id(doc_id, content_hash)
        chunk = TextChunk(chunk_id=chunk_id, content="brand new document content",
                          chunk_index=0, metadata={"file_path": str(txt), "chunk_index": 0, "doc_id": doc_id})
        pipeline.chunker.chunk.return_value = [chunk]

        meta = DocumentMetadata(title="New Doc", authors=[], institution=None, year=None,
                                abstract=None, keywords=[])
        pipeline.metadata_extractor.extract.return_value = meta
        pipeline.database.get_document.return_value = None
        pipeline.database.add_document.return_value = 1
        pipeline.graph_store.is_document_entity_extracted.return_value = False

        pipeline.ingest_file(txt)

        # ChunkTracker 应有记录
        assert pipeline.chunk_tracker.has_doc(doc_id)
        registered_ids = pipeline.chunk_tracker.get_chunk_ids(doc_id)
        assert len(registered_ids) == 1
        txt.unlink(missing_ok=True)
