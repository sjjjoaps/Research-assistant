"""
tests/test_document_status.py

Phase 1.1 验收测试：
1. DocumentStatusStore CRUD
2. IngestionPipeline 状态流转（mock 重型依赖）
3. 失败时状态正确写入
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.storage.document_status_store import (
    DocumentStatus,
    DocumentStatusStore,
    generate_doc_id,
)
from src.storage.extraction_cache import compute_file_hash


# ── 1. DocumentStatusStore CRUD ───────────────────────────────────────────────

class TestDocumentStatusStore(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = DocumentStatusStore(db_path=Path(self.tmp.name) / "test.db")
        self.store.init_db()
        self.doc_id = "a" * 32  # 模拟 32 位 MD5

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_upsert_and_get(self):
        self.store.upsert(DocumentStatus(
            file_path="paper.pdf", doc_id=self.doc_id,
            status="pending", current_step="等待入库",
        ))
        s = self.store.get(self.doc_id)
        self.assertIsNotNone(s)
        self.assertEqual(s.status, "pending")
        self.assertEqual(s.file_path, "paper.pdf")

    def test_upsert_updates_existing(self):
        self.store.upsert(DocumentStatus(
            file_path="paper.pdf", doc_id=self.doc_id,
            status="pending", current_step="等待入库",
        ))
        self.store.upsert(DocumentStatus(
            file_path="paper.pdf", doc_id=self.doc_id,
            status="parsing", current_step="解析文档",
        ))
        s = self.store.get(self.doc_id)
        self.assertEqual(s.status, "parsing")

    def test_get_by_path(self):
        self.store.upsert(DocumentStatus(
            file_path="paper.pdf", doc_id=self.doc_id,
            status="processed", current_step="入库完成",
        ))
        s = self.store.get_by_path("paper.pdf")
        self.assertIsNotNone(s)
        self.assertEqual(s.doc_id, self.doc_id)

    def test_mark_failed_existing_record(self):
        self.store.upsert(DocumentStatus(
            file_path="paper.pdf", doc_id=self.doc_id,
            status="parsing", current_step="解析文档",
        ))
        self.store.mark_failed(self.doc_id, "parsing", "文件损坏")
        s = self.store.get(self.doc_id)
        self.assertEqual(s.status, "failed")
        self.assertEqual(s.error_message, "文件损坏")
        self.assertEqual(s.current_step, "parsing")

    def test_mark_failed_creates_missing_record(self):
        """早期失败（记录尚未创建）时应自动补建。"""
        new_id = "b" * 32
        self.store.mark_failed(new_id, "初始化", "磁盘满", file_path="new.pdf")
        s = self.store.get(new_id)
        self.assertIsNotNone(s)
        self.assertEqual(s.status, "failed")
        self.assertEqual(s.file_path, "new.pdf")

    def test_delete(self):
        self.store.upsert(DocumentStatus(
            file_path="paper.pdf", doc_id=self.doc_id,
            status="processed", current_step="入库完成",
        ))
        self.assertTrue(self.store.delete(self.doc_id))
        self.assertIsNone(self.store.get(self.doc_id))
        self.assertFalse(self.store.delete(self.doc_id))

    def test_list_by_status(self):
        for i in range(3):
            self.store.upsert(DocumentStatus(
                file_path=f"p{i}.pdf", doc_id=f"{'c' * 31}{i}",
                status="processed", current_step="入库完成",
            ))
        self.store.upsert(DocumentStatus(
            file_path="fail.pdf", doc_id="d" * 32,
            status="failed", current_step="parsing",
        ))
        processed = self.store.list_by_status("processed")
        failed = self.store.list_by_status("failed")
        self.assertEqual(len(processed), 3)
        self.assertEqual(len(failed), 1)

    def test_invalid_status_raises(self):
        with self.assertRaises(ValueError):
            self.store.upsert(DocumentStatus(
                file_path="p.pdf", doc_id=self.doc_id,
                status="unknown_state", current_step="",
            ))


# ── 2. generate_doc_id 内容身份 ───────────────────────────────────────────────

class TestGenerateDocId(unittest.TestCase):

    def test_same_hash_same_id(self):
        fh = "abc123" * 5 + "ab"  # 32 chars
        self.assertEqual(generate_doc_id(fh), generate_doc_id(fh))

    def test_doc_id_equals_file_hash(self):
        fh = "f" * 32
        self.assertEqual(generate_doc_id(fh), fh)

    def test_different_hash_different_id(self):
        self.assertNotEqual(generate_doc_id("a" * 32), generate_doc_id("b" * 32))


# ── 3. IngestionPipeline 状态流转（mock 重型依赖）────────────────────────────

class TestIngestionPipelineStatus(unittest.TestCase):
    """
    mock 掉所有 I/O 和 LLM 调用，只验证状态机流转是否正确。
    """

    def _make_pipeline(self, tmp_dir: str):
        """构造一个所有重型依赖都被 mock 的 pipeline。"""
        with patch("src.ingestion_pipeline.DocumentParser"), \
             patch("src.ingestion_pipeline.DocumentChunker"), \
             patch("src.ingestion_pipeline.MetadataExtractor"), \
             patch("src.ingestion_pipeline.MetadataDatabase") as MockDB, \
             patch("src.ingestion_pipeline.VectorStore"), \
             patch("src.ingestion_pipeline.GraphStore"), \
             patch("src.ingestion_pipeline.EntityExtractor"):

            from src.ingestion_pipeline import IngestionPipeline

            pipeline = IngestionPipeline(enable_entity_extraction=False)

            # 替换 status_store 为指向临时目录的真实实例
            pipeline.status_store.close()
            pipeline.status_store = DocumentStatusStore(
                db_path=Path(tmp_dir) / "status.db"
            )
            pipeline.status_store.init_db()

            return pipeline

    def test_successful_ingest_reaches_processed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_pdf = Path(tmp) / "paper.txt"
            fake_pdf.write_text("hello world")

            pipeline = self._make_pipeline(tmp)
            try:
                pipeline.document_parser.parse.return_value = MagicMock()
                mock_chunk_a = MagicMock()
                mock_chunk_a.content = "chunk content a"
                mock_chunk_a.metadata = {"content_type": "text"}
                mock_chunk_b = MagicMock()
                mock_chunk_b.content = "chunk content b"
                mock_chunk_b.metadata = {"content_type": "text"}
                pipeline.chunker.chunk.return_value = [mock_chunk_a, mock_chunk_b]
                mock_meta = MagicMock()
                mock_meta.title = "Test Paper"
                mock_meta.authors = ["Author A"]
                mock_meta.keywords = ["kw1"]
                mock_meta.institution = None
                mock_meta.year = 2024
                mock_meta.abstract = "abstract"
                pipeline.metadata_extractor.extract.return_value = mock_meta
                pipeline.database.get_document.return_value = None
                pipeline.database.add_document.return_value = 1
                pipeline.graph_store.add_document_with_chunks.return_value = None

                result = pipeline.ingest_file(fake_pdf)

                self.assertIn("doc_id", result)
                self.assertEqual(result["chunk_count"], 2)

                s = pipeline.status_store.get(result["doc_id"])
                self.assertIsNotNone(s)
                self.assertEqual(s.status, "processed")
                self.assertIsNone(s.error_message)
            finally:
                pipeline.status_store.close()

    def test_failed_ingest_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_pdf = Path(tmp) / "paper.txt"
            fake_pdf.write_text("hello world")

            pipeline = self._make_pipeline(tmp)
            try:
                pipeline.document_parser.parse.side_effect = RuntimeError("解析器崩溃")

                with self.assertRaises(RuntimeError):
                    pipeline.ingest_file(fake_pdf)

                file_hash = compute_file_hash(fake_pdf)
                doc_id = generate_doc_id(file_hash)
                s = pipeline.status_store.get(doc_id)
                self.assertIsNotNone(s)
                self.assertEqual(s.status, "failed")
                self.assertEqual(s.current_step, "parsing")
                self.assertIn("解析器崩溃", s.error_message)
            finally:
                pipeline.status_store.close()


if __name__ == "__main__":
    unittest.main()
