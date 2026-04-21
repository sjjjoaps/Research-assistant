"""
Phase 2.4 精确文档删除测试

验收标准：
- 删除后该文档 chunk 无法再检索到
- 共享实体仍保留
- 独占实体被清理
- doc_id 不存在时抛出 KeyError
- 删除中途失败时状态变为 delete_failed
- Document 节点被删除
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.storage.chunk_tracker import ChunkTracker, compute_chunk_content_hash
from src.storage.document_status_store import DocumentStatus, DocumentStatusStore


def _build_pipeline(tmp_path: Path):
    """构造带真实 status_store / chunk_tracker 的 pipeline，其余 mock。"""
    from src.workflows.ingestion_pipeline import IngestionPipeline

    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline._enable_entity_extraction = False
    pipeline._entity_extractor = None

    pipeline.status_store = DocumentStatusStore(db_path=tmp_path / "doc_status.db")
    pipeline.status_store.init_db()
    pipeline.chunk_tracker = ChunkTracker(db_path=tmp_path / "chunk_tracker.db")
    pipeline.chunk_tracker.init_db()

    pipeline.document_parser = MagicMock()
    pipeline.chunker = MagicMock()
    pipeline.metadata_extractor = MagicMock()
    pipeline.database = MagicMock()
    pipeline.vector_store = MagicMock()
    pipeline.relation_vector_store = MagicMock()
    pipeline.graph_store = MagicMock()

    # 默认 mock 返回值
    pipeline.vector_store.delete_by_doc_id.return_value = 2
    pipeline.relation_vector_store.delete_by_doc_id.return_value = 1
    pipeline.graph_store.delete_document_chunks.return_value = 2
    pipeline.graph_store.delete_document_node.return_value = True
    pipeline.graph_store.delete_stale_relations.return_value = 0
    pipeline.graph_store.get_orphan_entity_ids.return_value = []
    pipeline.graph_store.delete_entities_by_ids.return_value = 0

    return pipeline


class TestDeleteDocument:
    def test_delete_removes_status_and_chunks(self, tmp_path):
        """删除后 status_store 和 chunk_tracker 中不再有该文档记录。"""
        pipeline = _build_pipeline(tmp_path)
        doc_id = "abc123"

        # 预置状态和 chunk 记录
        pipeline.status_store.upsert(DocumentStatus(
            file_path="/tmp/paper.pdf",
            doc_id=doc_id,
            status="processed",
            current_step="入库完成",
            chunk_ids=["c1", "c2"],
        ))
        pipeline.chunk_tracker.register_chunks(
            doc_id=doc_id,
            content_hashes=[compute_chunk_content_hash("chunk a"), compute_chunk_content_hash("chunk b")],
        )

        result = pipeline.delete_document(doc_id)

        assert pipeline.status_store.get(doc_id) is None
        assert pipeline.chunk_tracker.get_chunk_ids(doc_id) == []
        assert result["doc_id"] == doc_id
        assert result["deleted_vectors"] == 2
        assert result["deleted_relation_vectors"] == 1
        assert result["deleted_chunks"] == 2

    def test_delete_calls_faiss_and_neo4j(self, tmp_path):
        """删除时应依次调用 FAISS 删除、Neo4j chunk 删除、Document 节点删除、关系清理、实体清理。"""
        pipeline = _build_pipeline(tmp_path)
        doc_id = "def456"

        pipeline.status_store.upsert(DocumentStatus(
            file_path="/tmp/paper2.pdf",
            doc_id=doc_id,
            status="processed",
            current_step="入库完成",
        ))

        pipeline.delete_document(doc_id)

        pipeline.vector_store.delete_by_doc_id.assert_called_once_with(doc_id)
        pipeline.vector_store.save.assert_called_once()
        pipeline.relation_vector_store.delete_by_doc_id.assert_called_once_with(doc_id)
        pipeline.relation_vector_store.save.assert_called_once()
        pipeline.graph_store.delete_document_chunks.assert_called_once_with("/tmp/paper2.pdf")
        pipeline.graph_store.delete_document_node.assert_called_once_with("/tmp/paper2.pdf")
        pipeline.graph_store.delete_stale_relations.assert_called_once()
        pipeline.graph_store.get_orphan_entity_ids.assert_called_once()

    def test_delete_cleans_orphan_entities(self, tmp_path):
        """有孤立实体时应调用 delete_entities_by_ids。"""
        pipeline = _build_pipeline(tmp_path)
        doc_id = "ghi789"
        orphan_ids = ["e1", "e2", "e3"]
        pipeline.graph_store.get_orphan_entity_ids.return_value = orphan_ids
        pipeline.graph_store.delete_entities_by_ids.return_value = 3

        pipeline.status_store.upsert(DocumentStatus(
            file_path="/tmp/paper3.pdf",
            doc_id=doc_id,
            status="processed",
            current_step="入库完成",
        ))

        result = pipeline.delete_document(doc_id)

        pipeline.graph_store.delete_entities_by_ids.assert_called_once_with(orphan_ids)
        assert result["deleted_entities"] == 3

    def test_delete_preserves_shared_entities(self, tmp_path):
        """无孤立实体时不调用 delete_entities_by_ids（共享实体保留）。"""
        pipeline = _build_pipeline(tmp_path)
        doc_id = "jkl000"
        pipeline.graph_store.get_orphan_entity_ids.return_value = []

        pipeline.status_store.upsert(DocumentStatus(
            file_path="/tmp/shared.pdf",
            doc_id=doc_id,
            status="processed",
            current_step="入库完成",
        ))

        result = pipeline.delete_document(doc_id)

        pipeline.graph_store.delete_entities_by_ids.assert_not_called()
        assert result["deleted_entities"] == 0

    def test_delete_nonexistent_raises_key_error(self, tmp_path):
        """doc_id 不存在时应抛出 KeyError。"""
        pipeline = _build_pipeline(tmp_path)

        with pytest.raises(KeyError, match="missing_doc"):
            pipeline.delete_document("missing_doc")

    def test_delete_removes_sqlite_record(self, tmp_path):
        """删除时应调用 database.delete_document。"""
        pipeline = _build_pipeline(tmp_path)
        doc_id = "mno111"

        pipeline.status_store.upsert(DocumentStatus(
            file_path="/tmp/meta.pdf",
            doc_id=doc_id,
            status="processed",
            current_step="入库完成",
        ))

        pipeline.delete_document(doc_id)

        pipeline.database.delete_document.assert_called_once_with("/tmp/meta.pdf")

    def test_delete_sets_deleting_state_then_removes(self, tmp_path):
        """删除成功后状态记录应被完全移除（不留 deleting 状态）。"""
        pipeline = _build_pipeline(tmp_path)
        doc_id = "pqr222"

        pipeline.status_store.upsert(DocumentStatus(
            file_path="/tmp/state_test.pdf",
            doc_id=doc_id,
            status="processed",
            current_step="入库完成",
        ))

        pipeline.delete_document(doc_id)

        # 成功删除后状态记录应被清除
        assert pipeline.status_store.get(doc_id) is None

    def test_delete_failure_sets_delete_failed_state(self, tmp_path):
        """删除中途失败时，状态应变为 delete_failed 而非丢失记录。"""
        pipeline = _build_pipeline(tmp_path)
        doc_id = "stu333"

        pipeline.status_store.upsert(DocumentStatus(
            file_path="/tmp/fail_test.pdf",
            doc_id=doc_id,
            status="processed",
            current_step="入库完成",
        ))

        # 模拟 Neo4j 删除 chunk 时抛出异常
        pipeline.graph_store.delete_document_chunks.side_effect = RuntimeError("Neo4j connection lost")

        with pytest.raises(RuntimeError, match="Neo4j connection lost"):
            pipeline.delete_document(doc_id)

        # 状态应变为 delete_failed，记录仍存在
        s = pipeline.status_store.get(doc_id)
        assert s is not None
        assert s.status == "delete_failed"
        assert "Neo4j connection lost" in (s.error_message or "")


class TestDeleteDocumentAPI:
    """测试 API 层的路由行为（不依赖真实 pipeline）。"""

    def test_delete_returns_404_for_missing_doc(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from api.routers.documents import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        with patch("api.routers.documents.IngestionPipeline") as MockPipeline:
            mock_inst = MagicMock()
            mock_inst.delete_document.side_effect = KeyError("not_found")
            MockPipeline.return_value = mock_inst

            resp = client.delete("/documents/not_found")
            assert resp.status_code == 404
            # 确认 detail 不含多余引号
            assert resp.json()["detail"] == "not_found"

    def test_delete_returns_200_on_success(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from api.routers.documents import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        expected = {
            "doc_id": "abc123",
            "file_path": "/tmp/paper.pdf",
            "deleted_vectors": 3,
            "deleted_relation_vectors": 1,
            "deleted_chunks": 3,
            "deleted_relations": 1,
            "deleted_entities": 2,
        }

        with patch("api.routers.documents.IngestionPipeline") as MockPipeline:
            mock_inst = MagicMock()
            mock_inst.delete_document.return_value = expected
            MockPipeline.return_value = mock_inst

            resp = client.delete("/documents/abc123")
            assert resp.status_code == 200
            assert resp.json()["deleted_chunks"] == 3
            assert resp.json()["deleted_entities"] == 2
