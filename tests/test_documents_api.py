from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.documents import router
from src.storage.document_status_store import DocumentStatus


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _make_workspace_temp_file(name: str) -> Path:
    temp_dir = Path(".tmp_phase53_tests")
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / name
    file_path.write_text("hello", encoding="utf-8")
    return file_path


def test_list_documents_includes_status_details():
    client = _client()
    record = SimpleNamespace(
        id=1,
        file_path="C:/papers/a.pdf",
        doc_id="doc-1",
        title="Paper A",
        authors="Alice",
        institution="Lab",
        year=2024,
        abstract="abstract",
        keywords="graph,rag",
    )
    status = DocumentStatus(
        file_path=record.file_path,
        doc_id="doc-1",
        status="extracting",
        current_step="实体抽取",
        chunk_ids=["c1", "c2"],
        entity_ids=["e1"],
        relation_ids=["r1", "r2", "r3"],
        error_message="最近一次抽取超时",
        updated_at="2026-04-14T10:00:00+00:00",
    )

    with patch("api.routers.documents.MetadataDatabase") as MockDB, \
         patch("api.routers.documents._get_status_store") as mock_get_status_store:
        db = MagicMock()
        db.list_documents.return_value = [record]
        MockDB.return_value = db

        status_store = MagicMock()
        status_store.get.return_value = status
        mock_get_status_store.return_value = status_store

        resp = client.get("/documents")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    item = data[0]
    assert item["status"] == "extracting"
    assert item["current_step"] == "实体抽取"
    assert item["chunk_count"] == 2
    assert item["entity_count"] == 1
    assert item["relation_count"] == 3
    assert item["error_message"] == "最近一次抽取超时"
    assert item["updated_at"] == "2026-04-14T10:00:00+00:00"
    db.close.assert_called_once()
    status_store.close.assert_called_once()


def test_list_documents_defaults_status_details_when_missing():
    client = _client()
    record = SimpleNamespace(
        id=2,
        file_path="C:/papers/b.pdf",
        doc_id=None,
        title="Paper B",
        authors="Bob",
        institution=None,
        year=None,
        abstract=None,
        keywords="",
    )

    with patch("api.routers.documents.MetadataDatabase") as MockDB, \
         patch("api.routers.documents._get_status_store") as mock_get_status_store:
        db = MagicMock()
        db.list_documents.return_value = [record]
        MockDB.return_value = db

        status_store = MagicMock()
        mock_get_status_store.return_value = status_store

        resp = client.get("/documents")

    assert resp.status_code == 200
    item = resp.json()[0]
    assert item["status"] is None
    assert item["current_step"] is None
    assert item["chunk_count"] == 0
    assert item["entity_count"] == 0
    assert item["relation_count"] == 0
    assert item["error_message"] is None
    assert item["updated_at"] is None
    status_store.get.assert_not_called()
    status_store.close.assert_called_once()


def test_start_ingest_file_returns_doc_id_and_initial_status():
    client = _client()
    file_path = _make_workspace_temp_file("start_ingest.txt")

    with patch("api.routers.documents.compute_file_hash", return_value="a" * 32), \
         patch("api.routers.documents._get_status_store") as mock_get_status_store, \
         patch("api.routers.documents.IngestionPipeline") as MockPipeline:
        status_store = MagicMock()
        status_store.get.return_value = None
        mock_get_status_store.return_value = status_store

        resp = client.post(
            "/documents/ingest-file/start",
            json={"file_path": str(file_path), "enable_entity_extraction": True},
        )

    assert resp.status_code == 202
    data = resp.json()
    assert data["doc_id"] == "a" * 32
    assert data["accepted"] is True
    assert data["status"] == "pending"
    assert data["current_step"] == "等待入库"
    status_store.upsert.assert_called_once()
    MockPipeline.return_value.ingest_file.assert_called_once_with(str(file_path))
    MockPipeline.return_value.close.assert_called_once()
    status_store.close.assert_called_once()


def test_start_ingest_file_does_not_restart_running_job():
    client = _client()
    file_path = _make_workspace_temp_file("running_ingest.txt")
    existing_status = DocumentStatus(
        file_path=str(file_path),
        doc_id="b" * 32,
        status="chunking",
        current_step="文本切块",
    )

    with patch("api.routers.documents.compute_file_hash", return_value="b" * 32), \
         patch("api.routers.documents._get_status_store") as mock_get_status_store, \
         patch("api.routers.documents.IngestionPipeline") as MockPipeline:
        status_store = MagicMock()
        status_store.get.return_value = existing_status
        mock_get_status_store.return_value = status_store

        resp = client.post(
            "/documents/ingest-file/start",
            json={"file_path": str(file_path), "enable_entity_extraction": False},
        )

    assert resp.status_code == 202
    data = resp.json()
    assert data["accepted"] is False
    assert data["status"] == "chunking"
    assert data["current_step"] == "文本切块"
    status_store.upsert.assert_not_called()
    MockPipeline.return_value.ingest_file.assert_not_called()
    status_store.close.assert_called_once()
