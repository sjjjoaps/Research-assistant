from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from src.storage.relation_vector_store import RelationVectorRecord, RelationVectorStore


class _FakeFAISSStore:
    def __init__(self, docs):
        self.docstore = SimpleNamespace(_dict={str(i): doc for i, doc in enumerate(docs)})

    def add_documents(self, docs):
        start = len(self.docstore._dict)
        for offset, doc in enumerate(docs):
            self.docstore._dict[str(start + offset)] = doc

    def similarity_search(self, query, k=3):
        query_lower = query.lower()
        matches = [
            doc
            for doc in self.docstore._dict.values()
            if query_lower in doc.page_content.lower()
        ]
        return matches[:k]

    def save_local(self, path):
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "index.faiss").write_text("fake-index", encoding="utf-8")
        (save_dir / "index.pkl").write_text("fake-meta", encoding="utf-8")


def _fake_from_documents(docs, _embeddings):
    return _FakeFAISSStore(docs)


def _build_record(doc_id="doc-a", relation_key="e1::USES::e2"):
    return RelationVectorRecord(
        relation_key=relation_key,
        source_entity_id="e1",
        source_name="LightRAG",
        relation_type="USES",
        target_entity_id="e2",
        target_name="GraphStore",
        description="uses graph retrieval for global reasoning",
        doc_id=doc_id,
        file_path="/tmp/paper.pdf",
    )


def test_relation_vector_record_builds_search_text():
    record = _build_record()

    assert "LightRAG" in record.content
    assert "USES" in record.content
    assert "GraphStore" in record.content
    assert "global reasoning" in record.content


def test_relation_vector_store_add_search_and_delete(monkeypatch, tmp_path):
    monkeypatch.setattr("src.storage.relation_vector_store.FAISS.from_documents", _fake_from_documents)

    store = RelationVectorStore(index_dir=tmp_path / "relation_faiss")
    store.embedder = SimpleNamespace(langchain_embeddings=object())

    record_a = _build_record(doc_id="doc-a")
    record_b = RelationVectorRecord(
        relation_key="e3::BASED_ON::e4",
        source_entity_id="e3",
        source_name="GraphRAG",
        relation_type="BASED_ON",
        target_entity_id="e4",
        target_name="Knowledge Graph",
        description="based on knowledge graph augmentation",
        doc_id="doc-b",
        file_path="/tmp/paper2.pdf",
    )

    store.add_relations([record_a, record_b])

    results = store.similarity_search("global reasoning", k=3)
    assert len(results) == 1
    assert results[0].metadata["relation_key"] == "e1::USES::e2"

    deleted = store.delete_by_doc_id("doc-a")
    assert deleted == 1
    remaining = store.get_all_documents()
    assert len(remaining) == 1
    assert remaining[0].metadata["doc_id"] == "doc-b"


def test_relation_vector_store_saves_independent_index(monkeypatch, tmp_path):
    monkeypatch.setattr("src.storage.relation_vector_store.FAISS.from_documents", _fake_from_documents)

    index_dir = tmp_path / "relation_faiss"
    store = RelationVectorStore(index_dir=index_dir)
    store.embedder = SimpleNamespace(langchain_embeddings=object())
    store.add_relations([_build_record()])

    store.save()

    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "index.pkl").exists()
