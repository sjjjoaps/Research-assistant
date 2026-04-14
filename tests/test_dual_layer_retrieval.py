from types import SimpleNamespace

from src.retriever import RetrievedChunk


def test_local_retriever_fuses_graph_and_semantic(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval.local_retriever.GraphRetriever",
        lambda top_k, expand_entities=True: SimpleNamespace(
            retrieve=lambda query: [
                RetrievedChunk(content="graph evidence", file_path="a.pdf", chunk_index=1)
            ],
            close=lambda: None,
        ),
    )
    monkeypatch.setattr(
        "src.retrieval.local_retriever.SemanticRetriever",
        lambda top_k: SimpleNamespace(
            retrieve=lambda query: [
                RetrievedChunk(content="semantic evidence", file_path="b.pdf", chunk_index=2)
            ]
        ),
    )

    from src.retrieval.local_retriever import LocalRetriever

    results = LocalRetriever(top_k=3).retrieve("LightRAG uses graph")

    assert len(results) == 2
    assert any(item.content == "graph evidence" for item in results)
    assert any(item.content == "semantic evidence" for item in results)


def test_global_retriever_uses_relation_index_and_graph(monkeypatch):
    relation_docs = [
        SimpleNamespace(
            page_content="ignored",
            metadata={
                "source_name": "LightRAG",
                "relation_type": "USES",
                "target_name": "GraphStore",
                "description": "global reasoning",
                "file_path": "paper.pdf",
            },
        )
    ]
    monkeypatch.setattr(
        "src.retrieval.global_retriever.RelationVectorStore",
        lambda: SimpleNamespace(
            load=lambda: None,
            similarity_search=lambda query, k=10: relation_docs,
        ),
    )
    monkeypatch.setattr(
        "src.retrieval.global_retriever.GraphRetriever",
        lambda top_k, expand_entities=True: SimpleNamespace(
            retrieve=lambda query: [
                RetrievedChunk(content="graph relation evidence", file_path="c.pdf", chunk_index=3)
            ],
            close=lambda: None,
        ),
    )
    monkeypatch.setattr(
        "src.retrieval.global_retriever.KeywordExtractor",
        lambda: SimpleNamespace(
            extract=lambda query: SimpleNamespace(ll_keywords=["LightRAG"], hl_keywords=["研究趋势"])
        ),
    )

    from src.retrieval.global_retriever import GlobalRetriever

    results = GlobalRetriever(top_k=3).retrieve("LightRAG研究趋势")

    assert len(results) == 2
    assert any(item.section_type == "relation" for item in results)
    assert any("关系类型: USES" in item.content for item in results)


def test_mix_retriever_fuses_semantic_local_global(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval.mix_retriever.SemanticRetriever",
        lambda top_k: SimpleNamespace(
            retrieve=lambda query: [RetrievedChunk(content="semantic", file_path="s.pdf", chunk_index=1)]
        ),
    )
    monkeypatch.setattr(
        "src.retrieval.mix_retriever.LocalRetriever",
        lambda top_k: SimpleNamespace(
            retrieve=lambda query: [RetrievedChunk(content="local", file_path="l.pdf", chunk_index=2)],
            close=lambda: None,
        ),
    )
    monkeypatch.setattr(
        "src.retrieval.mix_retriever.GlobalRetriever",
        lambda top_k: SimpleNamespace(
            retrieve=lambda query: [RetrievedChunk(content="global", file_path="g.pdf", chunk_index=3)],
            close=lambda: None,
        ),
    )

    from src.retrieval.mix_retriever import MixRetriever

    results = MixRetriever(top_k=5).retrieve("overview")

    assert len(results) == 3
    contents = {item.content for item in results}
    assert contents == {"semantic", "local", "global"}
