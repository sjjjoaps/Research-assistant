from types import SimpleNamespace

from src.retrieval.keyword_extractor import KeywordExtractor, KeywordResult


def test_extract_low_level_keywords_from_specific_query():
    extractor = KeywordExtractor()

    result = extractor.extract("CLIP和GraphSAGE在OGB-Arxiv数据集上的表现差异是什么？")

    assert "CLIP" in result.ll_keywords
    assert "GraphSAGE" in result.ll_keywords
    assert "OGB-Arxiv" in result.ll_keywords
    assert result.hl_keywords == []


def test_extract_high_level_keywords_from_macro_query():
    extractor = KeywordExtractor()

    result = extractor.extract("多模态RAG研究趋势和应用挑战有哪些？")

    assert "多模态RAG" in result.ll_keywords
    assert "研究趋势" in result.hl_keywords
    assert "应用挑战" in result.hl_keywords


def test_llm_fallback_when_rule_extraction_returns_empty(monkeypatch):
    extractor = KeywordExtractor()
    monkeypatch.setattr(extractor, "_rule_extract", lambda query: KeywordResult(raw_query=query))
    monkeypatch.setattr(
        extractor,
        "_get_llm",
        lambda: SimpleNamespace(
            invoke=lambda _: {"content": '{"ll_keywords": ["LightRAG"], "hl_keywords": ["研究方向"]}'}
        ),
    )

    result = extractor.extract("a")

    assert result.ll_keywords == ["LightRAG"]
    assert result.hl_keywords == ["研究方向"]


def test_llm_failure_returns_empty_result(monkeypatch):
    extractor = KeywordExtractor()
    monkeypatch.setattr(extractor, "_rule_extract", lambda query: KeywordResult(raw_query=query))

    def _raise(_prompt):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        extractor,
        "_get_llm",
        lambda: SimpleNamespace(invoke=_raise),
    )

    result = extractor.extract("x")

    assert result.ll_keywords == []
    assert result.hl_keywords == []
