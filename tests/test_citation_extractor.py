"""
Phase 3.3 引用提取测试

验收标准：
- CitationRecord.ref_id 为 MD5 哈希，相同 raw_text 产生相同 ID
- _find_reference_section 能定位 References / Bibliography / 参考文献章节
- _split_references 支持编号格式和双换行兜底
- _parse_reference_rule 提取 DOI / 年份 / 引号标题
- _parse_reference_llm LLM 成功/失败均不抛异常
- extract 完整流程：无参考文献章节返回空列表
- LLM 仅对无标题条目调用
- CitationExtractor 实例化不触发 LLM 初始化（懒初始化）
- GraphStore citation 方法（create_reference_node / create_cites_relation /
  get_document_citations / delete_document_citations）
- IngestionPipeline enable_citation_extraction=False 不调用提取器
- IngestionPipeline enable_citation_extraction=True 调用并写入图
- API 404 for unknown doc_id / 200 with citation list
"""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

from src.ingestion.citation_extractor import CitationExtractor, CitationRecord


# ── CitationRecord ────────────────────────────────────────────────────────────

class TestCitationRecord:
    def test_ref_id_is_md5_of_raw_text(self):
        raw = "[1] Smith et al., 2020. A great paper."
        record = CitationRecord(raw_text=raw, title="", authors="", year="", doi="")
        expected = hashlib.md5(raw.encode("utf-8")).hexdigest()
        assert record.ref_id == expected

    def test_same_raw_text_same_ref_id(self):
        raw = "[2] Jones, 2019. Another paper."
        r1 = CitationRecord(raw_text=raw, title="", authors="", year="", doi="")
        r2 = CitationRecord(raw_text=raw, title="", authors="", year="", doi="")
        assert r1.ref_id == r2.ref_id

    def test_explicit_ref_id_not_overwritten(self):
        record = CitationRecord(
            raw_text="text", title="", authors="", year="", doi="", ref_id="custom_id"
        )
        assert record.ref_id == "custom_id"


# ── _find_reference_section ───────────────────────────────────────────────────

class TestFindReferenceSection:
    def _extractor(self) -> CitationExtractor:
        return CitationExtractor.__new__(CitationExtractor)

    def test_finds_references_section(self):
        text = "Introduction\nSome text.\n\nReferences\n[1] Smith 2020"
        e = self._extractor()
        section = e._find_reference_section(text)
        assert section is not None
        assert "[1] Smith 2020" in section

    def test_finds_bibliography_section(self):
        text = "Body text.\n\nBibliography\n[1] Jones 2019"
        e = self._extractor()
        section = e._find_reference_section(text)
        assert section is not None
        assert "Jones 2019" in section

    def test_finds_chinese_reference_section(self):
        text = "正文内容。\n\n参考文献\n[1] 张三 2021"
        e = self._extractor()
        section = e._find_reference_section(text)
        assert section is not None
        assert "张三 2021" in section

    def test_returns_none_when_no_reference_section(self):
        text = "Introduction\nSome text.\n\nConclusion\nFinal words."
        e = self._extractor()
        assert e._find_reference_section(text) is None


# ── _split_references ─────────────────────────────────────────────────────────

class TestSplitReferences:
    def _extractor(self) -> CitationExtractor:
        return CitationExtractor.__new__(CitationExtractor)

    def test_numbered_bracket_format(self):
        section = (
            "References\n"
            "\n[1] Smith et al., 2020. First paper.\n"
            "\n[2] Jones, 2019. Second paper.\n"
            "\n[3] Lee, 2018. Third paper."
        )
        e = self._extractor()
        refs = e._split_references(section)
        assert len(refs) >= 2

    def test_numbered_dot_format(self):
        section = (
            "References\n"
            "\n1. Smith et al., 2020. First paper.\n"
            "\n2. Jones, 2019. Second paper."
        )
        e = self._extractor()
        refs = e._split_references(section)
        assert len(refs) >= 2

    def test_double_newline_fallback(self):
        section = (
            "References\n\n"
            "Smith et al., 2020. A paper about things.\n\n"
            "Jones, 2019. Another paper about stuff."
        )
        e = self._extractor()
        refs = e._split_references(section)
        assert len(refs) >= 2

    def test_empty_section_returns_empty(self):
        e = self._extractor()
        refs = e._split_references("")
        assert refs == []


# ── _parse_reference_rule ─────────────────────────────────────────────────────

class TestParseReferenceRule:
    def _extractor(self) -> CitationExtractor:
        return CitationExtractor.__new__(CitationExtractor)

    def test_extracts_doi(self):
        ref = '[1] Smith, 2020. doi:10.1234/abc.def'
        e = self._extractor()
        record = e._parse_reference_rule(ref)
        assert record.doi == "10.1234/abc.def"

    def test_extracts_year(self):
        ref = '[2] Jones, 2019. Some paper title.'
        e = self._extractor()
        record = e._parse_reference_rule(ref)
        assert record.year == "2019"

    def test_extracts_quoted_title(self):
        ref = '[3] Lee, 2018. "Deep Learning for NLP". Journal of AI.'
        e = self._extractor()
        record = e._parse_reference_rule(ref)
        assert record.title == "Deep Learning for NLP"

    def test_no_info_returns_empty_fields(self):
        ref = 'Some random text without any structured info here at all'
        e = self._extractor()
        record = e._parse_reference_rule(ref)
        assert record.title == ""
        assert record.doi == ""
        assert record.year == ""
        assert record.raw_text == ref


# ── _parse_reference_llm ──────────────────────────────────────────────────────

class TestParseReferenceLLM:
    def _extractor_with_mock_llm(self, response_content: str) -> CitationExtractor:
        e = CitationExtractor.__new__(CitationExtractor)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=response_content)
        e._llm = mock_llm
        return e

    def test_llm_success_returns_structured_record(self):
        json_resp = '{"title": "A Great Paper", "authors": "Smith, J.", "year": "2020", "doi": "10.1234/x"}'
        e = self._extractor_with_mock_llm(json_resp)
        record = e._parse_reference_llm("Smith, J. (2020). A Great Paper.")
        assert record.title == "A Great Paper"
        assert record.authors == "Smith, J."
        assert record.year == "2020"
        assert record.doi == "10.1234/x"

    def test_llm_json_in_markdown_block(self):
        json_resp = '```json\n{"title": "Paper", "authors": "", "year": "2021", "doi": ""}\n```'
        e = self._extractor_with_mock_llm(json_resp)
        record = e._parse_reference_llm("Some ref text")
        assert record.title == "Paper"

    def test_llm_invalid_json_falls_back_gracefully(self):
        e = self._extractor_with_mock_llm("not valid json at all")
        record = e._parse_reference_llm("Some ref text")
        assert record.title == ""
        assert record.raw_text == "Some ref text"

    def test_llm_exception_falls_back_gracefully(self):
        e = CitationExtractor.__new__(CitationExtractor)
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API error")
        e._llm = mock_llm
        record = e._parse_reference_llm("Some ref text")
        assert record.title == ""
        assert record.raw_text == "Some ref text"


# ── CitationExtractor.extract ─────────────────────────────────────────────────

class TestCitationExtractorExtract:
    def test_no_reference_section_returns_empty(self):
        e = CitationExtractor.__new__(CitationExtractor)
        e._llm = None
        result = e.extract("Introduction\nSome text.\n\nConclusion\nFinal words.")
        assert result == []

    def test_extract_returns_citation_records(self):
        text = (
            "Introduction\nSome text.\n\n"
            "References\n"
            "\n[1] Smith, 2020. \"Deep Learning\". Journal of AI. doi:10.1234/dl\n"
            "\n[2] Jones, 2019. \"NLP Methods\". ACL. doi:10.5678/nlp"
        )
        e = CitationExtractor.__new__(CitationExtractor)
        e._llm = None
        results = e.extract(text)
        assert len(results) >= 2
        years = {r.year for r in results}
        assert "2020" in years or "2019" in years

    def test_llm_called_only_for_records_without_title(self):
        """规则能提取标题的条目不应触发 LLM 调用。"""
        text = (
            "References\n"
            "\n[1] Smith, 2020. \"Quoted Title Paper\". doi:10.1234/x\n"
            "\n[2] Jones, 2019. No quoted title here, just plain text."
        )
        e = CitationExtractor.__new__(CitationExtractor)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"title": "Inferred Title", "authors": "", "year": "2019", "doi": ""}'
        )
        e._llm = mock_llm

        results = e.extract(text)
        # LLM 只应被调用一次（第二条无引号标题）
        assert mock_llm.invoke.call_count == 1
        # 第一条有引号标题，不应调用 LLM
        titled = [r for r in results if r.title == "Quoted Title Paper"]
        assert len(titled) == 1

    def test_lazy_init_no_llm_on_instantiation(self):
        """实例化 CitationExtractor 不应触发 LLM 初始化。"""
        with patch("src.llm_client.get_llm") as mock_get_llm:
            e = CitationExtractor()
            mock_get_llm.assert_not_called()
            assert e._llm is None


# ── GraphStore citation 方法 ──────────────────────────────────────────────────

class TestGraphStoreCitations:
    def _make_graph_store(self):
        from src.graph_store import GraphStore
        gs = GraphStore.__new__(GraphStore)
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        gs.driver = mock_driver
        return gs, mock_session

    def test_create_reference_node_runs_merge(self):
        gs, session = self._make_graph_store()
        ref = CitationRecord(
            raw_text="Smith 2020", title="A Paper", authors="Smith", year="2020", doi="10.1/x"
        )
        gs.create_reference_node(ref)
        session.run.assert_called_once()
        call_args = session.run.call_args
        assert "MERGE" in call_args[0][0]

    def test_create_cites_relation_runs_merge(self):
        gs, session = self._make_graph_store()
        gs.create_cites_relation("/path/doc.pdf", "abc123")
        session.run.assert_called_once()
        call_args = session.run.call_args
        assert "CITES" in call_args[0][0]

    def test_get_document_citations_returns_list(self):
        gs, session = self._make_graph_store()
        mock_record = {
            "ref_id": "abc", "title": "T", "authors": "A",
            "year": "2020", "doi": "", "raw_text": "raw"
        }
        session.run.return_value = [mock_record]
        result = gs.get_document_citations("/path/doc.pdf")
        assert isinstance(result, list)
        assert result[0]["ref_id"] == "abc"

    def test_delete_document_citations_returns_count(self):
        gs, session = self._make_graph_store()
        mock_result = MagicMock()
        mock_result.single.return_value = {"deleted": 3}
        session.run.return_value = mock_result
        count = gs.delete_document_citations("/path/doc.pdf")
        assert isinstance(count, int)


# ── IngestionPipeline 引用提取集成 ────────────────────────────────────────────

class TestIngestionPipelineCitations:
    def test_citation_extractor_none_when_disabled(self):
        from src.ingestion_pipeline import IngestionPipeline
        pipeline = IngestionPipeline.__new__(IngestionPipeline)
        pipeline._enable_citation_extraction = False
        pipeline._citation_extractor = None
        assert pipeline._citation_extractor is None

    def test_citation_extractor_created_when_enabled(self):
        from src.ingestion_pipeline import IngestionPipeline
        pipeline = IngestionPipeline.__new__(IngestionPipeline)
        pipeline._enable_citation_extraction = True
        pipeline._citation_extractor = CitationExtractor()
        assert pipeline._citation_extractor is not None
        assert isinstance(pipeline._citation_extractor, CitationExtractor)


# ── Citations API ─────────────────────────────────────────────────────────────

class TestCitationsAPI:
    def _make_client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.routers.documents import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_404_for_unknown_doc_id(self):
        client = self._make_client()
        with patch("api.routers.documents._get_status_store") as mock_store_fn:
            mock_store = MagicMock()
            mock_store.get.return_value = None
            mock_store_fn.return_value = mock_store
            response = client.get("/documents/nonexistent_doc_id/citations")
        assert response.status_code == 404

    def test_200_with_citation_list(self):
        from src.storage.document_status_store import DocumentStatus
        client = self._make_client()

        mock_status = DocumentStatus(
            file_path="/tmp/paper.pdf",
            doc_id="test_doc_id",
            status="processed",
            current_step="入库完成",
        )
        mock_citations = [
            {
                "ref_id": "abc123",
                "title": "A Paper",
                "authors": "Smith",
                "year": "2020",
                "doi": "10.1/x",
                "raw_text": "[1] Smith 2020",
            }
        ]

        with (
            patch("api.routers.documents._get_status_store") as mock_store_fn,
            patch("api.routers.documents.GraphStore") as MockGraphStore,
        ):
            mock_store = MagicMock()
            mock_store.get.return_value = mock_status
            mock_store_fn.return_value = mock_store

            mock_gs = MagicMock()
            mock_gs.get_document_citations.return_value = mock_citations
            MockGraphStore.return_value = mock_gs

            response = client.get("/documents/test_doc_id/citations")

        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "test_doc_id"
        assert data["citation_count"] == 1
        assert data["citations"][0]["title"] == "A Paper"
