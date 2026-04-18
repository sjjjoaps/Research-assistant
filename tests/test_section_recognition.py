"""
Phase 3.2 章节结构识别测试

验收标准：
- SectionRecognizer 规则识别覆盖全部 10 种有效章节类型
- 全大写标题可被识别
- 前向填充正确传播 section_type
- LLM 兜底仅对 unknown 页面调用
- LLM 失败时降级为 "unknown"，不抛异常
- SectionRecognizer 实例化不触发 LLM 初始化（懒初始化）
- DocumentParser 默认不启用章节识别（向后兼容）
- DocumentParser 启用后 page_sections / page_section_titles 被填充
- 非 PDF 文件即使启用也不识别
- TextChunk.metadata 包含 section_type / section_title
- page_sections 为空时 chunk 的 section_type 为 "unknown"
- 多模态 chunk 也携带 section_type 键（值为 "unknown"）
- 可按 section_type 过滤 chunk
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.section_recognizer import SectionRecognizer, SECTION_TYPES
from src.ingestion.document_parser import ParsedDocument, DocumentParser
from src.ingestion.chunker import DocumentChunker
from src.ingestion.modal_processors import ModalContent


# ── SectionRecognizer 规则识别 ────────────────────────────────────────────────

class TestSectionRecognizerRuleBased:
    def _recognizer(self) -> SectionRecognizer:
        r = SectionRecognizer.__new__(SectionRecognizer)
        r._llm = MagicMock()
        r._llm.invoke.return_value = MagicMock(content="unknown")
        return r

    def _first_type(self, page_text: str) -> str:
        r = self._recognizer()
        types, _ = r.recognize_per_page([page_text])
        return types[0]

    def test_abstract_detected(self):
        assert self._first_type("Abstract\nThis paper presents...") == "abstract"

    def test_introduction_detected(self):
        assert self._first_type("1. Introduction\nRecent advances...") == "introduction"

    def test_related_work_detected(self):
        assert self._first_type("2. Related Work\nPrior studies...") == "related_work"

    def test_method_detected(self):
        assert self._first_type("3. Methodology\nWe propose...") == "method"

    def test_experiment_detected(self):
        assert self._first_type("4. Experiments\nWe evaluate...") == "experiment"

    def test_result_detected(self):
        assert self._first_type("5. Results\nTable 1 shows...") == "result"

    def test_discussion_detected(self):
        assert self._first_type("6. Discussion\nThe results indicate...") == "discussion"

    def test_conclusion_detected(self):
        assert self._first_type("7. Conclusion\nIn this paper...") == "conclusion"

    def test_reference_detected(self):
        assert self._first_type("References\n[1] Smith et al.") == "reference"

    def test_appendix_detected(self):
        assert self._first_type("Appendix A\nAdditional details...") == "appendix"

    def test_allcaps_heading_detected(self):
        assert self._first_type("INTRODUCTION\nRecent advances in...") == "introduction"

    def test_unknown_for_body_text(self):
        """长段落正文（无标题行）在规则阶段应为 unknown（前向填充前）。"""
        r = SectionRecognizer.__new__(SectionRecognizer)
        r._llm = None
        stype, _ = r._detect_heading_in_page(
            "This is a long paragraph of body text that does not start with any heading. "
            "It continues for many words without any section marker."
        )
        assert stype is None

    def test_long_line_not_matched_as_heading(self):
        """超过 80 字符的行不应被识别为标题。"""
        r = SectionRecognizer.__new__(SectionRecognizer)
        r._llm = None
        long_line = "Introduction " + "x" * 80
        stype, _ = r._detect_heading_in_page(long_line)
        assert stype is None


# ── 前向填充 ──────────────────────────────────────────────────────────────────

class TestSectionRecognizerForwardFill:
    def _recognizer_no_llm(self) -> SectionRecognizer:
        """返回一个 LLM 永远返回 unknown 的识别器（模拟 LLM 不可用）。"""
        r = SectionRecognizer.__new__(SectionRecognizer)
        r._llm = MagicMock()
        r._llm.invoke.return_value = MagicMock(content="unknown")
        return r

    def test_forward_fill_propagates_section(self):
        """page 0 有标题，pages 1-2 为正文 → 三页都应为同一 section_type。"""
        r = self._recognizer_no_llm()
        pages = [
            "Introduction\nThis paper...",
            "We propose a method that...",
            "Furthermore, our approach...",
        ]
        types, _ = r.recognize_per_page(pages)
        assert types == ["introduction", "introduction", "introduction"]

    def test_forward_fill_resets_on_new_heading(self):
        """page 0 Introduction，page 2 Method → pages 0-1 为 introduction，page 2 为 method。"""
        r = self._recognizer_no_llm()
        pages = [
            "Introduction\nThis paper...",
            "We review prior work...",
            "3. Methodology\nOur approach...",
            "We implement the model...",
        ]
        types, _ = r.recognize_per_page(pages)
        assert types[0] == "introduction"
        assert types[1] == "introduction"
        assert types[2] == "method"
        assert types[3] == "method"


# ── LLM 兜底 ─────────────────────────────────────────────────────────────────

class TestSectionRecognizerLLMFallback:
    def test_llm_called_only_for_unknown_pages(self):
        """前向填充无法覆盖的 unknown 页面才调用 LLM。"""
        r = SectionRecognizer.__new__(SectionRecognizer)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="method")
        r._llm = mock_llm

        # page 0 无标题（前向填充无法覆盖，因为前面没有已知 section）
        # page 1 有标题
        pages = [
            "Some body text without any heading marker here at all",  # unknown → LLM 调用
            "Introduction\nThis paper...",   # 规则识别 → introduction
        ]
        types, _ = r.recognize_per_page(pages)

        # LLM 应只被调用一次（第一页，前向填充无法覆盖）
        assert mock_llm.invoke.call_count == 1
        assert types[1] == "introduction"

    def test_llm_not_called_when_all_pages_classified(self):
        """所有页面规则识别成功时，LLM 不应被调用。"""
        r = SectionRecognizer.__new__(SectionRecognizer)
        mock_llm = MagicMock()
        r._llm = mock_llm

        pages = [
            "Abstract\nThis paper...",
            "1. Introduction\nRecent work...",
        ]
        r.recognize_per_page(pages)
        mock_llm.invoke.assert_not_called()

    def test_llm_failure_falls_back_to_unknown(self):
        """LLM 调用失败时该页保留 unknown，不抛异常。"""
        r = SectionRecognizer.__new__(SectionRecognizer)
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API error")
        r._llm = mock_llm

        pages = ["Some body text without any heading marker here"]
        types, _ = r.recognize_per_page(pages)
        assert types[0] == "unknown"

    def test_llm_lazy_init(self):
        """实例化 SectionRecognizer 不应触发 LLM 初始化。"""
        with patch("src.infrastructure.llm_client.get_llm") as mock_get_llm:
            r = SectionRecognizer()
            mock_get_llm.assert_not_called()
            assert r._llm is None


# ── recognize_per_page 输出结构 ───────────────────────────────────────────────

class TestSectionRecognizerPerPage:
    def test_recognize_per_page_returns_parallel_lists(self):
        """输出两个列表长度均与 pages 相同。"""
        r = SectionRecognizer.__new__(SectionRecognizer)
        r._llm = MagicMock()
        r._llm.invoke.return_value = MagicMock(content="unknown")

        pages = ["Abstract\nText", "Body text", "References\nList"]
        types, titles = r.recognize_per_page(pages)
        assert len(types) == len(pages)
        assert len(titles) == len(pages)

    def test_empty_pages_returns_empty_lists(self):
        r = SectionRecognizer()
        types, titles = r.recognize_per_page([])
        assert types == []
        assert titles == []


# ── DocumentParser 集成 ───────────────────────────────────────────────────────

class TestDocumentParserSectionRecognition:
    def test_default_no_section_recognition(self, tmp_path):
        """默认不启用章节识别，page_sections 为空列表。"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello world")

        parser = DocumentParser()
        doc = parser.parse(txt_file)
        assert doc.page_sections == []
        assert doc.page_section_titles == []

    def test_section_recognition_disabled_by_default_for_pdf(self, tmp_path):
        """DocumentParser() 默认不启用章节识别（向后兼容）。"""
        import fitz
        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Introduction\nSome text here.")
        doc.save(str(pdf_path))
        doc.close()

        parser = DocumentParser()
        result = parser.parse(pdf_path)
        assert result.page_sections == []

    def test_section_recognition_enabled_populates_page_sections(self, tmp_path):
        """启用 enable_section_recognition 时，PDF 解析应填充 page_sections。"""
        import fitz
        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Introduction\nSome text here.")
        doc.save(str(pdf_path))
        doc.close()

        mock_sections = (["introduction"], ["Introduction"])
        with patch("src.ingestion.section_recognizer.SectionRecognizer.recognize_per_page",
                   return_value=mock_sections):
            parser = DocumentParser(enable_section_recognition=True)
            result = parser.parse(pdf_path)

        assert result.page_sections == ["introduction"]
        assert result.page_section_titles == ["Introduction"]

    def test_section_recognition_not_applied_to_txt(self, tmp_path):
        """TXT 文件即使启用 enable_section_recognition，page_sections 也为空。"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Introduction\nSome text here.")

        parser = DocumentParser(enable_section_recognition=True)
        doc = parser.parse(txt_file)
        assert doc.page_sections == []


# ── DocumentChunker 章节元数据 ────────────────────────────────────────────────

class TestChunkerSectionMetadata:
    def _make_doc(self, raw_text: str, pages=None, page_sections=None,
                  page_section_titles=None, modal_contents=None) -> ParsedDocument:
        return ParsedDocument(
            file_path="/tmp/test.pdf",
            raw_text=raw_text,
            pages=pages or ([raw_text] if raw_text else []),
            page_sections=page_sections or [],
            page_section_titles=page_section_titles or [],
            modal_contents=modal_contents or [],
        )

    def test_section_type_written_to_chunk_metadata(self):
        """page_sections 有值时，文本 chunk 应携带正确的 section_type。"""
        text = "Introduction\nThis paper presents a novel approach."
        doc = self._make_doc(
            raw_text=text,
            pages=[text],
            page_sections=["introduction"],
            page_section_titles=["Introduction"],
        )
        chunker = DocumentChunker()
        chunks = chunker.chunk(doc)

        assert len(chunks) > 0
        assert chunks[0].metadata["section_type"] == "introduction"

    def test_section_title_written_to_chunk_metadata(self):
        """page_sections 有值时，文本 chunk 应携带正确的 section_title。"""
        text = "3. Methodology\nWe propose a new method."
        doc = self._make_doc(
            raw_text=text,
            pages=[text],
            page_sections=["method"],
            page_section_titles=["3. Methodology"],
        )
        chunker = DocumentChunker()
        chunks = chunker.chunk(doc)

        assert chunks[0].metadata["section_title"] == "3. Methodology"

    def test_unknown_when_page_sections_empty(self):
        """page_sections 为空时，所有文本 chunk 的 section_type 应为 'unknown'。"""
        doc = self._make_doc("Some text content for testing section type default.")
        chunker = DocumentChunker()
        chunks = chunker.chunk(doc)

        for c in chunks:
            assert c.metadata["section_type"] == "unknown"
            assert c.metadata["section_title"] == ""

    def test_modal_chunks_have_section_type_key(self):
        """多模态 chunk 也应携带 section_type 键（值为 'unknown'）。"""
        modal = ModalContent(
            content_type="image",
            raw_content="data:image/png;base64,abc",
            processed_text="图片描述",
            page_number=0,
            position_hint="page 1, image 1",
        )
        doc = self._make_doc("", modal_contents=[modal])
        chunker = DocumentChunker()
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert "section_type" in chunks[0].metadata
        assert chunks[0].metadata["section_type"] == "unknown"
        assert "section_title" in chunks[0].metadata

    def test_mismatched_page_section_titles_length_no_index_error(self):
        """page_section_titles 长度短于 page_sections 时不应抛 IndexError。"""
        text = "Introduction\n" + "Some content. " * 50
        doc = ParsedDocument(
            file_path="/tmp/test.pdf",
            raw_text=text,
            pages=[text],
            page_sections=["introduction"],
            page_section_titles=[],  # 故意为空，长度不匹配
        )
        chunker = DocumentChunker()
        # 不应抛异常
        chunks = chunker.chunk(doc)
        assert all(c.metadata["section_type"] == "introduction" for c in chunks)
        assert all(c.metadata["section_title"] == "" for c in chunks)

    def test_cursor_prevents_duplicate_content_mismatch(self):
        """重复内容（如页眉）不应因 find() 从头搜索而被错误映射到首次出现的页面。"""
        # page 0: method 章节，包含重复短语
        # page 1: conclusion 章节，也包含相同短语（模拟页眉/页脚）
        repeated = "University of Science"
        page0 = f"3. Methodology\n{repeated}\n" + "We propose a method. " * 40
        page1 = f"7. Conclusion\n{repeated}\n" + "In this paper we conclude. " * 40

        raw_text = page0 + "\n\n" + page1
        doc = ParsedDocument(
            file_path="/tmp/test.pdf",
            raw_text=raw_text,
            pages=[page0, page1],
            page_sections=["method", "conclusion"],
            page_section_titles=["3. Methodology", "7. Conclusion"],
        )
        chunker = DocumentChunker()
        chunks = chunker.chunk(doc)

        # 结论章节的 chunk 不应被错误标记为 method
        conclusion_chunks = [c for c in chunks if c.metadata["section_type"] == "conclusion"]
        assert len(conclusion_chunks) >= 1


# ── 按 section_type 过滤（VectorStore 路径）────────────────────────────────────

class TestSectionTypeFilterable:
    def test_section_type_in_metadata_filterable(self):
        """可按 metadata['section_type'] 过滤 chunk（内存 chunk list 验证）。"""
        # 使用足够长的文本确保 splitter 产生多个 chunk
        intro_text = "Introduction\n" + "This paper presents a novel approach. " * 30
        method_text = "3. Methodology\n" + "We propose a new method based on deep learning. " * 30

        raw_text = intro_text + "\n\n" + method_text
        doc = ParsedDocument(
            file_path="/tmp/test.pdf",
            raw_text=raw_text,
            pages=[intro_text, method_text],
            page_sections=["introduction", "method"],
            page_section_titles=["Introduction", "3. Methodology"],
        )
        chunker = DocumentChunker()
        chunks = chunker.chunk(doc)

        method_chunks = [c for c in chunks if c.metadata["section_type"] == "method"]
        intro_chunks = [c for c in chunks if c.metadata["section_type"] == "introduction"]

        assert len(method_chunks) >= 1
        assert len(intro_chunks) >= 1
        assert all("Methodology" in c.metadata["section_title"] for c in method_chunks)

    def test_vector_store_similarity_search_section_filter(self):
        """VectorStore.similarity_search 的 section_type 参数应过滤结果。"""
        from src.storage.vector_store import VectorStore
        from langchain_core.documents import Document

        store = VectorStore.__new__(VectorStore)
        store._lock = __import__("threading").Lock()

        # 构造两个 mock Document，section_type 不同
        doc_method = Document(
            page_content="We propose a deep learning method.",
            metadata={"section_type": "method", "doc_id": "d1", "chunk_index": 0},
        )
        doc_intro = Document(
            page_content="This paper introduces a new approach.",
            metadata={"section_type": "introduction", "doc_id": "d1", "chunk_index": 1},
        )

        # mock _store.similarity_search 返回两个文档
        mock_faiss = MagicMock()
        mock_faiss.similarity_search.return_value = [doc_method, doc_intro]
        store._store = mock_faiss

        # 不过滤：返回两个
        results = store.similarity_search("deep learning", k=5)
        assert len(results) == 2

        # 过滤 method：只返回 method chunk
        results_method = store.similarity_search("deep learning", k=5, section_type="method")
        assert len(results_method) == 1
        assert results_method[0].metadata["section_type"] == "method"

        # 过滤 introduction：只返回 introduction chunk
        results_intro = store.similarity_search("deep learning", k=5, section_type="introduction")
        assert len(results_intro) == 1
        assert results_intro[0].metadata["section_type"] == "introduction"

    def test_semantic_retriever_passes_section_type_to_vector_store(self):
        """SemanticRetriever.retrieve 应将 section_type 透传给 VectorStore。"""
        from src.retrieval.retriever import SemanticRetriever
        from langchain_core.documents import Document

        retriever = SemanticRetriever.__new__(SemanticRetriever)
        retriever.top_k = 3

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            Document(
                page_content="Method content",
                metadata={"section_type": "method", "file_path": "/tmp/a.pdf", "chunk_index": 0},
            )
        ]
        retriever.vector_store = mock_store

        results = retriever.retrieve("query", section_type="method")

        mock_store.similarity_search.assert_called_once_with(
            query="query", k=3, section_type="method"
        )
        assert len(results) == 1
        assert results[0].section_type == "method"
