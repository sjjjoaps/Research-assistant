"""
测试 section_chunker（Phase 10-1）

测试策略：
- chunk_by_section() 函数单元测试：
    1. 普通段落分块：段落不超 max_size 时保持完整，不在段落中间截断
    2. 超长段落：单段落超出 max_size 时在句子边界切分
    3. 极短内容：少于 max_size 的内容返回单个 chunk
    4. 空内容：返回空列表，不报错
    5. 章节类型 → chunk size 映射：method 比 abstract 允许更大 chunk
    6. overlap：相邻 chunk 尾部内容在下一 chunk 头部有重叠

- SectionChunker.chunk() 集成测试（Mock ParsedDocument）：
    7. 有章节信息时使用语义分块，各 chunk metadata 含正确 section_type
    8. 无章节信息（page_sections 为空）时 fallback 到 DocumentChunker
    9. 多模态 chunk 正确透传，section_type 固定为 "unknown"
   10. 相邻同 section_type 页面内容合并为同一 section 组
   11. 跨 section_type 页面分别分块，metadata 正确

注意：
- 所有测试均不依赖 LLM / FAISS / Neo4j
- ParsedDocument 用 MagicMock 构造，不实际解析 PDF
"""
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.section_chunker import (
    SECTION_CHUNK_SIZES,
    SectionChunker,
    chunk_by_section,
)


# ── 辅助工厂 ────────────────────────────────────────────────────────────────

def _make_doc(
    raw_text: str,
    pages: list[str],
    page_sections: list[str] = None,
    page_section_titles: list[str] = None,
    file_path: str = "test.pdf",
    modal_contents: list = None,
):
    doc = MagicMock()
    doc.raw_text            = raw_text
    doc.pages               = pages
    doc.page_sections       = page_sections or []
    doc.page_section_titles = page_section_titles or ([""] * len(pages) if pages else [])
    doc.file_path           = file_path
    doc.modal_contents      = modal_contents or []
    return doc


# ══════════════════════════════════════════════════════════════════════════
# Case 1-6：chunk_by_section() 函数单元测试
# ══════════════════════════════════════════════════════════════════════════

def test_paragraphs_stay_intact():
    """段落不超 max_size 时，单个段落作为完整 chunk，不被截断。"""
    content = "第一段内容，描述方法步骤。\n\n第二段内容，描述实验结果。"
    chunks = chunk_by_section(content, section_type="method", max_size=500)
    # 应返回 1 个 chunk（两段合并后仍在 500 以内）
    assert len(chunks) == 1
    assert "第一段内容" in chunks[0]
    assert "第二段内容" in chunks[0]
    print("[PASS] test_paragraphs_stay_intact")


def test_paragraph_boundary_split():
    """多段落超出 max_size 时，在段落边界切分，不在段落内部截断。"""
    para1 = "A" * 200
    para2 = "B" * 200
    para3 = "C" * 200
    content = f"{para1}\n\n{para2}\n\n{para3}"
    chunks = chunk_by_section(content, section_type="method", max_size=300)
    # 每个段落 200 字，max_size=300，para1+para2=400>300，应在段落边界切分
    assert len(chunks) >= 2, f"期望至少 2 个 chunk，实际 {len(chunks)}"
    # 每个 chunk 不应在段落中间截断（不存在 "A...B" 混合的情况）
    for chunk in chunks:
        has_a = "A" in chunk
        has_b = "B" in chunk
        has_c = "C" in chunk
        # 允许 overlap 导致尾部有少量混合，但主体不应出现三种字符全混
        assert not (has_a and has_b and has_c), f"段落不应全混入同一 chunk: {chunk[:50]}"
    print("[PASS] test_paragraph_boundary_split")


def test_short_content_single_chunk():
    """内容短于 max_size 时，返回单个 chunk。"""
    content = "这是摘要内容，描述了本文的主要贡献和研究方法。"
    chunks = chunk_by_section(content, section_type="abstract", max_size=300)
    assert len(chunks) == 1
    assert chunks[0] == content
    print("[PASS] test_short_content_single_chunk")


def test_empty_content_returns_empty():
    """空内容应返回空列表，不报错。"""
    chunks = chunk_by_section("", section_type="method")
    assert chunks == []
    chunks = chunk_by_section("   \n\n  ", section_type="abstract")
    assert chunks == []
    print("[PASS] test_empty_content_returns_empty")


def test_section_chunk_sizes_differentiate():
    """method 的 max_size 应大于 abstract，验证 SECTION_CHUNK_SIZES 差异化配置。"""
    assert SECTION_CHUNK_SIZES["method"]   > SECTION_CHUNK_SIZES["abstract"], \
        "method 应允许更大的 chunk"
    assert SECTION_CHUNK_SIZES["method"]   > SECTION_CHUNK_SIZES["conclusion"], \
        "method 应允许更大的 chunk"
    assert SECTION_CHUNK_SIZES["experiment"] > SECTION_CHUNK_SIZES["abstract"], \
        "experiment 应允许更大的 chunk"
    print("[PASS] test_section_chunk_sizes_differentiate")


def test_overlap_connects_adjacent_chunks():
    """overlap > 0 时，相邻 chunk 之间应有内容重叠（尾部出现在下一 chunk 头部）。"""
    # 构造 3 个段落，每段 180 字，max_size=200，overlap=30
    para1 = "段落一：" + "甲" * 176
    para2 = "段落二：" + "乙" * 176
    content = f"{para1}\n\n{para2}"
    chunks = chunk_by_section(content, section_type="method", max_size=200, overlap=30)
    if len(chunks) >= 2:
        # chunk[0] 的尾部 30 字应出现在 chunk[1] 中
        tail = chunks[0][-30:]
        assert tail in chunks[1], \
            f"overlap 未生效，chunk[0] 尾部不在 chunk[1] 中\ntail={tail!r}\nchunk[1]={chunks[1][:60]!r}"
    print("[PASS] test_overlap_connects_adjacent_chunks")


# ══════════════════════════════════════════════════════════════════════════
# Case 7-11：SectionChunker.chunk() 集成测试
# ══════════════════════════════════════════════════════════════════════════

def test_with_sections_uses_semantic_chunking():
    """有章节信息时，各 chunk 的 metadata 应含正确 section_type。"""
    page1 = "这是摘要段落。" * 5
    page2 = "这是方法段落。" * 5
    raw_text = page1 + "\n\n" + page2
    doc = _make_doc(
        raw_text=raw_text,
        pages=[page1, page2],
        page_sections=["abstract", "method"],
        page_section_titles=["摘要", "方法"],
    )
    chunker = SectionChunker()
    chunks = chunker.chunk(doc, doc_id="test_doc")
    assert len(chunks) > 0, "有内容的文档应产生至少 1 个 chunk"
    section_types = {c.metadata.get("section_type") for c in chunks}
    assert "abstract" in section_types or "method" in section_types, \
        f"section_type 未正确设置，实际: {section_types}"
    for c in chunks:
        assert c.metadata.get("content_type") == "text"
        assert "doc_id" in c.metadata
    print("[PASS] test_with_sections_uses_semantic_chunking")


def test_without_sections_fallback_to_document_chunker():
    """无章节信息时，应 fallback 到 DocumentChunker，仍能产生 chunk。"""
    text = "通用文本内容，没有章节识别信息。" * 20
    doc = _make_doc(
        raw_text=text,
        pages=[text],
        page_sections=[],   # 无章节信息
    )
    chunker = SectionChunker()
    chunks = chunker.chunk(doc)
    assert len(chunks) > 0, "无章节信息时 fallback 应仍能产生 chunk"
    print("[PASS] test_without_sections_fallback_to_document_chunker")


def test_modal_chunks_have_unknown_section_type():
    """多模态 chunk 的 section_type 应固定为 'unknown'。"""
    modal = MagicMock()
    modal.content_type    = "image"
    modal.processed_text  = "这是图片的描述文字。"
    modal.position_hint   = "第 2 页，图 1"
    modal.page_number     = 2

    doc = _make_doc(
        raw_text="",
        pages=[],
        page_sections=[],
        modal_contents=[modal],
    )
    chunker = SectionChunker()
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].metadata["section_type"] == "unknown"
    assert chunks[0].metadata["content_type"] == "image"
    assert "这是图片的描述文字" in chunks[0].content
    print("[PASS] test_modal_chunks_have_unknown_section_type")


def test_adjacent_same_section_merged():
    """相邻同 section_type 的页面应合并为同一组，而不是产生两组分块。"""
    page1 = "方法段落第一页。" * 10
    page2 = "方法段落第二页。" * 10
    raw_text = page1 + "\n\n" + page2
    doc = _make_doc(
        raw_text=raw_text,
        pages=[page1, page2],
        page_sections=["method", "method"],   # 两页同属 method
    )
    chunker = SectionChunker()
    chunks = chunker.chunk(doc)
    # 所有文本 chunk 都应属于 method 组
    text_chunks = [c for c in chunks if c.metadata.get("content_type") == "text"]
    for c in text_chunks:
        assert c.metadata["section_type"] == "method", \
            f"期望 method，实际 {c.metadata['section_type']}"
    print("[PASS] test_adjacent_same_section_merged")


def test_different_sections_split_correctly():
    """不同 section_type 的页面应分别分块，metadata 中的 section_type 对应正确。"""
    page_abs    = "摘要内容，简要说明本研究的目的与方法。"
    page_method = "方法内容，详细描述实验设计与数据处理步骤。"
    page_exp    = "实验结果，列出各组数据及显著性分析。"
    raw_text = "\n\n".join([page_abs, page_method, page_exp])
    doc = _make_doc(
        raw_text=raw_text,
        pages=[page_abs, page_method, page_exp],
        page_sections=["abstract", "method", "experiment"],
    )
    chunker = SectionChunker()
    chunks = chunker.chunk(doc)
    section_types = {c.metadata["section_type"] for c in chunks
                     if c.metadata.get("content_type") == "text"}
    assert "abstract"   in section_types, "应有 abstract 类型的 chunk"
    assert "method"     in section_types, "应有 method 类型的 chunk"
    assert "experiment" in section_types, "应有 experiment 类型的 chunk"
    print("[PASS] test_different_sections_split_correctly")


# ══════════════════════════════════════════════════════════════════════════
# Case 12-14：Review 补充测试
# ══════════════════════════════════════════════════════════════════════════

def test_pages_sections_length_mismatch_fallback():
    """page_sections 与 pages 长度不一致时，应 fallback 到 DocumentChunker，不报错。"""
    page1 = "第一页内容。" * 10
    page2 = "第二页内容。" * 10
    doc = _make_doc(
        raw_text=page1 + "\n\n" + page2,
        pages=[page1, page2],
        page_sections=["abstract"],   # 只有 1 个 section，但有 2 个 page → 不一致
    )
    chunker = SectionChunker()
    # 不应抛出异常，应 fallback 并产生 chunk
    chunks = chunker.chunk(doc)
    assert len(chunks) > 0, "长度不一致时 fallback 应仍能产生 chunk"
    print("[PASS] test_pages_sections_length_mismatch_fallback")


def test_single_sentence_exceeds_max_size_hard_cut():
    """单句超过 max_size 时，应硬切为多个 chunk，每块不超过 max_size 字符。"""
    # 构造一个没有句末标点的超长"句子"（无法在句子边界切分）
    long_sentence = "A" * 600   # 600 字，max_size=300
    chunks = chunk_by_section(long_sentence, section_type="abstract", max_size=300)
    assert len(chunks) >= 2, f"超长句子应被硬切为至少 2 个 chunk，实际 {len(chunks)}"
    # 每个 chunk 的正文部分（去掉 overlap 前缀后）不超过 max_size
    # 由于 overlap 会在 chunk 头部追加前一块的尾部，检查最后一块（无 overlap 追加）
    last_chunk_len = len(chunks[-1])
    assert last_chunk_len <= 300 + 50 + 10, \
        f"硬切后 chunk 不应远超 max_size，实际最后一块长度 {last_chunk_len}"
    print("[PASS] test_single_sentence_exceeds_max_size_hard_cut")


def test_same_type_different_title_not_merged():
    """同 section_type 但 section_title 不同的连续页面，不应合并为同一组。"""
    page1 = "3.1 数据预处理方法描述。" * 5
    page2 = "3.2 模型训练方法描述。" * 5
    raw_text = page1 + "\n\n" + page2
    doc = _make_doc(
        raw_text=raw_text,
        pages=[page1, page2],
        page_sections=["method", "method"],          # 同 section_type
        page_section_titles=["3.1 数据预处理", "3.2 模型训练"],  # 不同 title
    )
    chunker = SectionChunker()
    chunks = chunker.chunk(doc)
    text_chunks = [c for c in chunks if c.metadata.get("content_type") == "text"]
    # 两个不同 title 的 method 页面应产生两组 chunk，section_title 各自正确
    titles_in_chunks = {c.metadata.get("section_title") for c in text_chunks}
    assert "3.1 数据预处理" in titles_in_chunks, \
        f"应有 '3.1 数据预处理' 的 chunk，实际 titles: {titles_in_chunks}"
    assert "3.2 模型训练" in titles_in_chunks, \
        f"应有 '3.2 模型训练' 的 chunk，实际 titles: {titles_in_chunks}"
    print("[PASS] test_same_type_different_title_not_merged")


# ── 主入口 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 10-1  section_chunker 测试")
    print("=" * 60)

    tests = [
        # chunk_by_section 单元测试
        test_paragraphs_stay_intact,
        test_paragraph_boundary_split,
        test_short_content_single_chunk,
        test_empty_content_returns_empty,
        test_section_chunk_sizes_differentiate,
        test_overlap_connects_adjacent_chunks,
        # SectionChunker 集成测试
        test_with_sections_uses_semantic_chunking,
        test_without_sections_fallback_to_document_chunker,
        test_modal_chunks_have_unknown_section_type,
        test_adjacent_same_section_merged,
        test_different_sections_split_correctly,
        # Review 补充测试
        test_pages_sections_length_mismatch_fallback,
        test_single_sentence_exceeds_max_size_hard_cut,
        test_same_type_different_title_not_merged,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"结果：{passed} 通过，{failed} 失败")
    print("=" * 60)
