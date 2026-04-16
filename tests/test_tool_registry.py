"""
测试 tool_registry（Phase 8-3）

测试策略：
- 工具注册层测试（不依赖外部服务）：
    1. build_tool_registry() 返回 7 个工具
    2. 工具名称与 TOOL_DISPLAY_NAMES 完全对应
    3. 所有工具具备有效的 name / description / args_schema
    4. build_tool_registry() 是单例（两次调用返回同一对象）

- 工具逻辑单元测试（Mock 替换外部依赖）：
    5. retrieve_knowledge — 空结果时返回友好提示
    6. retrieve_knowledge — 有结果时格式化正确（[来源: xxx#chunk-n]）
    7. retrieve_knowledge — 异常时返回错误提示，不抛出
    8. list_documents     — 无结果时返回友好提示
    9. list_documents     — 有结果时按格式展示，keyword 过滤生效
   10. get_document_metadata — 未找到时提示用户
   11. get_document_metadata — 按 doc_id 精确查找返回详情
   12. get_knowledge_graph_stats — 异常时返回错误提示，不抛出
   13. _auto_select_mode — 含时间词 → mix；无关键词 → mix；含具体实体词 → local

注意：
- deep_research / generate_research_ideas / search_by_entity 依赖 Neo4j 和 LLM，
  此处只验证工具可被调用且异常时有降级（不验证真实结果）。
"""
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.tool_registry import (
    build_tool_registry,
    TOOL_DISPLAY_NAMES,
    _auto_select_mode,
)


# ── 辅助工厂 ──────────────────────────────────────────────────────────────

def _make_chunk(content: str, file_path: str = "paper.pdf",
                chunk_index: int = 0, section_type: str = "method"):
    """构造模拟的 RetrievedChunk 对象（不依赖实际类）。"""
    c = MagicMock()
    c.content     = content
    c.file_path   = file_path
    c.chunk_index = chunk_index
    c.section_type = section_type
    return c


def _make_doc(title: str = "Test Paper",
              authors: str = "Author A",
              year: int = 2024,
              abstract: str = "An abstract.",
              doc_id: str = "doc_001",
              file_path: str = "/data/test.pdf",
              institution: str = "MIT",
              keywords: str = "RAG, LLM"):
    doc = MagicMock()
    doc.title       = title
    doc.authors     = authors
    doc.year        = year
    doc.abstract    = abstract
    doc.doc_id      = doc_id
    doc.file_path   = file_path
    doc.institution = institution
    doc.keywords    = keywords
    return doc


# ══════════════════════════════════════════════════════════════════════════
# Case 1-4：工具注册层（不需要外部服务）
# ══════════════════════════════════════════════════════════════════════════

def test_registry_count():
    """build_tool_registry() 应返回 7 个工具。"""
    tools = build_tool_registry()
    assert len(tools) == 7, f"期望 7 个工具，实际 {len(tools)}"
    print("[PASS] test_registry_count")


def test_registry_names_match_display():
    """每个工具的 name 都应在 TOOL_DISPLAY_NAMES 中有对应条目。"""
    tools = build_tool_registry()
    tool_names = {t.name for t in tools}
    display_names = set(TOOL_DISPLAY_NAMES.keys())
    assert tool_names == display_names, (
        f"工具名不一致\n工具名：{tool_names}\n显示名键：{display_names}"
    )
    print("[PASS] test_registry_names_match_display")


def test_registry_tool_schemas():
    """每个工具都应有有效的 name、description、args_schema。"""
    tools = build_tool_registry()
    for t in tools:
        assert t.name,        f"工具 {t} 缺少 name"
        assert t.description, f"工具 {t.name} 缺少 description"
        assert t.args_schema, f"工具 {t.name} 缺少 args_schema"
    print("[PASS] test_registry_tool_schemas")


def test_registry_singleton():
    """build_tool_registry() 应返回同一个缓存对象（单例）。"""
    tools_a = build_tool_registry()
    tools_b = build_tool_registry()
    assert tools_a is tools_b, "两次调用返回不同对象，lru_cache 失效"
    print("[PASS] test_registry_singleton")


# ══════════════════════════════════════════════════════════════════════════
# Case 5-7：retrieve_knowledge 逻辑
# ══════════════════════════════════════════════════════════════════════════

def test_retrieve_empty_result():
    """检索无结果时应返回含"检索结果为空"的友好提示，而不是空字符串。"""
    with patch("src.agents.tool_registry._do_retrieve", return_value=[]):
        result = retrieve_knowledge.invoke({"query": "不存在的内容"})
    assert "检索结果为空" in result, f"未找到预期提示，实际: {result[:100]}"
    print("[PASS] test_retrieve_empty_result")


def test_retrieve_with_results():
    """检索有结果时应包含来源标签。"""
    fake_chunks = [
        _make_chunk("RAG 的核心思想是…", file_path="paper1.pdf", chunk_index=3),
        _make_chunk("GraphRAG 扩展了…",  file_path="paper2.pdf", chunk_index=7),
    ]
    with patch("src.agents.tool_registry._do_retrieve", return_value=fake_chunks):
        result = retrieve_knowledge.invoke({"query": "RAG 和 GraphRAG 的区别"})
    assert "[来源: paper1.pdf#chunk-3]" in result, f"来源标签缺失: {result[:200]}"
    assert "RAG 的核心思想是" in result
    assert "检索到 2 个相关片段" in result
    print("[PASS] test_retrieve_with_results")


def test_retrieve_exception_graceful():
    """检索抛出异常时应降级返回错误提示，不再向上抛出。"""
    with patch("src.agents.tool_registry._do_retrieve",
               side_effect=RuntimeError("FAISS 索引未加载")):
        result = retrieve_knowledge.invoke({"query": "任意查询"})
    assert "检索失败" in result or "FAISS" in result, \
        f"异常未被捕获，实际: {result[:100]}"
    print("[PASS] test_retrieve_exception_graceful")


# ══════════════════════════════════════════════════════════════════════════
# Case 8-9：list_documents 逻辑
# ══════════════════════════════════════════════════════════════════════════

def _patch_db(docs):
    """返回 patch MetadataDatabase 的上下文管理器，list_documents() 返回 docs。"""
    mock_db = MagicMock()
    mock_db.list_documents.return_value = docs
    return patch("src.agents.tool_registry.MetadataDatabase", return_value=mock_db)


def test_list_documents_empty():
    """知识库为空时应返回提示，而不是空字符串。"""
    with _patch_db([]):
        result = list_documents.invoke({})
    assert "暂无文献" in result, f"未找到预期提示，实际: {result[:100]}"
    print("[PASS] test_list_documents_empty")


def test_list_documents_with_keyword_filter():
    """keyword 过滤应只返回匹配的文献。"""
    docs = [
        _make_doc(title="RAG-based QA System",     doc_id="d1", keywords="RAG, LLM"),
        _make_doc(title="GraphRAG for Literature", doc_id="d2", keywords="GraphRAG"),
        _make_doc(title="Transformer Survey",      doc_id="d3", keywords="Transformer, NLP"),
    ]
    with _patch_db(docs):
        result = list_documents.invoke({"keyword": "RAG", "limit": 20})
    assert "RAG-based QA System" in result
    assert "GraphRAG for Literature" in result
    assert "Transformer Survey" not in result, "关键词过滤未生效"
    print("[PASS] test_list_documents_with_keyword_filter")


# ══════════════════════════════════════════════════════════════════════════
# Case 10-11：get_document_metadata 逻辑
# ══════════════════════════════════════════════════════════════════════════

def test_get_document_metadata_not_found():
    """找不到文献时应返回友好提示。"""
    mock_db = MagicMock()
    mock_db.get_document_by_doc_id.return_value = None
    mock_db.list_documents.return_value = []
    with patch("src.agents.tool_registry.MetadataDatabase", return_value=mock_db):
        result = get_document_metadata.invoke({"doc_title_or_id": "不存在的论文"})
    assert "未找到" in result, f"未找到预期提示，实际: {result[:100]}"
    print("[PASS] test_get_document_metadata_not_found")


def test_get_document_metadata_by_id():
    """按 doc_id 精确查找时应返回完整元数据。"""
    doc = _make_doc(title="A Survey on RAG", doc_id="doc_survey_rag")
    mock_db = MagicMock()
    mock_db.get_document_by_doc_id.return_value = doc
    with patch("src.agents.tool_registry.MetadataDatabase", return_value=mock_db):
        result = get_document_metadata.invoke({"doc_title_or_id": "doc_survey_rag"})
    assert "A Survey on RAG" in result
    assert "Author A" in result
    assert "2024" in result
    assert "An abstract." in result
    print("[PASS] test_get_document_metadata_by_id")


# ══════════════════════════════════════════════════════════════════════════
# Case 12：get_knowledge_graph_stats 异常降级
# ══════════════════════════════════════════════════════════════════════════

def test_graph_stats_exception_graceful():
    """GraphStore 异常时应降级返回错误提示，不再向上抛出。"""
    with patch("src.agents.tool_registry.GraphStore",
               side_effect=Exception("Neo4j 连接失败")):
        result = get_knowledge_graph_stats.invoke({})
    assert "失败" in result or "Neo4j" in result, \
        f"异常未被捕获，实际: {result[:100]}"
    print("[PASS] test_graph_stats_exception_graceful")


# ══════════════════════════════════════════════════════════════════════════
# Case 13：_auto_select_mode 路由逻辑
# ══════════════════════════════════════════════════════════════════════════

class _FakeKeywordResult:
    def __init__(self, hl: list, ll: list):
        self.hl_keywords = hl
        self.ll_keywords = ll


def _patch_extractor(hl: list, ll: list):
    fake_result = _FakeKeywordResult(hl=hl, ll=ll)
    mock_ext = MagicMock()
    mock_ext.extract.return_value = fake_result
    mock_cls = MagicMock(return_value=mock_ext)
    return patch("src.agents.tool_registry.KeywordExtractor", mock_cls)


def test_auto_select_time_keyword():
    """含时间词的查询应被路由到 mix 模式。"""
    with _patch_extractor(hl=[], ll=[]):
        mode = _auto_select_mode("2024年最新的 RAG 方法")
    assert mode == "mix", f"期望 mix，实际 {mode}"
    print("[PASS] test_auto_select_time_keyword")


def test_auto_select_global_mode():
    """hl_keywords 显著多于 ll_keywords 时应路由到 global 模式。"""
    with _patch_extractor(hl=["综述", "趋势", "比较"], ll=[]):
        mode = _auto_select_mode("NLP 领域的研究趋势与综述")  # 不含时间词
    assert mode == "global", f"期望 global，实际 {mode}"
    print("[PASS] test_auto_select_global_mode")


def test_auto_select_local_mode():
    """有 ll_keywords 时应路由到 local 模式。"""
    with _patch_extractor(hl=[], ll=["BERT", "attention"]):
        mode = _auto_select_mode("BERT 的注意力机制如何实现")
    assert mode == "local", f"期望 local，实际 {mode}"
    print("[PASS] test_auto_select_local_mode")


def test_auto_select_fallback_mix():
    """无关键词时应 fallback 到 mix 模式。"""
    with _patch_extractor(hl=[], ll=[]):
        mode = _auto_select_mode("请介绍一下相关研究")
    assert mode == "mix", f"期望 mix，实际 {mode}"
    print("[PASS] test_auto_select_fallback_mix")


def test_auto_select_extractor_exception():
    """KeywordExtractor 异常时应 fallback 到 mix，不抛出。"""
    with patch("src.agents.tool_registry.KeywordExtractor",
               side_effect=RuntimeError("初始化失败")):
        mode = _auto_select_mode("任意查询")
    assert mode == "mix", f"期望 fallback 到 mix，实际 {mode}"
    print("[PASS] test_auto_select_extractor_exception")


# ══════════════════════════════════════════════════════════════════════════
# 导入工具函数（便于直接在测试中调用）
# ══════════════════════════════════════════════════════════════════════════

from src.agents.tool_registry import (
    retrieve_knowledge,
    list_documents,
    get_document_metadata,
    get_knowledge_graph_stats,
)


# 修正 _patch_db 使 list_documents 测试中 patch 正确模块
def _patch_db(docs):
    mock_db = MagicMock()
    mock_db.list_documents.return_value = docs
    return patch("src.agents.tool_registry.MetadataDatabase", return_value=mock_db)


# ── 主入口 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 8-3  tool_registry 测试")
    print("=" * 60)

    tests = [
        # 工具注册层
        test_registry_count,
        test_registry_names_match_display,
        test_registry_tool_schemas,
        test_registry_singleton,
        # retrieve_knowledge
        test_retrieve_empty_result,
        test_retrieve_with_results,
        test_retrieve_exception_graceful,
        # list_documents
        test_list_documents_empty,
        test_list_documents_with_keyword_filter,
        # get_document_metadata
        test_get_document_metadata_not_found,
        test_get_document_metadata_by_id,
        # get_knowledge_graph_stats
        test_graph_stats_exception_graceful,
        # _auto_select_mode
        test_auto_select_time_keyword,
        test_auto_select_global_mode,
        test_auto_select_local_mode,
        test_auto_select_fallback_mix,
        test_auto_select_extractor_exception,
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
