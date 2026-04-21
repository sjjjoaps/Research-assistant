"""
测试 P5-Step 1：原生工具注册表（build_native_tool_registry）

测试覆盖：
1. build_native_tool_registry() 返回 8 个 Tool 对象
2. 每个 Tool 有 .name / .description / .invoke() / .to_openai_schema() / .args_schema
3. to_openai_schema() 格式符合 OpenAI function calling 规范
4. args_schema 返回与 parameters 相同的 dict（兼容性）
5. _infer_schema() — str/int/bool/float 基本类型映射
6. _infer_schema() — 有默认值的参数不计入 required
7. _infer_schema() — _PARAM_ENUMS 中的参数有 enum 约束
8. build_native_tool_registry() 单例（lru_cache）
9. Tool.invoke() — 正常调用转发到业务函数
10. Tool.invoke() — 业务函数异常时不向上抛出（由业务函数内部处理）
"""
import sys
import os
from typing import Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.tool_registry import (
    build_native_tool_registry,
    Tool,
    _infer_schema,
    _annotation_to_prop,
    _PARAM_ENUMS,
    retrieve_knowledge,
    list_documents,
    get_document_metadata,
    get_knowledge_graph_stats,
    _retrieve_knowledge_func,
    _list_documents_func,
    _get_document_metadata_func,
    _get_knowledge_graph_stats_func,
)


# ── 辅助工厂 ──────────────────────────────────────────────────────────────

def _make_chunk(content: str, file_path: str = "paper.pdf", chunk_index: int = 0,
                section_type: str = "method"):
    c = MagicMock()
    c.content      = content
    c.file_path    = file_path
    c.chunk_index  = chunk_index
    c.section_type = section_type
    return c


# ══════════════════════════════════════════════════════════════════════════
# Case 1：工具数量
# ══════════════════════════════════════════════════════════════════════════

def test_native_registry_count():
    """build_native_tool_registry() 应返回 8 个 Tool。"""
    tools = build_native_tool_registry()
    assert len(tools) == 8, f"期望 8 个工具，实际 {len(tools)}"
    print("[PASS] test_native_registry_count")


# ══════════════════════════════════════════════════════════════════════════
# Case 2：每个 Tool 的必要属性
# ══════════════════════════════════════════════════════════════════════════

def test_native_tool_attributes():
    """每个 Tool 必须有 name / description / args_schema，且 invoke/to_openai_schema 可调用。"""
    tools = build_native_tool_registry()
    for t in tools:
        assert isinstance(t, Tool), f"{t} 不是 Tool 实例"
        assert t.name,              f"工具缺少 name: {t}"
        assert t.description,       f"工具 {t.name} 缺少 description"
        assert isinstance(t.args_schema, dict), f"工具 {t.name} args_schema 不是 dict"
        assert callable(t.invoke),          f"工具 {t.name} invoke 不可调用"
        assert callable(t.to_openai_schema), f"工具 {t.name} to_openai_schema 不可调用"
    print("[PASS] test_native_tool_attributes")


# ══════════════════════════════════════════════════════════════════════════
# Case 3：to_openai_schema 格式
# ══════════════════════════════════════════════════════════════════════════

def test_to_openai_schema_format():
    """to_openai_schema() 必须含 type/function/function.name/function.description/function.parameters。"""
    tools = build_native_tool_registry()
    for t in tools:
        schema = t.to_openai_schema()
        assert schema.get("type") == "function", f"{t.name} schema.type 不为 'function'"
        fn = schema.get("function", {})
        assert fn.get("name") == t.name,        f"{t.name} schema.function.name 不匹配"
        assert fn.get("description"),           f"{t.name} schema.function.description 为空"
        assert "parameters" in fn,              f"{t.name} schema.function.parameters 缺失"
        params = fn["parameters"]
        assert params.get("type") == "object",  f"{t.name} parameters.type 不为 'object'"
        assert "properties" in params,          f"{t.name} parameters.properties 缺失"
    print("[PASS] test_to_openai_schema_format")


# ══════════════════════════════════════════════════════════════════════════
# Case 4：args_schema 与 parameters 一致
# ══════════════════════════════════════════════════════════════════════════

def test_args_schema_equals_parameters():
    """args_schema 应返回与 parameters 相同的 dict。"""
    tools = build_native_tool_registry()
    for t in tools:
        assert t.args_schema is t.parameters, \
            f"{t.name}: args_schema 与 parameters 不是同一对象"
    print("[PASS] test_args_schema_equals_parameters")


# ══════════════════════════════════════════════════════════════════════════
# Case 5：_infer_schema 基本类型映射
# ══════════════════════════════════════════════════════════════════════════

def test_infer_schema_basic_types():
    """str/int/bool/float 应映射到对应 JSON 类型。"""
    def sample_func(a: str, b: int, c: bool, d: float) -> str: ...
    schema = _infer_schema(sample_func)
    props = schema["properties"]
    assert props["a"]["type"] == "string",  f"str 未映射 string: {props['a']}"
    assert props["b"]["type"] == "integer", f"int 未映射 integer: {props['b']}"
    assert props["c"]["type"] == "boolean", f"bool 未映射 boolean: {props['c']}"
    assert props["d"]["type"] == "number",  f"float 未映射 number: {props['d']}"
    assert schema["required"] == ["a", "b", "c", "d"]
    print("[PASS] test_infer_schema_basic_types")


# ══════════════════════════════════════════════════════════════════════════
# Case 6：有默认值的参数不计入 required
# ══════════════════════════════════════════════════════════════════════════

def test_infer_schema_optional_not_required():
    """有默认值的参数不应出现在 required 列表中。"""
    def sample_func(query: str, top_k: int = 5, mode: str = "auto") -> str: ...
    schema = _infer_schema(sample_func)
    assert "query" in schema.get("required", []),   "必填参数 query 未在 required 中"
    assert "top_k" not in schema.get("required", []), "可选参数 top_k 不应在 required 中"
    assert "mode" not in schema.get("required", []),  "可选参数 mode 不应在 required 中"
    print("[PASS] test_infer_schema_optional_not_required")


# ══════════════════════════════════════════════════════════════════════════
# Case 7：_PARAM_ENUMS 枚举约束
# ══════════════════════════════════════════════════════════════════════════

def test_infer_schema_param_enums():
    """mode / retriever_mode / section_filter / memory_type 参数应有 enum 约束。"""
    def sample_func(query: str, mode: str = "auto", section_filter: str = "") -> str: ...
    schema = _infer_schema(sample_func)
    props = schema["properties"]
    assert "enum" in props.get("mode", {}), "mode 缺少 enum 约束"
    assert props["mode"]["enum"] == _PARAM_ENUMS["mode"]
    assert "enum" in props.get("section_filter", {}), "section_filter 缺少 enum 约束"
    print("[PASS] test_infer_schema_param_enums")


# ══════════════════════════════════════════════════════════════════════════
# Case 8：单例缓存
# ══════════════════════════════════════════════════════════════════════════

def test_native_registry_singleton():
    """两次调用 build_native_tool_registry() 应返回同一对象。"""
    a = build_native_tool_registry()
    b = build_native_tool_registry()
    assert a is b, "lru_cache 失效，两次调用返回不同对象"
    print("[PASS] test_native_registry_singleton")


# ══════════════════════════════════════════════════════════════════════════
# Case 9：Tool.invoke() 转发调用
# ══════════════════════════════════════════════════════════════════════════

def test_tool_invoke_forwards_to_func():
    """Tool.invoke() 应将参数传给 func 并返回其结果。"""
    fake_func = MagicMock(return_value="hello")
    t = Tool(name="test_tool", description="desc", func=fake_func,
             parameters={"type": "object", "properties": {"x": {"type": "string"}}})
    result = t.invoke({"x": "world"})
    fake_func.assert_called_once_with(x="world")
    assert result == "hello"
    print("[PASS] test_tool_invoke_forwards_to_func")


# ══════════════════════════════════════════════════════════════════════════
# Case 10：retrieve_knowledge Tool — 检索空结果
# ══════════════════════════════════════════════════════════════════════════

def test_native_retrieve_knowledge_empty():
    """原生 retrieve_knowledge 工具在空结果时应返回友好提示。"""
    tools = build_native_tool_registry()
    rk = next(t for t in tools if t.name == "retrieve_knowledge")
    with patch("src.agents.tool_registry._do_retrieve", return_value=[]):
        result = rk.invoke({"query": "不存在的内容"})
    assert "检索结果为空" in result, f"未找到预期提示: {result[:100]}"
    print("[PASS] test_native_retrieve_knowledge_empty")


# ══════════════════════════════════════════════════════════════════════════
# Case 11：retrieve_knowledge Tool — 有结果时格式化
# ══════════════════════════════════════════════════════════════════════════

def test_native_retrieve_knowledge_with_results():
    """原生 retrieve_knowledge 工具在有结果时应包含来源标签。"""
    tools = build_native_tool_registry()
    rk = next(t for t in tools if t.name == "retrieve_knowledge")
    fake_chunks = [
        _make_chunk("RAG 的核心思想是…", file_path="paper1.pdf", chunk_index=3),
    ]
    with patch("src.agents.tool_registry._do_retrieve", return_value=fake_chunks):
        result = rk.invoke({"query": "RAG"})
    assert "[来源: paper1.pdf#chunk-3]" in result
    assert "检索到 1 个相关片段" in result
    print("[PASS] test_native_retrieve_knowledge_with_results")


# ══════════════════════════════════════════════════════════════════════════
# Case 12：_annotation_to_prop — Optional[list[T]] 修复验证
# ══════════════════════════════════════════════════════════════════════════

def test_infer_schema_optional_list():
    """Optional[list[str]] 应被推断为 array，而不是 string。"""
    def sample_func(tags: Optional[list[str]] = None) -> str: ...
    schema = _infer_schema(sample_func)
    prop = schema["properties"]["tags"]
    assert prop["type"] == "array",  f"Optional[list[str]] 应为 array，实际: {prop}"
    assert prop["items"]["type"] == "string", f"items.type 应为 string，实际: {prop}"
    assert "tags" not in schema.get("required", []), "有默认值的参数不应在 required 中"
    print("[PASS] test_infer_schema_optional_list")


def test_annotation_to_prop_list():
    """_annotation_to_prop 对 list[int] 应返回 array of integer。"""
    prop = _annotation_to_prop(list[int])
    assert prop == {"type": "array", "items": {"type": "integer"}}, f"实际: {prop}"
    print("[PASS] test_annotation_to_prop_list")


def test_annotation_to_prop_optional_scalar():
    """_annotation_to_prop 对 Optional[int] 应返回 integer。"""
    prop = _annotation_to_prop(Optional[int])
    assert prop["type"] == "integer", f"Optional[int] 应为 integer，实际: {prop}"
    print("[PASS] test_annotation_to_prop_optional_scalar")


# ══════════════════════════════════════════════════════════════════════════
# Case 13-16：_xxx_func 与 @tool 版本语义一致性
# ══════════════════════════════════════════════════════════════════════════

def _make_doc(title="Test Paper", authors="Author A", year=2024,
              abstract="An abstract.", doc_id="doc_001",
              file_path="/data/test.pdf", institution="MIT", keywords="RAG"):
    doc = MagicMock()
    doc.title = title; doc.authors = authors; doc.year = year
    doc.abstract = abstract; doc.doc_id = doc_id; doc.file_path = file_path
    doc.institution = institution; doc.keywords = keywords
    return doc


def test_retrieve_knowledge_semantic_consistency():
    """_retrieve_knowledge_func 与 @tool retrieve_knowledge 空结果输出一致。"""
    with patch("src.agents.tool_registry._do_retrieve", return_value=[]):
        result_tool = retrieve_knowledge.invoke({"query": "test"})
        result_func = _retrieve_knowledge_func(query="test")
    assert result_tool == result_func, (
        f"两套实现输出不一致\n@tool: {result_tool!r}\n_func: {result_func!r}"
    )
    print("[PASS] test_retrieve_knowledge_semantic_consistency")


def test_list_documents_semantic_consistency():
    """_list_documents_func 与 @tool list_documents 空库时输出一致。"""
    mock_db = MagicMock()
    mock_db.list_documents.return_value = []
    with patch("src.agents.tool_registry.MetadataDatabase", return_value=mock_db):
        result_tool = list_documents.invoke({})
        result_func = _list_documents_func()
    assert result_tool == result_func, (
        f"两套实现输出不一致\n@tool: {result_tool!r}\n_func: {result_func!r}"
    )
    print("[PASS] test_list_documents_semantic_consistency")


def test_get_document_metadata_semantic_consistency():
    """_get_document_metadata_func 与 @tool get_document_metadata 未找到时输出一致。"""
    mock_db = MagicMock()
    mock_db.get_document_by_doc_id.return_value = None
    mock_db.list_documents.return_value = []
    with patch("src.agents.tool_registry.MetadataDatabase", return_value=mock_db):
        result_tool = get_document_metadata.invoke({"doc_title_or_id": "不存在"})
        result_func = _get_document_metadata_func(doc_title_or_id="不存在")
    assert result_tool == result_func, (
        f"两套实现输出不一致\n@tool: {result_tool!r}\n_func: {result_func!r}"
    )
    print("[PASS] test_get_document_metadata_semantic_consistency")


def test_get_graph_stats_semantic_consistency():
    """_get_knowledge_graph_stats_func 与 @tool 版本在异常时输出一致。"""
    with patch("src.agents.tool_registry.GraphStore",
               side_effect=Exception("Neo4j 连接失败")):
        result_tool = get_knowledge_graph_stats.invoke({})
        result_func = _get_knowledge_graph_stats_func()
    assert result_tool == result_func, (
        f"两套实现输出不一致\n@tool: {result_tool!r}\n_func: {result_func!r}"
    )
    print("[PASS] test_get_graph_stats_semantic_consistency")


# ── 主入口 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("P5-Step 1  原生工具注册表测试")
    print("=" * 60)

    tests = [
        test_native_registry_count,
        test_native_tool_attributes,
        test_to_openai_schema_format,
        test_args_schema_equals_parameters,
        test_infer_schema_basic_types,
        test_infer_schema_optional_not_required,
        test_infer_schema_param_enums,
        test_native_registry_singleton,
        test_tool_invoke_forwards_to_func,
        test_native_retrieve_knowledge_empty,
        test_native_retrieve_knowledge_with_results,
        test_infer_schema_optional_list,
        test_annotation_to_prop_list,
        test_annotation_to_prop_optional_scalar,
        test_retrieve_knowledge_semantic_consistency,
        test_list_documents_semantic_consistency,
        test_get_document_metadata_semantic_consistency,
        test_get_graph_stats_semantic_consistency,
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
