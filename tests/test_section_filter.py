"""
测试 section_filter 跨检索器统一透传（Phase 10-2）

测试策略：
- 各检索器 retrieve() 的 section_filter 行为单元测试（Mock 替换外部依赖）：
    1. LocalRetriever：section_filter 透传给 SemanticRetriever，GraphRetriever 结果后过滤
    2. LocalRetriever：空 section_filter 时，所有结果均返回（不过滤）
    3. MixRetriever：section_filter 透传给 SemanticRetriever 和 LocalRetriever，
       GlobalRetriever 结果后过滤
    4. HybridRetriever：section_filter 透传给 SemanticRetriever，BM25 结果后过滤
    5. GlobalRetriever：section_filter 在融合后后过滤（关系/图结果无 section_type）
    6. LightRAGDualRetriever：section_filter 在 merge_and_deduplicate 后后过滤
    7. _do_retrieve()：各 mode 路径均正确传递 section_filter（smoke test）
    8. section_filter="" 时结果集与不传 filter 等价（回归测试）

注意：
- 所有测试均不依赖 LLM / FAISS / Neo4j，用 MagicMock 替换所有外部依赖
- RetrievedChunk 用真实类（通过 src.retriever 导入），不 Mock
"""
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retriever import RetrievedChunk


# ── 辅助工厂 ──────────────────────────────────────────────────────────────────

def _chunk(content: str, section_type: str, file_path: str = "paper.pdf", chunk_index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        file_path=file_path,
        chunk_index=chunk_index,
        section_type=section_type,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Case 1-2：LocalRetriever
# ══════════════════════════════════════════════════════════════════════════════

def test_local_retriever_section_filter_applies():
    """section_filter='method' 时，LocalRetriever 应只返回 section_type=='method' 的 chunk。"""
    from src.retrieval.local_retriever import LocalRetriever

    method_chunk  = _chunk("方法描述", "method",  chunk_index=0)
    abstract_chunk = _chunk("摘要内容", "abstract", chunk_index=1)

    mock_sem = MagicMock()
    # SemanticRetriever 已在源头过滤，只返回 method chunk
    mock_sem.retrieve.return_value = [method_chunk]
    mock_sem_cls = MagicMock(return_value=mock_sem)

    mock_gr = MagicMock()
    # GraphRetriever 返回两种 chunk（它不支持原生过滤）
    mock_gr.retrieve.return_value = [method_chunk, abstract_chunk]
    mock_gr.close = MagicMock()
    mock_gr_cls = MagicMock(return_value=mock_gr)

    with patch("src.retrieval.local_retriever.SemanticRetriever", mock_sem_cls), \
         patch("src.retrieval.local_retriever.GraphRetriever", mock_gr_cls):
        retriever = LocalRetriever(top_k=5)
        chunks = retriever.retrieve("方法论问题", section_filter="method")

    # 所有返回 chunk 的 section_type 必须是 method
    assert all(c.section_type == "method" for c in chunks), \
        f"存在非 method chunk: {[c.section_type for c in chunks]}"
    # SemanticRetriever 必须收到 section_type="method"
    mock_sem.retrieve.assert_called_once()
    call_kwargs = mock_sem.retrieve.call_args
    assert call_kwargs.kwargs.get("section_type") == "method" or \
           (len(call_kwargs.args) > 1 and call_kwargs.args[1] == "method"), \
        f"SemanticRetriever 未收到 section_type=method，实际调用: {call_kwargs}"
    print("[PASS] test_local_retriever_section_filter_applies")


def test_local_retriever_no_filter_returns_all():
    """section_filter='' 时，LocalRetriever 应返回所有 chunk，不过滤。"""
    from src.retrieval.local_retriever import LocalRetriever

    method_chunk   = _chunk("方法描述", "method",   chunk_index=0)
    abstract_chunk = _chunk("摘要内容", "abstract", chunk_index=1)

    mock_sem = MagicMock()
    mock_sem.retrieve.return_value = [method_chunk, abstract_chunk]
    mock_sem_cls = MagicMock(return_value=mock_sem)

    mock_gr = MagicMock()
    mock_gr.retrieve.return_value = []
    mock_gr.close = MagicMock()
    mock_gr_cls = MagicMock(return_value=mock_gr)

    with patch("src.retrieval.local_retriever.SemanticRetriever", mock_sem_cls), \
         patch("src.retrieval.local_retriever.GraphRetriever", mock_gr_cls):
        retriever = LocalRetriever(top_k=5)
        chunks = retriever.retrieve("综合问题", section_filter="")

    section_types = {c.section_type for c in chunks}
    assert "method" in section_types and "abstract" in section_types, \
        f"无 filter 时应保留所有 section_type，实际: {section_types}"
    print("[PASS] test_local_retriever_no_filter_returns_all")


# ══════════════════════════════════════════════════════════════════════════════
# Case 3：MixRetriever
# ══════════════════════════════════════════════════════════════════════════════

def test_mix_retriever_section_filter_applies():
    """section_filter='experiment' 时，MixRetriever 应只返回 experiment 类型的 chunk。"""
    from src.retrieval.mix_retriever import MixRetriever

    exp_chunk  = _chunk("实验结果", "experiment", chunk_index=0)
    other_chunk = _chunk("引言内容", "introduction", chunk_index=1)

    mock_sem = MagicMock()
    mock_sem.retrieve.return_value = [exp_chunk]
    mock_sem_cls = MagicMock(return_value=mock_sem)

    # LocalRetriever：patch 整个类，返回 [exp_chunk, other_chunk]
    mock_local = MagicMock()
    mock_local.retrieve.return_value = [exp_chunk, other_chunk]
    mock_local.close = MagicMock()
    mock_local_cls = MagicMock(return_value=mock_local)

    # GlobalRetriever：返回 relation chunk（section_type='relation'）
    mock_global = MagicMock()
    mock_global.retrieve.return_value = [_chunk("关系结果", "relation", chunk_index=2)]
    mock_global.close = MagicMock()
    mock_global_cls = MagicMock(return_value=mock_global)

    with patch("src.retrieval.mix_retriever.SemanticRetriever", mock_sem_cls), \
         patch("src.retrieval.mix_retriever.LocalRetriever", mock_local_cls), \
         patch("src.retrieval.mix_retriever.GlobalRetriever", mock_global_cls):
        retriever = MixRetriever(top_k=10)
        chunks = retriever.retrieve("实验数据问题", section_filter="experiment")

    assert all(c.section_type == "experiment" for c in chunks), \
        f"存在非 experiment chunk: {[c.section_type for c in chunks]}"
    # LocalRetriever 应收到 section_filter='experiment'
    mock_local.retrieve.assert_called_once()
    local_call = mock_local.retrieve.call_args
    assert local_call.kwargs.get("section_filter") == "experiment" or \
           (len(local_call.args) > 1 and local_call.args[1] == "experiment"), \
        f"LocalRetriever 未收到 section_filter=experiment，实际: {local_call}"
    print("[PASS] test_mix_retriever_section_filter_applies")


# ══════════════════════════════════════════════════════════════════════════════
# Case 4：HybridRetriever
# ══════════════════════════════════════════════════════════════════════════════

def test_hybrid_retriever_section_filter_applies():
    """section_filter='abstract' 时，HybridRetriever 应只返回 abstract 类型的 chunk。"""
    from src.hybrid_retriever import HybridRetriever

    abs_chunk   = _chunk("摘要内容", "abstract",     file_path="paper.pdf", chunk_index=0)
    method_chunk = _chunk("方法内容", "method",      file_path="paper.pdf", chunk_index=1)

    mock_sem = MagicMock()
    mock_sem.retrieve.return_value = [abs_chunk]  # semantic 已过滤
    mock_sem_cls = MagicMock(return_value=mock_sem)

    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = [abs_chunk, method_chunk]  # BM25 未过滤
    mock_bm25_cls = MagicMock(return_value=mock_bm25)

    with patch("src.hybrid_retriever.SemanticRetriever", mock_sem_cls), \
         patch("src.hybrid_retriever.BM25Retriever", mock_bm25_cls):
        retriever = HybridRetriever(top_k=10)
        chunks = retriever.retrieve("摘要问题", section_filter="abstract")

    assert all(c.section_type == "abstract" for c in chunks), \
        f"存在非 abstract chunk: {[c.section_type for c in chunks]}"
    print("[PASS] test_hybrid_retriever_section_filter_applies")


# ══════════════════════════════════════════════════════════════════════════════
# Case 5：GlobalRetriever
# ══════════════════════════════════════════════════════════════════════════════

def test_global_retriever_section_filter_filters_out_relation_chunks():
    """GlobalRetriever 的关系 chunk 无 section_type 匹配，过滤后返回空列表。"""
    from src.retrieval.global_retriever import GlobalRetriever

    # relation chunk 的 section_type 固定是 "relation"，不匹配 "method"
    relation_chunk = _chunk("来源实体: A\n目标实体: B", "relation", chunk_index=0)

    mock_rel_vs = MagicMock()
    mock_rel_vs.similarity_search.return_value = []
    mock_rel_vs_cls = MagicMock(return_value=mock_rel_vs)

    mock_gr = MagicMock()
    mock_gr.retrieve.return_value = [relation_chunk]
    mock_gr.close = MagicMock()
    mock_gr_cls = MagicMock(return_value=mock_gr)

    mock_kw = MagicMock()
    mock_kw.extract.return_value = MagicMock(hl_keywords=[], ll_keywords=[])
    mock_kw_cls = MagicMock(return_value=mock_kw)

    with patch("src.retrieval.global_retriever.RelationVectorStore", mock_rel_vs_cls), \
         patch("src.retrieval.global_retriever.GraphRetriever", mock_gr_cls), \
         patch("src.retrieval.global_retriever.KeywordExtractor", mock_kw_cls):
        retriever = GlobalRetriever(top_k=5)
        # 传入 section_filter="method"，关系 chunk 不匹配，应被过滤掉
        chunks = retriever.retrieve("宏观趋势", section_filter="method")

    assert chunks == [], f"GlobalRetriever 过滤后应无结果，实际: {chunks}"
    print("[PASS] test_global_retriever_section_filter_filters_out_relation_chunks")


# ══════════════════════════════════════════════════════════════════════════════
# Case 6：LightRAGDualRetriever
# ══════════════════════════════════════════════════════════════════════════════

def test_lightrag_retriever_section_filter_post_filters():
    """LightRAGDualRetriever section_filter 在 merge 后后过滤，不匹配的 chunk 被丢弃。"""
    from src.retrieval.lightrag_retriever import LightRAGDualRetriever

    method_chunk = _chunk("方法内容", "method",   chunk_index=0)
    entity_chunk = _chunk("实体内容", "entity",   chunk_index=1)

    mock_kw_ext = MagicMock()
    mock_kw_ext.extract.return_value = MagicMock(ll_keywords=["method"], hl_keywords=[])

    mock_gr = MagicMock()
    mock_gr.retrieve.return_value = [method_chunk, entity_chunk]
    mock_gr.close = MagicMock()

    retriever = LightRAGDualRetriever(top_k=5)
    retriever._keyword_extractor = mock_kw_ext
    retriever._graph_retriever   = mock_gr
    retriever._graph_store       = None  # 不触发 High-Level 检索（hl_keywords 为空）

    chunks = retriever.retrieve("方法论查询", section_filter="method")

    assert all(c.section_type == "method" for c in chunks), \
        f"存在非 method chunk: {[c.section_type for c in chunks]}"
    print("[PASS] test_lightrag_retriever_section_filter_post_filters")


# ══════════════════════════════════════════════════════════════════════════════
# Case 7：_do_retrieve() 各 mode 路径传递 section_filter
# ══════════════════════════════════════════════════════════════════════════════





def test_do_retrieve_local_mode_passes_section_filter():
    """_do_retrieve(mode='local') 应把 section_filter 传给 LocalRetriever.retrieve()。"""
    from src.agents.tool_registry import _do_retrieve

    method_chunk = _chunk("方法内容", "method", chunk_index=0)

    mock_local_inst = MagicMock()
    mock_local_inst.retrieve.return_value = [method_chunk]
    mock_local_inst.close = MagicMock()
    mock_local_cls = MagicMock(return_value=mock_local_inst)

    mock_ltm_inst = MagicMock()
    mock_ltm_inst.get_best_mode.return_value = None
    mock_ltm_inst.record_strategy_result.return_value = None

    mock_ltm_module = MagicMock()
    mock_ltm_module.get_instance.return_value = mock_ltm_inst

    # LongTermMemory 和 classify_question_type 通过局部 import 在 _do_retrieve 内加载，
    # 需要 patch 源模块路径（src.core.long_term_memory）
    with patch("src.retrieval.local_retriever.LocalRetriever", mock_local_cls), \
         patch("src.core.long_term_memory.LongTermMemory", mock_ltm_module), \
         patch("src.core.long_term_memory.classify_question_type", return_value="specific"):
        _do_retrieve(
            query="方法论问题",
            mode="local",
            top_k=5,
            section_filter="method",
            year_from=0,
            year_to=0,
        )

    # LocalRetriever 必须被以 section_filter="method" 调用
    mock_local_inst.retrieve.assert_called_once()
    call_kwargs = mock_local_inst.retrieve.call_args
    sf_passed = call_kwargs.kwargs.get("section_filter") or \
                (call_kwargs.args[1] if len(call_kwargs.args) > 1 else None)
    assert sf_passed == "method", \
        f"_do_retrieve 未传 section_filter=method 给 LocalRetriever，实际: {call_kwargs}"
    print("[PASS] test_do_retrieve_local_mode_passes_section_filter")


# ══════════════════════════════════════════════════════════════════════════════
# Case 8：section_filter="" 回归测试
# ══════════════════════════════════════════════════════════════════════════════

def test_local_retriever_empty_filter_is_idempotent():
    """
    section_filter='' 时与 section_filter 未传时行为完全相同：
    无额外过滤，结果集不变。
    """
    from src.retrieval.local_retriever import LocalRetriever

    chunks_all = [
        _chunk("方法描述", "method",      chunk_index=0),
        _chunk("摘要内容", "abstract",    chunk_index=1),
        _chunk("结论内容", "conclusion",  chunk_index=2),
    ]

    mock_sem = MagicMock()
    mock_sem.retrieve.return_value = chunks_all
    mock_sem_cls = MagicMock(return_value=mock_sem)

    mock_gr = MagicMock()
    mock_gr.retrieve.return_value = []
    mock_gr.close = MagicMock()
    mock_gr_cls = MagicMock(return_value=mock_gr)

    with patch("src.retrieval.local_retriever.SemanticRetriever", mock_sem_cls), \
         patch("src.retrieval.local_retriever.GraphRetriever", mock_gr_cls):
        retriever = LocalRetriever(top_k=10)
        result_no_filter  = retriever.retrieve("综合查询")
        result_empty_str  = retriever.retrieve("综合查询", section_filter="")

    assert len(result_no_filter) == len(result_empty_str), \
        "空 filter 与无 filter 结果数量不同"
    assert {c.section_type for c in result_no_filter} == {c.section_type for c in result_empty_str}, \
        "空 filter 与无 filter 结果集不同"
    print("[PASS] test_local_retriever_empty_filter_is_idempotent")


# ── 主入口 ────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# Case 9-12：_do_retrieve() 补充 mix/hybrid/global/dual/semantic 路径
# ══════════════════════════════════════════════════════════════════════════════

def _ltm_patch():
    """公用 LongTermMemory patch helper（局部 import 路径）。"""
    mock_inst = MagicMock()
    mock_inst.get_best_mode.return_value = None
    mock_inst.record_strategy_result.return_value = None
    mock_module = MagicMock()
    mock_module.get_instance.return_value = mock_inst
    return mock_module


def _run_do_retrieve(mode: str, retriever_cls_patch_path: str, mock_inst: MagicMock, mock_cls: MagicMock):
    """通用辅助：patch 指定检索器类，运行 _do_retrieve，验证 section_filter 透传。"""
    from src.agents.tool_registry import _do_retrieve
    mock_ltm = _ltm_patch()
    with patch(retriever_cls_patch_path, mock_cls), \
         patch("src.core.long_term_memory.LongTermMemory", mock_ltm), \
         patch("src.core.long_term_memory.classify_question_type", return_value="specific"):
        _do_retrieve(query="测试", mode=mode, top_k=5,
                     section_filter="method", year_from=0, year_to=0)
    return mock_inst.retrieve.call_args


def test_do_retrieve_mix_mode_passes_section_filter():
    """_do_retrieve(mode='mix') 应把 section_filter 传给 MixRetriever.retrieve()。"""
    method_chunk = _chunk("方法内容", "method", chunk_index=0)
    mock_inst = MagicMock()
    mock_inst.retrieve.return_value = [method_chunk]
    mock_inst.close = MagicMock()
    mock_cls = MagicMock(return_value=mock_inst)

    call_args = _run_do_retrieve("mix", "src.retrieval.mix_retriever.MixRetriever", mock_inst, mock_cls)
    sf = call_args.kwargs.get("section_filter") or (call_args.args[1] if len(call_args.args) > 1 else None)
    assert sf == "method", f"mix 模式未传 section_filter，实际: {call_args}"
    print("[PASS] test_do_retrieve_mix_mode_passes_section_filter")


def test_do_retrieve_hybrid_mode_passes_section_filter():
    """_do_retrieve(mode='hybrid') 应把 section_filter 传给 HybridRetriever.retrieve()。"""
    method_chunk = _chunk("方法内容", "method", chunk_index=0)
    mock_inst = MagicMock()
    mock_inst.retrieve.return_value = [method_chunk]
    mock_cls = MagicMock(return_value=mock_inst)

    call_args = _run_do_retrieve("hybrid", "src.hybrid_retriever.HybridRetriever", mock_inst, mock_cls)
    sf = call_args.kwargs.get("section_filter") or (call_args.args[1] if len(call_args.args) > 1 else None)
    assert sf == "method", f"hybrid 模式未传 section_filter，实际: {call_args}"
    print("[PASS] test_do_retrieve_hybrid_mode_passes_section_filter")


def test_do_retrieve_global_mode_passes_section_filter():
    """_do_retrieve(mode='global') 应把 section_filter 传给 GlobalRetriever.retrieve()。"""
    mock_inst = MagicMock()
    mock_inst.retrieve.return_value = []
    mock_inst.close = MagicMock()
    mock_cls = MagicMock(return_value=mock_inst)

    call_args = _run_do_retrieve("global", "src.retrieval.global_retriever.GlobalRetriever", mock_inst, mock_cls)
    sf = call_args.kwargs.get("section_filter") or (call_args.args[1] if len(call_args.args) > 1 else None)
    assert sf == "method", f"global 模式未传 section_filter，实际: {call_args}"
    print("[PASS] test_do_retrieve_global_mode_passes_section_filter")


def test_do_retrieve_dual_mode_passes_section_filter():
    """_do_retrieve(mode='dual') 应把 section_filter 传给 LightRAGDualRetriever.retrieve()。"""
    mock_inst = MagicMock()
    mock_inst.retrieve.return_value = []
    mock_inst.close = MagicMock()
    mock_cls = MagicMock(return_value=mock_inst)

    call_args = _run_do_retrieve("dual", "src.retrieval.lightrag_retriever.LightRAGDualRetriever", mock_inst, mock_cls)
    sf = call_args.kwargs.get("section_filter") or (call_args.args[2] if len(call_args.args) > 2 else None)
    assert sf == "method", f"dual 模式未传 section_filter，实际: {call_args}"
    print("[PASS] test_do_retrieve_dual_mode_passes_section_filter")


def test_do_retrieve_semantic_mode_passes_section_filter():
    """_do_retrieve(mode='semantic') 应透传 section_filter 给 SemanticRetriever。"""
    from src.agents.tool_registry import _do_retrieve

    method_chunk = _chunk("方法内容", "method", chunk_index=0)
    mock_sem_inst = MagicMock()
    mock_sem_inst.retrieve.return_value = [method_chunk]
    mock_sem_cls = MagicMock(return_value=mock_sem_inst)

    mock_ltm = _ltm_patch()
    with patch("src.retriever.SemanticRetriever", mock_sem_cls), \
         patch("src.core.long_term_memory.LongTermMemory", mock_ltm), \
         patch("src.core.long_term_memory.classify_question_type", return_value="specific"):
        _do_retrieve(query="摘要信息", mode="semantic", top_k=5,
                     section_filter="method", year_from=0, year_to=0)

    call_args = mock_sem_inst.retrieve.call_args
    sf = call_args.kwargs.get("section_type") or (call_args.args[1] if len(call_args.args) > 1 else None)
    assert sf == "method", f"semantic 模式未传 section_type，实际: {call_args}"
    print("[PASS] test_do_retrieve_semantic_mode_passes_section_filter")


def test_fetch_k_expands_when_section_filter_set():
    """section_filter 非空时，fetch_k 应扩容（传入检索器的 top_k 应 > 原始 top_k）。"""
    from src.agents.tool_registry import _do_retrieve

    mock_inst = MagicMock()
    mock_inst.retrieve.return_value = []
    mock_inst.close = MagicMock()
    mock_cls = MagicMock(return_value=mock_inst)

    mock_ltm = _ltm_patch()
    with patch("src.retrieval.local_retriever.LocalRetriever", mock_cls), \
         patch("src.core.long_term_memory.LongTermMemory", mock_ltm), \
         patch("src.core.long_term_memory.classify_question_type", return_value="specific"):
        _do_retrieve(query="测试", mode="local", top_k=5,
                     section_filter="method", year_from=0, year_to=0)

    # LocalRetriever 被构造时传入的 top_k 应该是 5 * 4 = 20（fetch_k 扩容）
    ctor_args = mock_cls.call_args
    top_k_used = ctor_args.kwargs.get("top_k") or (ctor_args.args[0] if ctor_args.args else None)
    assert top_k_used is not None and top_k_used > 5, \
        f"section_filter 时 fetch_k 未扩容，LocalRetriever(top_k={top_k_used})"
    print("[PASS] test_fetch_k_expands_when_section_filter_set")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 10-2  section_filter 统一透传测试")
    print("=" * 60)

    tests = [
        test_local_retriever_section_filter_applies,
        test_local_retriever_no_filter_returns_all,
        test_mix_retriever_section_filter_applies,
        test_hybrid_retriever_section_filter_applies,
        test_global_retriever_section_filter_filters_out_relation_chunks,
        test_lightrag_retriever_section_filter_post_filters,
        test_do_retrieve_local_mode_passes_section_filter,
        test_local_retriever_empty_filter_is_idempotent,
        # Review 补充
        test_do_retrieve_mix_mode_passes_section_filter,
        test_do_retrieve_hybrid_mode_passes_section_filter,
        test_do_retrieve_global_mode_passes_section_filter,
        test_do_retrieve_dual_mode_passes_section_filter,
        test_do_retrieve_semantic_mode_passes_section_filter,
        test_fetch_k_expands_when_section_filter_set,
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
