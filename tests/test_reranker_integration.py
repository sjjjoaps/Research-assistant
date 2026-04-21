"""
Reranker 集成测试（P1-Step 4）

覆盖范围：
1. HybridRetriever — RERANKER_ENABLED=false，走原始 RRF 顺序，数量 == top_k
2. HybridRetriever — RERANKER_ENABLED=true，结果经 Reranker 重排，数量 <= top_k
3. HybridRetriever — Reranker API 超时，降级返回 RRF 顺序，长度和首元素正确
4. MixRetriever    — RERANKER_ENABLED=false，走原始 RRF 顺序
5. MixRetriever    — RERANKER_ENABLED=true，结果经 Reranker 重排
6. MixRetriever    — Reranker API 超时，降级返回 RRF 顺序，长度和首元素正确
"""
import sys
import os
from unittest.mock import MagicMock, patch

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.retriever import RetrievedChunk
from src.retrieval.reranker import get_reranker, APIReranker


# ── 辅助工厂 ──────────────────────────────────────────────────────────────────

def _chunk(content: str, idx: int = 0, section: str = "method") -> RetrievedChunk:
    return RetrievedChunk(content=content, file_path="paper.pdf",
                          chunk_index=idx, section_type=section)


def _mock_reranker(order: list[int], top_n: int = 3) -> MagicMock:
    """返回一个 mock reranker，rerank() 按 order 重排后截取 top_n。"""
    reranker = MagicMock()
    def _rerank(_, chunks):
        reordered = [chunks[i] for i in order if i < len(chunks)]
        return reordered[:top_n]
    reranker.rerank.side_effect = _rerank
    return reranker


# ── HybridRetriever 集成 ──────────────────────────────────────────────────────

def test_hybrid_reranker_disabled():
    """RERANKER_ENABLED=false 时 HybridRetriever 返回 RRF 原始顺序，数量 == top_k。"""
    from src.retrieval.hybrid_retriever import HybridRetriever

    chunks = [_chunk(f"doc{i}", i) for i in range(5)]
    retriever = HybridRetriever(top_k=3)

    # 修正 Finding 3：patch SemanticRetriever 类，控制其实例化后的行为
    mock_sem_instance = MagicMock()
    mock_sem_instance.retrieve.return_value = chunks[:4]
    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = chunks[1:]

    get_reranker.cache_clear()
    with patch("src.retrieval.hybrid_retriever.SemanticRetriever", return_value=mock_sem_instance), \
         patch.object(retriever, "bm25_retriever", mock_bm25), \
         patch("src.retrieval.hybrid_retriever.get_reranker", return_value=None):
        result = retriever.retrieve("query")

    # 修正 Finding 4：断言数量和首个元素内容
    assert len(result) == 3
    assert result[0].content == "doc1"   # doc1 同时出现在语义(rank2)和BM25(rank1)，RRF 得分最高
    print("[PASS] test_hybrid_reranker_disabled")


def test_hybrid_reranker_enabled():
    """RERANKER_ENABLED=true 时 HybridRetriever 结果经 Reranker 重排，数量 <= top_k。"""
    from src.retrieval.hybrid_retriever import HybridRetriever

    chunks = [_chunk(f"doc{i}", i) for i in range(5)]
    retriever = HybridRetriever(top_k=3)

    mock_sem_instance = MagicMock()
    mock_sem_instance.retrieve.return_value = chunks[:4]
    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = chunks[1:]

    # Reranker 将 fused 顺序反转（取前 top_n=3），top_k 裁剪确保数量正确
    mock_reranker = _mock_reranker(order=[3, 2, 1, 0], top_n=3)

    get_reranker.cache_clear()
    with patch("src.retrieval.hybrid_retriever.SemanticRetriever", return_value=mock_sem_instance), \
         patch.object(retriever, "bm25_retriever", mock_bm25), \
         patch("src.retrieval.hybrid_retriever.get_reranker", return_value=mock_reranker):
        result = retriever.retrieve("query")

    assert mock_reranker.rerank.called, "Reranker 未被调用"
    assert len(result) <= 3             # Finding 1 修复：不能超过 top_k
    assert result[0].content == "doc0"  # Reranker order=[3,2,1,0] → fused[3]=doc0 排首位
    print("[PASS] test_hybrid_reranker_enabled")


def test_hybrid_reranker_timeout_fallback():
    """Reranker API 超时时 HybridRetriever 降级，返回数量和首元素正确。"""
    from src.retrieval.hybrid_retriever import HybridRetriever

    chunks = [_chunk(f"doc{i}", i) for i in range(5)]
    retriever = HybridRetriever(top_k=2)

    mock_sem_instance = MagicMock()
    mock_sem_instance.retrieve.return_value = chunks[:3]
    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = chunks[1:]

    real_reranker = APIReranker(api_url="http://test/rerank", api_key="k",
                                top_n=2, timeout=1)

    get_reranker.cache_clear()
    with patch("src.retrieval.hybrid_retriever.SemanticRetriever", return_value=mock_sem_instance), \
         patch.object(retriever, "bm25_retriever", mock_bm25), \
         patch("src.retrieval.hybrid_retriever.get_reranker", return_value=real_reranker), \
         patch("requests.post", side_effect=requests.exceptions.Timeout("timeout")):
        result = retriever.retrieve("query")

    # 降级后 reranker 内部截取 top_n=2，再被 [:self.top_k=2] 裁剪
    assert len(result) == 2
    assert result[0].content == "doc1"  # 降级返回 fused[:top_n=2]，RRF 首位为 doc1
    print("[PASS] test_hybrid_reranker_timeout_fallback")


# ── MixRetriever 集成 ─────────────────────────────────────────────────────────

def test_mix_reranker_disabled():
    """RERANKER_ENABLED=false 时 MixRetriever 返回 RRF 原始顺序，数量 <= top_k。"""
    from src.retrieval.mix_retriever import MixRetriever

    chunks = [_chunk(f"doc{i}", i) for i in range(5)]
    retriever = MixRetriever(top_k=3)

    mock_sem = MagicMock()
    mock_sem.retrieve.return_value = chunks[:4]
    mock_local = MagicMock()
    mock_local.retrieve.return_value = chunks[1:]
    mock_global = MagicMock()
    mock_global.retrieve.return_value = []

    get_reranker.cache_clear()
    with patch.object(retriever, "semantic_retriever", mock_sem), \
         patch.object(retriever, "local_retriever", mock_local), \
         patch.object(retriever, "global_retriever", mock_global), \
         patch("src.retrieval.mix_retriever.get_reranker", return_value=None):
        result = retriever.retrieve("query")

    assert len(result) <= 3
    assert result[0].content in {"doc0", "doc1"}  # RRF 融合后靠前
    print("[PASS] test_mix_reranker_disabled")


def test_mix_reranker_enabled():
    """RERANKER_ENABLED=true 时 MixRetriever 结果经 Reranker 重排，数量 <= top_k。"""
    from src.retrieval.mix_retriever import MixRetriever

    chunks = [_chunk(f"doc{i}", i) for i in range(5)]
    retriever = MixRetriever(top_k=3)

    mock_sem = MagicMock()
    mock_sem.retrieve.return_value = chunks[:4]
    mock_local = MagicMock()
    mock_local.retrieve.return_value = chunks[1:]
    mock_global = MagicMock()
    mock_global.retrieve.return_value = []

    # Reranker 将 fused 反转，取前 3
    mock_reranker = _mock_reranker(order=[3, 2, 1, 0], top_n=3)

    get_reranker.cache_clear()
    with patch.object(retriever, "semantic_retriever", mock_sem), \
         patch.object(retriever, "local_retriever", mock_local), \
         patch.object(retriever, "global_retriever", mock_global), \
         patch("src.retrieval.mix_retriever.get_reranker", return_value=mock_reranker):
        result = retriever.retrieve("query")

    assert mock_reranker.rerank.called, "Reranker 未被调用"
    assert len(result) <= 3             # Finding 1 修复：不能超过 top_k
    # fuse_ranked_lists 三路融合后 fused[3] 即第4位；order=[3,2,1,0] → 首位为 doc3
    assert result[0].content == "doc3"
    print("[PASS] test_mix_reranker_enabled")


def test_mix_reranker_timeout_fallback():
    """Reranker API 超时时 MixRetriever 降级，返回数量和首元素正确。"""
    from src.retrieval.mix_retriever import MixRetriever

    chunks = [_chunk(f"doc{i}", i) for i in range(5)]
    retriever = MixRetriever(top_k=2)

    mock_sem = MagicMock()
    mock_sem.retrieve.return_value = chunks[:3]
    mock_local = MagicMock()
    mock_local.retrieve.return_value = chunks[1:]
    mock_global = MagicMock()
    mock_global.retrieve.return_value = []

    real_reranker = APIReranker(api_url="http://test/rerank", api_key="k",
                                top_n=2, timeout=1)

    get_reranker.cache_clear()
    with patch.object(retriever, "semantic_retriever", mock_sem), \
         patch.object(retriever, "local_retriever", mock_local), \
         patch.object(retriever, "global_retriever", mock_global), \
         patch("src.retrieval.mix_retriever.get_reranker", return_value=real_reranker), \
         patch("requests.post", side_effect=requests.exceptions.Timeout("timeout")):
        result = retriever.retrieve("query")

    assert len(result) == 2
    assert result[0].content in {"doc0", "doc1"}  # RRF 首位
    print("[PASS] test_mix_reranker_timeout_fallback")


# ── 主入口 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("P1-Step 4  Reranker 集成测试")
    print("=" * 60)

    tests = [
        test_hybrid_reranker_disabled,
        test_hybrid_reranker_enabled,
        test_hybrid_reranker_timeout_fallback,
        test_mix_reranker_disabled,
        test_mix_reranker_enabled,
        test_mix_reranker_timeout_fallback,
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
