"""
Reranker 专项测试（P1-Step 4）

覆盖范围：
1. 正常 API 返回，结果按 relevance_score 降序
2. API 超时，自动降级返回原始候选顺序（截取 top_n）
3. API 返回非 2xx，自动降级
4. 候选列表为空，返回空列表
5. 候选超过 max_candidates，自动截断后再调用 API
6. RERANKER_ENABLED=false，get_reranker() 返回 None
7. RERANKER_ENABLED=true 但 api_url 为空，get_reranker() 返回 None
8. API 返回 results 为空，降级返回原始顺序
"""
import sys
import os
from unittest.mock import MagicMock, patch

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.reranker import APIReranker, get_reranker
from src.retrieval.retriever import RetrievedChunk


# ── 辅助工厂 ──────────────────────────────────────────────────────────────────

def _chunk(content: str, idx: int = 0) -> RetrievedChunk:
    return RetrievedChunk(content=content, file_path="paper.pdf",
                          chunk_index=idx, section_type="method")


def _make_reranker(**kwargs) -> APIReranker:
    defaults = dict(api_url="http://reranker.test/rerank", api_key="test_key",
                    model="BAAI/bge-reranker-v2-m3", top_n=3,
                    timeout=5, max_candidates=10)
    defaults.update(kwargs)
    return APIReranker(**defaults)


def _mock_response(results: list[dict], status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"results": results}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# Case 1：正常 API 返回，结果按 relevance_score 降序
# ══════════════════════════════════════════════════════════════════════════════

def test_rerank_normal():
    chunks = [_chunk("A", 0), _chunk("B", 1), _chunk("C", 2)]
    api_results = [
        {"index": 2, "relevance_score": 0.95},
        {"index": 0, "relevance_score": 0.80},
        {"index": 1, "relevance_score": 0.60},
    ]
    reranker = _make_reranker(top_n=3)
    with patch("requests.post", return_value=_mock_response(api_results)):
        result = reranker.rerank("query", chunks)

    assert len(result) == 3
    assert result[0].content == "C"   # score 0.95
    assert result[1].content == "A"   # score 0.80
    assert result[2].content == "B"   # score 0.60


def test_rerank_top_n_limits_output():
    chunks = [_chunk(f"doc{i}", i) for i in range(5)]
    api_results = [{"index": i, "relevance_score": 1.0 - i * 0.1} for i in range(5)]
    reranker = _make_reranker(top_n=2)
    with patch("requests.post", return_value=_mock_response(api_results)):
        result = reranker.rerank("query", chunks)
    assert len(result) == 2


# ══════════════════════════════════════════════════════════════════════════════
# Case 2：API 超时，自动降级
# ══════════════════════════════════════════════════════════════════════════════

def test_rerank_timeout_fallback():
    chunks = [_chunk("A", 0), _chunk("B", 1), _chunk("C", 2), _chunk("D", 3)]
    reranker = _make_reranker(top_n=2)
    with patch("requests.post", side_effect=requests.exceptions.Timeout("timeout")):
        result = reranker.rerank("query", chunks)
    # 降级：返回原始顺序前 top_n 个
    assert len(result) == 2
    assert result[0].content == "A"
    assert result[1].content == "B"


# ══════════════════════════════════════════════════════════════════════════════
# Case 3：API 返回非 2xx，自动降级
# ══════════════════════════════════════════════════════════════════════════════

def test_rerank_http_error_fallback():
    chunks = [_chunk("X", 0), _chunk("Y", 1)]
    reranker = _make_reranker(top_n=2)
    with patch("requests.post", return_value=_mock_response([], status_code=500)):
        result = reranker.rerank("query", chunks)
    assert len(result) == 2
    assert result[0].content == "X"


# ══════════════════════════════════════════════════════════════════════════════
# Case 4：候选列表为空，返回空列表
# ══════════════════════════════════════════════════════════════════════════════

def test_rerank_empty_chunks():
    reranker = _make_reranker()
    result = reranker.rerank("query", [])
    assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# Case 5：候选超过 max_candidates，自动截断
# ══════════════════════════════════════════════════════════════════════════════

def test_rerank_truncates_to_max_candidates():
    chunks = [_chunk(f"doc{i}", i) for i in range(20)]
    reranker = _make_reranker(max_candidates=5, top_n=3)

    captured_payload = {}

    def fake_post(url, json, headers, timeout):  # noqa: ARG001
        captured_payload["documents"] = json["documents"]
        results = [{"index": i, "relevance_score": 1.0 - i * 0.1} for i in range(5)]
        return _mock_response(results)

    with patch("requests.post", side_effect=fake_post):
        result = reranker.rerank("query", chunks)

    # API 只收到 max_candidates=5 个文档
    assert len(captured_payload["documents"]) == 5
    assert len(result) == 3


# ══════════════════════════════════════════════════════════════════════════════
# Case 6：RERANKER_ENABLED=false，get_reranker() 返回 None
# ══════════════════════════════════════════════════════════════════════════════

def test_get_reranker_disabled():
    get_reranker.cache_clear()
    with patch("src.infrastructure.config.settings") as mock_settings:
        mock_settings.reranker_enabled = False
        result = get_reranker()
    assert result is None
    get_reranker.cache_clear()


# ══════════════════════════════════════════════════════════════════════════════
# Case 7：RERANKER_ENABLED=true 但 api_url 为空，get_reranker() 返回 None
# ══════════════════════════════════════════════════════════════════════════════

def test_get_reranker_no_url():
    get_reranker.cache_clear()
    with patch("src.infrastructure.config.settings") as mock_settings:
        mock_settings.reranker_enabled = True
        mock_settings.reranker_api_url = ""
        result = get_reranker()
    assert result is None
    get_reranker.cache_clear()


# ══════════════════════════════════════════════════════════════════════════════
# Case 8：API 返回 results 为空，降级返回原始顺序
# ══════════════════════════════════════════════════════════════════════════════

def test_rerank_empty_results_fallback():
    chunks = [_chunk("P", 0), _chunk("Q", 1)]
    reranker = _make_reranker(top_n=2)
    with patch("requests.post", return_value=_mock_response([])):
        result = reranker.rerank("query", chunks)
    # results 为空 → _call_api 抛 ValueError → 降级
    assert len(result) == 2
    assert result[0].content == "P"
