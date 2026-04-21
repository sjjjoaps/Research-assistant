"""
API Reranker（P1-Step 1）

封装外部 Reranker API 调用，统一输入输出，含超时/降级逻辑。

降级策略：API 调用失败时直接截取 chunks[:top_n]，保留多路召回原始顺序。
（chunks 此时已是 RRF 融合后的单列表，再走 RRF 无实际意义。）

Public API：
    APIReranker.rerank(query, chunks) → list[RetrievedChunk]
    get_reranker()                    → APIReranker | None
"""
from __future__ import annotations

import logging
from functools import lru_cache

import requests

from src.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


class APIReranker:
    """封装外部 Reranker API 调用，统一输入输出，含超时/降级。"""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = 5,
        timeout: int = 10,
        max_candidates: int = 50,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.top_n = top_n
        self.timeout = timeout
        self.max_candidates = max_candidates

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        对候选 chunk 列表精排，返回按相关性降序排列的 top_n 个 chunk。

        - 自动截断超过 max_candidates 的候选
        - API 失败时降级返回 chunks[:top_n]（保留原始顺序）
        """
        if not chunks:
            return []

        candidates = chunks[: self.max_candidates]

        try:
            return self._call_api(query, candidates)
        except Exception as exc:
            logger.warning("Reranker API 失败，降级返回原始候选顺序: %s", exc)
            return candidates[: self.top_n]

    # ── 内部实现 ──────────────────────────────────────────────────────────────

    def _call_api(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        调用外部 Reranker API（兼容 HuggingFace TEI / SiliconFlow / Jina 等标准接口）。

        请求格式（POST JSON）：
            {"model": "...", "query": "...", "documents": ["text1", "text2", ...]}

        响应格式（期望）：
            {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
        """
        documents = [c.content for c in chunks]
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            raise ValueError(f"Reranker API 返回空 results，响应体: {data}")

        # 按 relevance_score 降序排列，取 top_n
        sorted_results = sorted(results, key=lambda r: r.get("relevance_score", 0.0), reverse=True)
        top_results = sorted_results[: self.top_n]

        reranked: list[RetrievedChunk] = []
        for r in top_results:
            idx = r.get("index")
            if idx is not None and 0 <= idx < len(chunks):
                reranked.append(chunks[idx])

        if not reranked:
            raise ValueError("Reranker API 返回的 index 均无效，无法映射回 chunks")

        return reranked


# ── 单例工厂 ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_reranker() -> APIReranker | None:
    """
    返回全局 APIReranker 单例。

    当 RERANKER_ENABLED=false 或 api_url 为空时返回 None，
    调用方通过 `if reranker:` 判断是否启用，无需感知配置细节。
    """
    from src.infrastructure.config import settings

    if not settings.reranker_enabled:
        return None
    if not settings.reranker_api_url:
        logger.warning("RERANKER_ENABLED=true 但 RERANKER_API_URL 未配置，Reranker 已禁用")
        return None
    if not settings.reranker_api_key:
        logger.warning("RERANKER_ENABLED=true 但 RERANKER_API_KEY 未配置，请求将以空 Bearer token 发出")

    return APIReranker(
        api_url=settings.reranker_api_url,
        api_key=settings.reranker_api_key,
        model=settings.reranker_model,
        top_n=settings.reranker_top_n,
        timeout=settings.reranker_timeout,
        max_candidates=settings.reranker_max_candidates,
    )


def rerank_or_truncate(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """
    Reranker 可用时精排，否则直接截断。

    所有检索器的最终输出均应通过此函数，确保 Reranker 统一生效。
    """
    reranker = get_reranker()
    if reranker and chunks:
        return reranker.rerank(query, chunks)[:top_k]
    return chunks[:top_k]
