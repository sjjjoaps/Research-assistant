"""
混合检索模块
融合语义检索（FAISS）与关键词检索（BM25），使用 Reciprocal Rank Fusion（RRF）重排序

Reranker 可用时：合并两路候选池直接精排，跳过 RRF。
Reranker 不可用时：RRF 融合后截断。

Phase 10-2：retrieve() 新增 section_filter 参数，透传给 SemanticRetriever；
BM25Retriever 无法原生过滤，在 RRF 排序后对完整候选集过滤再裁至 top_k。
"""
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.reranker import get_reranker, rerank_or_truncate
from src.retrieval.retriever import RetrievedChunk, SemanticRetriever
from src.infrastructure.config import settings


def _rrf_score(rank: int) -> float:
    return 1.0 / (settings.rrf_k + rank)


class HybridRetriever:
    def __init__(
        self,
        top_k: int = 5,
        semantic_top_k: int | None = None,
        bm25_top_k: int | None = None,
    ) -> None:
        self.top_k = int(top_k)
        self.semantic_retriever = SemanticRetriever(
            top_k=semantic_top_k if semantic_top_k is not None else settings.hybrid_semantic_top_k
        )
        self.bm25_retriever = BM25Retriever(
            top_k=bm25_top_k if bm25_top_k is not None else settings.hybrid_bm25_top_k
        )

    @staticmethod
    def _chunk_key(chunk: RetrievedChunk) -> str:
        return f"{chunk.file_path}#{chunk.chunk_index}"

    def retrieve(self, query: str, section_filter: str = "") -> list[RetrievedChunk]:
        sf = section_filter if section_filter else None
        if section_filter:
            sem_retriever = SemanticRetriever(top_k=self.top_k * settings.retrieval_filter_expand)
        else:
            sem_retriever = self.semantic_retriever
        semantic_results = sem_retriever.retrieve(query, section_type=sf)
        bm25_results = self.bm25_retriever.retrieve(query)

        if get_reranker():
            # reranker 可用：合并去重后直接精排，跳过 RRF
            chunk_map: dict[str, RetrievedChunk] = {}
            for chunk in semantic_results + bm25_results:
                key = self._chunk_key(chunk)
                if key not in chunk_map:
                    chunk_map[key] = chunk
            candidates = list(chunk_map.values())
            if section_filter:
                candidates = [c for c in candidates if getattr(c, "section_type", "") == section_filter]
            return rerank_or_truncate(query, candidates, self.top_k)

        # reranker 不可用：RRF 融合
        chunk_map = {}
        for chunk in semantic_results + bm25_results:
            key = self._chunk_key(chunk)
            if key not in chunk_map:
                chunk_map[key] = chunk

        rrf_scores: dict[str, float] = {key: 0.0 for key in chunk_map}
        for rank, chunk in enumerate(semantic_results, start=1):
            rrf_scores[self._chunk_key(chunk)] += _rrf_score(rank)
        for rank, chunk in enumerate(bm25_results, start=1):
            rrf_scores[self._chunk_key(chunk)] += _rrf_score(rank)

        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        fused = [chunk_map[key] for key in sorted_keys]
        if section_filter:
            fused = [c for c in fused if getattr(c, "section_type", "") == section_filter]
        return fused[: self.top_k]
