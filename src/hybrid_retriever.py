"""
混合检索模块
融合语义检索（FAISS）与关键词检索（BM25），使用 Reciprocal Rank Fusion（RRF）重排序

Phase 10-2：retrieve() 新增 section_filter 参数，透传给 SemanticRetriever；
BM25Retriever 无法原生过滤，在 RRF 排序后对完整候选集过滤再裁至 top_k。

Phase 10-2 Review 修复：section_filter 非空时不先截断 sorted_keys 再过滤，
而是在完整 RRF 排序结果上过滤后再取 top_k，保证 BM25 后排但符合章节的结果不被丢弃。
同时语义路径按 top_k * 3 扩容以提供更多候选。

P1-Step 3：RRF 融合后接入 Reranker 精排（RERANKER_ENABLED=true 时生效）。
"""
from src.bm25_retriever import BM25Retriever
from src.retrieval.reranker import get_reranker
from src.retriever import RetrievedChunk, SemanticRetriever


_RRF_K = 60           # RRF 平滑常数，通常取 60
_FILTER_EXPAND = 3    # section_filter 时语义检索的扩容倍数


def _rrf_score(rank: int, k: int = _RRF_K) -> float:
    """RRF 分数：1 / (k + rank)，rank 从 1 开始"""
    return 1.0 / (k + rank)


class HybridRetriever:
    """
    混合检索器：语义 + BM25，使用 RRF 融合排名。

    Parameters
    ----------
    top_k : int
        最终返回结果数量
    semantic_top_k : int
        向量检索候选数量（>=top_k）
    bm25_top_k : int
        BM25 检索候选数量（>=top_k）
    """

    def __init__(
        self,
        top_k: int = 5,
        semantic_top_k: int = 10,
        bm25_top_k: int = 10,
    ) -> None:
        self.top_k = top_k
        self.semantic_retriever = SemanticRetriever(top_k=semantic_top_k)
        self.bm25_retriever = BM25Retriever(top_k=bm25_top_k)

    @staticmethod
    def _chunk_key(chunk: RetrievedChunk) -> str:
        """用来去重和对齐的唯一键"""
        return f"{chunk.file_path}#{chunk.chunk_index}"

    def retrieve(self, query: str, section_filter: str = "") -> list[RetrievedChunk]:
        """
        Args:
            query:          用户查询字符串。
            section_filter: 章节类型过滤（如 "method"、"abstract"），空字符串表示不过滤。
                            SemanticRetriever 在向量检索阶段原生支持；
                            BM25Retriever 结果在 RRF 排序后过滤（不先截断，过滤后再取 top_k）。
        """
        sf = section_filter if section_filter else None
        # section_filter 时扩大语义候选量，以弥补后过滤带来的候选损失
        sem_top_k = self.top_k * _FILTER_EXPAND if section_filter else self.top_k
        semantic_results = SemanticRetriever(top_k=sem_top_k).retrieve(query, section_type=sf)
        bm25_results = self.bm25_retriever.retrieve(query)

        # 按唯一键收集 chunk 对象（保留第一次出现）
        chunk_map: dict[str, RetrievedChunk] = {}
        for chunk in semantic_results + bm25_results:
            key = self._chunk_key(chunk)
            if key not in chunk_map:
                chunk_map[key] = chunk

        # 计算 RRF 融合分数
        rrf_scores: dict[str, float] = {key: 0.0 for key in chunk_map}

        for rank, chunk in enumerate(semantic_results, start=1):
            key = self._chunk_key(chunk)
            rrf_scores[key] += _rrf_score(rank)

        for rank, chunk in enumerate(bm25_results, start=1):
            key = self._chunk_key(chunk)
            rrf_scores[key] += _rrf_score(rank)

        # 按 RRF 分数降序排列完整候选列表（不先截断），过滤后再取 top_k
        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        fused = [chunk_map[key] for key in sorted_keys]

        if section_filter:
            fused = [c for c in fused if getattr(c, "section_type", "") == section_filter]

        reranker = get_reranker()
        if reranker:
            return reranker.rerank(query, fused)[: self.top_k]
        return fused[: self.top_k]
