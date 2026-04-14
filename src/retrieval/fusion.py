"""
检索结果融合工具

供 local / global / mix 检索器复用的 RRF 融合逻辑。
"""
from __future__ import annotations

from src.retriever import RetrievedChunk


_RRF_K = 60


def rrf_score(rank: int, k: int = _RRF_K) -> float:
    return 1.0 / (k + rank)


def chunk_key(chunk: RetrievedChunk) -> str:
    return f"{chunk.file_path}#{chunk.chunk_index}#{hash(chunk.content)}"


def fuse_ranked_lists(ranked_lists: list[list[RetrievedChunk]], top_k: int) -> list[RetrievedChunk]:
    chunk_map: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            key = chunk_key(chunk)
            if key not in chunk_map:
                chunk_map[key] = chunk
                scores[key] = 0.0
            scores[key] += rrf_score(rank)

    sorted_keys = sorted(scores, key=lambda item: scores[item], reverse=True)
    return [chunk_map[key] for key in sorted_keys[:top_k]]
