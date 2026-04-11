"""
Phase 11 混合检索 & 图检索测试
对比三种检索模式（semantic / hybrid / graph）在同一查询上的结果差异。

用法：
  python tests/test_hybrid_search.py --query "AmpAgent的核心方法是什么？"
  python tests/test_hybrid_search.py --query "图神经网络在电路设计中的应用" --top-k 5
"""

import sys
from pathlib import Path

# 把项目根目录加入 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

import argparse

from src.retriever import SemanticRetriever
from src.bm25_retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever
from src.graph_retriever import GraphRetriever
from src.retriever import RetrievedChunk


_SEPARATOR = "=" * 80
_DIVIDER = "-" * 40


def _print_chunks(chunks: list[RetrievedChunk], max_content_len: int = 300) -> None:
    if not chunks:
        print("  （无结果）")
        return
    for i, chunk in enumerate(chunks, start=1):
        source = f"{chunk.file_path}#chunk-{chunk.chunk_index}"
        preview = chunk.content[:max_content_len].replace("\n", " ")
        if len(chunk.content) > max_content_len:
            preview += "..."
        print(f"  [{i}] 来源: {source}")
        print(f"      内容: {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase11 检索对比测试")
    parser.add_argument("--query", required=True, help="检索查询")
    parser.add_argument("--top-k", type=int, default=3, help="每种检索器返回的 chunk 数量")
    parser.add_argument(
        "--mode",
        choices=["all", "semantic", "bm25", "hybrid", "graph"],
        default="all",
        help="测试模式：all 表示全部对比，其余单独运行某一检索器",
    )
    args = parser.parse_args()

    query = args.query
    top_k = args.top_k

    print(_SEPARATOR)
    print(f"查询: {query}")
    print(f"top_k: {top_k}")
    print(_SEPARATOR)

    modes_to_run = (
        ["semantic", "bm25", "hybrid", "graph"]
        if args.mode == "all"
        else [args.mode]
    )

    for mode in modes_to_run:
        print(f"\n▶  检索模式: {mode.upper()}")
        print(_DIVIDER)

        retriever = None
        try:
            if mode == "semantic":
                retriever = SemanticRetriever(top_k=top_k)
            elif mode == "bm25":
                retriever = BM25Retriever(top_k=top_k)
            elif mode == "hybrid":
                retriever = HybridRetriever(
                    top_k=top_k,
                    semantic_top_k=top_k * 2,
                    bm25_top_k=top_k * 2,
                )
            else:  # graph
                retriever = GraphRetriever(top_k=top_k, expand_entities=True)

            chunks = retriever.retrieve(query)
            _print_chunks(chunks)

        except Exception as exc:
            print(f"  ✗ 检索失败: {exc}")

        finally:
            if mode == "graph" and retriever is not None and hasattr(retriever, "close"):
                retriever.close()

    print("\n" + _SEPARATOR)
    print("对比测试完成。")
    print(_SEPARATOR)


if __name__ == "__main__":
    main()
