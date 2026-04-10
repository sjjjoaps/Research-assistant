"""
向量检索测试
用法：
python tests/test_vector_search.py --query "transformer attention mechanism"
"""

import sys
from pathlib import Path

# 把项目根目录加入 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

import argparse

from src.chunker import DocumentChunker
from src.document_parser import DocumentParser
from src.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="测试 FAISS 向量检索")
    parser.add_argument("--query", required=True, help="检索问题")
    args = parser.parse_args()

    raw_data_dir = Path("C:/Users/Administrator/Desktop/assitant/data/raw_data")
    candidate_files = [
        raw_data_dir / "AmpAgent.pdf",
        raw_data_dir / "AnaFlow.pdf",
        raw_data_dir / "AMSnet_KG_A_NetlistDataset_for_LLM_based_AMS_Circuit.pdf",
    ]
    files = [file_path for file_path in candidate_files if file_path.exists()]

    if not files:
        raise FileNotFoundError("未找到测试文档，请检查 raw_data 目录")

    document_parser = DocumentParser()
    chunker = DocumentChunker()
    vector_store = VectorStore(index_dir=Path(__file__).parent.parent / "data" / "test_faiss")

    all_chunks = []
    for file_path in files:
        parsed_document = document_parser.parse(file_path)
        chunks = chunker.chunk(parsed_document)
        all_chunks.extend(chunks)

    vector_store.add_chunks(all_chunks)
    results = vector_store.similarity_search(args.query, k=3)

    print("=" * 80)
    print(f"查询: {args.query}")
    print(f"入库 chunk 数量: {len(all_chunks)}")
    print("Top-3 检索结果:")

    for index, doc in enumerate(results, start=1):
        print(f"\n[{index}] 文件: {doc.metadata.get('file_path')}")
        print(f"Chunk 序号: {doc.metadata.get('chunk_index')}")
        print(doc.page_content[:500])
        print("-" * 80)


if __name__ == "__main__":
    main()
