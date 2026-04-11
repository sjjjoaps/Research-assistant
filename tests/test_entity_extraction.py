"""
实体与关系抽取测试
用法：
python tests/test_entity_extraction.py --file "C:/path/to/file.pdf" --max-chunks 3
"""

import sys
from pathlib import Path

# 把项目根目录加入 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

import argparse

from src.chunker import DocumentChunker
from src.document_parser import DocumentParser
from src.entity_extractor import EntityExtractor
from src.graph_store import GraphStore
from src.metadata_extractor import MetadataExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="测试实体与关系抽取")
    parser.add_argument("--file", required=True, help="待抽取文档路径")
    parser.add_argument("--max-chunks", type=int, default=3, help="最多处理 chunk 数")
    args = parser.parse_args()

    file_path = Path(args.file)

    document_parser = DocumentParser()
    chunker = DocumentChunker()
    metadata_extractor = MetadataExtractor()
    graph_store = GraphStore()
    graph_store.init_schema()

    # 确保 Document/Chunk 已存在
    parsed_document = document_parser.parse(file_path)
    chunks = chunker.chunk(parsed_document)
    metadata = metadata_extractor.extract(parsed_document)
    graph_store.add_document_with_chunks(str(file_path), metadata, chunks)

    extractor = EntityExtractor(graph_store)
    stats = extractor.extract_for_document(
        file_path=str(file_path),
        chunks=chunks,
        max_chunks=args.max_chunks,
    )

    print("=" * 80)
    print(f"文件: {file_path}")
    print(f"处理 chunk 数: {stats.processed_chunks}")
    print(f"新增实体数: {stats.created_entities}")
    print(f"新增关系数: {stats.created_relations}")
    print("请在 Neo4j Browser 中检查 (:Entity)、[:MENTIONS]、[:RELATES_TO]。")
    print("=" * 80)

    graph_store.close()


if __name__ == "__main__":
    main()
