"""
Neo4j 图写入测试
用法：
python tests/test_graph.py --file "C:/path/to/file.pdf"
"""

import sys
from pathlib import Path

# 把项目根目录加入 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

import argparse

from src.chunker import DocumentChunker
from src.document_parser import DocumentParser
from src.graph_store import GraphStore
from src.metadata_extractor import MetadataExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="测试 Neo4j 图写入")
    parser.add_argument("--file", required=True, help="待写入图数据库的文件路径")
    args = parser.parse_args()

    file_path = Path(args.file)

    document_parser = DocumentParser()
    chunker = DocumentChunker()
    extractor = MetadataExtractor()
    graph_store = GraphStore()

    graph_store.init_schema()

    parsed_document = document_parser.parse(file_path)
    chunks = chunker.chunk(parsed_document)
    metadata = extractor.extract(parsed_document)

    graph_store.add_document_with_chunks(str(file_path), metadata, chunks)

    print("=" * 80)
    print(f"文件: {file_path}")
    print(f"标题: {metadata.title}")
    print(f"Chunk 数量: {len(chunks)}")
    print("Neo4j 写入完成，请到 Neo4j Browser 中检查 Document/Chunk/[:HAS_CHUNK]。")
    print("=" * 80)

    graph_store.close()


if __name__ == "__main__":
    main()
