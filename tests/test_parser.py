"""
文档解析与切块测试
用法：
python tests/test_parser.py --file "C:/path/to/file.pdf"
"""

import sys
from pathlib import Path
# 把项目根目录加入 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

import argparse


from src.ingestion.chunker import DocumentChunker
from src.ingestion.document_parser import DocumentParser


def main() -> None:
    parser = argparse.ArgumentParser(description="测试文档解析与切块")
    parser.add_argument("--file", required=True, help="待解析文件路径")
    args = parser.parse_args()

    file_path = Path(args.file)

    document_parser = DocumentParser()
    chunker = DocumentChunker()

    parsed_document = document_parser.parse(file_path)
    chunks = chunker.chunk(parsed_document)

    print("=" * 80)
    print(f"文件路径: {parsed_document.file_path}")
    print(f"页面数量: {len(parsed_document.pages)}")
    print(f"总字符数: {len(parsed_document.raw_text)}")
    print(f"切块数量: {len(chunks)}")
    print("=" * 80)

    for chunk in chunks[:3]:
        print(f"\n[Chunk {chunk.chunk_index}]")
        print(chunk.content[:500])
        print("-" * 80)


if __name__ == "__main__":
    main()
