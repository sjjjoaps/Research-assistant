"""
元数据提取测试
用法：
python tests/test_metadata.py --file "C:/path/to/file.pdf"
"""

import sys
from pathlib import Path

# 把项目根目录加入 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

import argparse

from src.ingestion.document_parser import DocumentParser
from src.ingestion.metadata_extractor import MetadataExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="测试文档元数据提取")
    parser.add_argument("--file", required=True, help="待解析文件路径")
    args = parser.parse_args()

    file_path = Path(args.file)

    document_parser = DocumentParser()
    extractor = MetadataExtractor()

    parsed_document = document_parser.parse(file_path)
    metadata = extractor.extract(parsed_document)

    print("=" * 80)
    print(f"文件路径: {parsed_document.file_path}")
    print("元数据提取结果:")
    print(metadata.model_dump_json(indent=2, exclude_none=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
