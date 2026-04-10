"""
文献入库 CLI 入口
用法：
python ingest.py --file "C:/path/to/file.pdf"
python ingest.py --dir "C:/path/to/dir"
"""

import sys
from pathlib import Path

# 把项目根目录加入 Python 路径
sys.path.append(str(Path(__file__).parent))

import argparse

from src.ingestion_pipeline import IngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="文献入库工具")
    parser.add_argument("--file", help="单个文件路径")
    parser.add_argument("--dir", help="批量入库目录")
    args = parser.parse_args()

    if not args.file and not args.dir:
        parser.error("--file 和 --dir 至少提供一个")

    pipeline = IngestionPipeline()

    if args.file:
        result = pipeline.ingest_file(args.file)
        print("\n入库完成:")
        print(result)

    if args.dir:
        results = pipeline.ingest_directory(args.dir)
        print(f"\n批量入库完成，共成功 {len(results)} 篇文献")
        for result in results:
            print(result)


if __name__ == "__main__":
    main()
