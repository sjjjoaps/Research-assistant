"""
SQLite 元数据存储测试
用法：
python tests/test_database.py
"""

import sys
from pathlib import Path

# 把项目根目录加入 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

from src.storage.database import MetadataDatabase
from src.ingestion.metadata_extractor import DocumentMetadata


def main() -> None:
    test_db_path = Path(__file__).parent.parent / "data" / "test_metadata.db"

    # 先删旧库（如果存在）
    if test_db_path.exists():
        try:
            test_db_path.unlink()
        except:
            pass

    # 初始化数据库
    database = MetadataDatabase(db_path=test_db_path)
    database.init_db()

    # 测试数据
    sample_file_path = "C:/test/sample_paper.pdf"
    sample_metadata = DocumentMetadata(
        title="AmpAgent: LLM-based Analog Circuit Design Assistant",
        authors=["Alice Zhang", "Bob Li"],
        institution="Tsinghua University",
        year=2024,
        abstract="This paper presents an LLM-based framework for analog circuit design assistance.",
        keywords=["LLM", "Analog Circuit", "EDA"],
    )

    print("=" * 80)
    print("[1] 插入测试记录")
    record_id = database.add_document(sample_file_path, sample_metadata)
    print(f"插入成功，记录 ID: {record_id}")

    print("\n[2] 查询单条记录")
    record = database.get_document(sample_file_path)
    if record is not None:
        print(f"file_path: {record.file_path}")
        print(f"title: {record.title}")
        print(f"authors: {record.authors}")
        print(f"institution: {record.institution}")
        print(f"year: {record.year}")
        print(f"keywords: {record.keywords}")

    print("\n[3] 查询全部记录")
    records = database.list_documents()
    print(f"当前记录数: {len(records)}")

    print("\n[4] 删除记录")
    deleted = database.delete_document(sample_file_path)
    print(f"删除结果: {deleted}")

    print("\n[5] 再次查询全部记录")
    records = database.list_documents()
    print(f"当前记录数: {len(records)}")
    print("=" * 80)

    # ✅ 关键修复：先关闭数据库连接，再删文件
    database.close()

    # 安全删除测试库
    if test_db_path.exists():
        try:
            test_db_path.unlink()
        except Exception as e:
            print(f"清理临时文件: {e}")


if __name__ == "__main__":
    main()