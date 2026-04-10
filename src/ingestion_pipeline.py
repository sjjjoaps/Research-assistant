"""
文献入库流水线
串联解析、切块、元数据提取、SQLite 写入、FAISS 写入
"""
from pathlib import Path

from src.chunker import DocumentChunker
from src.database import MetadataDatabase
from src.document_parser import DocumentParser
from src.metadata_extractor import MetadataExtractor
from src.vector_store import VectorStore


class IngestionPipeline:
    def __init__(self) -> None:
        self.document_parser = DocumentParser()
        self.chunker = DocumentChunker()
        self.metadata_extractor = MetadataExtractor()
        self.database = MetadataDatabase()
        self.vector_store = VectorStore()

        self.database.init_db()
        self.vector_store.load()

    def ingest_file(self, file_path: str | Path) -> dict:
        path = Path(file_path)
        print(f"\n[1/5] 开始解析文档: {path}")
        parsed_document = self.document_parser.parse(path)

        print("[2/5] 开始切块")
        chunks = self.chunker.chunk(parsed_document)
        print(f"      切块完成，共 {len(chunks)} 个 chunk")

        print("[3/5] 开始提取元数据")
        metadata = self.metadata_extractor.extract(parsed_document)
        print(f"      元数据提取完成，标题: {metadata.title}")

        print("[4/5] 写入 SQLite")
        existing = self.database.get_document(str(path))
        if existing is None:
            record_id = self.database.add_document(str(path), metadata)
            print(f"      SQLite 写入完成，记录 ID: {record_id}")
        else:
            record_id = existing.id
            print(f"      文档已存在，跳过 SQLite 写入，记录 ID: {record_id}")

        print("[5/5] 写入 FAISS")
        self.vector_store.add_chunks(chunks)
        self.vector_store.save()
        print("      FAISS 写入完成")

        return {
            "file_path": str(path),
            "record_id": record_id,
            "title": metadata.title,
            "chunk_count": len(chunks),
        }

    def ingest_directory(self, directory: str | Path) -> list[dict]:
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")

        supported_files = []
        for pattern in ("*.pdf", "*.docx", "*.txt"):
            supported_files.extend(directory_path.glob(pattern))

        results = []
        for file_path in sorted(supported_files):
            try:
                result = self.ingest_file(file_path)
                results.append(result)
            except Exception as e:
                print(f"\n[ERROR] 入库失败: {file_path}\n原因: {e}")

        return results
