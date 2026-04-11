"""
文献入库流水线
串联解析、切块、元数据提取、SQLite 写入、FAISS 写入、Neo4j 图写入
可选：实体与关系抽取（通过 enable_entity_extraction 开关控制）
"""
from pathlib import Path

from src.chunker import DocumentChunker
from src.config import settings
from src.database import MetadataDatabase
from src.document_parser import DocumentParser
from src.entity_extractor import EntityExtractor
from src.graph_store import GraphStore
from src.metadata_extractor import MetadataExtractor
from src.vector_store import VectorStore


class IngestionPipeline:
    def __init__(self, enable_entity_extraction: bool | None = None) -> None:
        # 优先使用显式参数，未传则读取 config 开关
        self._enable_entity_extraction = (
            enable_entity_extraction
            if enable_entity_extraction is not None
            else settings.enable_entity_extraction
        )

        self.document_parser = DocumentParser()
        self.chunker = DocumentChunker()
        self.metadata_extractor = MetadataExtractor()
        self.database = MetadataDatabase()
        self.vector_store = VectorStore()
        self.graph_store = GraphStore()

        self.database.init_db()
        self.vector_store.load()
        self.graph_store.init_schema()

        # 只在开关打开时才初始化（避免无谓 LLM 实例化）
        self._entity_extractor: EntityExtractor | None = (
            EntityExtractor(self.graph_store) if self._enable_entity_extraction else None
        )

    def ingest_file(self, file_path: str | Path) -> dict:
        path = Path(file_path)
        total_steps = 7 if self._enable_entity_extraction else 6
        print(f"\n[1/{total_steps}] 开始解析文档: {path}")
        parsed_document = self.document_parser.parse(path)

        print(f"[2/{total_steps}] 开始切块")
        chunks = self.chunker.chunk(parsed_document)
        print(f"      切块完成，共 {len(chunks)} 个 chunk")

        print(f"[3/{total_steps}] 开始提取元数据")
        metadata = self.metadata_extractor.extract(parsed_document)
        print(f"      元数据提取完成，标题: {metadata.title}")

        print(f"[4/{total_steps}] 写入 SQLite")
        existing = self.database.get_document(str(path))
        if existing is None:
            record_id = self.database.add_document(str(path), metadata)
            print(f"      SQLite 写入完成，记录 ID: {record_id}")
        else:
            record_id = existing.id
            print(f"      文档已存在，跳过 SQLite 写入，记录 ID: {record_id}")

        print(f"[5/{total_steps}] 写入 FAISS")
        self.vector_store.add_chunks(chunks)
        self.vector_store.save()
        print("      FAISS 写入完成")

        print(f"[6/{total_steps}] 写入 Neo4j")
        self.graph_store.add_document_with_chunks(str(path), metadata, chunks)
        print("      Neo4j 写入完成")

        entity_count = 0
        relation_count = 0
        if self._enable_entity_extraction and self._entity_extractor is not None:
            if self.graph_store.is_document_entity_extracted(str(path)):
                print(f"[7/{total_steps}] 实体抽取 — 已处理过，跳过")
            else:
                print(f"[7/{total_steps}] 开始实体与关系抽取（最多 {settings.entity_extraction_max_chunks} 个 chunk）")
                stats = self._entity_extractor.extract_for_document(
                    file_path=str(path),
                    chunks=chunks,
                    max_chunks=settings.entity_extraction_max_chunks,
                )
                entity_count = stats.created_entities
                relation_count = stats.created_relations
                print(f"      实体抽取完成，新增实体 {entity_count} 个，关系 {relation_count} 条")

        return {
            "file_path": str(path),
            "record_id": record_id,
            "title": metadata.title,
            "chunk_count": len(chunks),
            "entity_count": entity_count,
            "relation_count": relation_count,
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

    def close(self) -> None:
        self.database.close()
        self.graph_store.close()
