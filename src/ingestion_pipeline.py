"""
文献入库流水线
串联解析、切块、元数据提取、SQLite 写入、FAISS 写入、Neo4j 图写入
可选：实体与关系抽取（通过 enable_entity_extraction 开关控制）

Phase 1.1 新增：
- 每个入库步骤前后更新 DocumentStatus
- 任意步骤失败时写入 failed 状态和错误信息
- 返回结果中包含 doc_id
"""
from pathlib import Path

from src.chunker import DocumentChunker
from src.config import settings
from src.database import MetadataDatabase
from src.document_parser import DocumentParser
from src.entity_extractor import EntityExtractor
from src.graph_store import GraphStore
from src.metadata_extractor import MetadataExtractor
from src.storage.document_status_store import DocumentStatus, DocumentStatusStore, generate_doc_id
from src.storage.extraction_cache import compute_file_hash
from src.vector_store import VectorStore


class IngestionPipeline:
    def __init__(self, enable_entity_extraction: bool | None = None) -> None:
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
        self.status_store = DocumentStatusStore()

        self.database.init_db()
        self.vector_store.load()
        self.graph_store.init_schema()
        self.status_store.init_db()

        self._entity_extractor: EntityExtractor | None = (
            EntityExtractor(self.graph_store) if self._enable_entity_extraction else None
        )

    def ingest_file(self, file_path: str | Path) -> dict:
        path = Path(file_path)
        total_steps = 7 if self._enable_entity_extraction else 6

        # ── 计算内容身份（FileNotFoundError 在此传播，无需状态记录）──────────────
        file_hash = compute_file_hash(path)
        doc_id = generate_doc_id(file_hash)

        # ── 初始化状态对象，贯穿整个流程 ─────────────────────────────────────────
        status = DocumentStatus(
            file_path=str(path),
            doc_id=doc_id,
            status="pending",
            current_step="等待入库",
        )
        self.status_store.upsert(status)
        current_step = "初始化"

        try:
            # [1] 解析
            current_step = "parsing"
            status.status, status.current_step = "parsing", "解析文档"
            self.status_store.upsert(status)
            print(f"\n[1/{total_steps}] 开始解析文档: {path}")
            parsed_document = self.document_parser.parse(path)

            # [2] 切块
            current_step = "chunking"
            status.status, status.current_step = "chunking", "文本切块"
            self.status_store.upsert(status)
            print(f"[2/{total_steps}] 开始切块")
            chunks = self.chunker.chunk(parsed_document)
            print(f"      切块完成，共 {len(chunks)} 个 chunk")
            # Phase 1.1：用顺序占位 ID 记录数量；Phase 1.2 接入 ChunkTracker 后替换为内容哈希 ID
            status.chunk_ids = [str(i) for i in range(len(chunks))]
            self.status_store.upsert(status)

            # [3] 元数据
            current_step = "metadata"
            status.status, status.current_step = "metadata", "提取元数据"
            self.status_store.upsert(status)
            print(f"[3/{total_steps}] 开始提取元数据")
            metadata = self.metadata_extractor.extract(parsed_document)
            print(f"      元数据提取完成，标题: {metadata.title}")

            # [4] 写入 SQLite + FAISS
            current_step = "indexing"
            status.status, status.current_step = "indexing", "写入 SQLite / FAISS"
            self.status_store.upsert(status)
            print(f"[4/{total_steps}] 写入 SQLite")
            existing = self.database.get_document(str(path))
            if existing is None:
                record_id = self.database.add_document(str(path), metadata, doc_id=doc_id)
                print(f"      SQLite 写入完成，记录 ID: {record_id}")
            else:
                record_id = existing.id
                print(f"      文档已存在，跳过 SQLite 写入，记录 ID: {record_id}")

            print(f"[5/{total_steps}] 写入 FAISS")
            self.vector_store.add_chunks(chunks)
            self.vector_store.save()
            print("      FAISS 写入完成")

            # [5] 写入 Neo4j
            current_step = "graph"
            status.status, status.current_step = "graph", "写入 Neo4j"
            self.status_store.upsert(status)
            print(f"[6/{total_steps}] 写入 Neo4j")
            self.graph_store.add_document_with_chunks(str(path), metadata, chunks)
            print("      Neo4j 写入完成")

            # [6] 实体抽取（可选）
            entity_count = 0
            relation_count = 0
            if self._enable_entity_extraction and self._entity_extractor is not None:
                if self.graph_store.is_document_entity_extracted(str(path)):
                    print(f"[7/{total_steps}] 实体抽取 — 已处理过，跳过")
                else:
                    current_step = "extracting"
                    status.status, status.current_step = "extracting", "实体与关系抽取"
                    self.status_store.upsert(status)
                    print(f"[7/{total_steps}] 开始实体与关系抽取（最多 {settings.entity_extraction_max_chunks} 个 chunk）")
                    stats = self._entity_extractor.extract_for_document(
                        file_path=str(path),
                        chunks=chunks,
                        max_chunks=settings.entity_extraction_max_chunks,
                    )
                    entity_count = stats.created_entities
                    relation_count = stats.created_relations
                    # Phase 1.1：占位 ID，Phase 2.1 替换为真实实体/关系 ID
                    status.entity_ids = [str(i) for i in range(entity_count)]
                    status.relation_ids = [str(i) for i in range(relation_count)]
                    self.status_store.upsert(status)
                    print(f"      实体抽取完成，新增实体 {entity_count} 个，关系 {relation_count} 条")

            # ── 完成 ──────────────────────────────────────────────────────────
            status.status, status.current_step = "processed", "入库完成"
            self.status_store.upsert(status)

        except Exception as e:
            # 状态写入失败不应掩盖原始入库异常，用 try/except 隔离
            try:
                self.status_store.mark_failed(doc_id, current_step, str(e), file_path=str(path))
            except Exception:
                pass
            raise

        return {
            "file_path": str(path),
            "doc_id": doc_id,
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
        self.status_store.close()
