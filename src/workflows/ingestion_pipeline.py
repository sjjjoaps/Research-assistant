"""
文献入库流水线
串联解析、切块、元数据提取、SQLite 写入、FAISS 写入、Neo4j 图写入
可选：实体与关系抽取（通过 enable_entity_extraction 开关控制）

Phase 2.3 变更：
- 集成 ChunkTracker，使用内容哈希生成稳定 chunk_id
- 增量检测：同一文档未变化时直接跳过，不重复入库
- 变更检测：同路径文件内容变化时，先清理旧 chunk/source 绑定再重建
- 实体/关系写入统一走 merge 路径（entity_extractor 已改为自然键）
- status.chunk_ids / entity_ids / relation_ids 更新为真实 ID

Phase 2.4 变更：
- 新增 delete_document(doc_id)：精确删除，只清理该文档独占数据，共享实体/关系保留
- 删除流程引入 deleting / delete_failed 状态机

Phase 3.1 变更：
- IngestionPipeline 新增 enable_modal_extraction 参数，透传给 DocumentParser
- 启用后 PDF 解析阶段会提取图片/表格并生成 LLM 描述，作为特殊 chunk 入库
- 多模态 chunk 的 metadata["content_type"] 为 "image" 或 "table"，可区分类型

Phase 3.2 变更：
- IngestionPipeline 新增 enable_section_recognition 参数，透传给 DocumentParser
- 启用后 PDF 解析阶段会识别每页章节类型，写入 chunk metadata["section_type"]
- 支持检索时按 section_type 过滤（如只检索 method / experiment 章节）

Phase 3.3 变更：
- IngestionPipeline 新增 enable_citation_extraction 参数
- 启用后在 Neo4j 写入后提取参考文献，创建 Reference 节点和 CITES 关系
- delete_document 同步删除文档的 CITES 关系（Reference 节点保留供共享）

Phase 10-1 变更：
- IngestionPipeline 新增 enable_section_chunking 参数（默认 True）
- 启用且文档有章节信息时，使用 SectionChunker 按语义边界分块
- 无章节信息或 enable_section_chunking=False 时，仍 fallback 到 DocumentChunker
- 新策略仅影响新入库文档，已入库文档需重新入库才能生效
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import threading

from src.ingestion.chunker import DocumentChunker
from src.ingestion.section_chunker import SectionChunker
from src.infrastructure.config import settings
from src.storage.database import MetadataDatabase
from src.ingestion.document_parser import DocumentParser
from src.ingestion.entity_extractor import EntityExtractor
from src.storage.graph_store import GraphStore
from src.ingestion.citation_extractor import CitationExtractor
from src.ingestion.metadata_extractor import MetadataExtractor
from src.storage.chunk_tracker import ChunkTracker, compute_chunk_content_hash
from src.storage.document_status_store import DocumentStatus, DocumentStatusStore, generate_doc_id
from src.storage.extraction_cache import compute_file_hash
from src.storage.relation_vector_store import RelationVectorStore
from src.storage.vector_store import VectorStore


@dataclass
class BatchIngestResult:
    """并发批量入库的汇总结果。"""
    total: int = 0
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0
    results: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)


class IngestionPipeline:
    def __init__(
        self,
        enable_entity_extraction: bool | None = None,
        enable_modal_extraction: bool | None = None,
        enable_section_recognition: bool | None = None,
        enable_citation_extraction: bool | None = None,
        enable_section_chunking: bool | None = None,
    ) -> None:
        self._enable_entity_extraction = (
            enable_entity_extraction
            if enable_entity_extraction is not None
            else settings.enable_entity_extraction
        )
        self._enable_citation_extraction = (
            enable_citation_extraction
            if enable_citation_extraction is not None
            else settings.enable_citation_extraction
        )

        _modal = enable_modal_extraction if enable_modal_extraction is not None else settings.enable_modal_extraction
        _section_recog = enable_section_recognition if enable_section_recognition is not None else settings.enable_section_recognition
        _section_chunk = enable_section_chunking if enable_section_chunking is not None else settings.enable_section_chunking

        self.document_parser = DocumentParser(
            enable_modal_extraction=_modal,
            enable_section_recognition=_section_recog,
            modal_max_images=settings.modal_max_images,
            modal_max_tables=settings.modal_max_tables,
        )
        self.chunker = SectionChunker() if _section_chunk else DocumentChunker()
        self.metadata_extractor = MetadataExtractor()
        self.database = MetadataDatabase()
        self.vector_store = VectorStore()
        self.relation_vector_store = RelationVectorStore()
        self.graph_store = GraphStore()
        self.status_store = DocumentStatusStore()
        self.chunk_tracker = ChunkTracker()

        self.database.init_db()
        self.vector_store.load()
        self.relation_vector_store.load()
        self.graph_store.init_schema()
        self.status_store.init_db()
        self.chunk_tracker.init_db()

        self._entity_extractor: EntityExtractor | None = (
            EntityExtractor(self.graph_store) if self._enable_entity_extraction else None
        )
        self._citation_extractor = (
            CitationExtractor() if enable_citation_extraction else None
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def ingest_file(self, file_path: str | Path) -> dict:
        path = Path(file_path)
        total_steps = 7 if self._enable_entity_extraction else 6
        if self._enable_citation_extraction:
            total_steps += 1

        # ── 计算文件身份 ──────────────────────────────────────────────────────
        file_hash = compute_file_hash(path)
        doc_id = generate_doc_id(file_hash)

        # ── 增量检测：未变化文档直接跳过 ──────────────────────────────────────
        existing_status = self.status_store.get(doc_id)
        if existing_status and existing_status.status == "processed":
            print(f"\n[SKIP] 文档未变化，跳过入库: {path}")
            # 从 SQLite 补全 record_id / title，保持返回结构与正常入库一致
            db_record = self.database.get_document(str(path))
            return {
                "file_path": str(path),
                "doc_id": doc_id,
                "skipped": True,
                "reason": "already_processed",
                "record_id": db_record.id if db_record else None,
                "title": db_record.title if db_record else "",
                "chunk_count": len(existing_status.chunk_ids),
                "entity_count": len(existing_status.entity_ids),
                "relation_count": len(existing_status.relation_ids),
            }

        # ── 变更检测：同路径但内容已变化，先清理旧数据 ────────────────────────
        old_status = self.status_store.get_by_path(str(path))
        if old_status and old_status.doc_id != doc_id:
            print(f"\n[UPDATE] 文档内容已变化，清理旧数据: {path}")
            self._cleanup_old_document(old_status)

        # ── 初始化状态对象 ────────────────────────────────────────────────────
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

            # [2] 切块（传入 doc_id 生成稳定 chunk_id）
            current_step = "chunking"
            status.status, status.current_step = "chunking", "文本切块"
            self.status_store.upsert(status)
            print(f"[2/{total_steps}] 开始切块")
            chunks = self.chunker.chunk(parsed_document, doc_id=doc_id)
            modal_count = sum(1 for c in chunks if c.metadata.get("content_type") in ("image", "table"))
            text_count = len(chunks) - modal_count
            print(f"      切块完成，共 {len(chunks)} 个 chunk（文本 {text_count}，多模态 {modal_count}）")

            # 注册 chunk 到 ChunkTracker
            content_hashes = [compute_chunk_content_hash(c.content) for c in chunks]
            chunk_ids = self.chunk_tracker.register_chunks(
                doc_id=doc_id,
                content_hashes=content_hashes,
                texts=[c.content for c in chunks],
            )
            status.chunk_ids = chunk_ids
            self.status_store.upsert(status)

            # [3] 元数据
            current_step = "metadata"
            status.status, status.current_step = "metadata", "提取元数据"
            self.status_store.upsert(status)
            print(f"[3/{total_steps}] 开始提取元数据")
            metadata = self.metadata_extractor.extract(parsed_document)
            print(f"      元数据提取完成，标题: {metadata.title}")

            # [4] 写入 SQLite
            current_step = "indexing"
            status.status, status.current_step = "indexing", "写入 SQLite / FAISS"
            self.status_store.upsert(status)
            print(f"[4/{total_steps}] 写入 SQLite")
            existing_db = self.database.get_document(str(path))
            if existing_db is None:
                record_id = self.database.add_document(str(path), metadata, doc_id=doc_id)
                print(f"      SQLite 写入完成，记录 ID: {record_id}")
            else:
                # 文档已存在（可能是内容变更后重新入库），更新元数据和 doc_id
                self.database.update_document(str(path), metadata, doc_id=doc_id)
                record_id = existing_db.id
                print(f"      SQLite 更新完成，记录 ID: {record_id}")

            # [5] 写入 FAISS
            print(f"[5/{total_steps}] 写入 FAISS")
            # Phase 9-2: 在写入 FAISS 前将 year/doc_id/title 注入 chunk.metadata，
            # 供 SemanticRetriever 构建 RetrievedChunk.year，支持时间感知过滤。
            if metadata.year is not None:
                for chunk in chunks:
                    chunk.metadata["year"] = metadata.year
            self.vector_store.add_chunks(chunks)
            self.vector_store.save()
            print("      FAISS 写入完成")

            # [6] 写入 Neo4j
            current_step = "graph"
            status.status, status.current_step = "graph", "写入 Neo4j"
            self.status_store.upsert(status)
            print(f"[6/{total_steps}] 写入 Neo4j")
            self.graph_store.add_document_with_chunks(str(path), metadata, chunks)
            print("      Neo4j 写入完成")

            # [7] 引用提取（可选）
            citation_count = 0
            if self._enable_citation_extraction and self._citation_extractor is not None:
                current_step = "citations"
                status.status, status.current_step = "citations", "引用提取"
                self.status_store.upsert(status)
                print(f"[7/{total_steps}] 开始引用提取")
                citations = self._citation_extractor.extract(parsed_document.raw_text)
                for ref in citations:
                    self.graph_store.create_reference_node(ref)
                    self.graph_store.create_cites_relation(str(path), ref.ref_id)
                citation_count = len(citations)
                print(f"      引用提取完成，共 {citation_count} 条")

            # [N] 实体抽取（可选）
            entity_count = 0
            relation_count = 0
            _entity_step = total_steps if self._enable_entity_extraction else None
            if self._enable_entity_extraction and self._entity_extractor is not None:
                if self.graph_store.is_document_entity_extracted(str(path)):
                    print(f"[{_entity_step}/{total_steps}] 实体抽取 — 已处理过，跳过")
                else:
                    current_step = "extracting"
                    status.status, status.current_step = "extracting", "实体与关系抽取"
                    self.status_store.upsert(status)
                    print(f"[{_entity_step}/{total_steps}] 开始实体与关系抽取（最多 {settings.entity_extraction_max_chunks} 个 chunk）")
                    stats = self._entity_extractor.extract_for_document(
                        file_path=str(path),
                        chunks=chunks,
                        max_chunks=settings.entity_extraction_max_chunks,
                    )
                    entity_count = stats.created_entities
                    relation_count = stats.created_relations
                    status.entity_ids = stats.entity_ids
                    status.relation_ids = stats.relation_keys
                    self.status_store.upsert(status)
                    self.relation_vector_store.add_relations(stats.relation_records)
                    self.relation_vector_store.save()
                    print(f"      实体抽取完成，新增实体 {entity_count} 个，关系 {relation_count} 条")

            # ── 完成 ──────────────────────────────────────────────────────────
            status.status, status.current_step = "processed", "入库完成"
            self.status_store.upsert(status)

        except Exception as e:
            try:
                self.status_store.mark_failed(doc_id, current_step, str(e), file_path=str(path))
            except Exception:
                pass
            raise

        return {
            "file_path": str(path),
            "doc_id": doc_id,
            "skipped": False,
            "record_id": record_id,
            "title": metadata.title,
            "chunk_count": len(chunks),
            "entity_count": entity_count,
            "relation_count": relation_count,
            "citation_count": citation_count,
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

    def ingest_files_concurrent(
        self,
        file_paths: list[str | Path],
        max_workers: int | None = None,
    ) -> BatchIngestResult:
        """并发入库多个文件。

        使用 ThreadPoolExecutor(max_workers) 控制并发数，单文件失败不影响其他文件。
        max_workers 默认读取 settings.ingestion_max_workers（INGESTION_MAX_WORKERS 环境变量）。

        线程安全保证：
        - FAISS 写入：VectorStore 内置 threading.Lock，add_chunks / save 均在锁内执行
        - Neo4j 实体/关系 upsert：GraphStore 内置细粒度锁，同一实体/关系串行化
        - 同 doc_id 并发重复处理：per-doc_id claim 锁，确保同一文档只被一个线程处理

        Args:
            file_paths: 待入库文件路径列表。
            max_workers: 最大并发线程数，默认 4。

        Returns:
            BatchIngestResult，包含成功/跳过/失败统计及各文件结果。
        """
        batch = BatchIngestResult(total=len(file_paths))
        # per-doc_id claim 锁：防止同 doc_id 被两个线程同时处理
        # key: doc_id, value: threading.Lock
        _claim_locks: dict[str, threading.Lock] = {}
        _claim_locks_meta = threading.Lock()

        def _get_claim_lock(doc_id: str) -> threading.Lock:
            with _claim_locks_meta:
                if doc_id not in _claim_locks:
                    _claim_locks[doc_id] = threading.Lock()
                return _claim_locks[doc_id]

        def _ingest_one(fp: str | Path) -> dict:
            path = Path(fp)
            file_hash = compute_file_hash(path)
            doc_id = generate_doc_id(file_hash)

            # 先做无锁快速检查，已处理则直接跳过（避免不必要的锁竞争）
            existing_status = self.status_store.get(doc_id)
            if existing_status and existing_status.status == "processed":
                db_record = self.database.get_document(str(path))
                return {
                    "file_path": str(path),
                    "doc_id": doc_id,
                    "skipped": True,
                    "reason": "already_processed",
                    "record_id": db_record.id if db_record else None,
                    "title": db_record.title if db_record else "",
                    "chunk_count": len(existing_status.chunk_ids),
                    "entity_count": len(existing_status.entity_ids),
                    "relation_count": len(existing_status.relation_ids),
                }

            # 持有 claim 锁后再次检查，防止两个线程同时通过上面的快速检查
            claim_lock = _get_claim_lock(doc_id)
            with claim_lock:
                existing_status = self.status_store.get(doc_id)
                if existing_status and existing_status.status == "processed":
                    db_record = self.database.get_document(str(path))
                    return {
                        "file_path": str(path),
                        "doc_id": doc_id,
                        "skipped": True,
                        "reason": "already_processed",
                        "record_id": db_record.id if db_record else None,
                        "title": db_record.title if db_record else "",
                        "chunk_count": len(existing_status.chunk_ids),
                        "entity_count": len(existing_status.entity_ids),
                        "relation_count": len(existing_status.relation_ids),
                    }
                # FAISS 和 Neo4j 的线程安全由各自内置锁保证，此处直接调用
                return self.ingest_file(path)

        with ThreadPoolExecutor(max_workers=max_workers or settings.ingestion_max_workers) as executor:
            future_to_path = {executor.submit(_ingest_one, fp): fp for fp in file_paths}
            for future in as_completed(future_to_path):
                fp = future_to_path[future]
                try:
                    result = future.result()
                    batch.results.append(result)
                    if result.get("skipped"):
                        batch.skipped += 1
                    else:
                        batch.succeeded += 1
                except Exception as e:
                    batch.failed += 1
                    batch.errors.append({"file_path": str(fp), "error": str(e)})
                    print(f"\n[ERROR] 并发入库失败: {fp}\n原因: {e}")

        return batch

    def delete_document(self, doc_id: str) -> dict:
        """精确删除文档，只清理该文档独占的数据，共享实体和关系保留。

        执行顺序：
        1. 查询状态记录，获取 file_path，标记为 deleting
        2. 删除 FAISS 中该 doc_id 的向量
        3. 删除 Neo4j 中该文档的 Chunk 节点（及 MENTIONS 关系）
        4. 删除 Neo4j 中的 Document 节点
        5. 删除失去所有 Chunk 支撑的 RELATES_TO 边
        6. 删除孤立实体（无任何 MENTIONS 来源）
        7. 删除 SQLite 元数据记录
        8. 清理 ChunkTracker 记录
        9. 删除状态记录

        中途任何步骤失败时，状态记录会被标记为 delete_failed，保留 file_path
        以便后续重试。

        Args:
            doc_id: 文档唯一标识。

        Returns:
            包含删除统计的字典。

        Raises:
            KeyError: doc_id 不存在时。
        """
        status = self.status_store.get(doc_id)
        if status is None:
            raise KeyError(f"未找到 doc_id={doc_id!r} 的文档记录")

        file_path = status.file_path
        print(f"\n[DELETE] 开始删除文档: {file_path} (doc_id={doc_id})")

        # 标记为删除中，防止并发重入
        status.status = "deleting"
        status.current_step = "删除中"
        self.status_store.upsert(status)

        try:
            # [1] FAISS 向量
            deleted_vectors = self.vector_store.delete_by_doc_id(doc_id)
            self.vector_store.save()
            print(f"  FAISS 删除向量: {deleted_vectors} 条")

            deleted_relation_vectors = self.relation_vector_store.delete_by_doc_id(doc_id)
            self.relation_vector_store.save()
            if deleted_relation_vectors:
                print(f"  关系索引删除向量: {deleted_relation_vectors} 条")

            # [2] Neo4j Chunk 节点及 MENTIONS 关系
            deleted_chunks = self.graph_store.delete_document_chunks(file_path)
            print(f"  Neo4j 删除 Chunk: {deleted_chunks} 个")

            # [3] Neo4j CITES 关系（Reference 节点保留供共享）
            deleted_citations = self.graph_store.delete_document_citations(file_path)
            if deleted_citations:
                print(f"  Neo4j 删除 CITES 关系: {deleted_citations} 条")

            # [4] Neo4j Document 节点
            self.graph_store.delete_document_node(file_path)
            print(f"  Neo4j 删除 Document 节点")

            # [4] 失去 Chunk 支撑的 RELATES_TO 边
            deleted_relations = self.graph_store.delete_stale_relations()
            print(f"  Neo4j 删除孤立关系: {deleted_relations} 条")

            # [5] 孤立实体
            orphan_ids = self.graph_store.get_orphan_entity_ids()
            deleted_entities = 0
            if orphan_ids:
                deleted_entities = self.graph_store.delete_entities_by_ids(orphan_ids)
            print(f"  Neo4j 删除孤立实体: {deleted_entities} 个")

            # [6] SQLite 元数据（优先按 doc_id 删，file_path 作为兜底）
            deleted_sqlite = self.database.delete_document_by_doc_id(doc_id)
            if not deleted_sqlite:
                deleted_sqlite = self.database.delete_document(file_path)
            if not deleted_sqlite:
                print(f"  [WARN] SQLite 未找到匹配记录 (doc_id={doc_id}, file_path={file_path})，跳过")

            # [7] ChunkTracker
            self.chunk_tracker.delete_by_doc(doc_id)

            # [8] 状态记录（最后删除，确保前面步骤均成功）
            self.status_store.delete(doc_id)

        except Exception as e:
            self.status_store.mark_failed(
                doc_id=doc_id,
                step="deleting",
                error=str(e),
                file_path=file_path,
            )
            # mark_failed 会将 status 改为 "failed"，这里手动改为 delete_failed
            s = self.status_store.get(doc_id)
            if s is not None:
                s.status = "delete_failed"
                self.status_store.upsert(s)
            raise

        print(f"  [DELETE] 完成")
        return {
            "doc_id": doc_id,
            "file_path": file_path,
            "deleted_vectors": deleted_vectors,
            "deleted_chunks": deleted_chunks,
            "deleted_relations": deleted_relations,
            "deleted_entities": deleted_entities,
            "deleted_relation_vectors": deleted_relation_vectors,
        }

    def close(self) -> None:
        self.database.close()
        self.graph_store.close()
        self.status_store.close()
        self.chunk_tracker.close()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _cleanup_old_document(self, old_status: DocumentStatus) -> None:
        """清理旧文档的 chunk/source 绑定，为增量更新做准备。

        执行顺序：
        1. 删除 FAISS 中旧 doc_id 的向量
        2. 删除 Neo4j 中旧文档的 Chunk 节点（及 MENTIONS 关系）
        3. 删除失去所有 Chunk 支撑的 RELATES_TO 边（旧文档独占的关系）
        4. 清理孤立实体（无任何 MENTIONS 来源的实体）
        5. 清理 ChunkTracker 中的旧记录
        6. 删除旧状态记录
        7. 重置 Neo4j 文档的实体抽取标志
        """
        old_doc_id = old_status.doc_id
        file_path = old_status.file_path

        deleted_vectors = self.vector_store.delete_by_doc_id(old_doc_id)
        print(f"      [cleanup] FAISS 删除向量: {deleted_vectors} 条")

        deleted_relation_vectors = self.relation_vector_store.delete_by_doc_id(old_doc_id)
        if deleted_relation_vectors:
            print(f"      [cleanup] 关系索引删除向量: {deleted_relation_vectors} 条")

        deleted_chunks = self.graph_store.delete_document_chunks(file_path)
        print(f"      [cleanup] Neo4j 删除 Chunk: {deleted_chunks} 个")

        # 清理旧文档的 CITES 关系（Reference 节点保留供共享）
        deleted_citations = self.graph_store.delete_document_citations(file_path)
        if deleted_citations:
            print(f"      [cleanup] Neo4j 删除 CITES 关系: {deleted_citations} 条")

        # 先清理失去 Chunk 支撑的 RELATES_TO 边，再清理孤立实体
        deleted_relations = self.graph_store.delete_stale_relations()
        if deleted_relations:
            print(f"      [cleanup] Neo4j 删除孤立关系: {deleted_relations} 条")

        orphan_ids = self.graph_store.get_orphan_entity_ids()
        if orphan_ids:
            deleted_entities = self.graph_store.delete_entities_by_ids(orphan_ids)
            print(f"      [cleanup] Neo4j 删除孤立实体: {deleted_entities} 个")

        self.chunk_tracker.delete_by_doc(old_doc_id)
        self.status_store.delete(old_doc_id)
        self.graph_store.reset_document_entity_extracted(file_path)
        self.relation_vector_store.save()
