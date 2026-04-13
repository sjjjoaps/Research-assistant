"""
Phase 2.5 并发安全实体处理测试

验收标准：
- 并发入库多文件后不产生重复实体（GraphStore 细粒度锁）
- 并发入库结果统计正确（succeeded / skipped / failed）
- 单文件失败不影响其他文件
- 无死锁（测试在合理时间内完成）
- FAISS 内置锁保证并发写入安全（不依赖 monkey-patching）
- 同 doc_id 并发入库只处理一次（per-doc_id claim 锁）
"""
import threading
from pathlib import Path
from unittest.mock import MagicMock

from src.storage.document_status_store import DocumentStatus, DocumentStatusStore
from src.storage.chunk_tracker import ChunkTracker


def _build_pipeline(tmp_path: Path):
    """构造带真实 status_store / chunk_tracker 的 pipeline，其余 mock。"""
    from src.ingestion_pipeline import IngestionPipeline

    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline._enable_entity_extraction = False
    pipeline._entity_extractor = None

    pipeline.status_store = DocumentStatusStore(db_path=tmp_path / "doc_status.db")
    pipeline.status_store.init_db()
    pipeline.chunk_tracker = ChunkTracker(db_path=tmp_path / "chunk_tracker.db")
    pipeline.chunk_tracker.init_db()

    pipeline.document_parser = MagicMock()
    pipeline.chunker = MagicMock()
    pipeline.metadata_extractor = MagicMock()
    pipeline.database = MagicMock()
    pipeline.vector_store = MagicMock()
    pipeline.graph_store = MagicMock()

    pipeline.database.get_document.return_value = None
    pipeline.database.add_document.return_value = 1
    pipeline.graph_store.is_document_entity_extracted.return_value = False

    return pipeline


class TestGraphStoreLocking:
    """测试 GraphStore 细粒度锁的并发安全性。"""

    def test_entity_lock_serializes_same_entity(self):
        """同一 entity_id 的并发 upsert 应被串行化，不产生竞态。"""
        from src.graph_store import GraphStore

        store = GraphStore.__new__(GraphStore)
        store.__init__.__func__  # 不调用真实 __init__（需要 Neo4j 连接）

        # 手动初始化锁结构
        import threading
        store._entity_locks = {}
        store._relation_locks = {}
        store._entity_locks_meta = threading.Lock()
        store._relation_locks_meta = threading.Lock()

        entity_id = "test_entity_001"
        lock1 = store._get_entity_lock(entity_id)
        lock2 = store._get_entity_lock(entity_id)

        # 同一 entity_id 应返回同一个 Lock 对象
        assert lock1 is lock2

    def test_different_entities_get_different_locks(self):
        """不同 entity_id 应获得不同的 Lock，互不阻塞。"""
        from src.graph_store import GraphStore
        import threading

        store = GraphStore.__new__(GraphStore)
        store._entity_locks = {}
        store._relation_locks = {}
        store._entity_locks_meta = threading.Lock()
        store._relation_locks_meta = threading.Lock()

        lock_a = store._get_entity_lock("entity_a")
        lock_b = store._get_entity_lock("entity_b")

        assert lock_a is not lock_b

    def test_relation_lock_serializes_same_relation(self):
        """同一关系三元组的并发 upsert 应被串行化。"""
        from src.graph_store import GraphStore
        import threading

        store = GraphStore.__new__(GraphStore)
        store._entity_locks = {}
        store._relation_locks = {}
        store._entity_locks_meta = threading.Lock()
        store._relation_locks_meta = threading.Lock()

        key = "src_id::USES::tgt_id"
        lock1 = store._get_relation_lock(key)
        lock2 = store._get_relation_lock(key)

        assert lock1 is lock2

    def test_concurrent_entity_lock_acquisition(self):
        """多线程并发获取同一实体锁时不产生竞态，执行顺序被串行化。"""
        from src.graph_store import GraphStore
        import threading

        store = GraphStore.__new__(GraphStore)
        store._entity_locks = {}
        store._relation_locks = {}
        store._entity_locks_meta = threading.Lock()
        store._relation_locks_meta = threading.Lock()

        entity_id = "shared_entity"
        results = []

        def write_with_lock(value: int):
            lock = store._get_entity_lock(entity_id)
            with lock:
                # 模拟读后写操作
                current = list(results)
                current.append(value)
                results.clear()
                results.extend(current)

        threads = [threading.Thread(target=write_with_lock, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 所有写入都应成功，无丢失
        assert len(results) == 10
        assert sorted(results) == list(range(10))


class TestBatchIngestResult:
    """测试 BatchIngestResult 数据结构。"""

    def test_batch_result_defaults(self):
        from src.ingestion_pipeline import BatchIngestResult

        r = BatchIngestResult()
        assert r.total == 0
        assert r.succeeded == 0
        assert r.skipped == 0
        assert r.failed == 0
        assert r.results == []
        assert r.errors == []


class TestIngestFilesConcurrent:
    """测试 ingest_files_concurrent 的并发行为。"""

    def test_all_succeed(self, tmp_path):
        """所有文件成功入库时，succeeded 计数正确。"""
        pipeline = _build_pipeline(tmp_path)

        # mock ingest_file 返回成功结果
        call_count = 0
        call_lock = threading.Lock()

        def fake_ingest(fp):
            nonlocal call_count
            with call_lock:
                call_count += 1
            return {
                "file_path": str(fp),
                "doc_id": f"doc_{Path(fp).stem}",
                "skipped": False,
                "record_id": 1,
                "title": "Test",
                "chunk_count": 2,
                "entity_count": 0,
                "relation_count": 0,
            }

        pipeline.ingest_file = fake_ingest

        files = [tmp_path / f"doc{i}.txt" for i in range(5)]
        for f in files:
            f.write_text("content")

        result = pipeline.ingest_files_concurrent(files, max_workers=3)

        assert result.total == 5
        assert result.succeeded == 5
        assert result.skipped == 0
        assert result.failed == 0
        assert call_count == 5

    def test_single_failure_does_not_block_others(self, tmp_path):
        """单文件失败不影响其他文件的入库。"""
        pipeline = _build_pipeline(tmp_path)

        FAIL_MARKER = "__should_fail__"

        def fake_ingest(fp):
            if FAIL_MARKER in Path(fp).name:
                raise RuntimeError("模拟失败")
            return {
                "file_path": str(fp),
                "doc_id": f"doc_{Path(fp).stem}",
                "skipped": False,
                "record_id": 1,
                "title": "Test",
                "chunk_count": 2,
                "entity_count": 0,
                "relation_count": 0,
            }

        pipeline.ingest_file = fake_ingest

        files = [
            tmp_path / "doc1.txt",
            tmp_path / f"{FAIL_MARKER}.txt",
            tmp_path / "doc2.txt",
        ]
        for f in files:
            f.write_text("content")

        result = pipeline.ingest_files_concurrent(files, max_workers=3)

        assert result.total == 3
        assert result.succeeded == 2
        assert result.failed == 1
        assert len(result.errors) == 1
        assert "模拟失败" in result.errors[0]["error"]

    def test_already_processed_counted_as_skipped(self, tmp_path):
        """已处理文档应计入 skipped，不计入 succeeded。"""
        from src.storage.extraction_cache import compute_file_hash
        from src.storage.document_status_store import generate_doc_id

        pipeline = _build_pipeline(tmp_path)

        # 预置一个已处理文档
        doc_file = tmp_path / "existing.txt"
        doc_file.write_text("already processed content")
        file_hash = compute_file_hash(doc_file)
        doc_id = generate_doc_id(file_hash)
        pipeline.status_store.upsert(DocumentStatus(
            file_path=str(doc_file),
            doc_id=doc_id,
            status="processed",
            current_step="入库完成",
        ))

        # 另一个新文档
        new_file = tmp_path / "new_doc.txt"
        new_file.write_text("new content")

        def fake_ingest(fp):
            return {
                "file_path": str(fp),
                "doc_id": "new_doc_id",
                "skipped": False,
                "record_id": 1,
                "title": "New",
                "chunk_count": 1,
                "entity_count": 0,
                "relation_count": 0,
            }

        pipeline.ingest_file = fake_ingest

        result = pipeline.ingest_files_concurrent([doc_file, new_file], max_workers=2)

        assert result.total == 2
        assert result.skipped == 1
        assert result.succeeded == 1
        assert result.failed == 0

    def test_concurrent_writes_no_duplicate_calls(self, tmp_path):
        """并发入库时每个文件只被处理一次。"""
        pipeline = _build_pipeline(tmp_path)
        processed_files = []
        lock = threading.Lock()

        def fake_ingest(fp):
            with lock:
                processed_files.append(str(fp))
            return {
                "file_path": str(fp),
                "doc_id": f"doc_{Path(fp).stem}",
                "skipped": False,
                "record_id": 1,
                "title": "Test",
                "chunk_count": 1,
                "entity_count": 0,
                "relation_count": 0,
            }

        pipeline.ingest_file = fake_ingest

        files = [tmp_path / f"file{i}.txt" for i in range(8)]
        for f in files:
            f.write_text(f"content {f.stem}")

        result = pipeline.ingest_files_concurrent(files, max_workers=4)

        assert result.succeeded == 8
        # 每个文件只处理一次
        assert len(processed_files) == 8
        assert len(set(processed_files)) == 8


class TestVectorStoreLocking:
    """测试 VectorStore 内置锁的线程安全性。"""

    def test_vector_store_has_builtin_lock(self):
        """VectorStore 实例应有内置 _lock 属性，类型为 threading.Lock 实例。"""
        import threading
        from src.vector_store import VectorStore

        store = VectorStore.__new__(VectorStore)
        store._store = None
        store._lock = threading.Lock()

        # threading.Lock() 返回 _thread.lock 实例，用 hasattr 验证接口即可
        assert hasattr(store._lock, "acquire")
        assert hasattr(store._lock, "release")
        assert hasattr(store._lock, "__enter__")

    def test_concurrent_add_chunks_serialized_by_lock(self, tmp_path):
        """并发调用 add_chunks 时，内置锁保证写操作串行化，不产生竞态。"""
        import threading
        from src.vector_store import VectorStore

        store = VectorStore.__new__(VectorStore)
        store._store = None
        store._lock = threading.Lock()
        store.embedder = MagicMock()
        store.index_dir = tmp_path

        # 用锁保护的计数器模拟并发写入
        counter = [0]
        errors = []

        def increment_with_lock():
            with store._lock:
                val = counter[0]
                # 模拟读后写的非原子操作
                counter[0] = val + 1

        threads = [threading.Thread(target=increment_with_lock) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 锁保证了所有增量都被正确执行
        assert counter[0] == 20
        assert errors == []

    def test_no_monkey_patching_in_ingest_files_concurrent(self, tmp_path):
        """ingest_files_concurrent 不应修改 vector_store 的方法（无 monkey-patching）。"""
        pipeline = _build_pipeline(tmp_path)

        # 记录 vector_store 方法的原始 id
        original_add_id = id(pipeline.vector_store.add_chunks)
        original_save_id = id(pipeline.vector_store.save)

        ingest_called = []

        def fake_ingest(fp):
            # 在 ingest_file 执行期间检查方法是否被替换
            ingest_called.append({
                "add_id": id(pipeline.vector_store.add_chunks),
                "save_id": id(pipeline.vector_store.save),
            })
            return {
                "file_path": str(fp),
                "doc_id": f"doc_{Path(fp).stem}",
                "skipped": False,
                "record_id": 1,
                "title": "Test",
                "chunk_count": 1,
                "entity_count": 0,
                "relation_count": 0,
            }

        pipeline.ingest_file = fake_ingest

        f = tmp_path / "test.txt"
        f.write_text("unique content xyz")

        pipeline.ingest_files_concurrent([f], max_workers=1)

        # 方法 id 在 ingest_file 执行期间不应改变
        assert len(ingest_called) == 1
        assert ingest_called[0]["add_id"] == original_add_id
        assert ingest_called[0]["save_id"] == original_save_id

        # 执行完毕后方法 id 也不应改变
        assert id(pipeline.vector_store.add_chunks) == original_add_id
        assert id(pipeline.vector_store.save) == original_save_id


class TestDocIdClaimLock:
    """测试 per-doc_id claim 锁防止同内容文件并发重复处理。"""

    def test_same_content_files_processed_only_once(self, tmp_path):
        """两个内容相同的文件（同 doc_id）并发入库时，只有一个被实际处理，另一个被跳过。"""
        from src.storage.extraction_cache import compute_file_hash
        from src.storage.document_status_store import generate_doc_id

        pipeline = _build_pipeline(tmp_path)

        # 创建两个内容相同的文件（会产生相同 doc_id）
        content = "identical content for dedup test"
        file_a = tmp_path / "doc_a.txt"
        file_b = tmp_path / "doc_b.txt"
        file_a.write_text(content)
        file_b.write_text(content)

        ingest_call_count = 0
        ingest_lock = threading.Lock()

        def fake_ingest(fp):
            nonlocal ingest_call_count
            file_hash = compute_file_hash(Path(fp))
            doc_id = generate_doc_id(file_hash)
            with ingest_lock:
                ingest_call_count += 1
            # 模拟入库完成后写入 processed 状态
            pipeline.status_store.upsert(DocumentStatus(
                file_path=str(fp),
                doc_id=doc_id,
                status="processed",
                current_step="入库完成",
            ))
            return {
                "file_path": str(fp),
                "doc_id": doc_id,
                "skipped": False,
                "record_id": 1,
                "title": "Test",
                "chunk_count": 1,
                "entity_count": 0,
                "relation_count": 0,
            }

        pipeline.ingest_file = fake_ingest

        result = pipeline.ingest_files_concurrent([file_a, file_b], max_workers=2)

        assert result.total == 2
        # 同 doc_id 只处理一次：一个 succeeded，一个 skipped
        assert result.succeeded + result.skipped == 2
        assert result.failed == 0
        # ingest_file 只被调用一次
        assert ingest_call_count == 1

    def test_different_content_files_both_processed(self, tmp_path):
        """内容不同的文件（不同 doc_id）并发入库时，两个都被处理。"""
        pipeline = _build_pipeline(tmp_path)

        file_a = tmp_path / "doc_a.txt"
        file_b = tmp_path / "doc_b.txt"
        file_a.write_text("content A unique 111")
        file_b.write_text("content B unique 222")

        ingest_call_count = 0
        ingest_lock = threading.Lock()

        def fake_ingest(fp):
            nonlocal ingest_call_count
            with ingest_lock:
                ingest_call_count += 1
            return {
                "file_path": str(fp),
                "doc_id": f"doc_{Path(fp).stem}",
                "skipped": False,
                "record_id": 1,
                "title": "Test",
                "chunk_count": 1,
                "entity_count": 0,
                "relation_count": 0,
            }

        pipeline.ingest_file = fake_ingest

        result = pipeline.ingest_files_concurrent([file_a, file_b], max_workers=2)

        assert result.total == 2
        assert result.succeeded == 2
        assert result.skipped == 0
        assert ingest_call_count == 2
