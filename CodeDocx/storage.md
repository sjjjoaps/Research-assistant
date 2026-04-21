# 存储层（storage）

目录：`src/storage/`

## 模块清单

| 文件 | 主要类/函数 | 职责 |
|---|---|---|
| `database.py` | `MetadataDatabase` | SQLite 元数据库（SQLAlchemy），存储文档元信息，支持 delete_document_by_doc_id |
| `vector_store.py` | `VectorStore` | FAISS chunk 向量索引，语义检索 |
| `relation_vector_store.py` | `RelationVectorStore` | FAISS 关系向量索引，LightRAG Global 检索 |
| `graph_store.py` | `GraphStore` | Neo4j 图谱写入与查询（elementId 去重，细粒度锁） |
| `document_status_store.py` | `DocumentStatusStore`, `DocumentStatus`, `generate_doc_id()` | 文档状态机持久化（SQLite） |
| `chunk_tracker.py` | `ChunkTracker`, `compute_chunk_content_hash()`, `generate_chunk_id()` | Chunk ID 注册与增量去重（SQLite） |
| `extraction_cache.py` | `compute_file_hash()`, `ExtractionCache` | 文件哈希与提取结果缓存 |
