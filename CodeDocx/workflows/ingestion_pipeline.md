# ingestion_pipeline.py

## 所在层次
工作流层 `src/workflows/`

## 主体功能
文献入库流水线，串联所有入库阶段：解析 → 切块 → 元数据 → SQLite → FAISS → Neo4j → 实体抽取 → 引用提取。
支持增量检测（内容未变则跳过）、变更检测（内容变化则先清理再重建）、并发批量入库。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `BatchIngestResult` (dataclass) | 批量入库汇总：`total / succeeded / skipped / failed / results / errors` |
| `IngestionPipeline.__init__(enable_entity_extraction, enable_modal_extraction, enable_section_recognition, enable_citation_extraction, enable_section_chunking)` | 初始化各阶段开关和组件 |
| `IngestionPipeline.ingest(file_path)` | 单文档入库主流程，更新 `DocumentStatusStore` 状态机 |
| `IngestionPipeline.batch_ingest(file_paths, max_workers)` | 并发批量入库，使用 `ThreadPoolExecutor` |
| `IngestionPipeline.delete_document(doc_id)` | 精确删除：只清理该文档独占数据，共享实体/关系保留 |

## 调用关系
- **被调用方**：`api/routers/documents.py`（上传接口触发入库）
- **依赖方**：`DocumentParser`、`SectionChunker`/`DocumentChunker`、`MetadataExtractor`、`MetadataDatabase`、`VectorStore`、`GraphStore`、`EntityExtractor`、`CitationExtractor`、`ChunkTracker`、`DocumentStatusStore`、`RelationVectorStore`

## 注意事项
- 增量检测：`doc_id`（文件内容 MD5）未变化时直接跳过，不重复入库
- 变更检测：同路径文件内容变化时，先清理旧 chunk/source 绑定再重建
- `enable_section_chunking=True`（默认）时使用 `SectionChunker`，无章节信息时 fallback 到 `DocumentChunker`
- 删除流程引入 `deleting / delete_failed` 状态机，支持失败重试
- 状态流转：`pending → parsing → chunking → metadata → indexing → graph → citations → extracting → processed`
