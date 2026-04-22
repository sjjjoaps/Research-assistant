# entity_vector_store.py

## 所在层次
存储层 `src/storage/`

## 主体功能
为 Entity 节点维护独立的 FAISS 向量索引，支持实体文本写入、检索和按文档删除。
对齐 LightRAG §3.2 Low-Level 检索中的 `entities_vdb`，content 格式为 `"{entity_name}\n{description}"`。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `EntityVectorRecord` (dataclass) | 实体向量记录：`entity_id / entity_name / entity_type / description / doc_id / file_path` |
| `EntityVectorRecord.content` | 嵌入文本：`"{entity_name}\n{description}"`（与 LightRAG entities_vdb 格式一致） |
| `EntityVectorStore.add_entities(records)` | 批量写入实体向量，元数据含 `doc_id` 支持精确删除 |
| `EntityVectorStore.similarity_search(query, k)` | 按查询检索最相关的实体记录，返回 `list[Document]` |
| `EntityVectorStore.delete_by_doc_id(doc_id)` | 按 doc_id 删除文档贡献的实体向量（重建索引） |
| `EntityVectorStore.save()` / `load()` | 持久化/加载实体 FAISS 索引 |

## 调用关系
- **被调用方**：`LightRAGDualRetriever`（Low-Level 实体向量召回）、`ingestion_pipeline`（写入实体向量）
- **依赖方**：`langchain_community.vectorstores.FAISS`、`src/infrastructure/embedder.Embedder`

## 注意事项
- 每条实体按文档贡献写入一份向量，同一实体被多文档引用时有多条记录
- 内置 `threading.Lock` 保证并发写入安全
- 索引目录由 `settings.entity_faiss_index_dir` 配置（默认 `data/entity_faiss/`），与 chunk 和关系索引目录分离
- 索引为空时 `similarity_search` 返回空列表，`LightRAGDualRetriever` 会自动降级到 `GraphRetriever`
