# relation_vector_store.py

## 所在层次
存储层 `src/storage/`

## 主体功能
为 Entity-RELATES_TO 关系维护独立的 FAISS 向量索引，支持关系文本写入、检索和按文档删除。
与 chunk 索引分离，磁盘目录独立保存。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `RelationVectorRecord` (dataclass) | 关系向量记录：`relation_key / source_entity_id / source_name / relation_type / target_entity_id / target_name / description / doc_id / file_path` |
| `RelationVectorStore.add_relations(records)` | 批量写入关系向量，元数据含 `doc_id` 支持精确删除 |
| `RelationVectorStore.similarity_search(query, k)` | 按查询检索最相关的关系记录 |
| `RelationVectorStore.delete_by_doc_id(doc_id)` | 按 doc_id 删除文档贡献的关系向量（重建索引） |
| `RelationVectorStore.save()` / `load()` | 持久化/加载关系 FAISS 索引 |

## 调用关系
- **被调用方**：`GlobalRetriever`（关系向量召回）、`ingestion_pipeline`（写入关系向量）
- **依赖方**：`langchain_community.vectorstores.FAISS`、`src/infrastructure/embedder.Embedder`

## 注意事项
- 每条关系按文档贡献写入一份向量，同一关系被多文档引用时有多条记录
- 内置 `threading.Lock` 保证并发写入安全
- 索引目录由 `settings.relation_faiss_index_dir` 配置，与 chunk 索引目录分离
