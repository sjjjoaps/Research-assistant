# chunk_tracker.py

## 所在层次
存储层 `src/storage/`

## 主体功能
持久化记录 `doc_id → chunk_ids` 映射，为增量更新和精确删除提供基础。
chunk_id 基于 `doc_id + 内容哈希 + 文档内出现序号` 生成，保证稳定性。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `generate_chunk_id(doc_id, content_hash, occurrence_index)` | 生成稳定 chunk_id（MD5），同文档内相同内容不同位置产生不同 ID |
| `compute_chunk_content_hash(content)` | 计算 chunk 文本内容的 MD5 |
| `ChunkTracker.save_chunks(doc_id, chunk_ids)` | 持久化 doc_id → chunk_ids 映射 |
| `ChunkTracker.get_chunk_ids(doc_id)` | 查询指定文档的所有 chunk_id |
| `ChunkTracker.delete_by_doc_id(doc_id)` | 删除文档的 chunk 追踪记录 |

## 调用关系
- **被调用方**：`ingestion_pipeline`（写入/查询 chunk 映射）、`DocumentChunker`/`SectionChunker`（生成 chunk_id）
- **依赖方**：`SQLAlchemy`、`src/infrastructure/config.py`

## 注意事项
- `occurrence_index` 参与主键，防止同文档内重复内容互相覆盖
- 文件内容变更后 `doc_id` 改变，旧 chunk_id 自动失效（不需要显式清理）
- 同内容不同路径的文档共享同一 `doc_id`（去重语义）
