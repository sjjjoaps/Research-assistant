# 工作流层（workflows）

目录：`src/workflows/`

## 模块清单

| 文件 | 主要类/函数 | 职责 |
|---|---|---|
| `ingestion_pipeline.py` | `IngestionPipeline`, `BatchIngestResult` | 文档入库主流水线，协调解析→分块→元数据→向量→图存储 |

## 增量入库逻辑

`IngestionPipeline.ingest_file()` 流程：

1. 计算文件哈希 → 生成 `doc_id`
2. 查询 `DocumentStatusStore`：若已处理且哈希未变 → 跳过（`skipped=True`）
3. 若路径相同但哈希变化 → 调用 `_cleanup_old_document()` 清理旧向量/图数据
4. 解析（含多模态提取）→ 分块 → 元数据提取 → 写入向量库/图库
5. 注册 chunk ID 到 `ChunkTracker`，更新 `DocumentStatusStore` 状态为 `processed`

## 删除逻辑

`delete_document(doc_id)` 优先按 `doc_id` 查找 SQLite 记录，找不到再按 `file_path` 降级，解决路径不一致导致的静默删除失败。

## 状态流转

```
pending → parsing → chunking → metadata → indexing → graph → citations → extracting → processed
                                                                                      ↘ failed
```
