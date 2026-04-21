# document_status_store.py

## 所在层次
存储层 `src/storage/`

## 主体功能
文档状态机存储，记录每个入库文档的当前处理阶段、关联 ID 和错误信息。
支持断点续传和失败定位。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `VALID_STATUSES` (frozenset) | 合法状态集合（11 种） |
| `generate_doc_id(file_hash)` | 基于文件内容 MD5 生成稳定 doc_id（与路径无关） |
| `DocumentStatus` (dataclass) | 状态记录：`doc_id / file_path / status / chunk_ids / entity_ids / relation_ids / error_message / created_at / updated_at` |
| `DocumentStatusStore.create(doc_id, file_path)` | 创建初始 pending 状态记录 |
| `DocumentStatusStore.update_status(doc_id, status, **kwargs)` | 更新状态和关联数据 |
| `DocumentStatusStore.get(doc_id)` | 查询单文档状态 |
| `DocumentStatusStore.list_all()` | 列出所有文档状态 |
| `DocumentStatusStore.delete(doc_id)` | 删除状态记录 |

## 调用关系
- **被调用方**：`ingestion_pipeline`（每阶段更新状态）、`api/routers/documents.py`（状态查询接口）
- **依赖方**：`SQLAlchemy`、`src/infrastructure/config.py`

## 注意事项
- 状态流转：`pending → parsing → chunking → metadata → indexing → graph → citations → extracting → processed`
- 任意阶段均可跳转到 `failed`；`processed → deleting → delete_failed`
- `doc_id` 基于文件内容 MD5，文件重命名/移动后 doc_id 不变（身份延续）
