# database.py

## 所在层次
存储层 `src/storage/`

## 主体功能
SQLite 元数据存储，使用 SQLAlchemy 2.0 ORM 管理论文元数据的增删查改。
存储文档的 `title / authors / year / abstract / keywords / doc_id` 等字段。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `DocumentRecord` (ORM Model) | SQLAlchemy 映射：`documents` 表，含 `file_path(unique) / doc_id / title / authors / year / abstract / keywords` |
| `MetadataDatabase.__init__(db_path?)` | 初始化 SQLite 引擎和 session_factory |
| `MetadataDatabase.init_db()` | 建表 + 迁移：为已有 DB 补加 `doc_id` 列 |
| `MetadataDatabase.upsert_document(file_path, metadata)` | 新增或更新文档元数据 |
| `MetadataDatabase.update_document(file_path, metadata, doc_id)` | 更新文档内容及 doc_id |
| `MetadataDatabase.list_documents()` | 列出所有文档记录 |
| `MetadataDatabase.get_document(file_path)` | 按路径查询单文档 |
| `MetadataDatabase.delete_document_by_doc_id(doc_id)` | 按 doc_id 删除记录 |

## 调用关系
- **被调用方**：`ingestion_pipeline`（写入元数据）、`tool_registry.list_documents/get_document_metadata`、`GlobalRetriever`（year 缓存）
- **依赖方**：`SQLAlchemy 2.0`、`src/infrastructure/config.py`（`sqlite_path`）

## 注意事项
- `doc_id` 列通过迁移 DDL 添加（`ALTER TABLE ADD COLUMN`），保证向后兼容
- `file_path` 为唯一键，重复 upsert 自动更新已有记录
