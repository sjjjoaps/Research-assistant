# vector_store.py

## 所在层次
存储层 `src/storage/`

## 主体功能
FAISS 向量存储封装，支持 chunk 入库、相似度检索、持久化保存与加载。
内置 `threading.Lock` 保证多线程并发入库安全（FAISS 本身不是线程安全的）。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `VectorStore.__init__(index_dir?)` | 初始化 FAISS 存储路径和 Embedder |
| `VectorStore.add_chunks(chunks)` | 批量入库 TextChunk，写入 `doc_id / section_type` 等元数据 |
| `VectorStore.similarity_search(query, k, section_type?)` | 向量相似度检索，可选按 `section_type` 过滤 |
| `VectorStore.get_all_documents()` | 返回 docstore 中全部 Document（供 BM25 构建索引） |
| `VectorStore.delete_by_doc_id(doc_id)` | 重建索引排除指定文档的向量（保守实现） |
| `VectorStore.save()` | 持久化到磁盘；`_store=None` 时主动删除磁盘文件防止复现 |
| `VectorStore.load()` | 从磁盘加载 FAISS 索引 |

## 调用关系
- **被调用方**：`SemanticRetriever`、`BM25Retriever`、`ingestion_pipeline`
- **依赖方**：`langchain_community.vectorstores.FAISS`、`src/infrastructure/embedder.Embedder`

## 注意事项
- 所有写操作（`add_chunks / save / delete_by_doc_id`）均在 `threading.Lock` 内执行
- `delete_by_doc_id` 通过重建索引实现，大索引时耗时较长
- `save()` 在 `_store=None` 时主动删除磁盘文件，防止重启后已删向量复现
