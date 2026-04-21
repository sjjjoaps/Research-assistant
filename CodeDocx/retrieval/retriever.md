# retriever.py

## 所在层次
检索层 `src/retrieval/`

## 主体功能
封装 FAISS 语义向量检索，返回带来源信息的 `RetrievedChunk` 列表。
同时定义了整个检索层的核心数据结构 `RetrievedChunk`（被所有检索器复用）。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `RetrievedChunk` (dataclass) | 检索结果数据结构，含 `content/file_path/chunk_index/section_type/entity_id/year` |
| `SemanticRetriever.__init__(top_k)` | 加载 FAISS VectorStore |
| `SemanticRetriever.retrieve(query, section_type?)` | 执行语义向量检索，可选按 `section_type` 章节类型过滤 |

## 调用关系
- **被调用方**：`HybridRetriever`、`LocalRetriever`、`MixRetriever`、`LightRAGDualRetriever`、`qa_chain.py`
- **依赖方**：`src/storage/vector_store.py`（FAISS 封装）

## 注意事项
- `section_type` 过滤由 `VectorStore.similarity_search()` 原生支持（元数据字段筛选）
- `entity_id` 字段供 LightRAG one-hop 扩展使用，普通语义检索结果该字段为空字符串
- `year` 字段从 FAISS 元数据中提取，由 `ingestion_pipeline` 写入；提取失败时为 `None`
