# embedder.py

## 所在层次
基础设施层 `src/infrastructure/`

## 主体功能
文本嵌入封装，对外暴露统一的 `embed_texts / embed_query` 接口。
当前使用 DashScope（阿里云）嵌入模型，通过 `langchain_embeddings` 属性兼容 LangChain FAISS。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `Embedder.__init__()` | 初始化 `DashScopeEmbeddings`，从 `settings` 读取模型名和 API Key |
| `Embedder.embed_texts(texts)` | 批量 embedding，返回与输入等长的向量列表 |
| `Embedder.embed_query(query)` | 单条查询 embedding |
| `Embedder.langchain_embeddings` (property) | 返回底层 `DashScopeEmbeddings` 实例，供 FAISS 直接使用 |

## 调用关系
- **被调用方**：`VectorStore`、`RelationVectorStore`（通过 `langchain_embeddings` 属性）
- **依赖方**：`langchain_community.embeddings.DashScopeEmbeddings`、`src/infrastructure/config.settings`

## 注意事项
- 嵌入维度由 `settings.embedding_dim` 配置（默认 1024），需与 FAISS 索引维度一致
- 空列表输入时 `embed_texts` 直接返回 `[]`，不调用 API
