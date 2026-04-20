# bm25_retriever.py

## 所在层次
检索层 `src/retrieval/`

## 主体功能
基于 BM25Okapi 算法的稀疏关键词检索器。
从 FAISS docstore 中加载全部 chunk 文本，构建倒排索引，支持精确关键词匹配。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `BM25Retriever.__init__(top_k)` | 加载 VectorStore 全量文档，调用 `_build_index()` 构建 BM25 索引 |
| `BM25Retriever._tokenize(text)` | 静态方法，按非单词字符切分并转小写，生成词元列表 |
| `BM25Retriever._build_index()` | 从 docstore 提取所有 chunk，构建 `BM25Okapi` 对象 |
| `BM25Retriever.retrieve(query)` | BM25 检索，返回 score > 0 的 top_k 个 `RetrievedChunk` |

## 调用关系
- **被调用方**：`HybridRetriever`
- **依赖方**：`rank_bm25.BM25Okapi`、`src/storage/vector_store.py`（提供全量文档）

## 注意事项
- 不支持 `section_type` 过滤（BM25 索引不含章节元数据），需在 RRF 融合后过滤
- score ≤ 0 的结果被丢弃（无关文档），保证返回质量
- 索引在 `__init__` 时一次性构建，不支持动态更新；若 VectorStore 发生变更需重新实例化
