# 检索层（retrieval）

目录：`src/retrieval/`

## 模块清单

| 文件 | 主要类/函数 | 职责 |
|---|---|---|
| `retriever.py` | `SemanticRetriever`, `RetrievedChunk` | FAISS 语义检索，支持 section_type 过滤 |
| `bm25_retriever.py` | `BM25Retriever` | BM25 关键词检索 |
| `graph_retriever.py` | `GraphRetriever` | Neo4j 实体感知检索（top_k 统一 int，KeywordExtractor 扩展） |
| `hybrid_retriever.py` | `HybridRetriever` | FAISS + BM25 + RRF 融合，可接 Reranker |
| `local_retriever.py` | `LocalRetriever` | 图检索优先 + 语义补充（LightRAG Local 模式） |
| `global_retriever.py` | `GlobalRetriever` | 关系向量 + 图关系优先（LightRAG Global 模式） |
| `mix_retriever.py` | `MixRetriever` | semantic + local + global 三路 RRF 融合 |
| `lightrag_retriever.py` | `LightRAGDualRetriever` | LightRAG 双极检索（local + global 并行） |
| `reranker.py` | `APIReranker`, `get_reranker()` | 外部 Reranker API 调用，超时降级 |
| `keyword_extractor.py` | `KeywordExtractor` | LLM 关键词提取（ll_keywords + hl_keywords 双层），含 isinstance 类型守卫 |
| `fusion.py` | `fuse_ranked_lists()` | RRF（Reciprocal Rank Fusion）融合算法 |
| `time_filter.py` | `TimeFilter` | 时间感知过滤（按年份范围过滤 RetrievedChunk） |

## 检索模式对比

| 模式 | 类 | 适用场景 |
|------|-----|----------|
| `semantic` | `SemanticRetriever` | 自然语言语义问答 |
| `hybrid` | `HybridRetriever` | 一般性查询（FAISS + BM25） |
| `graph` | `GraphRetriever` | 实体关联问题（Neo4j 实体命中） |
| `local` | `LocalRetriever` | 具体方法/模型/数据集（LightRAG Local） |
| `global` | `GlobalRetriever` | 趋势/主题/方向（LightRAG Global） |
| `mix` | `MixRetriever` | 复杂综合问题（三路融合） |

## 重要约束

所有 Retriever 的 `__init__` 均执行 `self.top_k = int(top_k)`，防止 LLM 工具调用传入字符串时导致 Neo4j `LIMIT` 参数类型错误（`"5" * 3 == "555"`）。
