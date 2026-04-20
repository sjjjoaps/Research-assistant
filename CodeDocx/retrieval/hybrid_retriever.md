# hybrid_retriever.py

## 所在层次
检索层 `src/retrieval/`

## 主体功能
混合检索器：融合语义检索（FAISS）与 BM25 关键词检索，通过 Reciprocal Rank Fusion（RRF）重排序。
可选接入外部 Reranker 进行精排（`RERANKER_ENABLED=true` 时生效）。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `_rrf_score(rank, k=60)` | 计算单条记录的 RRF 分数：`1 / (k + rank)` |
| `HybridRetriever.__init__(top_k, semantic_top_k, bm25_top_k)` | 初始化语义和 BM25 两路检索器 |
| `HybridRetriever._chunk_key(chunk)` | 生成 `file_path#chunk_index` 唯一键，用于去重与分数聚合 |
| `HybridRetriever.retrieve(query, section_filter)` | 双路召回 → RRF 融合 → 可选 section 过滤 → 可选 Reranker 精排 |

## 调用关系
- **被调用方**：`src/qa_chain.py`（通过 `_do_retrieve` 工具注册表）、`tool_registry.py`
- **依赖方**：`SemanticRetriever`、`BM25Retriever`、`get_reranker()`

## 注意事项
- `section_filter` 非空时语义路径按 `top_k * 3` 扩容，避免后过滤导致候选不足
- RRF 在完整排序结果上过滤后再裁至 `top_k`，不先截断（防止 BM25 后排但符合章节的结果丢失）
- RRF 平滑常数 `_RRF_K = 60`（标准值，通常无需调整）
