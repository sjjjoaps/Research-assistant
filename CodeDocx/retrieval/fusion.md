# fusion.py

## 所在层次
检索层 `src/retrieval/`

## 主体功能
多路检索结果融合工具，实现 Reciprocal Rank Fusion（RRF）算法。
供 `LocalRetriever`、`GlobalRetriever`、`MixRetriever` 复用，避免重复实现。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `rrf_score(rank, k=60)` | 计算 RRF 分数：`1 / (k + rank)`，rank 从 1 开始 |
| `chunk_key(chunk)` | 生成去重键：`file_path#chunk_index#hash(content)` |
| `fuse_ranked_lists(ranked_lists, top_k)` | 接收多路有序列表，RRF 融合后返回 top_k 个 `RetrievedChunk` |

## 调用关系
- **被调用方**：`LocalRetriever`、`GlobalRetriever`、`MixRetriever`
- **依赖方**：`src/retrieval/retriever.RetrievedChunk`

## 注意事项
- `chunk_key` 包含 `hash(content)` 以区分内容不同但 `(file_path, chunk_index)` 相同的 chunk（如关系 chunk）
- `HybridRetriever` 有自己的内联 RRF 实现（不调用此模块），两者逻辑等价但 key 生成略有差异
- RRF 平滑常数 `_RRF_K = 60`，与 `hybrid_retriever.py` 保持一致
