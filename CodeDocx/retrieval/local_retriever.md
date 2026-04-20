# local_retriever.md

## 所在层次
检索层 `src/retrieval/`

## 主体功能
LightRAG Local 模式检索器，面向具体实体/方法/数据集问题的精确定位。
融合图检索（实体邻域）与语义检索两路，使用 RRF 合并结果。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `LocalRetriever.__init__(top_k)` | 初始化 `SemanticRetriever(top_k*2)` 和 `GraphRetriever(top_k*2)` |
| `LocalRetriever.retrieve(query, section_filter)` | 图路 + 语义路 → RRF 融合 → 可选 section 后过滤 → top_k |
| `LocalRetriever.close()` | 关闭 GraphRetriever 的 Neo4j 连接 |

## 调用关系
- **被调用方**：`MixRetriever`、`LightRAGDualRetriever`（间接）
- **依赖方**：`GraphRetriever`、`SemanticRetriever`、`fuse_ranked_lists()`

## 注意事项
- `GraphRetriever` 结果不含 `section_type` 语义，需在融合后做后过滤
- `section_filter` 非空时融合候选池扩大到 `top_k * 3`（`_FILTER_EXPAND = 3`），防止早截断
- `SemanticRetriever` 原生支持 `section_type` 过滤（向量检索阶段内部处理）
