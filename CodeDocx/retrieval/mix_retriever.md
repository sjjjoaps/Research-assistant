# mix_retriever.py

## 所在层次
检索层 `src/retrieval/`

## 主体功能
LightRAG Mix 模式检索器，综合 Semantic + Local + Global 三路结果。
适用于无明确实体/主题倾向的综合性问题。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `MixRetriever.__init__(top_k)` | 初始化三路检索器，各自 top_k 扩容为 `top_k * 2` |
| `MixRetriever.retrieve(query, section_filter)` | 三路并行检索 → RRF 融合 → 可选 section 后过滤 → 可选 Reranker 精排 |
| `MixRetriever.close()` | 关闭 LocalRetriever 和 GlobalRetriever 的连接 |

## 调用关系
- **被调用方**：`LightRAGDualRetriever`（当关键词提取失败时作为 fallback）、`tool_registry.py`
- **依赖方**：`SemanticRetriever`、`LocalRetriever`、`GlobalRetriever`、`fuse_ranked_lists()`、`get_reranker()`

## 注意事项
- `GlobalRetriever` 结果（`section_type="relation/entity"`）在 section_filter 后过滤通常为空，属设计预期
- `section_filter` 时融合候选池扩大到 `top_k * 3`（`_FILTER_EXPAND = 3`），过滤后再裁至 `top_k`
- 三路检索器包含 Neo4j 连接，使用完毕应调用 `close()` 释放资源
