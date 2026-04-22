# lightrag_retriever.py

## 所在层次
检索层 `src/retrieval/`

## 主体功能
实现 LightRAG §3.2 Dual-Level Retrieval Paradigm（双极检索范式），完全对齐论文原始流程。

检索三步流程：
1. **双极关键词提取**：`KeywordExtractor` → `ll_keywords`（具体实体）+ `hl_keywords`（宏观主题）
2. **双路向量召回**：
   - Low-Level：`ll_keywords → EntityVectorStore.similarity_search() → entity_ids → 图邻域扩展`（对齐 LightRAG `entities_vdb`）
   - High-Level：`hl_keywords → GraphStore.search_by_relations() → 关系描述向量召回`（对齐 LightRAG `relationships_vdb`）
3. **One-hop 邻居扩展**：从双路结果的 entity_id 集合出发，补充一跳邻居实体描述

降级策略：entity FAISS 索引为空（尚未建索引）时，Low-Level 自动降级到 `GraphRetriever`（Neo4j 关键词匹配）。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `LightRAGDualRetriever.__init__(top_k)` | 声明懒加载字段（含 `_entity_vector_store`），不触发 Neo4j/FAISS 连接 |
| `LightRAGDualRetriever.retrieve(query, top_k, section_filter)` | 主检索入口：关键词提取 → 双路召回 → one-hop 扩展 → 去重合并 → rerank |
| `LightRAGDualRetriever._get_entity_vector_store()` | 懒加载 `EntityVectorStore`，首次调用时 `load()` 索引 |
| `LightRAGDualRetriever._entity_ids_to_chunks(graph_store, entity_ids, k)` | Low-Level 核心：entity_id → `get_one_hop_neighbors()` → `[实体检索]` 格式 chunk |
| `LightRAGDualRetriever._relations_to_chunks(relations)` | 将 GraphStore 关系数据转为 `[关系检索]` 格式的 `RetrievedChunk` |
| `LightRAGDualRetriever._neighbors_to_chunks(neighbors)` | 将 one-hop 邻居实体转为 `[邻居实体]` 格式的 `RetrievedChunk` |
| `LightRAGDualRetriever._collect_entity_ids(chunks)` | 从 chunks 收集 entity_id（支持 `src::tgt` 拆分格式） |
| `LightRAGDualRetriever._merge_and_deduplicate(low, high, expanded, top_k)` | 三路合并去重，优先级：Low > High > Expanded |
| `LightRAGDualRetriever.close()` | 释放 GraphRetriever 和 GraphStore 连接 |

## 调用关系
- **被调用方**：`MixRetriever`（主路径）、`tool_registry.py`（global/dual 模式路由）
- **依赖方**：`KeywordExtractor`、`EntityVectorStore`、`GraphRetriever`（降级备用）、`GraphStore`（懒加载）

## 注意事项
- 所有外部依赖懒加载（`_get_xxx()` 方法），避免构造时触发连接
- Low-Level 优先使用 `EntityVectorStore` FAISS 向量匹配；索引为空时降级到 `GraphRetriever`
- `entity_id` 格式 `"source_id::target_id"` 由 `_relations_to_chunks` 生成，`_collect_entity_ids` 负责拆分
- `section_filter` 与 LightRAG 结果（`section_type="entity/relation"`）语义不一致，过滤后大概率为空
- 路由逻辑在 `tool_registry._do_retrieve()` 中实现，此类只负责 dual 路径的核心执行
