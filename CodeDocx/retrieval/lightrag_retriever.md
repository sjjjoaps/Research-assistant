# lightrag_retriever.py

## 所在层次
检索层 `src/retrieval/`

## 主体功能
实现 LightRAG §3.2 Dual-Level Retrieval Paradigm（双极检索范式）。
自动路由三条检索路径：dual（双路）/ local（实体精确）/ global（关系宏观）/ mix（综合兜底）。

检索三步流程：
1. 双极关键词提取（`ll_keywords` + `hl_keywords`）
2. Low-Level（GraphRetriever）+ High-Level（关系向量）双路召回
3. One-hop 邻居扩展（补充关联实体描述）

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `LightRAGDualRetriever.__init__(top_k)` | 声明懒加载字段，不触发 Neo4j/FAISS 连接 |
| `LightRAGDualRetriever.retrieve(query, top_k, section_filter)` | 主检索入口：关键词提取 → 双路召回 → one-hop 扩展 → 去重合并 |
| `LightRAGDualRetriever._relations_to_chunks(relations)` | 将 GraphStore 关系数据转为 `[关系检索]` 格式的 `RetrievedChunk` |
| `LightRAGDualRetriever._neighbors_to_chunks(neighbors)` | 将 one-hop 邻居实体转为 `[邻居实体]` 格式的 `RetrievedChunk` |
| `LightRAGDualRetriever._collect_entity_ids(chunks)` | 从 chunks 收集 entity_id（支持 `src::tgt` 拆分格式） |
| `LightRAGDualRetriever._merge_and_deduplicate(low, high, expanded, top_k)` | 三路合并去重，优先级：Low > High > Expanded |
| `LightRAGDualRetriever.close()` | 释放 GraphRetriever 和 GraphStore 连接 |

## 调用关系
- **被调用方**：`tool_registry.py`（`_do_retrieve` 中按检索模式路由）
- **依赖方**：`KeywordExtractor`、`GraphRetriever`、`GraphStore`（懒加载）

## 注意事项
- 所有外部依赖懒加载（`_get_xxx()` 方法），避免构造时触发连接
- `entity_id` 格式 `"source_id::target_id"` 由 `_relations_to_chunks` 生成，`_collect_entity_ids` 负责拆分
- `section_filter` 与 LightRAG 结果（`section_type="entity/relation"`）语义不一致，过滤后大概率为空
- 路由逻辑在 `tool_registry._do_retrieve()` 中实现，此类只负责 dual 路径的核心执行
