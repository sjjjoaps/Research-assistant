# graph_store.py

## 所在层次
存储层 `src/storage/`

## 主体功能
Neo4j 图数据库封装，负责创建约束、写入节点/关系、图谱查询和可视化数据获取。
内置细粒度进程内锁，防止并发写入同一实体/关系时描述丢失。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `GraphStore.create_document_node(file_path, metadata)` | 创建/更新 Document 节点 |
| `GraphStore.create_chunk_node(chunk)` | 写入 Chunk 节点 |
| `GraphStore.upsert_entity_with_merge(entity_id, name, type, desc, merger)` | 合并写入实体（描述合并，不覆盖） |
| `GraphStore.upsert_relation_with_merge(source_id, target_id, type, desc, merger)` | 合并写入关系（描述合并，不覆盖） |
| `GraphStore.create_reference_node(ref)` / `create_cites_relation()` | 引用节点与 CITES 关系管理 |
| `GraphStore.get_entity_relations()` | 读取所有 RELATES_TO 边（用 elementId 去重，替代废弃的 id()） |
| `GraphStore.create_community_node(...)` / `create_belongs_to_relation()` | 社区节点与 BELONGS_TO 关系写入 |
| `GraphStore.search_by_relations(keywords, k)` | 按关键词匹配关系描述（LightRAG High-Level 召回） |
| `GraphStore.get_one_hop_neighbors(entity_ids)` | 获取实体集合的一跳邻居（LightRAG one-hop 扩展） |
| `GraphStore.get_graph_stats()` | 查询节点数/关系数统计 |
| `GraphStore.get_subgraph(limit, search, node_types)` | 查询可视化子图数据 |
| `GraphStore.delete_document_chunks(file_path)` | 删除文档的 Chunk 节点及 MENTIONS 关系 |
| `GraphStore.delete_stale_relations()` | 删除无 Chunk 共现支撑的 RELATES_TO 边 |

## 调用关系
- **被调用方**：`EntityExtractor`、`GraphRetriever`、`GlobalRetriever`、`LightRAGDualRetriever`、`ingestion_pipeline`、`CommunityDetector`、`api/routers/graph.py`、`api/routers/community.py`
- **依赖方**：`neo4j.GraphDatabase`、`src/infrastructure/config.py`

## 注意事项
- 细粒度锁：同一实体/关系串行化，不同实体互不阻塞（锁字典随实体数量增长，不自动回收）
- `get_entity_relations()` 使用 `elementId(s) < elementId(t)` 去重（Neo4j 5.x 废弃 `id()`）
- `upsert_entity_with_merge` 使用 MERGE 模式保证幂等，兼容旧节点的 `description_list` 为空情况
