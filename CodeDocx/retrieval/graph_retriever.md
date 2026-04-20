# graph_retriever.py

## 所在层次
检索层 `src/retrieval/`

## 主体功能
基于 Neo4j 图数据库的实体感知检索器。
通过关键词匹配实体节点，沿 `MENTIONS` 关系回溯到 Chunk，并可选扩展 1-hop 邻居实体描述。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `GraphRetriever.__init__(top_k, expand_entities)` | 初始化 Neo4j driver 和 `KeywordExtractor` |
| `GraphRetriever._extract_keywords(text)` | 调用 `KeywordExtractor` 提取 ll+hl 关键词，兜底朴素切词 |
| `GraphRetriever._find_matching_chunks(keywords)` | Cypher 查询：关键词 → 实体 → Chunk，按命中次数排序 |
| `GraphRetriever._get_neighbor_entities(entity_name)` | 查询指定实体的 1-hop `RELATES_TO` 邻居实体名称 |
| `GraphRetriever.retrieve(query)` | 完整检索流程：提取关键词 → 匹配 Chunk → 去重 → 可选邻居扩展 |
| `GraphRetriever.close()` | 关闭 Neo4j driver 连接 |

## 调用关系
- **被调用方**：`LocalRetriever`、`GlobalRetriever`、`LightRAGDualRetriever`
- **依赖方**：`neo4j.GraphDatabase`、`KeywordExtractor`、`src/infrastructure/config.py`（Neo4j 连接参数）

## 注意事项
- Cypher 查询使用 `toLower() CONTAINS toLower()` 做大小写不敏感模糊匹配
- `expand_entities=True` 时每个匹配实体都会额外查询邻居，可能产生多次 Neo4j 往返
- `entity_id` 字段取第一个匹配实体 ID，供 LightRAG one-hop 扩展使用
- `year` 字段通过 `Document -[:HAS_CHUNK]-> Chunk` 反查，旧数据可能为 `None`
