# global_retriever.py

## 所在层次
检索层 `src/retrieval/`

## 主体功能
LightRAG Global 模式检索器，面向宏观主题、趋势、关系结构问题。
融合关系向量检索（`RelationVectorStore`）与图实体检索（`GraphRetriever`）。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `GlobalRetriever.__init__(top_k)` | 初始化 `RelationVectorStore`、`GraphRetriever`、`KeywordExtractor` |
| `GlobalRetriever._get_year_cache()` | 懒加载 `file_path→year` 映射（从 `MetadataDatabase` 读取一次后缓存） |
| `GlobalRetriever._relation_docs_to_chunks(query)` | 关系向量检索 → 构建 `[关系检索]` 格式的 `RetrievedChunk` 列表 |
| `GlobalRetriever.retrieve(query, section_filter)` | 关系向量路 + 图路 → RRF 融合 → 可选 section 后过滤 → top_k |
| `GlobalRetriever.close()` | 关闭 GraphRetriever 的 Neo4j 连接 |

## 调用关系
- **被调用方**：`MixRetriever`、`LightRAGDualRetriever`（High-Level 路）
- **依赖方**：`RelationVectorStore`、`GraphRetriever`、`KeywordExtractor`、`MetadataDatabase`（year 缓存）

## 注意事项
- GlobalRetriever 产出的 `section_type` 固定为 `"relation"` 或 `"entity"`，与文本章节类型不兼容
- 搭配 `section_filter` 时过滤后结果大概率为空；如需章节过滤建议改用 local/semantic 模式
- `year` 字段通过 `file_path→year` 缓存附加到关系 chunk，旧索引中可能为 `None`
