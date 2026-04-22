# entity_extractor.py

## 所在层次
摄入层 `src/ingestion/`

## 主体功能
从 chunk 文本中抽取学术实体与关系，写入 Neo4j 图数据库，同时生成实体向量记录供 `EntityVectorStore` 索引。
支持跨文档实体合并（基于自然键），并通过 `DescriptionMerger` 合并多次命中的描述。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `EntityItem` (Pydantic) | 实体结构：`name / entity_type / description` |
| `RelationItem` (Pydantic) | 关系结构：`source_name / target_name / relation_type / description` |
| `ExtractionResult` (Pydantic) | LLM 输出结构：`entities / relations` |
| `EntityExtractionStats` (dataclass) | 统计信息：处理 chunk 数 / 创建实体数 / 关系数 / `entity_records` / `relation_records` |
| `EntityExtractor.extract_for_document(chunks, file_path)` | 主入口：逐 chunk 调用 LLM 抽取，写入 Neo4j，填充 `entity_records` 和 `relation_records`，返回统计 |

## 调用关系
- **被调用方**：`src/workflows/ingestion_pipeline.py`（入库流水线实体抽取阶段）
- **依赖方**：`src/storage/graph_store.GraphStore`（upsert 写入）、`DescriptionMerger`、`src/agents/prompt_loader.load_prompt_pair("relation_extractor")`

## 注意事项
- 实体 ID 基于 `normalized_name + entity_type` 生成（MD5），实现跨文档实体合并
- 使用 `upsert_entity_with_merge / upsert_relation_with_merge`，并发安全由 `GraphStore` 内置细粒度锁保证
- `stats.entity_records` 填充 `EntityVectorRecord` 列表，由 `ingestion_pipeline` 写入 `EntityVectorStore`
- `stats.relation_records` 填充 `RelationVectorRecord` 列表，由 `ingestion_pipeline` 写入 `RelationVectorStore`
- 支持的实体类型：`Method / Model / Dataset / Task / Concept / Metric`
- 支持的关系类型：`USES / PROPOSES / EVALUATES_ON / APPLIES_TO / BASED_ON`
