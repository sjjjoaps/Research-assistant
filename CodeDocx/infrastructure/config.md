# config.py

## 所在层次
基础设施层 `src/infrastructure/`

## 主体功能
全局配置管理，使用 `pydantic-settings` 从 `.env` 文件和环境变量读取配置。
通过模块级单例 `settings` 供全项目访问。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `Settings` (BaseSettings) | 配置类，所有字段均有默认值，通过 `alias` 对应环境变量名 |
| `settings` | 模块级单例，`Settings()` 实例 |

主要配置分组：
- **LLM**：`model_name / base_url / api_key / embedding_model_name / embedding_dim`
- **LLM 稳定性**：`llm_timeout_seconds / llm_max_retries`
- **入库开关**：`enable_entity_extraction / enable_modal_extraction / enable_section_recognition / enable_citation_extraction / enable_section_chunking`
- **文本切块**：`chunk_size / chunk_overlap / section_chunk_overlap`
- **混合检索**：`rrf_k / retrieval_filter_expand / hybrid_semantic_top_k / hybrid_bm25_top_k`
- **Agent 行为**：`master_agent_max_iterations / agent_max_history_turns`
- **并发入库**：`ingestion_max_workers`
- **存储路径**：`data_dir / sqlite_path / faiss_index_dir / relation_faiss_index_dir / entity_faiss_index_dir`
- **Neo4j**：`neo4j_uri / neo4j_username / neo4j_password`
- **Reranker**：`reranker_enabled / reranker_api_url / reranker_api_key / reranker_model / reranker_top_n / reranker_timeout / reranker_max_candidates`
- **费用估算**：`price_per_1k_prompt / price_per_1k_completion`
- **工具限流**：`tool_call_max_per_window / tool_call_window_seconds`

## 调用关系
- **被调用方**：几乎所有模块（通过 `from src.infrastructure.config import settings` 导入）
- **依赖方**：`pydantic_settings.BaseSettings`、`.env` 文件

## 注意事项
- `.env` 路径固定为 `ROOT_DIR/.env`（项目根目录）
- `extra="ignore"` 忽略未知环境变量，不抛异常
- 配置变更需重启服务（`settings` 为模块导入时实例化的单例）
