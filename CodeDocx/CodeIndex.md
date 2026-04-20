# 代码索引（CodeIndex）

本文档是项目各层代码文档的导航索引，对应 `src/` 目录的分层架构。

## 层级结构

| 层 | 目录 | 文档目录 | 职责 |
|---|---|---|---|
| 基础设施层 | `src/infrastructure/` | [infrastructure/](infrastructure/) | 配置、LLM客户端（text+vision双工厂）、嵌入、Token追踪、弹性工具、长期记忆、JSON工具 |
| 存储层 | `src/storage/` | [storage/](storage/) | 向量库、图数据库（Neo4j 5.x）、元数据库、文档状态、Chunk追踪、关系向量、解析缓存 |
| 检索层 | `src/retrieval/` | [retrieval/](retrieval/) | 语义/BM25/图/混合/Local/Global/Mix/LightRAG双极检索、Reranker、时间过滤（top_k 统一 int） |
| 入库层 | `src/ingestion/` | [ingestion/](ingestion/) | 文档解析（多模态）、分块、章节分块、元数据提取、实体/关系/引用提取、社区检测、多模态处理 |
| 工作流层 | `src/workflows/` | [workflows/](workflows/) | 入库流水线（增量/批量/删除，delete_by_doc_id 优先） |
| Agent层 | `src/agents/` | [agents/](agents/) | MasterAgent（两阶段工具调用）、QA/DeepResearch/Idea Agent、会话管理、工具注册表、记忆提取 |
| API层 | `api/` | [api/](api/) | FastAPI 路由（含 /agent SSE）、Schema 数据契约 |

## 入口文件

- `main.py` — FastAPI 应用启动入口（v1.1.0，含 /agent 路由）
- `app.py` — Streamlit 备用前端入口
- `frontend/` — React/TypeScript 主前端（Vite，端口 5173）
- `src/qa_chain.py` — 问答链核心逻辑（传统 QA 接口兼容层）

## 重要设计决策

| 决策 | 说明 |
|------|------|
| 工具调用两阶段 | MasterAgent 先遍历所有 tool_calls 发送 ToolStartEvent，再执行，确保 UI 立即显示工具卡片 |
| top_k 统一 int 转换 | 所有 Retriever `__init__` 中 `self.top_k = int(top_k)`，防止 LLM 传入字符串导致 Neo4j LIMIT 类型错误 |
| elementId 替代 id | Neo4j 5.x 废弃 `id()` 函数，`get_entity_relations()` 改用 `elementId()` 去重 |
| delete_document_by_doc_id 优先 | 删除时先按 doc_id 查找，找不到再按 file_path，解决路径不一致导致的静默删除失败 |
| extract_json 替代正则 | 所有 JSON 解析统一使用 `json_utils.extract_json()`，兼容 markdown 包裹、嵌套对象 |
| 社区检测自适应阈值 | `max(2, round(entity_count / 20))`，基于 Entity 数而非总节点数 |
| 多模态默认开启 | `ENABLE_MODAL_EXTRACTION=true`，默认最多 20 张图片 + 20 个表格，使用 vision LLM |

## 子目录文档

### 基础设施层 `infrastructure/`
| 文件 | 文档 |
|---|---|
| `config.py` | [infrastructure/config.md](infrastructure/config.md) |
| `llm_client.py` | [infrastructure/llm_client.md](infrastructure/llm_client.md) |
| `embedder.py` | [infrastructure/embedder.md](infrastructure/embedder.md) |
| `token_tracker.py` | [infrastructure/token_tracker.md](infrastructure/token_tracker.md) |
| `resilience.py` | [infrastructure/resilience.md](infrastructure/resilience.md) |
| `long_term_memory.py` | [infrastructure/long_term_memory.md](infrastructure/long_term_memory.md) |

### 存储层 `storage/`
| 文件 | 文档 |
|---|---|
| `vector_store.py` | [storage/vector_store.md](storage/vector_store.md) |
| `graph_store.py` | [storage/graph_store.md](storage/graph_store.md) |
| `database.py` | [storage/database.md](storage/database.md) |
| `relation_vector_store.py` | [storage/relation_vector_store.md](storage/relation_vector_store.md) |
| `chunk_tracker.py` | [storage/chunk_tracker.md](storage/chunk_tracker.md) |
| `document_status_store.py` | [storage/document_status_store.md](storage/document_status_store.md) |
| `extraction_cache.py` | [storage/extraction_cache.md](storage/extraction_cache.md) |

### 检索层 `retrieval/`
| 文件 | 文档 |
|---|---|
| `retriever.py` | [retrieval/retriever.md](retrieval/retriever.md) |
| `bm25_retriever.py` | [retrieval/bm25_retriever.md](retrieval/bm25_retriever.md) |
| `hybrid_retriever.py` | [retrieval/hybrid_retriever.md](retrieval/hybrid_retriever.md) |
| `graph_retriever.py` | [retrieval/graph_retriever.md](retrieval/graph_retriever.md) |
| `local_retriever.py` | [retrieval/local_retriever.md](retrieval/local_retriever.md) |
| `global_retriever.py` | [retrieval/global_retriever.md](retrieval/global_retriever.md) |
| `mix_retriever.py` | [retrieval/mix_retriever.md](retrieval/mix_retriever.md) |
| `lightrag_retriever.py` | [retrieval/lightrag_retriever.md](retrieval/lightrag_retriever.md) |
| `reranker.py` | [retrieval/reranker.md](retrieval/reranker.md) |
| `fusion.py` | [retrieval/fusion.md](retrieval/fusion.md) |
| `keyword_extractor.py` | [retrieval/keyword_extractor.md](retrieval/keyword_extractor.md) |
| `time_filter.py` | [retrieval/time_filter.md](retrieval/time_filter.md) |

### 摄入层 `ingestion/`
| 文件 | 文档 |
|---|---|
| `document_parser.py` | [ingestion/document_parser.md](ingestion/document_parser.md) |
| `chunker.py` | [ingestion/chunker.md](ingestion/chunker.md) |
| `section_chunker.py` | [ingestion/section_chunker.md](ingestion/section_chunker.md) |
| `section_recognizer.py` | [ingestion/section_recognizer.md](ingestion/section_recognizer.md) |
| `metadata_extractor.py` | [ingestion/metadata_extractor.md](ingestion/metadata_extractor.md) |
| `entity_extractor.py` | [ingestion/entity_extractor.md](ingestion/entity_extractor.md) |
| `citation_extractor.py` | [ingestion/citation_extractor.md](ingestion/citation_extractor.md) |
| `community_detector.py` | [ingestion/community_detector.md](ingestion/community_detector.md) |
| `description_merger.py` | [ingestion/description_merger.md](ingestion/description_merger.md) |
| `modal_processors.py` | [ingestion/modal_processors.md](ingestion/modal_processors.md) |

### 工作流层 `workflows/`
| 文件 | 文档 |
|---|---|
| `ingestion_pipeline.py` | [workflows/ingestion_pipeline.md](workflows/ingestion_pipeline.md) |

### Agent层 `agents/`
| 文件 | 文档 |
|---|---|
| `base_agent.py` | [agents/base_agent.md](agents/base_agent.md) |
| `master_agent.py` | [agents/master_agent.md](agents/master_agent.md) |
| `qa_agent.py` | [agents/qa_agent.md](agents/qa_agent.md) |
| `deep_research_agent.py` | [agents/deep_research_agent.md](agents/deep_research_agent.md) |
| `idea_agent.py` | [agents/idea_agent.md](agents/idea_agent.md) |
| `session_manager.py` | [agents/session_manager.md](agents/session_manager.md) |
| `tool_registry.py` | [agents/tool_registry.md](agents/tool_registry.md) |
| `tool_prompt_loader.py` | [agents/tool_prompt_loader.md](agents/tool_prompt_loader.md) |
| `prompt_loader.py` | [agents/prompt_loader.md](agents/prompt_loader.md) |
| `memory_extractor.py` | [agents/memory_extractor.md](agents/memory_extractor.md) |
| `events.py` | [agents/events.md](agents/events.md) |

### API层 `api/`
| 文件 | 文档 |
|---|---|
| `schemas.py` | [api/schemas.md](api/schemas.md) |
| `routers/` | [api/routers.md](api/routers.md) |

## 测试

测试文件位于 `tests/`，命名规则为 `test_<模块名>.py`。
