# tool_registry.py

## 所在层次
Agent 层 `src/agents/`

## 主体功能
8 个核心工具的定义与注册，封装检索器/Agent 为 LLM 可调用的 `@tool`。
通过 `build_tool_registry()` 一次性构建工具列表（`@lru_cache` 单例）。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `TOOL_DISPLAY_NAMES` (dict) | 工具 → 前端展示消息映射（SSE ToolStartEvent 使用） |
| `retrieve_knowledge(query, mode, top_k, section_filter, year_from, year_to)` | 知识库多模式检索，支持 auto/semantic/hybrid/graph/local/global/mix/dual |
| `deep_research(question, retriever_mode, use_community)` | 调用 `DeepResearchAgent` 生成深度研究报告 |
| `generate_research_ideas(question, retriever_mode)` | 调用 `IdeaAgent` 生成研究 Idea |
| `list_documents()` | 查询 `MetadataDatabase` 返回文献清单 |
| `get_document_metadata(file_path)` | 查询单篇文献元数据 |
| `search_by_entity(entity_name, entity_type?, top_k)` | 知识图谱实体及关系查询 |
| `get_knowledge_graph_stats()` | 返回图谱节点数、关系数统计 |
| `save_user_memory(title, body, memory_type)` | 将用户偏好/规则写入长期记忆 |
| `_do_retrieve(query, mode, top_k, section_filter, year_from, year_to)` | 检索核心逻辑，含路由/时间过滤/长期记忆 |
| `build_tool_registry()` | `@lru_cache` 工具列表工厂，返回 `list[StructuredTool]` |

## 调用关系
- **被调用方**：`MasterAgent`（`build_tool_registry()` 获取工具列表）
- **依赖方**：所有检索器（懒加载）、`DeepResearchAgent`、`IdeaAgent`、`MetadataDatabase`、`GraphStore`、`LongTermMemory`

## 注意事项
- `auto` 模式路由：优先查询 `LongTermMemory.get_best_mode()`，无历史最优时回退启发式 `_auto_select_mode()`
- `dual` 模式触发 `LightRAGDualRetriever`（Phase 9-1）
- 所有工具返回格式化 Markdown 字符串，供 LLM 直接引用
- 顶层导入使用 `try/except` 兜底，外部服务未启动时导入本模块不会失败
