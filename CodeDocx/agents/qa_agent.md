# qa_agent.py

## 所在层次
Agent 层 `src/agents/`

## 主体功能
基于 `BaseAgent` 的文献问答 Agent，支持多轮上下文 + 多种检索模式 + RAG 回答。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `RetrieverMode` (Literal) | 检索模式枚举：`semantic / hybrid / graph / local / global / mix` |
| `_make_retriever(mode, top_k)` | 工厂函数：按模式创建对应检索器实例 |
| `QAAgent.__init__(retriever_mode, top_k)` | 初始化检索器和 LLM 链 |
| `QAAgent.run_turn(thread_id, user_input)` | 主入口：检索 → 构建 prompt → LLM 回答 → 写入历史 |

## 调用关系
- **被调用方**：`tool_registry.retrieve_knowledge`（作为 MasterAgent 工具的底层实现）
- **依赖方**：`BaseAgent`、各检索器（按 mode 懒加载）、`src/agents/prompt_loader.load_prompt_pair("qa_agent")`

## 注意事项
- 检索模式通过 `retriever_mode` 参数切换，默认 `"semantic"`
- `run_turn` 返回值包含 `ll_keywords / hl_keywords`（Phase 4.1 新增），供调用方记录
- 历史轮数由 `BaseAgent.max_history_turns` 控制（默认 5 轮）
