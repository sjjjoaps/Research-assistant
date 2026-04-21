# master_agent.py

## 所在层次
Agent 层 `src/agents/`

## 主体功能
GraphAssistant 的核心 Agent，实现工具调用的 for-break 安全阀循环。
路由用户请求到 8 个工具（检索、深度研究、Idea 生成、文献查询等），SSE 流式输出。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `MasterAgent.run_stream(session_id, user_input)` | 主入口（AsyncGenerator）：加载历史 → LLM for-break 循环 → SSE 事件流 |
| `_accumulate_tool_calls(pending, chunk_calls)` | 增量累积流式 tool_call chunks 为 `list[dict]` |
| `_finalize_tool_calls(pending)` | 将字符串 args 解析为 dict（JSON parse） |
| `_accumulate_tokens(totals, chunk)` | 多字段兼容的 token 统计（支持不同 provider 字段名） |
| `_estimate_cost(tokens)` | 从 settings 读取定价，估算人民币费用 |
| `_collect_new_messages(messages, base_length)` | 提取本轮新增消息（含 HumanMessage）供 SessionManager 持久化 |
| `_fire_memory_extractor(user_input, ai_response, session_id)` | 后台线程触发记忆提取 |

## 两阶段工具调用设计

```python
# 第一阶段：立即发送所有 ToolStartEvent（UI 立即显示工具卡片）
for tool_call in final_tool_calls:
    yield ToolStartEvent(tool_name=tool_name, ...)

# 第二阶段：依次执行工具
for tool_call in final_tool_calls:
    result_content = await self._execute_tool_safe(...)
    yield ToolEndEvent(...)
```

## 调用关系
- **被调用方**：`api/routers/agent.py`（SSE 流式接口）
- **依赖方**：`SessionManager`、`build_tool_registry()`、`get_native_llm()`、`LongTermMemory`、`ToolCallLimiter`

## 注意事项
- for-break 循环终止条件：LLM 输出 `tool_calls` 为空（即进入最终回答轮）
- 含 `tool_calls` 的中间文本不透传前端（缓冲后丢弃），仅最终回答文本流式发送
- `save_user_memory` 工具调用后设置 `memory_written_this_turn` 标志，抑制后台自动提取
- `usage` 事件包含 prompt/completion token 数与估算费用（CNY）
- 后台记忆提取在 daemon thread 中运行，失败只记 WARNING，不影响主流程
