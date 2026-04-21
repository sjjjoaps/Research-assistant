# events.py

## 所在层次
Agent 层 `src/agents/`

## 主体功能
定义 MasterAgent 向前端推送的所有 SSE 事件类型，是前后端 SSE 协议的数据契约。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `EventType` | Literal 类型别名，枚举 9 种事件名称 |
| `BaseEvent` | 所有事件的基类，提供 `to_sse()` 序列化为 SSE 协议字符串 |
| `SessionStartEvent` | 会话开始，携带 `session_id` |
| `ThinkingEvent` | Agent 进入新一轮推理，携带 `iteration` 轮次 |
| `ToolStartEvent` | 工具调用开始，携带 `tool_name` 和 `display_message`（前端展示文案） |
| `ToolEndEvent` | 工具调用结束，携带 `result_summary` 和 `elapsed_ms` |
| `TextDeltaEvent` | 流式文本增量，携带 `delta` 字符串 |
| `SourcesEvent` | 检索来源列表，携带 `sources` 数组 |
| `UsageEvent` | Token 用量与费用，携带 `total_tokens`/`prompt_tokens`/`completion_tokens`/`estimated_cost_cny` |
| `DoneEvent` | 对话结束，携带 `session_id` |
| `ErrorEvent` | 错误事件，携带 `message` 和 `code` |

## 调用关系
- **被调用方**：`src/agents/master_agent.py`（构造并 yield 各类事件）、`api/routers/agent.py`（接收 AsyncGenerator 并转发 SSE）
- **依赖方**：Python 标准库 `dataclasses`、`json`

## 注意事项
- `to_sse()` 输出格式：`event: {type}\ndata: {json}\n\n`，符合 W3C SSE 规范
- 所有事件字段均通过 `dataclasses.asdict()` 序列化，嵌套对象需确保可 JSON 序列化
- `sources` 字段类型为 `list`（无强类型约束），前端需按约定解析其内部结构
