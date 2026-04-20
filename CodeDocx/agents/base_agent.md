# base_agent.py

## 所在层次
Agent 层 `src/agents/`

## 主体功能
所有 Agent 的基类，提供统一的会话状态管理骨架和抽象接口约束。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `Turn` (dataclass) | 单轮对话记录：`role / content / sources` |
| `BaseAgent.__init__(max_history_turns)` | 初始化内存态会话字典 `{thread_id: list[Turn]}` |
| `BaseAgent.start_thread(thread_id)` | 初始化会话线程（已存在则不覆盖） |
| `BaseAgent.append_user_message(thread_id, text)` | 写入用户消息 |
| `BaseAgent.append_assistant_message(thread_id, text, sources)` | 写入 Assistant 消息 |
| `BaseAgent.get_history(thread_id, max_turns?)` | 获取最近 `max_history_turns` 轮历史（每轮 = user + assistant 各一条） |

## 调用关系
- **被调用方**：`QAAgent`、`DeepResearchAgent`、`IdeaAgent`（均继承此类）
- **依赖方**：无外部依赖

## 注意事项
- 会话状态为内存态，进程重启后丢失；持久化由 `SessionManager` 负责
- `max_history_turns * 2` 条消息传给 LLM，防止 prompt 过大
- 子类必须实现 `run_turn()` 和 `run_stream()` 两个抽象方法
