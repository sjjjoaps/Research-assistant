# session_manager.py

## 所在层次
Agent 层 `src/agents/`

## 主体功能
JSONL 会话持久化管理器，支持跨进程会话恢复和自动压缩（compact）。
数据存储在 `data/sessions/{session_id}.jsonl` 和 `.meta.json`。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `SessionManager.save_turn(session_id, turn_id, user_input, new_messages, sources, token_usage, cost_cny)` | 追加写入本轮消息到 JSONL |
| `SessionManager.load(session_id)` | 加载历史消息，用 `covers_turn_ids` 集合跳过已压缩轮次 |
| `SessionManager.compact(session_id)` | 压缩旧轮次为摘要，追加 `type=summary` 行，重置 meta token 计数 |
| `SessionManager._compact_if_needed(session_id)` | 检查 `total_tokens` 是否超阈值，自动触发压缩 |
| `SessionManager._build_history_text(messages)` | 从 `new_messages` 提取 AI 最终回答，生成完整对话文本供摘要 |

## 调用关系
- **被调用方**：`MasterAgent`（每轮 turn end 后调用 `save_turn`）
- **依赖方**：`src/agents/prompt_loader`（加载 `session_compact_system` 压缩提示词）

## 注意事项
- `turn_id` 使用 `time.time_ns()` 纳秒时间戳，全局唯一
- 压缩时保留所有 summary 行，按时序合并注入 `SystemMessage`（不伪造 HumanMessage/AIMessage）
- `covers_turn_ids` 集合成员判断跳过旧轮，不依赖字符串顺序比较（避免顺序脆弱性）
- 压缩后重置 `meta.total_tokens` 为保留轮次之和，防止压缩后立即再次触发
