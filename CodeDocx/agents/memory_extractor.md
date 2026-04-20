# memory_extractor.py

## 所在层次
Agent 层 `src/agents/`

## 主体功能
turn-end 后台记忆提取器（Phase 9-5-3）。
分析单轮对话，通过 LLM 判断是否有值得归档的长期记忆，若有则写入 `data/memory/`。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `MemoryExtractor.extract_and_save(user_input, ai_response, session_id)` | 主入口：LLM 判断 → 三层去重 → 写入长期记忆 |
| `MemoryExtractor._should_save(user_input, ai_response)` | 调用 LLM 返回 `{should_save, title, body, type}` JSON |
| `MemoryExtractor._is_duplicate(title, body, memory_type)` | 三层去重：slug 碰撞 / title 匹配 / description 词重叠率 > 0.7 |

## 调用关系
- **被调用方**：`MasterAgent._fire_memory_extractor()`（后台线程，仅在主模型未主动写记忆时触发）
- **依赖方**：`src/infrastructure/long_term_memory.LongTermMemory`、`src/agents/prompt_loader.load_system_prompt("memory_extractor")`

## 注意事项
- 全程 `try/except` 包裹，失败只记 WARNING，不影响主流程
- 只允许写入 `data/memory/` 目录，不允许修改业务文件
- LLM 调用 `temperature=0.0` 保证确定性
- 词重叠去重阈值 `_SIMILARITY_THRESHOLD = 0.7`
