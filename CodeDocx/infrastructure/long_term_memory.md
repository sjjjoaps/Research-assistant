# long_term_memory.py

## 所在层次
基础设施层 `src/infrastructure/`

## 主体功能
文件系统长期记忆，双重职责：
1. **用户偏好记忆**：将用户偏好/规则存储为 Markdown 文件（`data/memory/{slug}.md`）和 `MEMORY.md` 索引
2. **检索策略记忆**：按问题类型记录最优检索模式，供 `auto` 模式路由使用

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `classify_question_type(query)` | 按关键词特征分类问题类型（temporal/mixed/specific/abstract/general） |
| `LongTermMemory.save_memory(title, body, memory_type)` | 写入用户记忆文件 + 更新 MEMORY.md 索引 |
| `LongTermMemory.list_memories()` | 列出所有记忆条目（从 MEMORY.md 索引读取） |
| `LongTermMemory.get_memory(slug)` | 读取单条记忆内容 |
| `LongTermMemory.record_strategy_result(query_type, mode, success)` | 记录某问题类型下某检索模式的成功/失败 |
| `LongTermMemory.get_best_mode(query_type)` | 查询历史最优检索模式（至少 5 个样本才给出建议） |

## 调用关系
- **被调用方**：`tool_registry.save_user_memory`（写入记忆）、`MasterAgent._fire_memory_extractor()`、`tool_registry._do_retrieve()`（auto 路由）
- **依赖方**：文件系统（`data/memory/` 目录）、`classify_question_type`、`KeywordExtractor`（分类依赖）

## 注意事项
- 记忆文件格式：YAML frontmatter（`type / created_at / session_id`）+ Markdown body
- `_MIN_SAMPLES = 5`：样本不足时 `get_best_mode()` 返回 `None`，回退到启发式路由
- `temporal` 类查询跳过 LTM 路由（时效性最强，启发式更可靠）
- slug 由 `_slugify(title)` 生成，仅含小写字母/数字/下划线/连字符
