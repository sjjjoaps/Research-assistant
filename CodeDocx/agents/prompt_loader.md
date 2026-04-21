# prompt_loader.py

## 所在层次
Agent 层 `src/agents/`

## 主体功能
全项目统一 Prompt 加载入口，从 `prompt/` 目录读取 `.md` 文件。
支持新合并格式（单文件）和旧拆分格式（`_system.md / _human.md`）兼容加载。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `load_system_prompt(name)` | 加载 system prompt，`@lru_cache` 缓存 |
| `load_human_template(name)` | 加载 human prompt template，`@lru_cache` 缓存 |
| `load_prompt_pair(name)` | 同时返回 `(system, human)` 元组 |
| `reload_prompt(name)` | 清除指定 name 的 lru_cache（热更新使用） |
| `load_master_agent_system_prompt()` | MasterAgent 专用加载，含结构校验 |
| `validate_system_prompt(prompt)` | 校验 MasterAgent system prompt 的必要章节 |

## 调用关系
- **被调用方**：几乎所有模块（`EntityExtractor`、`MetadataExtractor`、`KeywordExtractor` 等）
- **依赖方**：文件系统（`prompt/` 目录）

## 注意事项
- 加载优先级：`prompt/{name}.md`（新合并格式）→ `prompt/{name}_system.md`（旧格式）→ 空字符串 + WARNING
- 合并格式分隔：`## System Prompt` 章节 + `---` 分隔符 + `## Human Prompt Template` 章节
- 文件不存在时不抛异常，仅打印 WARNING，保证服务正常启动
- `@lru_cache` 保证同名文件仅读取一次（除非调用 `reload_prompt()`）
