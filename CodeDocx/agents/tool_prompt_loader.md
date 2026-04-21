# tool_prompt_loader.py

## 所在层次
Agent 层 `src/agents/`

## 主体功能
从 `prompt/tools/{tool_name}.md` 加载工具描述文本，供 `tool_registry.py` 在注册工具时注入给 MasterAgent。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `load_tool_description(tool_name, fallback="")` | 加载指定工具的描述文本；带 `@lru_cache(maxsize=32)` 避免重复磁盘 I/O |
| `_resolve_tool_prompt_path(tool_name)` | 依次尝试相对路径和绝对路径，返回第一个存在的 `.md` 文件路径 |

## 调用关系
- **被调用方**：`src/agents/tool_registry.py`（`build_tool_registry()` 中为每个工具注入描述）
- **依赖方**：`prompt/tools/*.md` 文件

## 注意事项
- 文件不存在时降级策略：优先使用传入的 `fallback`（取前 200 字），否则返回 `"Tool: {tool_name}"`，保证启动不中断
- 路径解析顺序：先尝试相对路径 `prompt/tools/`，再尝试绝对路径（`__file__` 推导），适配不同工作目录启动场景
- `lru_cache` 缓存以 `(tool_name, fallback)` 为 key；若需热重载工具描述，需先调用 `load_tool_description.cache_clear()`
