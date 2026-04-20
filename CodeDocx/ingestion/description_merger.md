# description_merger.py

## 所在层次
摄入层 `src/ingestion/`

## 主体功能
实体/关系描述合并器，在多文档命中同一实体时合并描述而非覆盖。
策略：先去重 → 短描述直接拼接 → 长描述 LLM 摘要（失败时回退到直接拼接）。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `DescriptionMerger.__init__(max_direct_chars, separator)` | 初始化合并参数（默认 2000 字符 / `\n---\n` 分隔） |
| `DescriptionMerger.merge(existing, new_desc)` | 主入口：去重 → 判断字符长度 → 直接拼接或 LLM 摘要 |

## 调用关系
- **被调用方**：`EntityExtractor`（upsert 实体/关系时调用）
- **依赖方**：`src/agents/prompt_loader.load_prompt_pair("description_merger")`、`src/infrastructure/llm_client.get_llm()`

## 注意事项
- 跨文档描述合并依赖实体自然键（Phase 2.1 完成后生效）
- `max_direct_chars=2000` 是 LLM 摘要触发阈值，可通过构造参数调整
- LLM 摘要失败时安全回退到直接拼接，不抛出异常
