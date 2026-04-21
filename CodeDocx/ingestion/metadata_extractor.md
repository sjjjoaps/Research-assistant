# metadata_extractor.py

## 所在层次
摄入层 `src/ingestion/`

## 主体功能
从文档首页文本中提取结构化元数据（标题、作者、摘要、关键词、发表年份）。
优先使用 LLM 结构化输出，失败时降级为正则规则提取。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `DocumentMetadata` (Pydantic BaseModel) | 元数据结构：`title / authors / institution / year / abstract / keywords` |
| `_regex_extract(text)` | 正则降级提取：年份 / Abstract 段落 / Keywords 段落 |
| `MetadataExtractor.extract(parsed_doc)` | 主入口：LLM 结构化提取，失败时调用 `_regex_extract` 兜底 |

## 调用关系
- **被调用方**：`src/workflows/ingestion_pipeline.py`（入库流水线第三阶段）
- **依赖方**：`src/agents/prompt_loader.load_prompt_pair("metadata_extractor")`、`src/infrastructure/llm_client.get_llm()`

## 注意事项
- LLM 调用失败时所有字段降级为 `_regex_extract` 结果（`title` 填 `"Unknown"`）
- 摘要匹配使用 `re.DOTALL`，支持多段落 Abstract
- `year` 字段限制在 2000~2029 范围内防止噪声匹配
