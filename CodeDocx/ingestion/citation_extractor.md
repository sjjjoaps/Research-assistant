# citation_extractor.py

## 所在层次
摄入层 `src/ingestion/`

## 主体功能
从学术论文全文中定位参考文献章节，解析单条引用，生成结构化 `CitationRecord`。
两阶段识别策略：规则优先 → LLM 兜底。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `CitationRecord` (dataclass) | 引用结构：`raw_text / title / authors / year / doi / venue` |
| `CitationExtractor.extract(raw_text)` | 主入口：定位参考文献章节 → 切分单条引用 → 规则解析 → LLM 兜底 |
| `CitationExtractor._locate_reference_section(text)` | 正则定位 References / Bibliography / 参考文献 章节 |
| `CitationExtractor._split_references(section_text)` | 按编号格式 `[1]` 或 `1.` 切分单条引用 |
| `CitationExtractor._rule_parse(raw)` | 正则提取 DOI / 年份 / 引号内标题 |

## 调用关系
- **被调用方**：`src/workflows/ingestion_pipeline.py`（入库流水线引用提取阶段）
- **依赖方**：`src/agents/prompt_loader.load_system_prompt("citation_extractor_fallback")`、`src/infrastructure/llm_client`（LLM 兜底时懒加载）

## 注意事项
- 单文档最多处理 `_MAX_REFERENCES = 100` 条引用，防止超大文献列表消耗过多 LLM 调用
- 单条引用 < 10 字符直接跳过
- LLM 兜底仅对规则未提取到标题的条目触发，不全量调用
- 任何降级点均不中断整体提取流程
