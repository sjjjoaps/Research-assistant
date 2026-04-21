# section_recognizer.py

## 所在层次
摄入层 `src/ingestion/`

## 主体功能
从 PDF 页面文本识别学术论文章节类型，为 chunk 提供 `section_type` 元数据。
两阶段识别策略：规则正则 → LLM 兜底，外加前向填充。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `SECTION_TYPES` (frozenset) | 支持的 11 种章节类型（abstract/introduction/method 等） |
| `SectionRecognizer.recognize(pages)` | 主入口：规则匹配 → LLM 兜底 → 前向填充，返回每页 section_type 列表 |
| `SectionRecognizer._rule_match(page_text)` | 检查页面前 10 行中的标题关键词（含全大写标题） |
| `SectionRecognizer._llm_classify(page_text)` | 发送页面前 200 字符给 LLM，返回 section_type 字符串 |

## 调用关系
- **被调用方**：`DocumentParser`（`enable_section_recognition=True` 时调用）
- **依赖方**：`src/infrastructure/llm_client.get_llm()`（LLM 兜底时懒加载）

## 注意事项
- 规则优先：`_HEADING_PATTERNS` 列表包含 10 类章节的正则模式（含编号前缀，如 `2. Method`）
- 前向填充：已知 section_type 填充相邻未识别页面，避免长章节内页全部返回 "unknown"
- LLM 初始化失败时跳过兜底，保留 "unknown"；单页 LLM 失败不中断整体识别
