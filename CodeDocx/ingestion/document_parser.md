# document_parser.py

## 所在层次
摄入层 `src/ingestion/`

## 主体功能
文档解析模块，支持 PDF / DOCX / TXT 三种格式，统一返回 `ParsedDocument` 对象。
可选启用多模态内容提取（图片/表格）和章节结构识别。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `ParsedDocument` (dataclass) | 解析结果：`file_path / raw_text / pages / modal_contents / page_sections / page_section_titles` |
| `DocumentParser.__init__(enable_modal_extraction, enable_section_recognition)` | 初始化解析器，两个开关默认均为 False |
| `DocumentParser.parse(file_path)` | 主入口，按扩展名路由到对应解析方法 |

## 调用关系
- **被调用方**：`src/workflows/ingestion_pipeline.py`（入库流水线第一阶段）
- **依赖方**：`fitz`（PyMuPDF，PDF 解析）、`docx`（DOCX 解析）、`SectionRecognizer`（可选）、`modal_processors`（可选）

## 注意事项
- `page_sections` / `page_section_titles` 与 `pages` 列表平行（索引对应同一页）
- `enable_modal_extraction` 仅对 PDF 生效，启用后调用 LLM 生成图片/表格描述，增加入库耗时
- `enable_section_recognition` 仅对 PDF 生效，启用后识别每页章节类型
- 不支持的扩展名会抛出 `ValueError`
