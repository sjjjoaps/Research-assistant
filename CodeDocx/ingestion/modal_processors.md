# modal_processors.py

## 所在层次
入库层 `src/ingestion/`

## 主体功能
多模态内容处理器，从 PDF 中提取图片和表格，调用视觉 LLM 生成可检索的文本描述。
借鉴 RAG-Anything 的 processor 分层架构，不依赖其底层框架。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `ModalContent` (dataclass) | 多模态内容结构：`content_type / raw_content / processed_text / page_number / position_hint` |
| `BaseModalProcessor` (ABC) | 处理器基类，定义 `process(raw_content) → str` 接口 |
| `ImageProcessor.process(raw_content)` | base64 图片 → `get_vision_llm()` → 文本描述 |
| `TableProcessor.process(raw_content)` | 表格文本 → LLM → 结构化描述 |
| `extract_modal_contents_from_pdf(file_path, max_images, max_tables)` | 从 PDF 提取图片和表格，调用对应处理器生成描述 |

## 配置参数

- `max_images`：每文档最多处理的图片数（默认 20，来自 `settings.modal_max_images`）
- `max_tables`：每文档最多处理的表格数（默认 20，来自 `settings.modal_max_tables`）
- 视觉 LLM：`get_vision_llm()`，使用 `settings.vision_model_name`（默认 `qwen-vl-plus`）

## 调用关系
- **被调用方**：`DocumentParser`（`enable_modal_extraction=True` 时调用）
- **依赖方**：`fitz`（PyMuPDF，图片提取）、`pdfplumber`（表格提取）、`get_vision_llm()`、`load_prompt_pair("modal_image/modal_table")`

## 注意事项
- 仅处理 PDF 文档，DOCX/TXT 暂不支持
- `pdfplumber` 未安装时打印警告并跳过表格提取，不中断图片提取
- 任何单项处理失败均跳过该项，`processed_text` 为空的 `ModalContent` 会被 chunker 过滤
- LLM 初始化失败时 `process()` 返回空字符串（安全降级）
- 可用 `demo1.py` 单独测试多模态提取效果
