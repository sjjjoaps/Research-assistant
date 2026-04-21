# section_chunker.py

## 所在层次
摄入层 `src/ingestion/`

## 主体功能
层次化语义分块器（Phase 10-1），借鉴 AgenticRAG HiChunk 思想。
按章节语义边界分块，不同章节给予差异化 chunk size，优先在段落边界切分。
与 `DocumentChunker` 有相同的公开 API，可直接替换。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `SECTION_CHUNK_SIZES` (dict) | 章节 → chunk size 配置（method/experiment 等细节章节 size 更大） |
| `chunk_by_section(text, section_type, doc_id, base_index)` | 单章节分块：优先段落边界 → 句子边界 → 硬切 |
| `SectionChunker.__init__()` | 初始化，内部持有 `DocumentChunker` 作为 fallback |
| `SectionChunker.chunk(parsed_doc, doc_id?)` | 主入口：相邻同章节页面合并 → 按章节分块；无章节信息时 fallback 到 `DocumentChunker` |

## 调用关系
- **被调用方**：`src/workflows/ingestion_pipeline.py`（替换 `DocumentChunker`）
- **依赖方**：`DocumentChunker`（fallback）、`TextChunk`、`src/storage/chunk_tracker`

## 注意事项
- 相邻页面合并条件：`section_type` 相同 且 `section_title` 相同（或任一为空）
- `page_sections` 为空或与 `pages` 长度不一致时，自动 fallback 到固定大小分块
- overlap 以"在末尾切片"方式追加，不把 tail 当独立段落重新拼接
- 多模态 chunk 处理逻辑与 `DocumentChunker` 完全相同（直接复用）
