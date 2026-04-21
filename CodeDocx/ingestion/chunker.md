# chunker.py

## 所在层次
摄入层 `src/ingestion/`

## 主体功能
文本切块模块，基于 LangChain `RecursiveCharacterTextSplitter` 对解析后的文档进行切块。
同时将多模态内容（图片/表格描述）转为特殊 `TextChunk`，并为每个 chunk 分配 `section_type`。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `TextChunk` (dataclass) | 切块结果：`chunk_id / content / chunk_index / metadata` |
| `_find_page_for_offset(offset, page_offsets)` | 二分查找：根据字符偏移量定位所属页面索引 |
| `DocumentChunker.__init__(chunk_size, chunk_overlap)` | 初始化 splitter（默认 1000/200） |
| `DocumentChunker.chunk(parsed_doc, doc_id?)` | 主入口：文本切块 + 多模态 chunk + section_type 分配 |

## 调用关系
- **被调用方**：`src/workflows/ingestion_pipeline.py`（入库流水线第二阶段）
- **依赖方**：`langchain_text_splitters.RecursiveCharacterTextSplitter`、`src/storage/chunk_tracker`（生成稳定 chunk_id）、`ParsedDocument`

## 注意事项
- 传入 `doc_id` 时使用内容哈希生成稳定 `chunk_id`，支持增量更新（相同内容不重复入库）
- 多模态 chunk 的 `section_type` 固定为 `"unknown"`，`content_type` 字段区分 `"image"/"table"`
- `processed_text` 为空的 `ModalContent` 会被跳过（LLM 描述失败时不产生无效 chunk）
- 分隔符优先级：`\n\n` > `\n` > `。` > `. ` > ` ` > `""`（中英文混合友好）
