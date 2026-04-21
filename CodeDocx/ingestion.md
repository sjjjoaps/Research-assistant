# 入库层（ingestion）

目录：`src/ingestion/`

## 模块清单

| 文件 | 主要类/函数 | 职责 |
|---|---|---|
| `document_parser.py` | `DocumentParser`, `ParsedDocument` | PDF/DOCX/TXT 解析，支持多模态提取（modal_max_images/tables 参数） |
| `chunker.py` | `DocumentChunker`, `TextChunk` | 固定大小文本切块 |
| `section_chunker.py` | `SectionChunker` | 章节感知切块（优先保留结构） |
| `section_recognizer.py` | `SectionRecognizer` | 章节类型识别（method/result/intro 等） |
| `metadata_extractor.py` | `MetadataExtractor`, `DocumentMetadata` | LLM 元数据提取（降级为正则），含 setdefault 类型守卫 |
| `entity_extractor.py` | `EntityExtractor` | LLM 实体/关系抽取，写入 GraphStore |
| `citation_extractor.py` | `CitationExtractor` | LLM 引用关系抽取，写入 Reference 节点 |
| `description_merger.py` | `DescriptionMerger` | 合并重复实体描述（多文档去重） |
| `community_detector.py` | `CommunityDetector`, `CommunityStats` | Louvain 社区检测 + LLM 摘要生成，写入 Community 节点 |
| `modal_processors.py` | `ImageProcessor`, `TableProcessor`, `extract_modal_contents_from_pdf()` | 图片/表格视觉 LLM 描述生成（使用 get_vision_llm()） |

## 多模态提取

`DocumentParser` 在 `enable_modal_extraction=True` 时调用 `extract_modal_contents_from_pdf()`：
- `ImageProcessor`：base64 图片 → `get_vision_llm()` → 文本描述
- `TableProcessor`：表格文本 → LLM → 结构化描述
- 每文档最多提取 `modal_max_images` 张图片 + `modal_max_tables` 个表格（默认各 20）
- 任何单项失败均跳过，不中断主流程

## 社区检测触发条件

`CommunityDetector.run(min_community_size)` 需满足：
1. Neo4j 中存在 Entity 节点且有 `RELATES_TO` 关系
2. Louvain 划分后各社区实体数 ≥ `min_community_size`
3. 自适应阈值（前端）：`max(2, round(entity_count / 20))`
