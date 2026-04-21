"""
文本切块模块
基于 LangChain RecursiveCharacterTextSplitter 对解析后的文档进行切块

Phase 2.3 变更：
- chunk() 新增可选 doc_id 参数
- 传入 doc_id 时使用内容哈希生成稳定 chunk_id，支持增量更新

Phase 3.1 变更：
- chunk() 将 ParsedDocument.modal_contents 中的图片/表格描述转为特殊 TextChunk
- 多模态 chunk 的 metadata 包含 content_type 字段（"image" / "table"），可区分类型
- processed_text 为空的 ModalContent 会被跳过（LLM 描述失败时不产生无效 chunk）

Phase 3.2 变更：
- chunk() 读取 ParsedDocument.page_sections / page_section_titles，为文本 chunk 分配
  section_type 和 section_title 元数据
- 通过页面偏移量映射将 chunk 内容定位到所属页面，再取该页的章节类型
- 多模态 chunk 的 section_type 固定为 "unknown"（已有 page_number 可供后续扩展）
"""
import bisect
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.document_parser import ParsedDocument
from src.storage.chunk_tracker import compute_chunk_content_hash, generate_chunk_id


@dataclass
class TextChunk:
    chunk_id: str
    content: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _find_page_for_offset(offset: int, page_offsets: list[int]) -> int:
    """二分查找：返回包含 offset 的页面索引。"""
    idx = bisect.bisect_right(page_offsets, offset) - 1
    return max(0, idx)


class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ". ", " ", ""],
        )

    def _build_page_offset_map(self, raw_text: str, pages: list[str]) -> list[int]:
        """返回每页在 raw_text 中的起始偏移量列表。"""
        offsets: list[int] = []
        pos = 0
        for page in pages:
            idx = raw_text.find(page, pos)
            offsets.append(idx if idx != -1 else pos)
            pos = offsets[-1] + len(page)
        return offsets

    def chunk(self, document: ParsedDocument, doc_id: str | None = None) -> list[TextChunk]:
        """切块文档，同时将多模态内容转为特殊 chunk。

        Args:
            document: 已解析的文档对象。
            doc_id: 可选，文档唯一标识。传入时使用内容哈希生成稳定 chunk_id，
                    支持增量更新和精确删除；不传时退回到路径+索引格式（向后兼容）。

        Returns:
            TextChunk 列表，包含普通文本 chunk 和多模态 chunk（如有）。
            多模态 chunk 的 metadata["content_type"] 为 "image" 或 "table"。
            文本 chunk 的 metadata["section_type"] 由 page_sections 决定（若可用）。
        """
        chunks: list[TextChunk] = []
        occurrence_counter: dict[str, int] = {}
        global_index = 0

        # ── 章节映射（若 page_sections 已填充）────────────────────────────────
        _has_sections = bool(document.page_sections)
        _page_offsets: list[int] = []
        if _has_sections and document.pages:
            _page_offsets = self._build_page_offset_map(document.raw_text, document.pages)

        # ── 普通文本 chunk ────────────────────────────────────────────────────
        if document.raw_text.strip():
            parts = self.splitter.split_text(document.raw_text)
            # 维护递增搜索游标，避免重复内容（页眉/页脚/overlap）被错误映射到首次出现位置
            _search_cursor = 0
            for content in parts:
                if doc_id:
                    content_hash = compute_chunk_content_hash(content)
                    occ = occurrence_counter.get(content_hash, 0)
                    chunk_id = generate_chunk_id(doc_id, content_hash, occ)
                    occurrence_counter[content_hash] = occ + 1
                else:
                    chunk_id = f"{document.file_path}::chunk::{global_index}"

                section_type = "unknown"
                section_title = ""
                if _has_sections and _page_offsets:
                    chunk_offset = document.raw_text.find(content, _search_cursor)
                    if chunk_offset != -1:
                        page_idx = _find_page_for_offset(chunk_offset, _page_offsets)
                        if 0 <= page_idx < len(document.page_sections):
                            section_type = document.page_sections[page_idx]
                            if (document.page_section_titles
                                    and page_idx < len(document.page_section_titles)):
                                section_title = document.page_section_titles[page_idx]
                        # 游标推进到本 chunk 末尾（减去 overlap 长度，允许下一 chunk 从重叠区开始）
                        _search_cursor = max(_search_cursor, chunk_offset + len(content) - self.splitter._chunk_overlap)

                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        content=content,
                        chunk_index=global_index,
                        metadata={
                            "file_path": document.file_path,
                            "chunk_index": global_index,
                            "doc_id": doc_id or "",
                            "content_type": "text",
                            "section_type": section_type,
                            "section_title": section_title,
                        },
                    )
                )
                global_index += 1

        # ── 多模态 chunk ──────────────────────────────────────────────────────
        for modal in document.modal_contents:
            if not modal.processed_text.strip():
                # LLM 描述失败时跳过，不产生无效 chunk
                continue

            content = (
                f"[{modal.content_type.upper()}] {modal.position_hint}\n"
                f"{modal.processed_text}"
            )
            if doc_id:
                content_hash = compute_chunk_content_hash(content)
                occ = occurrence_counter.get(content_hash, 0)
                chunk_id = generate_chunk_id(doc_id, content_hash, occ)
                occurrence_counter[content_hash] = occ + 1
            else:
                chunk_id = f"{document.file_path}::modal::{global_index}"

            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    content=content,
                    chunk_index=global_index,
                    metadata={
                        "file_path": document.file_path,
                        "chunk_index": global_index,
                        "doc_id": doc_id or "",
                        "content_type": modal.content_type,
                        "page_number": modal.page_number,
                        "position_hint": modal.position_hint,
                        "section_type": "unknown",
                        "section_title": "",
                    },
                )
            )
            global_index += 1

        return chunks
