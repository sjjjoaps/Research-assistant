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
"""
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.document_parser import ParsedDocument
from src.storage.chunk_tracker import compute_chunk_content_hash, generate_chunk_id


@dataclass
class TextChunk:
    chunk_id: str
    content: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ". ", " ", ""],
        )

    def chunk(self, document: ParsedDocument, doc_id: str | None = None) -> list[TextChunk]:
        """切块文档，同时将多模态内容转为特殊 chunk。

        Args:
            document: 已解析的文档对象。
            doc_id: 可选，文档唯一标识。传入时使用内容哈希生成稳定 chunk_id，
                    支持增量更新和精确删除；不传时退回到路径+索引格式（向后兼容）。

        Returns:
            TextChunk 列表，包含普通文本 chunk 和多模态 chunk（如有）。
            多模态 chunk 的 metadata["content_type"] 为 "image" 或 "table"。
        """
        chunks: list[TextChunk] = []
        occurrence_counter: dict[str, int] = {}
        global_index = 0

        # ── 普通文本 chunk ────────────────────────────────────────────────────
        if document.raw_text.strip():
            parts = self.splitter.split_text(document.raw_text)
            for content in parts:
                if doc_id:
                    content_hash = compute_chunk_content_hash(content)
                    occ = occurrence_counter.get(content_hash, 0)
                    chunk_id = generate_chunk_id(doc_id, content_hash, occ)
                    occurrence_counter[content_hash] = occ + 1
                else:
                    chunk_id = f"{document.file_path}::chunk::{global_index}"

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
                    },
                )
            )
            global_index += 1

        return chunks
