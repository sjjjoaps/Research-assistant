"""
文本切块模块
基于 LangChain RecursiveCharacterTextSplitter 对解析后的文档进行切块

Phase 2.3 变更：
- chunk() 新增可选 doc_id 参数
- 传入 doc_id 时使用内容哈希生成稳定 chunk_id，支持增量更新
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
        """切块文档。

        Args:
            document: 已解析的文档对象。
            doc_id: 可选，文档唯一标识。传入时使用内容哈希生成稳定 chunk_id，
                    支持增量更新和精确删除；不传时退回到路径+索引格式（向后兼容）。
        """
        if not document.raw_text.strip():
            return []

        parts = self.splitter.split_text(document.raw_text)
        chunks: list[TextChunk] = []
        occurrence_counter: dict[str, int] = {}

        for index, content in enumerate(parts):
            if doc_id:
                content_hash = compute_chunk_content_hash(content)
                occ = occurrence_counter.get(content_hash, 0)
                chunk_id = generate_chunk_id(doc_id, content_hash, occ)
                occurrence_counter[content_hash] = occ + 1
            else:
                chunk_id = f"{document.file_path}::chunk::{index}"

            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    content=content,
                    chunk_index=index,
                    metadata={
                        "file_path": document.file_path,
                        "chunk_index": index,
                        "doc_id": doc_id or "",
                    },
                )
            )

        return chunks
