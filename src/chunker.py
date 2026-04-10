"""
文本切块模块
基于 LangChain RecursiveCharacterTextSplitter 对解析后的文档进行切块
"""
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.document_parser import ParsedDocument


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

    def chunk(self, document: ParsedDocument) -> list[TextChunk]:
        if not document.raw_text.strip():
            return []

        parts = self.splitter.split_text(document.raw_text)
        chunks: list[TextChunk] = []

        for index, content in enumerate(parts):
            chunks.append(
                TextChunk(
                    chunk_id=f"{document.file_path}::chunk::{index}",
                    content=content,
                    chunk_index=index,
                    metadata={
                        "file_path": document.file_path,
                        "chunk_index": index,
                    },
                )
            )

        return chunks
