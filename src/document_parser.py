"""
文档解析模块
支持 PDF / DOCX / TXT 三种格式，并统一返回解析结果对象
采用 PyMuPDF 解析 PDF 文件，PyDocx 解析 DOCX 文件，直接读取 TXT 文件内容
"""
from dataclasses import dataclass
from pathlib import Path

import fitz
from docx import Document as DocxDocument


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@dataclass
class ParsedDocument:
    file_path: str
    raw_text: str
    pages: list[str]


class DocumentParser:
    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件类型: {suffix}")

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        if suffix == ".pdf":
            return self._parse_pdf(path)
        if suffix == ".docx":
            return self._parse_docx(path)
        return self._parse_txt(path)

    def _parse_pdf(self, path: Path) -> ParsedDocument:
        pages: list[str] = []

        with fitz.open(path) as pdf:
            for page in pdf:
                text = page.get_text("text").strip()
                if text:
                    pages.append(text)

        raw_text = "\n\n".join(pages)
        return ParsedDocument(file_path=str(path), raw_text=raw_text, pages=pages)

    def _parse_docx(self, path: Path) -> ParsedDocument:
        doc = DocxDocument(path)
        paragraphs = [paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()]
        raw_text = "\n".join(paragraphs)
        return ParsedDocument(file_path=str(path), raw_text=raw_text, pages=[raw_text] if raw_text else [])

    def _parse_txt(self, path: Path) -> ParsedDocument:
        raw_text = path.read_text(encoding="utf-8").strip()
        return ParsedDocument(file_path=str(path), raw_text=raw_text, pages=[raw_text] if raw_text else [])
