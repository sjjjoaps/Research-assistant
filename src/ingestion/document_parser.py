"""
文档解析模块
支持 PDF / DOCX / TXT 三种格式，并统一返回解析结果对象
采用 PyMuPDF 解析 PDF 文件，PyDocx 解析 DOCX 文件，直接读取 TXT 文件内容

Phase 3.1 变更：
- ParsedDocument 新增 modal_contents 字段，存储从 PDF 提取的图片/表格内容
- DocumentParser 新增 enable_modal_extraction 开关（默认 False，避免影响现有流程）
- PDF 解析时可选调用 extract_modal_contents_from_pdf 提取多模态内容

Phase 3.2 变更：
- ParsedDocument 新增 page_sections / page_section_titles 字段（与 pages 平行）
- DocumentParser 新增 enable_section_recognition 开关（默认 False）
- PDF 解析时可选调用 SectionRecognizer 识别每页章节类型
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import fitz
from docx import Document as DocxDocument

if TYPE_CHECKING:
    from src.ingestion.modal_processors import ModalContent


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@dataclass
class ParsedDocument:
    file_path: str
    raw_text: str
    pages: list[str]
    modal_contents: list["ModalContent"] = field(default_factory=list)
    page_sections: list[str] = field(default_factory=list)        # 与 pages 平行，每页的 section_type
    page_section_titles: list[str] = field(default_factory=list)  # 与 pages 平行，每页的章节标题


class DocumentParser:
    def __init__(
        self,
        enable_modal_extraction: bool = True,
        enable_section_recognition: bool = True,
        modal_max_images: int = 20,
        modal_max_tables: int = 20,
    ) -> None:
        self.enable_modal_extraction = enable_modal_extraction
        self.enable_section_recognition = enable_section_recognition
        self.modal_max_images = modal_max_images
        self.modal_max_tables = modal_max_tables

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

        modal_contents = []
        if self.enable_modal_extraction:
            from src.ingestion.modal_processors import extract_modal_contents_from_pdf
            modal_contents = extract_modal_contents_from_pdf(
                str(path),
                max_images=self.modal_max_images,
                max_tables=self.modal_max_tables,
            )

        page_sections: list[str] = []
        page_section_titles: list[str] = []
        if self.enable_section_recognition and pages:
            from src.ingestion.section_recognizer import SectionRecognizer
            recognizer = SectionRecognizer()
            page_sections, page_section_titles = recognizer.recognize_per_page(pages)

        return ParsedDocument(
            file_path=str(path),
            raw_text=raw_text,
            pages=pages,
            modal_contents=modal_contents,
            page_sections=page_sections,
            page_section_titles=page_section_titles,
        )

    def _parse_docx(self, path: Path) -> ParsedDocument:
        doc = DocxDocument(path)
        paragraphs = [paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()]
        raw_text = "\n".join(paragraphs)
        return ParsedDocument(file_path=str(path), raw_text=raw_text, pages=[raw_text] if raw_text else [])

    def _parse_txt(self, path: Path) -> ParsedDocument:
        raw_text = path.read_text(encoding="utf-8").strip()
        return ParsedDocument(file_path=str(path), raw_text=raw_text, pages=[raw_text] if raw_text else [])
