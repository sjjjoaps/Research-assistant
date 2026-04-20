"""
多模态内容处理器模块

从 PDF 中提取图片、表格，调用 LLM 生成可检索的文本描述，
供后续切块、向量化和图谱写入使用。

借鉴 RAG-Anything 的 processor 分层架构，但不依赖其底层框架：
- ModalContent：统一的多模态内容数据结构
- BaseModalProcessor：处理器基类，定义 process() 接口
- ImageProcessor：图片描述生成
- TableProcessor：表格内容描述生成

Phase 3.1 实现范围：
- 仅处理 PDF 文档（DOCX/TXT 暂不支持图片/表格提取）
- 图片：base64 编码后调用视觉模型生成描述
- 表格：将表格文本内容发送给 LLM 生成结构化描述
- 无法处理时降级为空描述，不中断入库流程

降级边界：
- LLM 初始化失败（API_KEY 未配置、模型不可用）：process() 返回空字符串
- 单张图片/表格处理失败：跳过该项，继续处理其余内容
- pdfplumber 未安装：打印警告，跳过表格提取，不中断图片提取
"""
from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass


from src.agents.prompt_loader import load_prompt_pair

logger = logging.getLogger(__name__)

_image_sys, _image_human = load_prompt_pair("modal_image")
_table_sys, _table_human = load_prompt_pair("modal_table")


@dataclass
class ModalContent:
    """多模态内容的统一数据结构。

    Attributes:
        content_type: 内容类型，"image" / "table"
        raw_content: 原始内容（图片为 base64 字节串，表格为文本）
        processed_text: LLM 生成的可检索文本描述
        page_number: 所在页码（0-based）
        position_hint: 位置提示，如 "page 3, image 2"
        caption: 图表标题（如能提取）
    """
    content_type: str
    raw_content: str
    processed_text: str
    page_number: int
    position_hint: str
    caption: str = ""


class BaseModalProcessor(ABC):
    """多模态处理器基类。"""

    @abstractmethod
    def process(self, raw_content: str, page_number: int, position_hint: str, caption: str = "") -> str:
        """将原始内容转换为可检索文本描述。

        Args:
            raw_content: 原始内容（图片 base64 或表格文本）
            page_number: 所在页码
            position_hint: 位置提示
            caption: 可选标题

        Returns:
            LLM 生成的文本描述；失败时返回空字符串。
        """


class ImageProcessor(BaseModalProcessor):
    """图片处理器：调用视觉模型生成图片描述。

    LLM 采用懒初始化：首次调用 process() 时才创建，初始化失败时降级返回空描述。
    使用 VISION_MODEL_NAME 配置的视觉模型（如 qwen-vl-plus）。
    """

    _SYSTEM_PROMPT = _image_sys
    _HUMAN_PROMPT = _image_human

    def __init__(self) -> None:
        self._llm = None  # 懒初始化，首次 process() 时创建

    def _get_llm(self):
        if self._llm is None:
            from src.infrastructure.llm_client import get_vision_llm
            self._llm = get_vision_llm(temperature=0.0)
        return self._llm

    def process(self, raw_content: str, page_number: int, position_hint: str, caption: str = "") -> str:
        """用视觉模型描述图片内容。

        Returns:
            图片的文本描述；LLM 不可用或图片无法处理时返回空字符串。
        """
        try:
            llm = self._get_llm()
            caption_hint = f"图片标题：{caption}\n" if caption else ""
            prompt = self._HUMAN_PROMPT.format(
                caption_hint=caption_hint,
                position_hint=position_hint,
                page_number=page_number + 1,
            )
            messages = [{"role": "user", "content": [
                {"type": "text", "text": f"{self._SYSTEM_PROMPT}\n\n{prompt}"},
                {"type": "image_url", "image_url": {"url": raw_content}},
            ]}]
            resp = llm.invoke(messages)
            return str(resp.get("content") or "").strip()
        except Exception as e:
            logger.debug("图片描述生成失败（%s）: %s", position_hint, e)
            return ""


class TableProcessor(BaseModalProcessor):
    """表格处理器：将表格文本内容发送给 LLM 生成结构化描述。

    LLM 采用懒初始化：首次调用 process() 时才创建，初始化失败时降级返回空描述。
    """

    _SYSTEM_PROMPT = _table_sys
    _HUMAN_PROMPT = _table_human

    def __init__(self) -> None:
        self._llm = None  # 懒初始化，首次 process() 时创建

    def _get_llm(self):
        if self._llm is None:
            from src.infrastructure.llm_client import get_native_llm
            self._llm = get_native_llm(temperature=0.0)
        return self._llm

    def process(self, raw_content: str, page_number: int, position_hint: str, caption: str = "") -> str:
        """用 LLM 描述表格内容。

        Returns:
            表格的文本描述；LLM 不可用时返回空字符串。
        """
        try:
            llm = self._get_llm()
            caption_hint = f"表格标题：{caption}\n" if caption else ""
            content = (
                f"{self._SYSTEM_PROMPT}\n\n"
                + self._HUMAN_PROMPT.format(
                    caption_hint=caption_hint,
                    position_hint=position_hint,
                    page_number=page_number + 1,
                    raw_content=raw_content,
                )
            )
            resp = llm.invoke([{"role": "user", "content": content}])
            return str(resp.get("content") or "").strip()
        except Exception as e:
            logger.debug("表格描述生成失败（%s）: %s", position_hint, e)
            return ""


def extract_modal_contents_from_pdf(pdf_path: str, max_images: int = 20, max_tables: int = 20) -> list[ModalContent]:
    """从 PDF 文件中提取图片和表格，生成 ModalContent 列表。

    使用 PyMuPDF 提取图片，使用 pdfplumber 提取表格。
    结果按页码顺序排列（同页内先图片后表格）。
    两者均有数量上限，防止超大文档消耗过多 LLM 调用。

    Args:
        pdf_path: PDF 文件路径。
        max_images: 最多处理的图片数量，默认 20。
        max_tables: 最多处理的表格数量，默认 20。

    Returns:
        ModalContent 列表，按页码顺序排列（同页内先图片后表格）。
    """
    import fitz

    image_processor = ImageProcessor()
    table_processor = TableProcessor()

    # 检查 pdfplumber 是否可用
    try:
        import pdfplumber
        _pdfplumber_available = True
    except ImportError:
        logger.warning(
            "pdfplumber 未安装，表格提取将被跳过。"
            "如需表格提取，请运行: pip install pdfplumber>=0.10.0"
        )
        _pdfplumber_available = False

    # 预先收集各页的表格文本（避免重复打开 PDF）
    page_tables: dict[int, list[str]] = {}
    if _pdfplumber_available:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables() or []
                    texts = []
                    for table in tables:
                        if not table:
                            continue
                        rows = ["\t".join(cell or "" for cell in row) for row in table]
                        table_text = "\n".join(rows).strip()
                        if table_text:
                            texts.append(table_text)
                    if texts:
                        page_tables[page_num] = texts
        except Exception as e:
            logger.warning("pdfplumber 提取表格失败: %s", e)

    results: list[ModalContent] = []
    image_count = 0
    table_count = 0

    # 按页遍历，同页内先图片后表格，保证结果按页码顺序排列
    try:
        with fitz.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf):
                # 图片
                if image_count < max_images:
                    image_list = page.get_images(full=True)
                    for img_index, img_info in enumerate(image_list):
                        if image_count >= max_images:
                            break
                        xref = img_info[0]
                        try:
                            base_image = pdf.extract_image(xref)
                            img_bytes = base_image["image"]
                            img_ext = base_image.get("ext", "png")
                            b64 = base64.b64encode(img_bytes).decode("utf-8")
                            data_uri = f"data:image/{img_ext};base64,{b64}"
                            position_hint = f"page {page_num + 1}, image {img_index + 1}"
                            description = image_processor.process(
                                raw_content=data_uri,
                                page_number=page_num,
                                position_hint=position_hint,
                            )
                            results.append(ModalContent(
                                content_type="image",
                                raw_content=data_uri,
                                processed_text=description,
                                page_number=page_num,
                                position_hint=position_hint,
                            ))
                            image_count += 1
                        except Exception as e:
                            logger.debug("图片提取失败（page %d, img %d）: %s", page_num + 1, img_index + 1, e)
                            continue

                # 表格（同页）
                if table_count < max_tables and page_num in page_tables:
                    for tbl_index, table_text in enumerate(page_tables[page_num]):
                        if table_count >= max_tables:
                            break
                        position_hint = f"page {page_num + 1}, table {tbl_index + 1}"
                        description = table_processor.process(
                            raw_content=table_text,
                            page_number=page_num,
                            position_hint=position_hint,
                        )
                        results.append(ModalContent(
                            content_type="table",
                            raw_content=table_text,
                            processed_text=description,
                            page_number=page_num,
                            position_hint=position_hint,
                        ))
                        table_count += 1
    except Exception as e:
        logger.warning("PDF 多模态提取失败: %s", e)

    return results
