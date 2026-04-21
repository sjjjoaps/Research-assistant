"""
层次化语义分块（Phase 10-1）

借鉴 AgenticRAG HiChunk 思想：按章节语义边界分块，替代固定字符数的硬切分。
核心思路：
  - 不同章节内容密度不同（method 比 abstract 细节更多），给予差异化 chunk size
  - 优先在段落边界（双换行）切分，保持自然语义单元完整
  - 只有当段落超出 max_size 时，才在句子边界二次切分；句子也超长时硬切
  - 与现有 DocumentChunker 完全兼容：相同的输入/输出格式（ParsedDocument → list[TextChunk]）
  - 向后兼容：未识别 section_type 的文档自动 fallback 到 DocumentChunker 的固定大小分块

设计要点：
  [1] SECTION_CHUNK_SIZES：章节 → chunk size 的配置字典（模块级常量，不动态覆盖）
  [2] chunk_by_section()：单章节分块函数，优先段落边界，超长时才强制切分；
      overlap 以"生成 chunk 后在末尾切片"方式追加，不把 tail 当独立段落重新拼接
  [3] SectionChunker：完整的文档切块器，与 DocumentChunker 有相同的公开 API
  [4] 多模态 chunk 的处理与 DocumentChunker 完全相同（直接复用逻辑）
  [5] 若文档未启用章节识别（page_sections 为空），或 pages / page_sections 长度不一致，
      退回 DocumentChunker 的固定大小分块
  [6] 相邻页面合并条件：section_type 相同 且 section_title 相同（或任一为空）

被 ingestion_pipeline.py 引用：
  from src.ingestion.section_chunker import SectionChunker
  self.chunker = SectionChunker()   # 替换 DocumentChunker()
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from src.ingestion.document_parser import ParsedDocument
from src.infrastructure.config import settings
from src.storage.chunk_tracker import compute_chunk_content_hash, generate_chunk_id

# 复用 DocumentChunker 的 TextChunk 结构，保持一致性
from src.ingestion.chunker import TextChunk

logger = logging.getLogger(__name__)

# ── 章节 chunk size 配置（字符数） ──────────────────────────────────────────
# 依据各章节内容密度调整：method/experiment 细节多允许更大 chunk，
# abstract/conclusion 精炼信息适合更小 chunk（保持完整性）。
# 当前为模块级常量；如需运行时配置，可在 IngestionPipeline 初始化时
# 通过 SectionChunker(size_overrides={...}) 传入覆盖字典。
SECTION_CHUNK_SIZES: dict[str, int] = {
    "abstract":     300,
    "introduction": 400,
    "related_work": 500,
    "method":       700,   # 方法：细节多，允许更大 chunk
    "experiment":   600,   # 实验：结果/数据密集
    "result":       600,
    "discussion":   500,
    "conclusion":   300,
    "reference":    400,
    "appendix":     500,
    "unknown":      500,   # 默认 fallback
}

# 段落内按句子边界二次切分时识别的句末标点
_SENTENCE_END_RE = re.compile(r"(?<=[。！？.!?])\s*")


def chunk_by_section(
    section_content: str,
    section_type:    str,
    max_size:        Optional[int] = None,
    overlap:         int | None = None,
) -> list[str]:
    """
    按段落边界优先分割，在超出 max_size 时才强制切分（[2]）。

    比固定字符数分块更自然，不会在段落中间截断内容。

    Args:
        section_content: 要分块的文本内容
        section_type:    章节类型，用于从 SECTION_CHUNK_SIZES 查找默认 max_size
        max_size:        最大 chunk 字符数；None 时从 SECTION_CHUNK_SIZES 查找
        overlap:         相邻 chunk 的字符重叠数；None 时从 settings.section_chunk_overlap 读取

    Returns:
        分块后的字符串列表，每块正文部分不超过 max_size 字符。

    切分策略（优先级从高到低）：
        1. 段落边界（双换行）：保持段落完整性，首选切分点
        2. 句子边界（。！？.!? 后）：段落超长时的二次切分点
        3. 硬切（每 max_size 字符一刀）：单句也超长时的兜底，保证 chunk 数量有界
    """
    if max_size is None:
        max_size = SECTION_CHUNK_SIZES.get(section_type, SECTION_CHUNK_SIZES["unknown"])
    if overlap is None:
        overlap = settings.section_chunk_overlap

    # 拆分段落（双换行或单换行后跟空白行）
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_content) if p.strip()]
    if not paragraphs:
        return []

    # ── 第一阶段：段落边界合并 ──────────────────────────────────────────────
    # 把连续小段落合并成不超过 max_size 的组，超长段落单独处理
    raw_chunks: list[str] = []
    current_parts: list[str] = []
    current_len: int = 0

    def _flush():
        if current_parts:
            raw_chunks.append("\n\n".join(current_parts))

    for para in paragraphs:
        if len(para) > max_size:
            # 先 flush 当前积累
            _flush()
            current_parts.clear()
            current_len = 0

            # ── 第二阶段：句子边界切分超长段落 ──────────────────────────────
            sentences = [s.strip() for s in _SENTENCE_END_RE.split(para) if s.strip()]
            sent_buf: list[str] = []
            sent_len = 0
            for sent in sentences:
                # ── 第三阶段：单句超长时硬切 ─────────────────────────────────
                if len(sent) > max_size:
                    if sent_buf:
                        raw_chunks.append(" ".join(sent_buf))
                        sent_buf.clear()
                        sent_len = 0
                    # 按 max_size 硬切
                    for start in range(0, len(sent), max_size):
                        raw_chunks.append(sent[start:start + max_size])
                    continue

                if sent_len + len(sent) > max_size and sent_buf:
                    raw_chunks.append(" ".join(sent_buf))
                    sent_buf = [sent]
                    sent_len = len(sent)
                else:
                    sent_buf.append(sent)
                    sent_len += len(sent)

            if sent_buf:
                raw_chunks.append(" ".join(sent_buf))
            continue

        # 普通段落：能放入当前 chunk → 合并；否则 flush 后新建
        if current_len + len(para) + 2 > max_size and current_parts:
            _flush()
            current_parts = [para]
            current_len = len(para)
        else:
            current_parts.append(para)
            current_len += len(para) + 2  # +2 for "\n\n"

    _flush()

    # ── 第二步：按 overlap 为每个 chunk 末尾补切片 ─────────────────────────
    # [Fix-3] 不把 tail 当独立段落重新参与合并，而是在已生成的 chunk 末尾
    # 直接切片追加到下一 chunk 开头，避免产生不自然的残缺前缀。
    if overlap > 0 and len(raw_chunks) > 1:
        result: list[str] = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            tail = raw_chunks[i - 1][-overlap:]
            result.append(tail + raw_chunks[i])
        return [c for c in result if c.strip()]

    return [c for c in raw_chunks if c.strip()]


# ── SectionChunker ──────────────────────────────────────────────────────────

class SectionChunker:
    """
    层次化语义切块器（Phase 10-1），替换 DocumentChunker。

    公开 API 与 DocumentChunker 完全相同：
        chunk(document: ParsedDocument, doc_id: str | None = None) -> list[TextChunk]

    策略：
      - 文档已启用章节识别（page_sections 非空且与 pages 等长）→ 语义分块
      - 否则 → fallback 到 DocumentChunker（固定大小）
      - 多模态 chunk（图片/表格）→ 与 DocumentChunker 完全一致

    注意：
      - enable_section_chunking=True 但未开启章节识别（enable_section_recognition=False）时，
        page_sections 为空，等价于 fallback 到 DocumentChunker，行为与默认一致。

    用法：
        from src.ingestion.section_chunker import SectionChunker
        pipeline.chunker = SectionChunker()
    """

    def __init__(
        self,
        overlap:        int = 50,
        size_overrides: Optional[dict[str, int]] = None,
    ) -> None:
        """
        Args:
            overlap:        chunk 之间的字符重叠数，默认 50 字符（保持上下文连贯）。
            size_overrides: 可选，覆盖部分章节的 chunk size，如 {"method": 800}。
        """
        self._overlap = overlap
        self._chunk_sizes = dict(SECTION_CHUNK_SIZES)
        if size_overrides:
            self._chunk_sizes.update(size_overrides)
        # 延迟导入，避免循环导入
        self._fallback_chunker: object | None = None

    # ── 公开接口 ────────────────────────────────────────────────────────────

    def chunk(
        self,
        document: ParsedDocument,
        doc_id: str | None = None,
    ) -> list[TextChunk]:
        """
        切块文档。

        若文档有章节信息且 pages / page_sections 长度一致，使用层次化语义分块；
        否则 fallback 到 DocumentChunker 的固定大小分块（[5]）。

        Args:
            document: 已解析的文档对象（ParsedDocument）
            doc_id:   可选，文档唯一标识；传入时生成稳定 chunk_id

        Returns:
            TextChunk 列表（含文本 chunk 和多模态 chunk）
        """
        # [Fix-2] 同时校验 pages 和 page_sections 的可用性与长度一致性
        pages    = document.pages or []
        sections = document.page_sections or []

        use_semantic = (
            bool(sections)          # page_sections 非空
            and bool(pages)         # pages 非空
            and len(sections) == len(pages)  # 长度一致才能安全对齐
        )

        if use_semantic:
            logger.debug(
                "SectionChunker: 文档 %s 已识别章节（%d 页），使用层次化语义分块",
                document.file_path, len(pages),
            )
            return self._chunk_with_sections(document, doc_id)
        else:
            if sections and pages and len(sections) != len(pages):
                logger.warning(
                    "SectionChunker: 文档 %s 的 page_sections(%d) 与 pages(%d) 长度不一致，"
                    "fallback 到 DocumentChunker",
                    document.file_path, len(sections), len(pages),
                )
            else:
                logger.debug(
                    "SectionChunker: 文档 %s 无章节信息，fallback 到 DocumentChunker",
                    document.file_path,
                )
            return self._fallback(document, doc_id)

    # ── 内部方法 ────────────────────────────────────────────────────────────

    def _chunk_with_sections(
        self,
        document: ParsedDocument,
        doc_id: str | None,
    ) -> list[TextChunk]:
        """
        按章节语义边界分块（核心逻辑）。

        流程：
          1. 将 pages 按 (section_type, section_title) 分组，
             相邻且类型/标题均相同（或 title 为空）的页面合并
          2. 对每组调用 chunk_by_section()
          3. 生成 TextChunk，保留 section_type / section_title / doc_id 元数据
          4. 多模态 chunk 照常处理
        """
        chunks: list[TextChunk] = []
        occurrence_counter: dict[str, int] = {}
        global_index = 0

        # ── 1. 聚合相邻同 (section_type, section_title) 的页面 ──────────────
        pages    = document.pages
        sections = document.page_sections
        titles   = document.page_section_titles or [""] * len(pages)

        # 将连续同类型同标题的页面合并为 (type, title, content) 组
        page_groups: list[tuple[str, str, str]] = []
        if pages:
            cur_type  = sections[0]
            cur_title = titles[0] if titles else ""
            cur_text  = pages[0]

            for i in range(1, len(pages)):
                st = sections[i]
                tt = titles[i] if i < len(titles) else ""

                # [Fix-5] 合并条件：section_type 相同，且 section_title 相同或任一为空
                same_type  = (st == cur_type)
                same_title = (tt == cur_title) or (not tt) or (not cur_title)

                if same_type and same_title:
                    cur_text += "\n\n" + pages[i]
                    # 若当前 title 为空，用遇到的第一个非空 title
                    if not cur_title and tt:
                        cur_title = tt
                else:
                    page_groups.append((cur_type, cur_title, cur_text))
                    cur_type, cur_title, cur_text = st, tt, pages[i]

            page_groups.append((cur_type, cur_title, cur_text))

        # ── 2. 对每个 section 组分块 ────────────────────────────────────────
        for section_type, section_title, section_content in page_groups:
            max_size = self._chunk_sizes.get(section_type, self._chunk_sizes["unknown"])
            sub_chunks = chunk_by_section(
                section_content=section_content,
                section_type=section_type,
                max_size=max_size,
                overlap=self._overlap,
            )
            for text in sub_chunks:
                text = text.strip()
                if not text:
                    continue

                if doc_id:
                    content_hash = compute_chunk_content_hash(text)
                    occ = occurrence_counter.get(content_hash, 0)
                    chunk_id = generate_chunk_id(doc_id, content_hash, occ)
                    occurrence_counter[content_hash] = occ + 1
                else:
                    chunk_id = f"{document.file_path}::chunk::{global_index}"

                chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    content=text,
                    chunk_index=global_index,
                    metadata={
                        "file_path":     document.file_path,
                        "chunk_index":   global_index,
                        "doc_id":        doc_id or "",
                        "content_type":  "text",
                        "section_type":  section_type,
                        "section_title": section_title,
                    },
                ))
                global_index += 1

        # ── 3. 多模态 chunk（与 DocumentChunker 一致）──────────────────────
        for modal in document.modal_contents:
            if not modal.processed_text.strip():
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

            chunks.append(TextChunk(
                chunk_id=chunk_id,
                content=content,
                chunk_index=global_index,
                metadata={
                    "file_path":     document.file_path,
                    "chunk_index":   global_index,
                    "doc_id":        doc_id or "",
                    "content_type":  modal.content_type,
                    "page_number":   modal.page_number,
                    "position_hint": modal.position_hint,
                    "section_type":  "unknown",
                    "section_title": "",
                },
            ))
            global_index += 1

        logger.info(
            "SectionChunker: 文档 %s 语义分块完成，共 %d chunks（%d 个 section 组）",
            document.file_path, len(chunks), len(page_groups),
        )
        return chunks

    def _fallback(
        self,
        document: ParsedDocument,
        doc_id: str | None,
    ) -> list[TextChunk]:
        """Fallback：文档无章节信息或长度不一致时，委托给 DocumentChunker（[5]）。"""
        if self._fallback_chunker is None:
            from src.ingestion.chunker import DocumentChunker
            self._fallback_chunker = DocumentChunker()
        return self._fallback_chunker.chunk(document, doc_id=doc_id)
