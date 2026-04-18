"""
引用提取模块

从学术论文全文中定位参考文献章节，解析单条引用，生成结构化 CitationRecord。

识别策略（两阶段）：
1. 规则优先：正则提取 DOI / 年份 / 引号内标题
2. LLM 兜底：规则未能提取标题的条目发送给 LLM 结构化解析

降级边界：
- LLM 初始化失败：跳过 LLM 兜底，保留规则解析结果
- 单条 LLM 调用失败：该条目保留规则解析结果，不中断整体提取
- 未找到参考文献章节：返回空列表
- 单条引用过短（< 10 字符）：跳过
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field

from src.agents.prompt_loader import load_system_prompt

logger = logging.getLogger(__name__)

# ── 正则常量 ──────────────────────────────────────────────────────────────────

_DOI_RE = re.compile(r"10\.\d{4,}/\S+")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_TITLE_QUOTED_RE = re.compile(r'"([^"]{10,})"')
# 编号格式：[1] 或 1.
_NUMBERED_RE = re.compile(r"\n\s*(?:\[(\d+)\]|(\d+)\.\s)")
# 参考文献章节标题
_REF_SECTION_RE = re.compile(
    r"(?:^|\n)(References?|Bibliography|参考文献)\s*\n",
    re.IGNORECASE,
)

_LLM_PROMPT = load_system_prompt("citation_extractor_fallback")

_MAX_REFERENCES = 100  # 单文档最多处理引用数，防止超大文档消耗过多 LLM 调用


@dataclass
class CitationRecord:
    """单条引用的结构化表示。"""
    raw_text: str
    title: str
    authors: str
    year: str
    doi: str
    ref_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.ref_id:
            self.ref_id = hashlib.md5(self.raw_text.encode("utf-8")).hexdigest()


class CitationExtractor:
    """引用提取器。

    LLM 采用懒初始化：首次需要 LLM 兜底时才创建，初始化失败时降级跳过。
    """

    def __init__(self) -> None:
        self._llm = None  # 懒初始化

    def _get_llm(self):
        if self._llm is None:
            from src.infrastructure.llm_client import get_llm
            self._llm = get_llm(temperature=0.0)
        return self._llm

    def extract(self, raw_text: str) -> list[CitationRecord]:
        """从文档全文中提取引用列表。

        Args:
            raw_text: 文档全文（ParsedDocument.raw_text）。

        Returns:
            CitationRecord 列表；未找到参考文献章节时返回空列表。
        """
        ref_section = self._find_reference_section(raw_text)
        if ref_section is None:
            return []

        raw_refs = self._split_references(ref_section)
        raw_refs = [r for r in raw_refs if len(r.strip()) >= 10][:_MAX_REFERENCES]

        results: list[CitationRecord] = []
        for ref_text in raw_refs:
            record = self._parse_reference_rule(ref_text.strip())
            if not record.title:
                record = self._parse_reference_llm(ref_text.strip())
            results.append(record)

        return results

    def _find_reference_section(self, raw_text: str) -> str | None:
        """定位参考文献章节，返回该章节及其后所有内容。"""
        match = _REF_SECTION_RE.search(raw_text)
        if match:
            return raw_text[match.start():]
        return None

    def _split_references(self, ref_section: str) -> list[str]:
        """将参考文献章节切分为单条引用。

        优先级：编号格式 > 作者年份格式 > 双换行分段。
        """
        # 尝试编号格式切分
        numbered_positions = [m.start() for m in _NUMBERED_RE.finditer(ref_section)]
        if len(numbered_positions) >= 2:
            parts = []
            for i, pos in enumerate(numbered_positions):
                end = numbered_positions[i + 1] if i + 1 < len(numbered_positions) else len(ref_section)
                parts.append(ref_section[pos:end].strip())
            return [p for p in parts if p]

        # 兜底：双换行分段（跳过章节标题行）
        paragraphs = [p.strip() for p in ref_section.split("\n\n") if p.strip()]
        # 过滤掉章节标题行（通常很短且不含年份）
        return [p for p in paragraphs if len(p) >= 10]

    def _parse_reference_rule(self, ref_text: str) -> CitationRecord:
        """规则解析单条引用。"""
        doi_match = _DOI_RE.search(ref_text)
        year_match = _YEAR_RE.search(ref_text)
        title_match = _TITLE_QUOTED_RE.search(ref_text)

        title = title_match.group(1) if title_match else ""

        return CitationRecord(
            raw_text=ref_text,
            title=title,
            authors="",
            year=year_match.group(0) if year_match else "",
            doi=doi_match.group(0) if doi_match else "",
        )

    def _parse_reference_llm(self, ref_text: str) -> CitationRecord:
        """LLM 解析复杂引用，失败时返回仅含 raw_text 的 CitationRecord。"""
        try:
            llm = self._get_llm()
            prompt = _LLM_PROMPT.format(text=ref_text[:500])
            response = llm.invoke(prompt)
            content = response.content.strip()
            # 提取 JSON（可能被 markdown 代码块包裹）
            if "```" in content:
                content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`").strip()
            data = json.loads(content)
            return CitationRecord(
                raw_text=ref_text,
                title=str(data.get("title", "")),
                authors=str(data.get("authors", "")),
                year=str(data.get("year", "")),
                doi=str(data.get("doi", "")),
            )
        except Exception as e:
            logger.debug("LLM 引用解析失败: %s", e)
            return CitationRecord(
                raw_text=ref_text,
                title="",
                authors="",
                year="",
                doi="",
            )
