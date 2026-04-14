"""
章节结构识别模块

从 PDF 页面文本中识别学术论文的章节类型，为 chunk 提供 section_type 元数据。

识别策略（两阶段）：
1. 规则优先：正则匹配页面前 10 行中的标题关键词（含全大写标题）
2. LLM 兜底：规则未识别的页面发送前 200 字符给 LLM 分类
3. 前向填充：在规则遍历和 LLM 兜底之间，用已知 section_type 填充相邻页面

支持的 section_type（11 类）：
    abstract / introduction / related_work / method / experiment /
    result / discussion / conclusion / reference / appendix / unknown

降级边界：
- LLM 初始化失败：跳过 LLM 兜底，保留 "unknown"
- 单页 LLM 调用失败：该页保留 "unknown"，不中断整体识别
"""
from __future__ import annotations

import bisect
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SECTION_TYPES = frozenset({
    "abstract", "introduction", "related_work", "method",
    "experiment", "result", "discussion", "conclusion",
    "reference", "appendix", "unknown",
})

_HEADING_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(abstract)$", re.I), "abstract"),
    (re.compile(r"^(\d+\.?\s*)?(introduction)$", re.I), "introduction"),
    (re.compile(r"^(\d+\.?\s*)?(related\s+work|literature\s+review|background)$", re.I), "related_work"),
    (re.compile(r"^(\d+\.?\s*)?(method(ology|s)?|approach|proposed\s+method)$", re.I), "method"),
    (re.compile(r"^(\d+\.?\s*)?(experiment(s|al\s+setup)?|evaluation|experimental\s+results)$", re.I), "experiment"),
    (re.compile(r"^(\d+\.?\s*)?(results?)$", re.I), "result"),
    (re.compile(r"^(\d+\.?\s*)?(discussion)$", re.I), "discussion"),
    (re.compile(r"^(\d+\.?\s*)?(conclusion(s)?|summary)$", re.I), "conclusion"),
    (re.compile(r"^(references?|bibliography)$", re.I), "reference"),
    (re.compile(r"^(appendix|appendices|supplementary)(\s+[A-Z])?$", re.I), "appendix"),
]

_LLM_PROMPT = (
    "You are classifying a section of an academic paper.\n"
    "Given the following text (first 200 characters of a page), identify which section type it belongs to.\n"
    "Choose exactly one from: abstract, introduction, related_work, method, experiment, "
    "result, discussion, conclusion, reference, appendix, unknown.\n"
    "Respond with only the section type label, nothing else.\n\n"
    "Text:\n{text}"
)


@dataclass
class SectionInfo:
    """章节信息，描述连续若干页属于同一章节。"""
    section_type: str
    section_title: str
    start_page: int  # 0-based
    end_page: int    # inclusive, 0-based


class SectionRecognizer:
    """章节结构识别器。

    LLM 采用懒初始化：首次需要 LLM 兜底时才创建，初始化失败时降级跳过。
    """

    def __init__(self) -> None:
        self._llm = None  # 懒初始化

    def _get_llm(self):
        if self._llm is None:
            from src.llm_client import get_llm
            self._llm = get_llm(temperature=0.0)
        return self._llm

    def recognize_per_page(self, pages: list[str]) -> tuple[list[str], list[str]]:
        """识别每页所属章节类型。

        Args:
            pages: 每页文本列表（与 ParsedDocument.pages 对应）。

        Returns:
            (section_types, section_titles) 两个平行列表，长度与 pages 相同。
        """
        if not pages:
            return [], []

        # 初始化：所有页面默认 unknown
        page_results: list[tuple[str, str]] = [("unknown", "")] * len(pages)

        # 第一遍：规则识别
        for i, page_text in enumerate(pages):
            stype, stitle = self._detect_heading_in_page(page_text)
            if stype is not None:
                page_results[i] = (stype, stitle or "")

        # 第二遍：前向填充（用已知 section_type 填充相邻 unknown 页）
        current_type = "unknown"
        current_title = ""
        for i in range(len(page_results)):
            if page_results[i][0] != "unknown":
                current_type = page_results[i][0]
                current_title = page_results[i][1]
            else:
                page_results[i] = (current_type, current_title)

        # 第三遍：LLM 兜底（仅对仍为 unknown 的页面）
        for i, page_text in enumerate(pages):
            if page_results[i][0] == "unknown":
                classified = self._classify_with_llm(page_text)
                if classified != "unknown":
                    page_results[i] = (classified, "")

        section_types = [r[0] for r in page_results]
        section_titles = [r[1] for r in page_results]
        return section_types, section_titles

    def _detect_heading_in_page(self, page_text: str) -> tuple[str | None, str | None]:
        """在页面前 10 行中检测章节标题。

        Returns:
            (section_type, heading_text) 若找到标题，否则 (None, None)。
        """
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        candidates = lines[:10]

        for line in candidates:
            if len(line) > 80:
                continue

            # 直接正则匹配
            for pattern, section_type in _HEADING_PATTERNS:
                if pattern.match(line):
                    return section_type, line

            # 全大写短行：转小写后再匹配
            if line.isupper() and len(line) <= 40:
                normalized = line.lower()
                for pattern, section_type in _HEADING_PATTERNS:
                    if pattern.match(normalized):
                        return section_type, line

        return None, None

    def _classify_with_llm(self, page_text: str) -> str:
        """用 LLM 对页面文本进行章节分类。

        Returns:
            section_type 字符串；LLM 不可用或返回无效标签时返回 "unknown"。
        """
        try:
            llm = self._get_llm()
            prompt = _LLM_PROMPT.format(text=page_text[:200])
            response = llm.invoke(prompt)
            label = response.content.strip().lower()
            return label if label in SECTION_TYPES else "unknown"
        except Exception as e:
            logger.debug("LLM 章节分类失败: %s", e)
            return "unknown"
