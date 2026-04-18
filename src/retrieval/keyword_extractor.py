"""
查询关键词提取器

从用户查询中分离出两类关键词：
- ll_keywords（低层）：实体、模型、数据集、方法名等可直接用于定位证据的术语
- hl_keywords（高层）：研究趋势、主题、方向、问题域等宏观语义

Phase 4.1 实现策略：
1. 规则优先：短语识别 + 分类规则，优先覆盖常见学术问法
2. LLM 兜底：规则无法提取时再走结构化抽取
3. 最终失败返回空结果，不影响主流程
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_MAX_QUERY_LENGTH = 300
_MAX_KEYWORDS_PER_GROUP = 8

_HL_ROOTS_ZH = (
    "趋势",
    "综述",
    "挑战",
    "现状",
    "进展",
    "演变",
    "对比",
    "比较",
    "评估",
    "方向",
    "问题",
    "应用",
)
_HL_ROOTS_EN = (
    "trend",
    "survey",
    "overview",
    "challenge",
    "progress",
    "comparison",
    "review",
    "evolution",
    "evaluation",
    "direction",
    "application",
)

_ZH_LL_SUFFIXES = (
    "模型",
    "方法",
    "算法",
    "框架",
    "网络",
    "系统",
    "数据集",
    "基准",
    "任务",
    "论文",
    "编码器",
    "解码器",
    "预训练",
    "微调",
    "检索",
)
_EN_LL_HINTS = (
    "bert",
    "gpt",
    "rag",
    "clip",
    "llama",
    "transformer",
    "diffusion",
    "retrieval",
    "prompt",
    "agent",
    "benchmark",
    "dataset",
    "encoder",
    "decoder",
)
_ZH_TOPIC_HINTS = (
    "研究",
    "领域",
    "任务",
    "方向",
    "应用",
    "发展",
    "技术",
    "范式",
)
_QUESTION_FILLERS = (
    "什么",
    "哪些",
    "哪个",
    "如何",
    "怎么",
    "为什么",
    "是否",
    "以及",
    "还有",
    "请问",
    "一下",
)
_EN_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "for",
    "to",
    "and",
    "or",
    "in",
    "on",
    "with",
    "by",
    "is",
    "are",
    "be",
    "what",
    "which",
    "how",
    "why",
}
_HL_ONLY_WORDS = {
    "trend",
    "trends",
    "survey",
    "overview",
    "review",
    "progress",
    "challenge",
    "challenges",
    "direction",
    "directions",
}

from src.agents.prompt_loader import load_system_prompt

_PHRASE_RE = re.compile(
    r"[A-Z][A-Za-z0-9.+\-]*(?:\s+[A-Z]?[A-Za-z0-9.+\-]+){0,5}"
    r"|[A-Za-z][A-Za-z0-9.+\-]{1,31}"
    r"|[\u4e00-\u9fff]{2,20}"
)
_MIXED_TERM_RE = re.compile(
    r"[\u4e00-\u9fff]{2,12}[A-Za-z][A-Za-z0-9.+\-]{1,20}"
    r"|[A-Za-z][A-Za-z0-9.+\-]{1,20}[\u4e00-\u9fff]{2,12}"
)
_HL_PHRASE_RE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9.+\-]{0,10}(?:研究趋势|发展趋势|应用挑战|研究方向|研究现状|技术进展|主要挑战)"
)

_LLM_PROMPT = load_system_prompt("keyword_extractor_fallback")


@dataclass
class KeywordResult:
    """关键词提取结果。"""

    ll_keywords: list[str] = field(default_factory=list)
    hl_keywords: list[str] = field(default_factory=list)
    raw_query: str = ""


class KeywordExtractor:
    """两阶段关键词提取器：规则优先 + LLM 兜底。"""

    def __init__(self) -> None:
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from src.infrastructure.llm_client import get_llm

            self._llm = get_llm(temperature=0.0)
        return self._llm

    def extract(self, query: str) -> KeywordResult:
        """提取低层和高层关键词。"""
        normalized_query = (query or "").strip()
        if not normalized_query:
            return KeywordResult(raw_query=query)

        result = self._rule_extract(normalized_query)
        if not result.ll_keywords and not result.hl_keywords:
            result = self._llm_extract(normalized_query)
        return result

    def _rule_extract(self, query: str) -> KeywordResult:
        phrases = self._extract_phrases(query)
        if not phrases:
            return KeywordResult(raw_query=query)

        ll_keywords: list[str] = []
        hl_keywords: list[str] = []
        seen_ll: set[str] = set()
        seen_hl: set[str] = set()

        for phrase in phrases:
            cleaned = self._clean_phrase(phrase)
            if not cleaned:
                continue

            category = self._classify_phrase(cleaned)
            if category == "hl":
                for item in self._expand_high_level_phrase(cleaned):
                    self._append_unique(hl_keywords, seen_hl, item)
            elif category == "ll":
                self._append_unique(ll_keywords, seen_ll, cleaned)

        ll_keywords = self._trim_keywords(ll_keywords)
        ll_lookup = {item.casefold() for item in ll_keywords}
        hl_keywords = self._trim_keywords(
            [item for item in hl_keywords if item.casefold() not in ll_lookup]
        )
        return KeywordResult(ll_keywords=ll_keywords, hl_keywords=hl_keywords, raw_query=query)

    def _extract_phrases(self, query: str) -> list[str]:
        compact_query = re.sub(r"\s+", " ", query).strip()
        positioned_phrases: list[tuple[int, str]] = []
        for pattern in (_MIXED_TERM_RE, _HL_PHRASE_RE, _PHRASE_RE):
            for match in pattern.finditer(compact_query):
                positioned_phrases.append((match.start(), match.group(0).strip()))

        positioned_phrases.sort(key=lambda item: item[0])
        phrases = [text for _, text in positioned_phrases]
        if not phrases and compact_query:
            phrases = compact_query.split()
        return phrases

    def _classify_phrase(self, phrase: str) -> str | None:
        phrase_lower = phrase.lower()

        if phrase_lower in _EN_STOPWORDS or phrase in _QUESTION_FILLERS:
            return None

        if self._is_high_level_phrase(phrase, phrase_lower):
            return "hl"
        if self._is_low_level_phrase(phrase, phrase_lower):
            return "ll"

        if re.search(r"[\u4e00-\u9fff]", phrase):
            if 2 <= len(phrase) <= 8:
                return "ll"
            return None

        if " " in phrase and len(phrase.split()) >= 2:
            return "ll"
        if len(phrase) >= 3:
            return "ll"
        return None

    def _is_high_level_phrase(self, phrase: str, phrase_lower: str) -> bool:
        if any(root in phrase_lower for root in _HL_ROOTS_EN):
            return True
        if any(root in phrase for root in _HL_ROOTS_ZH):
            return True
        if phrase_lower in _HL_ONLY_WORDS:
            return True
        if re.fullmatch(r"[\u4e00-\u9fff]{4,20}", phrase) and any(hint in phrase for hint in _ZH_TOPIC_HINTS):
            return True
        if " " in phrase and any(root in phrase_lower.split() for root in _HL_ONLY_WORDS):
            return True
        return False

    def _is_low_level_phrase(self, phrase: str, phrase_lower: str) -> bool:
        if any(hint in phrase_lower for hint in _EN_LL_HINTS):
            return True
        if any(phrase.endswith(suffix) for suffix in _ZH_LL_SUFFIXES):
            return True
        if re.search(r"[A-Z]{2,}", phrase):
            return True
        if re.search(r"\d", phrase) and re.search(r"[A-Za-z]", phrase):
            return True
        if "-" in phrase and re.search(r"[A-Za-z]", phrase):
            return True
        if re.fullmatch(r"[\u4e00-\u9fff]{2,8}", phrase) and not any(hint in phrase for hint in _ZH_TOPIC_HINTS):
            return True
        return False

    def _llm_extract(self, query: str) -> KeywordResult:
        try:
            llm = self._get_llm()
            prompt = _LLM_PROMPT.format(query=query[:_MAX_QUERY_LENGTH])
            response = llm.invoke(prompt)
            content = str(response.content).strip()
            if "```" in content:
                content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`").strip()

            data = json.loads(content)
            ll = self._trim_keywords([str(k) for k in data.get("ll_keywords", []) if k])
            hl = self._trim_keywords([str(k) for k in data.get("hl_keywords", []) if k])
            return KeywordResult(ll_keywords=ll, hl_keywords=hl, raw_query=query)
        except Exception as exc:
            logger.debug("LLM 关键词提取失败: %s", exc)
            return KeywordResult(raw_query=query)

    def _expand_high_level_phrase(self, phrase: str) -> list[str]:
        parts = [phrase]
        if "和" in phrase:
            parts = [item for item in re.split(r"[和及与]", phrase) if item]

        expanded: list[str] = []
        for item in parts:
            normalized = item
            if "研究趋势" in normalized:
                normalized = "研究趋势"
            elif "应用挑战" in normalized:
                normalized = "应用挑战"
            elif "研究方向" in normalized:
                normalized = "研究方向"
            elif "研究现状" in normalized:
                normalized = "研究现状"
            elif "技术进展" in normalized:
                normalized = "技术进展"
            elif "主要挑战" in normalized:
                normalized = "主要挑战"
            expanded.append(normalized)
        return expanded

    @staticmethod
    def _clean_phrase(phrase: str) -> str:
        cleaned = phrase.strip(" ,，。！？!?()（）[]【】\"'“”‘’")
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"(有哪些|是什么|是什么问题|差异是什么|表现差异是什么|如何|怎么样)$", "", cleaned)
        cleaned = cleaned.strip("的在上中与和及 ")
        if cleaned.casefold() in _EN_STOPWORDS or cleaned in _QUESTION_FILLERS:
            return ""
        return cleaned

    @staticmethod
    def _append_unique(items: list[str], seen: set[str], value: str) -> None:
        key = value.casefold()
        if not value or key in seen:
            return
        seen.add(key)
        items.append(value)

    @staticmethod
    def _trim_keywords(items: list[str]) -> list[str]:
        trimmed: list[str] = []
        seen: set[str] = set()
        for item in items:
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            trimmed.append(item)
            if len(trimmed) >= _MAX_KEYWORDS_PER_GROUP:
                break
        return trimmed
