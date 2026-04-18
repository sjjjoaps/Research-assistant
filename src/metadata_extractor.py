"""
元数据提取模块
从文档首页文本中提取标题、作者、摘要、关键词、发表年份
优先使用 LLM 结构化输出，失败时降级为正则规则提取
"""
import re
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agents.prompt_loader import load_prompt_pair
from src.document_parser import ParsedDocument
from src.llm_client import get_llm


# ── 元数据数据结构 ────────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    title: str = Field(description="论文标题，若无法确定则填 'Unknown'")
    authors: list[str] = Field(description="作者列表，若无法确定则为空列表")
    institution: Optional[str] = Field(default=None, description="发表机构，若无则为 null")
    year: Optional[int] = Field(default=None, description="发表年份，4 位数字，无法确定则为 null")
    abstract: Optional[str] = Field(default=None, description="论文摘要原文，若无则为 null")
    keywords: list[str] = Field(default_factory=list, description="关键词列表，若无则为空列表")


# ── Prompt ────────────────────────────────────────────────────────────────────

_meta_sys, _meta_human = load_prompt_pair("metadata_extractor")
_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _meta_sys), ("human", _meta_human)]
)


# ── 正则降级提取 ──────────────────────────────────────────────────────────────

def _regex_extract(text: str) -> DocumentMetadata:
    """当 LLM 调用失败时，用正则规则做最低限度的提取"""
    # 年份：匹配 (2020) 或 2020 这样 2000~2029 范围的数字
    year_match = re.search(r"\b(20[0-2]\d)\b", text)
    year = int(year_match.group(1)) if year_match else None

    # 摘要：匹配 Abstract 段落
    abstract_match = re.search(
        r"(?i)abstract[.\s:—-]*(.+?)(?=\n\s*\n|\nintroduction|\nkeywords|\n1\b)",
        text,
        re.DOTALL,
    )
    abstract = abstract_match.group(1).strip() if abstract_match else None

    # 关键词：匹配 Keywords 段落的第一行
    keywords: list[str] = []
    kw_match = re.search(r"(?i)keywords?[.\s:—-]*(.+)", text)
    if kw_match:
        raw = kw_match.group(1).strip()
        keywords = [k.strip() for k in re.split(r"[;,·•]", raw) if k.strip()]

    return DocumentMetadata(
        title="Unknown",
        authors=[],
        year=year,
        abstract=abstract,
        keywords=keywords,
    )


# ── 主提取器类 ────────────────────────────────────────────────────────────────

class MetadataExtractor:
    def __init__(self) -> None:
        llm = get_llm(temperature=0.0)
        # with_structured_output 让 LLM 直接返回 Pydantic 模型
        self._chain = _PROMPT | llm.with_structured_output(DocumentMetadata)

    def extract(self, document: ParsedDocument) -> DocumentMetadata:
        """
        从 ParsedDocument 中提取元数据。
        取首页前 5000 字符送入 LLM；失败时降级为正则提取。
        """
        # 优先取第一页，没有则取全文开头
        first_page = document.pages[0] if document.pages else document.raw_text
        context = first_page[:5000]
    
        try:
            result = self._chain.invoke({"context": context})
            return result
        except Exception as e:
            print(f"[MetadataExtractor] LLM 提取失败，降级为正则提取。原因: {e}")
            return _regex_extract(context)
