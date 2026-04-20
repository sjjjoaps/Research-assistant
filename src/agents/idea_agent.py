"""
Idea Agent
消费 DeepResearchAgent 产出的 ResearchReport，生成固定模板的 Idea 报告。
不直接检索，避免超出已有证据范围。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.agents.deep_research_agent import ResearchReport
from src.agents.prompt_loader import load_prompt_pair
from src.infrastructure.llm_client import get_native_llm
from src.infrastructure.json_utils import extract_json


class IdeaReport(BaseModel):
    research_gaps: list[str] = Field(
        description="已发现的研究空白（必须基于研究报告中的证据）"
    )
    method_comparisons: list[str] = Field(
        description="现有方法之间的对比、差异与权衡"
    )
    suggested_directions: list[str] = Field(
        description="可尝试的研究方向或改进方向"
    )
    evidence_basis: str = Field(
        description="上述分析依赖的关键证据与来源概述"
    )
    confidence_note: str = Field(
        description="对证据充分度的说明；若证据不足必须明确写出"
    )


_idea_sys, _idea_human = load_prompt_pair("idea_agent")


def _parse_idea_report(content: str) -> IdeaReport:
    """Parse JSON from LLM output into IdeaReport, with graceful fallback."""
    # Step 1: extract JSON structure
    try:
        data = extract_json(content)
    except ValueError:
        # LLM returned prose or truncated JSON — wrap raw content as best-effort report
        return IdeaReport(
            research_gaps=[],
            method_comparisons=[],
            suggested_directions=[],
            evidence_basis=content[:1000] if content.strip() else "（模型未返回结构化内容）",
            confidence_note="JSON 解析失败，以上为模型原始输出，请重试。",
        )

    # Step 2: ensure we have a dict
    if isinstance(data, list):
        # LLM returned a list — treat as suggested_directions
        return IdeaReport(
            research_gaps=[],
            method_comparisons=[],
            suggested_directions=[str(item) for item in data],
            evidence_basis="（模型返回了列表格式，字段映射不完整）",
            confidence_note="输出格式异常，建议重试。",
        )
    if not isinstance(data, dict):
        return IdeaReport(
            research_gaps=[],
            method_comparisons=[],
            suggested_directions=[],
            evidence_basis="（模型返回了非预期格式）",
            confidence_note="输出格式异常，建议重试。",
        )

    # Step 3: fill missing fields with safe defaults before Pydantic validation
    data.setdefault("research_gaps", [])
    data.setdefault("method_comparisons", [])
    data.setdefault("suggested_directions", [])
    data.setdefault("evidence_basis", "（未提供）")
    data.setdefault("confidence_note", "（未提供）")

    # Coerce list fields: if LLM returned a string, wrap it
    for list_field in ("research_gaps", "method_comparisons", "suggested_directions"):
        if isinstance(data[list_field], str):
            data[list_field] = [data[list_field]] if data[list_field].strip() else []

    try:
        return IdeaReport(**data)
    except Exception as exc:
        return IdeaReport(
            research_gaps=[],
            method_comparisons=[],
            suggested_directions=[],
            evidence_basis=str(data),
            confidence_note=f"字段校验失败（{exc}），以上为原始解析内容。",
        )


class IdeaAgent:
    """无状态 Agent：将研究报告转化为固定模板 Idea 输出"""

    def __init__(self) -> None:
        self._llm = get_native_llm(temperature=0.2)

    def _invoke(self, question: str, report_summary: str) -> IdeaReport:
        resp = self._llm.invoke([
            {"role": "system", "content": _idea_sys},
            {"role": "user",   "content": _idea_human.format(
                question=question, report_summary=report_summary
            )},
        ])
        return _parse_idea_report(str(resp.get("content") or ""))

    def generate(self, question: str, report: ResearchReport) -> IdeaReport:
        return self._invoke(question, report.final_report)

    def generate_from_markdown(self, question: str, report_markdown: str) -> IdeaReport:
        return self._invoke(question, report_markdown)

    @staticmethod
    def to_markdown(result: IdeaReport) -> str:
        """将结构化 IdeaReport 渲染为 Markdown"""
        lines = [
            "# Idea 报告",
            "",
            "## Research Gaps",
            *[f"- {item}" for item in result.research_gaps],
            "",
            "## 方法对比",
            *[f"- {item}" for item in result.method_comparisons],
            "",
            "## 建议方向",
            *[f"- {item}" for item in result.suggested_directions],
            "",
            "## 证据依据",
            result.evidence_basis,
            "",
            "## 证据充分度说明",
            result.confidence_note,
        ]
        return "\n".join(lines)
