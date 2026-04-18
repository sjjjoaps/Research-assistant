"""
Idea Agent
消费 DeepResearchAgent 产出的 ResearchReport，生成固定模板的 Idea 报告。
不直接检索，避免超出已有证据范围。
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.agents.deep_research_agent import ResearchReport
from src.agents.prompt_loader import load_prompt_pair
from src.llm_client import get_llm


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
_IDEA_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _idea_sys), ("human", _idea_human)]
)


class IdeaAgent:
    """无状态 Agent：将研究报告转化为固定模板 Idea 输出"""

    def __init__(self) -> None:
        llm = get_llm(temperature=0.2)
        self.chain = _IDEA_PROMPT | llm.with_structured_output(IdeaReport)

    def generate(self, question: str, report: ResearchReport) -> IdeaReport:
        """基于研究报告生成结构化 Idea 报告"""
        return self.chain.invoke(
            {
                "question": question,
                "report_summary": report.final_report,
            }
        )

    def generate_from_markdown(self, question: str, report_markdown: str) -> IdeaReport:
        """直接接收 Markdown 文本生成 Idea 报告（供 API 层调用，无需完整 ResearchReport）"""
        return self.chain.invoke(
            {
                "question": question,
                "report_summary": report_markdown,
            }
        )

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
