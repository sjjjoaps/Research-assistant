"""
Deep Research Agent
原生 Plan-Execute-Report 多步推理 Agent

执行流程（五步固定编排）：
  Step1 Plan      → LLM 把问题拆成若干子问题
  Step2 Retrieve  → 对每个子问题执行检索
  Step3 Analyze   → 对每个子问题生成证据约束结论
  Step4 Community → 若 use_community=True，附加社区摘要视角
  Step5 Report    → 汇总生成完整 Markdown 研究报告
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.llm_client import get_llm
from src.retriever import RetrievedChunk
from src.token_tracker import TokenUsage

RetrieverMode = Literal["semantic", "hybrid", "graph"]


# ──────────────────────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SubQuestionResult:
    question: str
    chunks: list[RetrievedChunk]
    sources: list[str]
    analysis: str
    token_usage: TokenUsage | None


@dataclass
class ResearchReport:
    question: str
    sub_questions: list[str]
    sub_results: list[SubQuestionResult]
    community_insights: str       # "" if not enabled
    final_report: str             # Markdown 格式研究报告
    all_sources: list[str]        # 去重合并
    total_token_usage: dict       # {prompt, completion, total, cost_cny}


# ──────────────────────────────────────────────────────────────────────────────
# 子问题规划器（structured output）
# ──────────────────────────────────────────────────────────────────────────────

class SubQuestionList(BaseModel):
    sub_questions: list[str] = Field(
        description="3~5 个具体、可独立检索的子问题，彼此不重叠",
        min_length=2,
        max_length=5,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

_PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是学术研究规划助手。请将用户的研究问题拆解成 {max_n} 个以内的子问题，每个子问题应当具体、可独立检索，且彼此不重叠。聚焦于：定义/现状/方法/局限/趋势/空白 等角度。",
        ),
        ("human", "研究问题：{question}"),
    ]
)

_ANALYZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是学术文献分析助手。请严格基于给定检索上下文回答子问题，"
            "不得编造未出现的信息。若证据不足，请明确写明'证据不足'并说明。",
        ),
        (
            "human",
            "子问题：{sub_question}\n\n"
            "检索上下文：\n{context}\n\n"
            "请输出：\n1) 证据约束结论\n2) 主要局限\n3) 引用来源编号",
        ),
    ]
)

_REPORT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是学术研究报告生成助手。请基于子问题分析结果，"
            "生成一份结构完整的 Markdown 研究报告。"
            "报告应包含：研究背景、子问题分析摘要、方法对比、主要不足与局限、初步 Research Gaps、参考来源。"
            "严格基于给定证据，不得编造。",
        ),
        (
            "human",
            "研究问题：{question}\n\n"
            "子问题分析结果：\n{sub_analyses}\n\n"
            "{community_section}"
            "请生成完整 Markdown 研究报告：",
        ),
    ]
)


# ──────────────────────────────────────────────────────────────────────────────
# DeepResearchAgent
# ──────────────────────────────────────────────────────────────────────────────

def _make_retriever(mode: RetrieverMode, top_k: int):
    """检索器工厂（与 QAAgent 保持一致）"""
    if mode == "hybrid":
        from src.hybrid_retriever import HybridRetriever
        return HybridRetriever(top_k=top_k, semantic_top_k=top_k * 2, bm25_top_k=top_k * 2)
    if mode == "graph":
        from src.graph_retriever import GraphRetriever
        return GraphRetriever(top_k=top_k, expand_entities=True)
    from src.retriever import SemanticRetriever
    return SemanticRetriever(top_k=top_k)


def _build_context(chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
    """与 QAAgent 保持一致的上下文格式"""
    blocks: list[str] = []
    sources: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        src = f"{chunk.file_path}#chunk-{chunk.chunk_index}"
        sources.append(src)
        blocks.append(f"[{i}] 来源: {src}\n内容: {chunk.content}")
    return "\n\n".join(blocks), sources


def _aggregate_token_usage(usages: list[TokenUsage | None]) -> dict:
    """累加多步 token 统计"""
    prompt = sum(u.prompt_tokens for u in usages if u)
    completion = sum(u.completion_tokens for u in usages if u)
    total = sum(u.total_tokens for u in usages if u)
    cost = sum(u.estimated_cost_cny for u in usages if u)
    return {"prompt": prompt, "completion": completion, "total": total, "cost_cny": round(cost, 6)}


class DeepResearchAgent(BaseAgent):
    """
    原生多步研究 Agent。

    Parameters
    ----------
    top_k : int
        每个子问题检索 chunk 数量
    max_subquestions : int
        子问题上限（LLM 生成时的约束）
    retriever_mode : RetrieverMode
        检索模式：semantic / hybrid / graph
    use_community : bool
        是否附加社区摘要视角（需要先运行 CommunityDetector）
    """

    def __init__(
        self,
        top_k: int = 5,
        max_subquestions: int = 4,
        retriever_mode: RetrieverMode = "hybrid",
        use_community: bool = False,
    ) -> None:
        super().__init__(max_history_turns=5)
        self.top_k = top_k
        self.max_subquestions = max_subquestions
        self.retriever_mode = retriever_mode
        self.use_community = use_community
        self.retriever = _make_retriever(retriever_mode, top_k)
        self.llm = get_llm(temperature=0.2)
        # structured output 链（规划器）
        self._plan_chain = _PLAN_PROMPT | self.llm.with_structured_output(SubQuestionList)
        # 普通文本链
        self._analyze_chain = _ANALYZE_PROMPT | self.llm
        self._report_chain = _REPORT_PROMPT | self.llm

    # ── Step 1：规划 ──────────────────────────────────────────────────────────

    def _plan(self, question: str) -> list[str]:
        result = self._plan_chain.invoke(
            {"question": question, "max_n": self.max_subquestions}
        )
        return result.sub_questions[: self.max_subquestions]

    # ── Step 2+3：检索 + 局部分析 ─────────────────────────────────────────────

    def _retrieve_and_analyze(self, sub_question: str) -> SubQuestionResult:
        chunks = self.retriever.retrieve(sub_question)
        context, sources = _build_context(chunks)

        if not context:
            return SubQuestionResult(
                question=sub_question,
                chunks=chunks,
                sources=[],
                analysis="证据不足：未检索到相关内容。",
                token_usage=None,
            )

        message = self._analyze_chain.invoke(
            {"sub_question": sub_question, "context": context}
        )
        token_usage = TokenUsage.from_langchain_message(message, self.llm.model_name)
        return SubQuestionResult(
            question=sub_question,
            chunks=chunks,
            sources=sources,
            analysis=str(message.content),
            token_usage=token_usage,
        )

    # ── Step 4：社区视角（可选） ───────────────────────────────────────────────

    def _get_community_section(self) -> str:
        if not self.use_community:
            return ""
        try:
            from src.graph_store import GraphStore
            graph_store = GraphStore()
            summaries = graph_store.get_community_summaries(limit=5)
            graph_store.close()
            if not summaries:
                return ""
            lines = ["相关研究社区视角："]
            for s in summaries:
                lines.append(f"- 社区 {s['label']} ({s['entity_count']} 个实体): {s['summary']}")
            return "\n".join(lines) + "\n\n"
        except Exception:
            return ""

    # ── Step 5：总报告 ────────────────────────────────────────────────────────

    def _generate_report(
        self,
        question: str,
        sub_results: list[SubQuestionResult],
        community_section: str,
    ) -> tuple[str, TokenUsage | None]:
        sub_analyses = []
        for i, r in enumerate(sub_results, start=1):
            sub_analyses.append(f"### 子问题 {i}：{r.question}\n{r.analysis}")
        sub_analyses_text = "\n\n".join(sub_analyses)

        community_prompt = f"社区视角补充：\n{community_section}\n" if community_section else ""

        message = self._report_chain.invoke(
            {
                "question": question,
                "sub_analyses": sub_analyses_text,
                "community_section": community_prompt,
            }
        )
        token_usage = TokenUsage.from_langchain_message(message, self.llm.model_name)
        return str(message.content), token_usage

    # ── 主入口 ────────────────────────────────────────────────────────────────

    def research(self, thread_id: str, question: str) -> ResearchReport:
        """执行完整五步研究流程，返回 ResearchReport"""
        self.append_user_message(thread_id, question)

        # Step1 Plan
        sub_questions = self._plan(question)

        # Step2+3 Retrieve + Analyze
        sub_results: list[SubQuestionResult] = []
        for sub_q in sub_questions:
            sub_results.append(self._retrieve_and_analyze(sub_q))

        # Step4 Community
        community_insights = self._get_community_section()

        # Step5 Report
        final_report, report_token = self._generate_report(question, sub_results, community_insights)

        # 汇总来源（去重保序）
        all_sources: list[str] = []
        seen: set[str] = set()
        for r in sub_results:
            for src in r.sources:
                if src not in seen:
                    seen.add(src)
                    all_sources.append(src)

        # 汇总 token
        all_usages: list[TokenUsage | None] = [r.token_usage for r in sub_results] + [report_token]
        total_token_usage = _aggregate_token_usage(all_usages)

        self.append_assistant_message(thread_id, final_report, all_sources)

        return ResearchReport(
            question=question,
            sub_questions=sub_questions,
            sub_results=sub_results,
            community_insights=community_insights,
            final_report=final_report,
            all_sources=all_sources,
            total_token_usage=total_token_usage,
        )

    def run_turn(self, thread_id: str, user_input: str) -> dict:
        """满足 BaseAgent 抽象方法约束，包装 research()"""
        report = self.research(thread_id, user_input)
        return {
            "answer": report.final_report,
            "sources": report.all_sources,
            "thread_id": thread_id,
            "retriever_mode": self.retriever_mode,
            "token_usage": report.total_token_usage,
            "research_report": report,
        }
