"""
Deep Research Agent
原生 Plan-Execute-Report 多步推理 Agent

执行流程（五步固定编排）：
  Step1 Plan      → LLM 把问题拆成若干子问题
  Step2 Retrieve  → 对每个子问题执行检索
  Step3 Analyze   → 对每个子问题生成证据约束结论
  Step4 Community → 若 use_community=True，附加社区摘要视角
  Step5 Report    → 汇总生成完整 Markdown 研究报告

Phase 4.1 变更：
- 引入 KeywordExtractor，对每个子问题提取关键词并以 debug 日志记录
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.agents.prompt_loader import load_prompt_pair
from src.infrastructure.llm_client import get_native_llm
from src.infrastructure.json_utils import extract_json
from src.retrieval.retriever import RetrievedChunk
from src.retrieval.keyword_extractor import KeywordExtractor
from src.infrastructure.token_tracker import TokenUsage

logger = logging.getLogger(__name__)

RetrieverMode = Literal["semantic", "hybrid", "graph", "local", "global", "mix"]


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

_plan_sys, _plan_human = load_prompt_pair("deep_research_plan")
_plan_human = _plan_human or "研究问题：{question}"

_analyze_sys, _analyze_human = load_prompt_pair("deep_research_analyze")

_report_sys, _report_human = load_prompt_pair("deep_research_report")


# ──────────────────────────────────────────────────────────────────────────────
# DeepResearchAgent
# ──────────────────────────────────────────────────────────────────────────────

def _make_retriever(mode: RetrieverMode, top_k: int):
    """检索器工厂（与 QAAgent 保持一致）"""
    if mode == "hybrid":
        from src.retrieval.hybrid_retriever import HybridRetriever
        return HybridRetriever(top_k=top_k, semantic_top_k=top_k * 2, bm25_top_k=top_k * 2)
    if mode == "graph":
        from src.retrieval.graph_retriever import GraphRetriever
        return GraphRetriever(top_k=top_k, expand_entities=True)
    if mode == "local":
        from src.retrieval.local_retriever import LocalRetriever
        return LocalRetriever(top_k=top_k)
    if mode == "global":
        from src.retrieval.global_retriever import GlobalRetriever
        return GlobalRetriever(top_k=top_k)
    if mode == "mix":
        from src.retrieval.mix_retriever import MixRetriever
        return MixRetriever(top_k=top_k)
    from src.retrieval.retriever import SemanticRetriever
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
        检索模式：semantic / hybrid / graph / local / global / mix
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
        self.max_subquestions = int(max_subquestions)
        self.retriever_mode = retriever_mode
        self.use_community = use_community
        self.retriever = _make_retriever(retriever_mode, top_k)
        self._llm = get_native_llm(temperature=0.2)
        self._keyword_extractor = KeywordExtractor()

    # ── Step 1：规划 ──────────────────────────────────────────────────────────

    def _plan(self, question: str) -> list[str]:
        resp = self._llm.invoke([
            {"role": "system", "content": _plan_sys},
            {"role": "user",   "content": _plan_human.format(
                question=question, max_n=self.max_subquestions
            )},
        ])
        content = str(resp.get("content") or "").strip()

        # 优先尝试 JSON 格式（{"sub_questions": [...]}）
        try:
            data = extract_json(content)
            if isinstance(data, list):
                questions = [str(q) for q in data if q]
                return questions[: self.max_subquestions]
            if isinstance(data, dict):
                data.setdefault("sub_questions", [])
                result = SubQuestionList(**data)
                return result.sub_questions[: self.max_subquestions]
        except (ValueError, Exception):
            pass

        # 降级：解析纯文本编号列表（"1. xxx\n2. xxx" 或 "- xxx"）
        questions: list[str] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            # 去掉 "1." / "1)" / "-" / "*" 等前缀
            cleaned = re.sub(r"^[\d]+[.)、]\s*|^[-*•]\s*", "", line).strip()
            if cleaned:
                questions.append(cleaned)

        if questions:
            return questions[: self.max_subquestions]

        raise ValueError(f"无法从规划响应中提取子问题: {content[:200]!r}")

    # ── Step 2+3：检索 + 局部分析 ─────────────────────────────────────────────

    def _retrieve_and_analyze(self, sub_question: str) -> SubQuestionResult:
        kw = self._keyword_extractor.extract(sub_question)
        logger.debug(
            "子问题关键词 — ll: %s, hl: %s | 问题: %s",
            kw.ll_keywords, kw.hl_keywords, sub_question,
        )
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

        resp = self._llm.invoke([
            {"role": "system", "content": _analyze_sys},
            {"role": "user",   "content": _analyze_human.format(
                sub_question=sub_question, context=context
            )},
        ])
        token_usage = TokenUsage.from_native_response(resp, self._llm.model_name)
        return SubQuestionResult(
            question=sub_question,
            chunks=chunks,
            sources=sources,
            analysis=str(resp.get("content") or "").strip(),
            token_usage=token_usage,
        )

    # ── Step 4：社区视角（可选） ───────────────────────────────────────────────

    def _get_community_section(self) -> str:
        if not self.use_community:
            return ""
        try:
            from src.storage.graph_store import GraphStore
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

        resp = self._llm.invoke([
            {"role": "system", "content": _report_sys},
            {"role": "user",   "content": _report_human.format(
                question=question,
                sub_analyses=sub_analyses_text,
                community_section=community_prompt,
            )},
        ])
        token_usage = TokenUsage.from_native_response(resp, self._llm.model_name)
        return str(resp.get("content") or "").strip(), token_usage

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
