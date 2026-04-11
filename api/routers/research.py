"""
研究路由
POST /research — 深度研究（Plan-Execute-Report）
POST /idea     — 基于研究报告 Markdown 生成 Idea 报告
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import IdeaRequest, IdeaResponse, ResearchRequest, ResearchResponse
from src.agents.deep_research_agent import DeepResearchAgent
from src.agents.idea_agent import IdeaAgent

router = APIRouter(tags=["research"])

# 进程内单例缓存：(top_k, max_subquestions, retriever_mode, use_community) -> DeepResearchAgent
_research_agents: dict[tuple, DeepResearchAgent] = {}


def _get_research_agent(
    top_k: int,
    max_subquestions: int,
    retriever_mode: str,
    use_community: bool,
) -> DeepResearchAgent:
    key = (top_k, max_subquestions, retriever_mode, use_community)
    if key not in _research_agents:
        _research_agents[key] = DeepResearchAgent(
            top_k=top_k,
            max_subquestions=max_subquestions,
            retriever_mode=retriever_mode,
            use_community=use_community,
        )
    return _research_agents[key]


@router.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    """执行深度研究流程，返回子问题列表与完整 Markdown 研究报告"""
    agent = _get_research_agent(
        req.top_k, req.max_subquestions, req.retriever_mode, req.use_community
    )
    try:
        report = agent.research(thread_id=req.thread_id, question=req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ResearchResponse(
        question=report.question,
        sub_questions=report.sub_questions,
        final_report=report.final_report,
        all_sources=report.all_sources,
        community_insights=report.community_insights,
        total_token_usage=report.total_token_usage,
    )


@router.post("/idea", response_model=IdeaResponse)
def generate_idea(req: IdeaRequest):
    """基于研究报告 Markdown 生成结构化 Idea 报告"""
    try:
        agent = IdeaAgent()
        idea = agent.generate_from_markdown(req.question, req.report_markdown)
        markdown = IdeaAgent.to_markdown(idea)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return IdeaResponse(
        research_gaps=idea.research_gaps,
        method_comparisons=idea.method_comparisons,
        suggested_directions=idea.suggested_directions,
        evidence_basis=idea.evidence_basis,
        confidence_note=idea.confidence_note,
        markdown=markdown,
    )
