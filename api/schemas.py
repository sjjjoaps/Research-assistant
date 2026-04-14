"""
API 请求/响应 Pydantic 模型
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# 通用
# ──────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


# ──────────────────────────────────────────────────────────────────────────────
# 文献管理
# ──────────────────────────────────────────────────────────────────────────────

class DocumentItem(BaseModel):
    id: int
    file_path: str
    doc_id: str | None = None
    title: str
    authors: str
    institution: str | None
    year: int | None
    abstract: str | None
    keywords: str
    status: str | None = None          # 来自 DocumentStatusStore


class DocumentStatusResponse(BaseModel):
    doc_id: str
    file_path: str
    status: str
    current_step: str
    chunk_count: int
    entity_count: int
    relation_count: int
    error_message: str | None
    updated_at: str


class IngestFileRequest(BaseModel):
    file_path: str = Field(..., description="待入库文件的绝对路径")
    enable_entity_extraction: bool = Field(default=False, description="是否启用实体抽取")


class IngestDirectoryRequest(BaseModel):
    directory: str = Field(..., description="待入库目录的绝对路径")
    enable_entity_extraction: bool = Field(default=False, description="是否启用实体抽取")


class IngestFileResult(BaseModel):
    file_path: str
    doc_id: str
    skipped: bool = False
    reason: str | None = None          # 跳过时说明原因，如 "already_processed"
    record_id: int | None = None
    title: str = ""
    chunk_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    citation_count: int = 0


class DeleteDocumentResult(BaseModel):
    doc_id: str
    file_path: str
    deleted_vectors: int
    deleted_relation_vectors: int = 0
    deleted_chunks: int
    deleted_relations: int
    deleted_entities: int


class CitationItem(BaseModel):
    ref_id: str
    title: str
    authors: str
    year: str
    doi: str
    raw_text: str


class DocumentCitationsResponse(BaseModel):
    doc_id: str
    file_path: str
    citation_count: int
    citations: list[CitationItem]


# ──────────────────────────────────────────────────────────────────────────────
# 问答
# ──────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    thread_id: str = Field(default="default", description="会话 ID")
    top_k: int = Field(default=3, ge=1, le=20, description="检索 chunk 数量")
    max_history: int = Field(default=5, ge=1, le=20, description="保留历史轮数")
    retriever_mode: Literal["semantic", "hybrid", "graph"] = Field(
        default="hybrid", description="检索模式"
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    thread_id: str
    retriever_mode: str
    token_usage: dict | None
    ll_keywords: list[str] = Field(default_factory=list)
    hl_keywords: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# 深度研究
# ──────────────────────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    question: str = Field(..., description="研究问题")
    thread_id: str = Field(default="research", description="会话 ID")
    top_k: int = Field(default=5, ge=1, le=20, description="每个子问题检索 chunk 数量")
    max_subquestions: int = Field(default=4, ge=2, le=5, description="子问题上限")
    retriever_mode: Literal["semantic", "hybrid", "graph"] = Field(
        default="hybrid", description="检索模式"
    )
    use_community: bool = Field(default=False, description="是否附加社区视角")


class ResearchResponse(BaseModel):
    question: str
    sub_questions: list[str]
    final_report: str
    all_sources: list[str]
    community_insights: str
    total_token_usage: dict


# ──────────────────────────────────────────────────────────────────────────────
# Idea 生成
# ──────────────────────────────────────────────────────────────────────────────

class IdeaRequest(BaseModel):
    question: str = Field(..., description="原始研究问题")
    report_markdown: str = Field(..., description="研究报告 Markdown 文本")


class IdeaResponse(BaseModel):
    research_gaps: list[str]
    method_comparisons: list[str]
    suggested_directions: list[str]
    evidence_basis: str
    confidence_note: str
    markdown: str
