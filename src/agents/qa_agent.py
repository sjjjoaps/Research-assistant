"""
QAAgent
基于 BaseAgent 的具体实现：多轮上下文 + 可选检索模式 + RAG 回答

支持多种检索模式（通过 retriever_mode 参数切换）：
  - "semantic"  ：仅 FAISS 向量检索（默认）
  - "hybrid"    ：FAISS + BM25 融合检索（RRF）
  - "graph"     ：基于 Neo4j 实体匹配的图检索
  - "local"     ：图检索 + 语义检索，适合具体问题
  - "global"    ：关系索引 + 图关系，适合宏观问题
  - "mix"       ：semantic + local + global 综合融合

Phase 4.1 变更：
- 引入 KeywordExtractor，run_turn 返回值新增 ll_keywords / hl_keywords
"""
from __future__ import annotations

from typing import Literal

from src.agents.base_agent import BaseAgent, Turn
from src.agents.prompt_loader import load_prompt_pair
from src.infrastructure.llm_client import get_native_llm
from src.retrieval.retriever import RetrievedChunk, SemanticRetriever
from src.retrieval.keyword_extractor import KeywordExtractor
from src.infrastructure.token_tracker import TokenUsage

RetrieverMode = Literal["semantic", "hybrid", "graph", "local", "global", "mix"]


_qa_sys, _qa_human = load_prompt_pair("qa_agent")


def _make_retriever(mode: RetrieverMode, top_k: int):
    """工厂函数：按模式创建对应检索器"""
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
        from src.retrieval.lightrag_retriever import LightRAGDualRetriever
        return LightRAGDualRetriever(top_k=top_k)
    if mode == "mix":
        from src.retrieval.mix_retriever import MixRetriever
        return MixRetriever(top_k=top_k)
    # 默认 semantic
    return SemanticRetriever(top_k=top_k)


class QAAgent(BaseAgent):
    """
    多轮 RAG 问答 Agent。

    Parameters
    ----------
    top_k : int
        每轮检索返回的 chunk 数量
    max_history_turns : int
        携带到 prompt 的最大历史轮数
    retriever_mode : RetrieverMode
        检索模式："semantic" / "hybrid" / "graph" / "local" / "global" / "mix"
    """

    def __init__(
        self,
        top_k: int = 3,
        max_history_turns: int = 5,
        retriever_mode: RetrieverMode = "semantic",
    ) -> None:
        super().__init__(max_history_turns=max_history_turns)
        self.top_k = top_k
        self.retriever_mode = retriever_mode
        self.retriever = _make_retriever(retriever_mode, top_k)
        self._llm = get_native_llm(temperature=0.1)
        self._keyword_extractor = KeywordExtractor()

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
        context_blocks: list[str] = []
        sources: list[str] = []

        for i, chunk in enumerate(chunks, start=1):
            source = f"{chunk.file_path}#chunk-{chunk.chunk_index}"
            sources.append(source)
            context_blocks.append(
                f"[{i}] 来源: {source}\n"
                f"内容: {chunk.content}"
            )

        return "\n\n".join(context_blocks), sources

    @staticmethod
    def _build_history_text(history: list[Turn]) -> str:
        if not history:
            return "（无）"

        lines: list[str] = []
        for turn in history:
            role_cn = "用户" if turn.role == "user" else "助手"
            lines.append(f"{role_cn}: {turn.content}")
        return "\n".join(lines)

    def run_turn(self, thread_id: str, user_input: str) -> dict:
        self.append_user_message(thread_id, user_input)

        history = self.get_history(thread_id)
        history_text = self._build_history_text(history[:-1])

        kw = self._keyword_extractor.extract(user_input)
        chunks = self.retriever.retrieve(user_input)
        context, sources = self._build_context(chunks)

        if not context:
            answer = "未检索到相关内容，请先执行文献入库后再提问。"
            self.append_assistant_message(thread_id, answer, [])
            return {
                "answer": answer,
                "sources": [],
                "thread_id": thread_id,
                "retriever_mode": self.retriever_mode,
                "token_usage": None,
                "ll_keywords": kw.ll_keywords,
                "hl_keywords": kw.hl_keywords,
            }

        resp = self._llm.invoke([
            {"role": "system", "content": _qa_sys},
            {"role": "user",   "content": _qa_human.format(
                history=history_text, question=user_input, context=context
            )},
        ])
        token_usage = TokenUsage.from_native_response(resp, self._llm.model_name)

        answer = str(resp.get("content") or "").strip()
        self.append_assistant_message(thread_id, answer, sources)

        return {
            "answer": answer,
            "sources": sources,
            "thread_id": thread_id,
            "retriever_mode": self.retriever_mode,
            "token_usage": token_usage,
            "ll_keywords": kw.ll_keywords,
            "hl_keywords": kw.hl_keywords,
        }
