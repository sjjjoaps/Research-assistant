"""
QAAgent
基于 BaseAgent 的具体实现：多轮上下文 + 可选检索模式 + RAG 回答

支持三种检索模式（通过 retriever_mode 参数切换）：
  - "semantic"  ：仅 FAISS 向量检索（默认）
  - "hybrid"    ：FAISS + BM25 融合检索（RRF）
  - "graph"     ：基于 Neo4j 实体匹配的图检索
"""
from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate

from src.agents.base_agent import BaseAgent, Turn
from src.llm_client import get_llm
from src.retriever import RetrievedChunk, SemanticRetriever
from src.token_tracker import TokenUsage

RetrieverMode = Literal["semantic", "hybrid", "graph"]


def _load_prompt(filename: str) -> str:
    return open(f"prompt/{filename}", "r", encoding="utf-8").read()


_QA_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _load_prompt("qa_agent_system.md")),
        ("human", _load_prompt("qa_agent_human.md")),
    ]
)


def _make_retriever(mode: RetrieverMode, top_k: int):
    """工厂函数：按模式创建对应检索器"""
    if mode == "hybrid":
        from src.hybrid_retriever import HybridRetriever
        return HybridRetriever(top_k=top_k, semantic_top_k=top_k * 2, bm25_top_k=top_k * 2)
    if mode == "graph":
        from src.graph_retriever import GraphRetriever
        return GraphRetriever(top_k=top_k, expand_entities=True)
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
        检索模式："semantic" / "hybrid" / "graph"
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
        self.llm = get_llm(temperature=0.1)
        self.chain = _QA_AGENT_PROMPT | self.llm

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
        history_text = self._build_history_text(history[:-1])  # 不把当前 user_input 重复写入 history

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
            }

        message = self.chain.invoke(
            {
                "history": history_text,
                "question": user_input,
                "context": context,
            }
        )
        token_usage = TokenUsage.from_langchain_message(message, self.llm.model_name)

        answer = str(message.content)
        self.append_assistant_message(thread_id, answer, sources)

        return {
            "answer": answer,
            "sources": sources,
            "thread_id": thread_id,
            "retriever_mode": self.retriever_mode,
            "token_usage": token_usage,
        }
