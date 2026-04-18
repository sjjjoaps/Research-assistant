"""
基础 RAG 问答链
流程：检索 -> 拼接上下文 -> LLM 生成 -> 返回答案与来源
"""
from langchain_core.prompts import ChatPromptTemplate

from src.agents.prompt_loader import load_prompt_pair
from src.infrastructure.llm_client import get_llm
from src.retrieval.retriever import RetrievedChunk, SemanticRetriever


_qa_sys, _qa_human = load_prompt_pair("qa_chain")
_QA_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _qa_sys), ("human", _qa_human)]
)


class QAChain:
    def __init__(self, top_k: int = 3) -> None:
        self.retriever = SemanticRetriever(top_k=top_k)
        self.llm = get_llm(temperature=0.1)
        self.chain = _QA_PROMPT | self.llm

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

    def ask(self, question: str) -> dict:
        chunks = self.retriever.retrieve(question)
        context, sources = self._build_context(chunks)

        if not context:
            return {
                "answer": "未检索到相关内容，请先执行文献入库后再提问。",
                "sources": [],
            }

        message = self.chain.invoke({"question": question, "context": context})
        return {
            "answer": message.content,
            "sources": sources,
        }
