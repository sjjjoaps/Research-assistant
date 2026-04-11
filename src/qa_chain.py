"""
基础 RAG 问答链
流程：检索 -> 拼接上下文 -> LLM 生成 -> 返回答案与来源
"""
from langchain_core.prompts import ChatPromptTemplate

from src.llm_client import get_llm
from src.retriever import RetrievedChunk, SemanticRetriever


_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是学术文献问答助手。请严格基于给定上下文回答问题，不要编造。"
            "如果上下文不足，请明确说明。"
            "回答后给出来源编号（如 [1], [2]）。",
        ),
        (
            "human",
            "问题：{question}\n\n"
            "上下文：\n{context}\n\n"
            "请输出：\n"
            "1) 简洁答案\n"
            "2) 引用来源编号",
        ),
    ]
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
