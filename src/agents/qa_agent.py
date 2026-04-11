"""
QAAgent
基于 BaseAgent 的首个具体实现：多轮上下文 + 语义检索 + RAG 回答
"""
from langchain_core.prompts import ChatPromptTemplate

from src.agents.base_agent import BaseAgent, Turn
from src.llm_client import get_llm
from src.retriever import RetrievedChunk, SemanticRetriever


_QA_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是学术文献问答助手。请严格基于给定上下文回答问题，不要编造。"
            "如果上下文不足，请明确说明。"
            "回答要简洁清晰，并在结尾标注来源编号（如 [1], [2]）。",
        ),
        (
            "human",
            "历史对话：\n{history}\n\n"
            "当前问题：{question}\n\n"
            "检索上下文：\n{context}\n\n"
            "请输出：\n"
            "1) 回答\n"
            "2) 引用来源编号",
        ),
    ]
)


class QAAgent(BaseAgent):
    def __init__(self, top_k: int = 3, max_history_turns: int = 5) -> None:
        super().__init__(max_history_turns=max_history_turns)
        self.top_k = top_k
        self.retriever = SemanticRetriever(top_k=top_k)
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
            }

        message = self.chain.invoke(
            {
                "history": history_text,
                "question": user_input,
                "context": context,
            }
        )

        answer = str(message.content)
        self.append_assistant_message(thread_id, answer, sources)

        return {
            "answer": answer,
            "sources": sources,
            "thread_id": thread_id,
        }
