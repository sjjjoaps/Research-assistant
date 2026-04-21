"""
Embedding 封装模块（兼容转发层）
实际实现已迁移至 src/infrastructure/embedder.py（P2-Step 2）
"""
from langchain_community.embeddings import DashScopeEmbeddings

from src.infrastructure.config import settings


class Embedder:
    """对外暴露统一的 embed_texts / embed_query 接口"""

    def __init__(self) -> None:
        self._model = DashScopeEmbeddings(
            model=settings.embedding_model_name,
            dashscope_api_key=settings.api_key,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量 embedding，返回与输入等长的向量列表"""
        if not texts:
            return []
        return self._model.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        """单条查询 embedding"""
        return self._model.embed_query(query)

    # 将自身暴露为 LangChain Embeddings 接口，供 FAISS 直接使用
    @property
    def langchain_embeddings(self) -> DashScopeEmbeddings:
        return self._model
