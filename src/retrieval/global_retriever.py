"""
Global 检索器（备用路径）

定位宏观主题、趋势、关系结构问题，优先融合：
- RelationVectorStore：关系级向量召回
- GraphRetriever：图中实体关联补充

注意：此检索器现已作为备用路径保留。
主路径已切换为 LightRAGDualRetriever（实现 LightRAG §3.2 双极检索范式），
仅在 LightRAGDualRetriever 不可用时由 MixRetriever 和 tool_registry 降级使用。

与 LightRAGDualRetriever 的主要区别：
- 使用 RelationVectorStore（独立关系向量索引）而非 GraphStore.search_by_relations()
- 无双极关键词分级提取，不做 one-hop 邻居扩展
- year 从 MetadataDatabase 缓存获取，而非 Graph JOIN 回查
"""
from __future__ import annotations

from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.fusion import fuse_ranked_lists
from src.retrieval.keyword_extractor import KeywordExtractor
from src.retrieval.retriever import RetrievedChunk
from src.storage.relation_vector_store import RelationVectorStore


class GlobalRetriever:
    def __init__(self, top_k: int = 5) -> None:
        self.top_k = int(top_k)
        self.relation_vector_store = RelationVectorStore()
        self.relation_vector_store.load()
        self.graph_retriever = GraphRetriever(top_k=top_k * 2, expand_entities=True)
        self.keyword_extractor = KeywordExtractor()
        # Phase 9-2: file_path → year 缓存，懒加载，供关系 chunk 携带年份
        self._year_cache: dict[str, int | None] | None = None

    def _get_year_cache(self) -> dict[str, int | None]:
        """懒加载 file_path→year 映射，从 MetadataDatabase 读取一次后缓存。"""
        if self._year_cache is not None:
            return self._year_cache
        try:
            from src.storage.database import MetadataDatabase
            db = MetadataDatabase()
            docs = db.list_documents()
            self._year_cache = {
                d.file_path: d.year
                for d in docs
                if d.file_path
            }
        except Exception:
            self._year_cache = {}
        return self._year_cache

    def _relation_docs_to_chunks(self, query: str) -> list[RetrievedChunk]:
        keywords = self.keyword_extractor.extract(query)
        relation_query = " ".join(keywords.hl_keywords or keywords.ll_keywords or [query])
        docs = self.relation_vector_store.similarity_search(relation_query, k=self.top_k * 2)

        # Phase 9-2: 构建年份缓存，使关系 chunk 能携带 year
        year_cache = self._get_year_cache()

        chunks: list[RetrievedChunk] = []
        for doc in docs:
            metadata = doc.metadata
            file_path = str(metadata.get("file_path", "relation_index"))
            content = (
                f"[关系检索]\n"
                f"来源实体: {metadata.get('source_name', '')}\n"
                f"关系类型: {metadata.get('relation_type', '')}\n"
                f"目标实体: {metadata.get('target_name', '')}\n"
                f"关系描述: {metadata.get('description', '')}\n"
            ).strip()
            chunks.append(
                RetrievedChunk(
                    content=content,
                    file_path=file_path,
                    chunk_index=-1,
                    section_type="relation",
                    year=year_cache.get(file_path),   # 可能为 None（旧索引无 year）
                )
            )
        return chunks

    def retrieve(self, query: str, section_filter: str = "") -> list[RetrievedChunk]:
        """
        Args:
            query:          用户查询字符串。
            section_filter: 章节类型过滤（安全兜底）。
                            GlobalRetriever 结果的 section_type 固定为 "relation"/"entity"，
                            与文本章节类型不一致，过滤后通常为空。
                            有 section_filter 时先扩大融合池再过滤，保证候选充分。
        """
        relation_results = self._relation_docs_to_chunks(query)
        graph_results = self.graph_retriever.retrieve(query)

        # section_filter 时扩大融合候选池，保证后过滤有足够候选再裁至 top_k
        fuse_k = self.top_k * 3 if section_filter else self.top_k
        fused = fuse_ranked_lists([relation_results, graph_results], top_k=fuse_k)

        if section_filter:
            fused = [c for c in fused if getattr(c, "section_type", "") == section_filter]

        return fused[: self.top_k]

    def close(self) -> None:
        self.graph_retriever.close()
