"""
Global 检索器

定位宏观主题、趋势、关系结构问题，优先融合：
- RelationVectorStore：关系级向量召回
- GraphRetriever：图中实体关联补充

Phase 9-2 新增：
- _build_year_cache()：在首次检索时从 SQLite 构建 file_path→year 缓存，
  使关系 chunk 能携带 year 参与时间感知过滤

Phase 10-2：retrieve() 新增 section_filter 参数（安全兜底）。
注意：GlobalRetriever 的输出均为 section_type="relation"/"entity"，
与文本章节类型（method/abstract 等）语义不一致。
因此 global 模式与 section_filter 组合通常意义不大，过滤后结果大概率为空。
如需按章节过滤，建议改用 local/semantic 模式。
"""
from __future__ import annotations

from src.graph_retriever import GraphRetriever
from src.retrieval.fusion import fuse_ranked_lists
from src.retrieval.keyword_extractor import KeywordExtractor
from src.retriever import RetrievedChunk
from src.storage.relation_vector_store import RelationVectorStore


class GlobalRetriever:
    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
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
            from src.database import MetadataDatabase
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
