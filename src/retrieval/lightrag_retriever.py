"""
LightRAG 双极检索器（Phase 9-1）

实现 LightRAG §3.2 Dual-Level Retrieval Paradigm：

    查询类型        检索级别        关键词类型        检索来源
    ──────────    ──────────    ──────────────    ────────────────────────
    Specific       Low-Level      ll_keywords       实体向量匹配
    Abstract       High-Level     hl_keywords       关系向量匹配
    Both           Dual（双路）   ll + hl           Low-Level + High-Level
    Neither        Mix            —                 fallback 到 MixRetriever

三步流程（对应论文 §3.2）：
    (i)  双极关键词提取：KeywordExtractor → ll_keywords + hl_keywords
    (ii) 双路向量匹配：
         - Low-Level  (ll_keywords) → GraphRetriever（实体邻域精确召回）
         - High-Level (hl_keywords) → GraphStore.search_by_relations()（关系描述语义召回）
    (iii) 高阶关联扩展 (one-hop neighbor expansion)：
         从 (ii) 获得的实体 ID 集合出发，调用 GraphStore.get_one_hop_neighbors()
         补充 {v_i | v_i ∈ V ∧ (v_i ∈ N_v ∨ v_i ∈ N_e)}

自动路由（由 _auto_select_mode() 调用）：
    has_ll ∧ has_hl → "dual"   双极检索
    has_ll only     → "local"  Local 检索（实体精确）
    has_hl only     → "global" Global 检索（关系宏观）
    neither         → "mix"    Mix 检索（默认综合）
"""
from __future__ import annotations

import logging

from src.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


class LightRAGDualRetriever:
    """
    LightRAG 双极检索器：Low-Level + High-Level 双路召回 + one-hop 扩展。

    设计原则：
    - 懒加载所有外部依赖（GraphStore、GraphRetriever、KeywordExtractor），
      避免构造时触发 Neo4j / FAISS 连接。
    - 复用已有的 GraphRetriever 作为 Low-Level 实体召回（避免重复逻辑）。
    - High-Level 召回通过 GraphStore.search_by_relations() 直接查询关系描述。
    - one-hop 扩展通过 GraphStore.get_one_hop_neighbors() 实现。
    - 结果去重：以 (file_path, chunk_index) 或 entity_id 为主键去重，
      保留 Low-Level > High-Level > Expanded 的优先级顺序。
    """

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self._keyword_extractor = None
        self._graph_retriever = None
        self._graph_store = None

    def _get_keyword_extractor(self):
        if self._keyword_extractor is None:
            from src.retrieval.keyword_extractor import KeywordExtractor
            self._keyword_extractor = KeywordExtractor()
        return self._keyword_extractor

    def _get_graph_retriever(self):
        if self._graph_retriever is None:
            from src.graph_retriever import GraphRetriever
            self._graph_retriever = GraphRetriever(
                top_k=self.top_k * 2, expand_entities=True
            )
        return self._graph_retriever

    def _get_graph_store(self):
        if self._graph_store is None:
            from src.graph_store import GraphStore
            self._graph_store = GraphStore()
        return self._graph_store

    # ── 主检索入口 ───────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        执行 LightRAG 双极检索，返回去重后的 RetrievedChunk 列表。

        Args:
            query:  用户查询字符串。
            top_k:  返回上限（None 时使用初始化时的 self.top_k）。

        Returns:
            去重后的 RetrievedChunk 列表，长度 ≤ top_k。
        """
        k = top_k if top_k is not None else self.top_k

        # (i) 双极关键词提取
        try:
            extractor = self._get_keyword_extractor()
            kw = extractor.extract(query)
            ll_kws = kw.ll_keywords   # 低层：具体实体/方法关键词
            hl_kws = kw.hl_keywords   # 高层：宏观主题/趋势关键词
        except Exception as exc:
            logger.warning("LightRAGDualRetriever 关键词提取失败，fallback 到空列表: %s", exc)
            ll_kws, hl_kws = [], []

        logger.debug("LightRAG 双极关键词 ll=%s hl=%s", ll_kws, hl_kws)

        # (ii-a) Low-Level 检索：GraphRetriever 以原始 query 召回实体邻域 chunk
        low_chunks: list[RetrievedChunk] = []
        if ll_kws:
            try:
                gr = self._get_graph_retriever()
                low_chunks = gr.retrieve(query)
            except Exception as exc:
                logger.warning("LightRAG Low-Level 检索失败: %s", exc)

        # (ii-b) High-Level 检索：关系描述向量匹配，转换为 RetrievedChunk
        high_chunks: list[RetrievedChunk] = []
        if hl_kws:
            try:
                gs = self._get_graph_store()
                relations = gs.search_by_relations(keywords=hl_kws, k=k * 2)
                high_chunks = self._relations_to_chunks(relations)
            except Exception as exc:
                logger.warning("LightRAG High-Level 检索失败: %s", exc)

        # (iii) one-hop 邻居扩展
        expanded_chunks: list[RetrievedChunk] = []
        try:
            entity_ids = self._collect_entity_ids(low_chunks + high_chunks)
            if entity_ids:
                gs = self._get_graph_store()
                neighbors = gs.get_one_hop_neighbors(entity_ids)
                expanded_chunks = self._neighbors_to_chunks(neighbors)
        except Exception as exc:
            logger.warning("LightRAG one-hop 扩展失败: %s", exc)

        return self._merge_and_deduplicate(low_chunks, high_chunks, expanded_chunks, k)

    # ── 辅助方法 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _relations_to_chunks(relations: list[dict]) -> list[RetrievedChunk]:
        """
        将 GraphStore.search_by_relations() 的返回结果转为 RetrievedChunk。

        每条关系产生一个 chunk，content 格式与 GlobalRetriever 一致。
        entity_id 字段同时记录 source_id 和 target_id（以 "src::tgt" 分隔），
        供 _collect_entity_ids() 拆分出两端实体，确保 one-hop 扩展覆盖关系的两端。

        Phase 9-2: 从 relation row 的 year 字段提取年份（由 search_by_relations 通过
        Chunk→Document JOIN 回查），使 High-Level 结果能参与时间感知过滤。
        """
        chunks: list[RetrievedChunk] = []
        for rel in relations:
            source_id = rel.get("source_id", "") or ""
            target_id = rel.get("target_id", "") or ""
            content = (
                f"[关系检索]\n"
                f"来源实体: {rel.get('source_name', '')}\n"
                f"关系类型: {rel.get('relation_type', '')}\n"
                f"目标实体: {rel.get('target_name', '')}\n"
                f"关系描述: {rel.get('description', '')}\n"
            ).strip()
            # entity_id 存储 "source_id::target_id"，_collect_entity_ids() 会拆分
            combined_id = f"{source_id}::{target_id}" if source_id and target_id else (source_id or target_id)
            # 提取年份（search_by_relations 通过 Document JOIN 回查，可能为 None）
            raw_year = rel.get("year")
            try:
                year: int | None = int(raw_year) if raw_year is not None else None
            except (ValueError, TypeError):
                year = None
            chunks.append(
                RetrievedChunk(
                    content=content,
                    file_path="relation_index",
                    chunk_index=-1,
                    section_type="relation",
                    entity_id=combined_id,
                    year=year,
                )
            )
        return chunks

    @staticmethod
    def _neighbors_to_chunks(neighbors: list[dict]) -> list[RetrievedChunk]:
        """
        将 GraphStore.get_one_hop_neighbors() 的返回结果转为 RetrievedChunk。
        content 包含邻居实体的描述信息，用于补充上下文。
        """
        chunks: list[RetrievedChunk] = []
        for nb in neighbors:
            name = nb.get("name", "")
            desc = nb.get("description", "")
            rel_type = nb.get("relation_type", "")
            if not (name or desc):
                continue
            content = (
                f"[邻居实体]\n"
                f"实体: {name}\n"
                + (f"关联关系: {rel_type}\n" if rel_type else "")
                + (f"描述: {desc}" if desc else "")
            ).strip()
            chunks.append(
                RetrievedChunk(
                    content=content,
                    file_path="graph_neighbor",
                    chunk_index=-1,
                    section_type="entity",
                    entity_id=nb.get("entity_id", ""),
                )
            )
        return chunks

    @staticmethod
    def _collect_entity_ids(chunks: list[RetrievedChunk]) -> set[str]:
        """
        从 chunks 中收集非空 entity_id，用于 one-hop 扩展查询。

        支持两种格式：
        - 普通 entity_id（GraphRetriever 产生的 Low-Level chunk）
        - "source_id::target_id"（_relations_to_chunks 产生的 High-Level chunk）
          → 拆分后两端实体 ID 均纳入扩展集合，确保关系两端都被覆盖
        """
        ids: set[str] = set()
        for c in chunks:
            eid = getattr(c, "entity_id", "") or ""
            if not eid:
                continue
            if "::" in eid:
                for part in eid.split("::", 1):
                    if part:
                        ids.add(part)
            else:
                ids.add(eid)
        return ids

    @staticmethod
    def _merge_and_deduplicate(
        low: list[RetrievedChunk],
        high: list[RetrievedChunk],
        expanded: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        合并三路结果并去重，优先级：Low-Level > High-Level > Expanded。

        去重主键：(file_path, chunk_index)，relation/entity chunk 则以 entity_id 为键。
        """
        seen: set[tuple] = set()
        merged: list[RetrievedChunk] = []

        for chunk in low + high + expanded:
            fp = getattr(chunk, "file_path", "")
            ci = getattr(chunk, "chunk_index", -1)
            eid = getattr(chunk, "entity_id", "")

            # relation / entity chunk 以 entity_id 去重；普通 chunk 以 (fp, ci) 去重
            if fp in ("relation_index", "graph_neighbor") and eid:
                key: tuple = ("entity", eid)
            else:
                key = (fp, ci)

            if key in seen:
                continue
            seen.add(key)
            merged.append(chunk)

            if len(merged) >= top_k:
                break

        return merged

    def close(self) -> None:
        """释放 GraphRetriever 和 GraphStore 持有的连接。"""
        if self._graph_retriever is not None:
            try:
                self._graph_retriever.close()
            except Exception:
                pass
        if self._graph_store is not None:
            try:
                self._graph_store.close()
            except Exception:
                pass
