"""
图检索模块
通过关键词匹配 Neo4j 中的实体节点，再沿 MENTIONS 关系回溯到 Chunk，
收集 1-hop 邻居 Entity 的描述，拼装成 RetrievedChunk 返回。

检索流程：
  query → KeywordExtractor 提取 ll+hl 关键词 → 匹配 Entity.name → 找到关联 Chunk
        → 扩展 1-hop 邻居 Entity → 构建增强上下文

Phase 4.1 变更：
- 引入 KeywordExtractor 替换朴素切词，优先使用 ll_keywords，兜底 hl_keywords
- _extract_keywords 由静态方法改为实例方法
"""
import re

from src.config import settings
from src.retriever import RetrievedChunk
from src.retrieval.keyword_extractor import KeywordExtractor

from neo4j import GraphDatabase


class GraphRetriever:
    """
    基于 Neo4j 图数据的实体感知检索器。

    Parameters
    ----------
    top_k : int
        最多返回的 chunk 数量
    expand_entities : bool
        是否将匹配实体的 1-hop 邻居实体名称追加到 chunk 内容末尾
    """

    def __init__(self, top_k: int = 5, expand_entities: bool = True) -> None:
        self.top_k = top_k
        self.expand_entities = expand_entities
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        self._keyword_extractor = KeywordExtractor()

    def close(self) -> None:
        self._driver.close()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _extract_keywords(self, text: str) -> list[str]:
        """使用 KeywordExtractor 提取查询关键词，兜底朴素切词。

        优先使用 ll_keywords（实体/方法名），再追加 hl_keywords（主题词）。
        若两者均为空则退化为原有朴素切词，保证鲁棒性。
        """
        result = self._keyword_extractor.extract(text)
        keywords = result.ll_keywords + result.hl_keywords
        if not keywords:
            # 兜底：原有朴素切词
            tokens = [t.strip() for t in re.split(r"[\s,，。！？!?、\-/()（）]+", text) if t.strip()]
            keywords = [t for t in tokens if len(t) > 1]
        return keywords

    def _find_matching_chunks(self, keywords: list[str]) -> list[dict]:
        """
        对每个关键词做大小写不敏感的实体名模糊匹配，
        返回匹配到的 chunk 信息列表（去重，按匹配次数排序）。

        Returns list of dicts:
          {chunk_id, content, chunk_index, file_path, matched_entity_names,
           matched_entity_ids, year, hit_count}
        """
        if not keywords:
            return []

        # Phase 9-2: 通过 Document -[:HAS_CHUNK]-> Chunk 反查年份
        query = """
        UNWIND $keywords AS kw
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower(kw)
        WITH e
        MATCH (c:Chunk)-[:MENTIONS]->(e)
        OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c)
        RETURN
            c.id          AS chunk_id,
            c.content     AS content,
            c.chunk_index AS chunk_index,
            c.file_path   AS file_path,
            collect(e.name) AS matched_entity_names,
            collect(e.id)   AS matched_entity_ids,
            d.year          AS year,
            count(e)      AS hit_count
        ORDER BY hit_count DESC
        LIMIT $limit
        """
        with self._driver.session() as session:
            result = session.run(query, keywords=keywords, limit=self.top_k * 3)
            return [dict(record) for record in result]

    def _get_neighbor_entities(self, entity_name: str) -> list[str]:
        """返回与指定实体名相连的所有 1-hop 邻居实体名称列表"""
        query = """
        MATCH (e:Entity {name: $name})-[:RELATES_TO]-(neighbor:Entity)
        RETURN neighbor.name AS name
        LIMIT 10
        """
        with self._driver.session() as session:
            result = session.run(query, name=entity_name)
            return [record["name"] for record in result if record["name"]]

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        keywords = self._extract_keywords(query)
        if not keywords:
            return []

        raw_chunks = self._find_matching_chunks(keywords)
        if not raw_chunks:
            return []

        # 去重（同一 chunk_id 只保留一次），保持排序
        seen: set[str] = set()
        deduped: list[dict] = []
        for row in raw_chunks:
            cid = row["chunk_id"]
            if cid not in seen:
                seen.add(cid)
                deduped.append(row)

        results: list[RetrievedChunk] = []
        for row in deduped[: self.top_k]:
            content = row["content"] or ""

            # 可选：追加 1-hop 邻居实体名称，丰富上下文
            if self.expand_entities:
                neighbor_names: list[str] = []
                for entity_name in (row.get("matched_entity_names") or []):
                    neighbors = self._get_neighbor_entities(entity_name)
                    neighbor_names.extend(neighbors)

                if neighbor_names:
                    unique_neighbors = list(dict.fromkeys(neighbor_names))  # 保序去重
                    content = content + f"\n[相关实体: {', '.join(unique_neighbors)}]"

            # 取第一个匹配实体 ID，用于 LightRAG one-hop 扩展（entity_id 链路）
            matched_ids: list = row.get("matched_entity_ids") or []
            primary_entity_id = matched_ids[0] if matched_ids else ""

            # Phase 9-2: 从 Document 节点携带的 year 字段提取年份
            raw_year = row.get("year")
            try:
                year: int | None = int(raw_year) if raw_year is not None else None
            except (ValueError, TypeError):
                year = None

            results.append(
                RetrievedChunk(
                    content=content,
                    file_path=str(row.get("file_path") or "unknown"),
                    chunk_index=int(row.get("chunk_index") or -1),
                    entity_id=str(primary_entity_id),
                    year=year,
                )
            )

        return results
