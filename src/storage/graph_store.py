"""
Neo4j 图存储模块
负责创建约束、写入 Document/Chunk/Entity 节点及关系

Phase 2.2 新增：
- upsert_entity_with_merge  — 写入实体时合并描述，不覆盖
- upsert_relation_with_merge — 写入关系时合并描述，不覆盖

Phase 2.3 新增：
- delete_document_chunks     — 删除文档的 Chunk 节点及 MENTIONS 关系
- get_orphan_entity_ids      — 查找无 MENTIONS 来源的孤立实体
- delete_entities_by_ids     — 批量删除实体节点
- reset_document_entity_extracted — 重置实体抽取标志，允许重新抽取

Phase 2.5 新增：
- _entity_lock / _relation_lock — 细粒度进程内锁，防止并发写入同一实体/关系时描述丢失

Phase 3.3 新增：
- create_reference_node      — 创建/更新 Reference 占位节点
- create_cites_relation      — 创建 Document -[:CITES]-> Reference 关系
- get_document_citations     — 查询文档的所有引用
- delete_document_citations  — 删除文档的所有 CITES 关系（Reference 节点保留）

Phase 5.2 新增：
- get_graph_stats            — 查询图谱节点/关系统计
- get_subgraph               — 查询可视化子图数据

Phase 9-1 新增（LightRAG 双极检索支持）：
- search_by_relations        — 按 hl_keywords 匹配关系描述（High-Level 召回）
- get_one_hop_neighbors      — 获取给定实体集合的一跳邻居（one-hop 扩展）
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from neo4j import GraphDatabase

from src.ingestion.chunker import TextChunk
from src.infrastructure.config import settings
from src.ingestion.metadata_extractor import DocumentMetadata

if TYPE_CHECKING:
    from src.ingestion.description_merger import DescriptionMerger


class GraphStore:
    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        # 细粒度进程内锁：key -> Lock
        # 同一实体/关系的并发 upsert 会串行化，不同实体互不阻塞
        # 已知 tradeoff：锁字典随入库实体/关系数量单调增长，不会自动回收。
        # 对于长期运行且实体量极大的服务，可考虑 LRU 淘汰策略；
        # 当前批量入库场景下内存占用可接受（每个 Lock 对象约 50 字节）。
        self._entity_locks: dict[str, threading.Lock] = {}
        self._relation_locks: dict[str, threading.Lock] = {}
        self._entity_locks_meta = threading.Lock()   # 保护 _entity_locks 字典本身
        self._relation_locks_meta = threading.Lock() # 保护 _relation_locks 字典本身

    def _get_entity_lock(self, entity_id: str) -> threading.Lock:
        with self._entity_locks_meta:
            if entity_id not in self._entity_locks:
                self._entity_locks[entity_id] = threading.Lock()
            return self._entity_locks[entity_id]

    def _get_relation_lock(self, relation_key: str) -> threading.Lock:
        with self._relation_locks_meta:
            if relation_key not in self._relation_locks:
                self._relation_locks[relation_key] = threading.Lock()
            return self._relation_locks[relation_key]

    def close(self) -> None:
        self.driver.close()

    def init_schema(self) -> None:
        queries = [
            "CREATE CONSTRAINT document_file_path_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.file_path IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE INDEX document_title_index IF NOT EXISTS FOR (d:Document) ON (d.title)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE CONSTRAINT community_id_unique IF NOT EXISTS FOR (cm:Community) REQUIRE cm.id IS UNIQUE",
            "CREATE CONSTRAINT reference_id_unique IF NOT EXISTS FOR (r:Reference) REQUIRE r.id IS UNIQUE",
        ]
        with self.driver.session() as session:
            for query in queries:
                session.run(query)

    def create_document_node(self, file_path: str, metadata: DocumentMetadata) -> None:
        query = """
        MERGE (d:Document {file_path: $file_path})
        SET d.title = $title,
            d.authors = $authors,
            d.institution = $institution,
            d.year = $year,
            d.abstract = $abstract,
            d.keywords = $keywords
        """
        with self.driver.session() as session:
            session.run(
                query,
                file_path=file_path,
                title=metadata.title,
                authors=metadata.authors,
                institution=metadata.institution,
                year=metadata.year,
                abstract=metadata.abstract,
                keywords=metadata.keywords,
            )

    def create_chunk_node(self, chunk: TextChunk) -> None:
        query = """
        MERGE (c:Chunk {id: $chunk_id})
        SET c.content = $content,
            c.chunk_index = $chunk_index,
            c.file_path = $file_path,
            c.content_type = $content_type,
            c.page_number = $page_number,
            c.position_hint = $position_hint,
            c.section_type = $section_type,
            c.section_title = $section_title
        """
        with self.driver.session() as session:
            session.run(
                query,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                file_path=chunk.metadata.get("file_path"),
                content_type=chunk.metadata.get("content_type", "text"),
                page_number=chunk.metadata.get("page_number"),
                position_hint=chunk.metadata.get("position_hint"),
                section_type=chunk.metadata.get("section_type", "unknown"),
                section_title=chunk.metadata.get("section_title", ""),
            )

    def create_has_chunk_relation(self, file_path: str, chunk_id: str) -> None:
        query = """
        MATCH (d:Document {file_path: $file_path})
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        with self.driver.session() as session:
            session.run(query, file_path=file_path, chunk_id=chunk_id)

    def create_entity_node(self, entity_id: str, name: str, entity_type: str, description: str | None) -> None:
        query = """
        MERGE (e:Entity {id: $entity_id})
        SET e.name = $name,
            e.type = $entity_type,
            e.description = $description
        """
        with self.driver.session() as session:
            session.run(
                query,
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
                description=description,
            )

    def upsert_entity_with_merge(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        description: str | None,
        merger: "DescriptionMerger",
    ) -> bool:
        """写入实体节点，合并描述而不是覆盖。

        读取已有 description_list，追加新描述后调用 merger 生成合并摘要，
        同时将原始列表和合并摘要一并写回 Neo4j。

        并发安全：对同一 entity_id 的写入通过进程内细粒度锁串行化，
        不同实体的写入互不阻塞。

        Args:
            entity_id: 实体唯一 ID。
            name: 实体名称。
            entity_type: 实体类型。
            description: 本次新增描述（可为 None）。
            merger: DescriptionMerger 实例。

        Returns:
            True 表示新建节点，False 表示更新已有节点。
        """
        lock = self._get_entity_lock(entity_id)
        with lock:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity {id: $id})
                    RETURN coalesce(e.description_list, []) AS dl,
                           coalesce(e.description, '') AS legacy_desc
                    """,
                    id=entity_id,
                ).single()
                is_new = result is None
                existing: list[str] = list(result["dl"]) if result else []
                # 兼容 Phase 2.2 之前写入的旧节点：description_list 为空但 description 有值时，
                # 将旧 description 作为第一条历史描述，避免合并时静默丢失。
                if result and not existing:
                    legacy = (result["legacy_desc"] or "").strip()
                    if legacy:
                        existing = [legacy]

                new_desc = (description or "").strip()
                all_descs = merger._deduplicate(existing + ([new_desc] if new_desc else []))
                merged = merger.merge(all_descs)

                session.run(
                    """
                    MERGE (e:Entity {id: $id})
                    SET e.name = $name,
                        e.type = $type,
                        e.description = $desc,
                        e.description_list = $dl
                    """,
                    id=entity_id,
                    name=name,
                    type=entity_type,
                    desc=merged,
                    dl=all_descs,
                )
                return is_new

    def upsert_relation_with_merge(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relation_type: str,
        description: str | None,
        merger: "DescriptionMerger",
    ) -> bool:
        """写入关系，合并描述而不是覆盖。

        并发安全：对同一 (source, relation_type, target) 三元组的写入通过
        进程内细粒度锁串行化，不同关系的写入互不阻塞。

        Args:
            source_entity_id: 起点实体 ID。
            target_entity_id: 终点实体 ID。
            relation_type: 关系类型。
            description: 本次新增描述（可为 None）。
            merger: DescriptionMerger 实例。

        Returns:
            True 表示新建关系，False 表示更新已有关系。
        """
        relation_key = f"{source_entity_id}::{relation_type}::{target_entity_id}"
        lock = self._get_relation_lock(relation_key)
        with lock:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (s:Entity {id: $sid})-[r:RELATES_TO {relation_type: $rt}]->(t:Entity {id: $tid})
                    RETURN coalesce(r.description_list, []) AS dl,
                           coalesce(r.description, '') AS legacy_desc
                    """,
                    sid=source_entity_id,
                    tid=target_entity_id,
                    rt=relation_type,
                ).single()
                is_new = result is None
                existing: list[str] = list(result["dl"]) if result else []
                # 兼容旧关系节点：description_list 为空但 description 有值时保留历史描述。
                if result and not existing:
                    legacy = (result["legacy_desc"] or "").strip()
                    if legacy:
                        existing = [legacy]

                new_desc = (description or "").strip()
                all_descs = merger._deduplicate(existing + ([new_desc] if new_desc else []))
                merged = merger.merge(all_descs)

                session.run(
                    """
                    MATCH (s:Entity {id: $sid})
                    MATCH (t:Entity {id: $tid})
                    MERGE (s)-[r:RELATES_TO {relation_type: $rt}]->(t)
                    SET r.description = $desc,
                        r.description_list = $dl
                    """,
                    sid=source_entity_id,
                    tid=target_entity_id,
                    rt=relation_type,
                    desc=merged,
                    dl=all_descs,
                )
                return is_new

    def create_mentions_relation(self, chunk_id: str, entity_id: str) -> None:
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MATCH (e:Entity {id: $entity_id})
        MERGE (c)-[:MENTIONS]->(e)
        """
        with self.driver.session() as session:
            session.run(query, chunk_id=chunk_id, entity_id=entity_id)

    def create_relates_to_relation(self, source_entity_id: str, target_entity_id: str, relation_type: str) -> None:
        query = """
        MATCH (s:Entity {id: $source_entity_id})
        MATCH (t:Entity {id: $target_entity_id})
        MERGE (s)-[r:RELATES_TO {relation_type: $relation_type}]->(t)
        """
        with self.driver.session() as session:
            session.run(
                query,
                source_entity_id=source_entity_id,
                target_entity_id=target_entity_id,
                relation_type=relation_type,
            )

    def is_document_entity_extracted(self, file_path: str) -> bool:
        query = """
        MATCH (d:Document {file_path: $file_path})
        RETURN coalesce(d.entity_extracted, false) AS entity_extracted
        """
        with self.driver.session() as session:
            result = session.run(query, file_path=file_path).single()
            if result is None:
                return False
            return bool(result["entity_extracted"])

    def mark_document_entity_extracted(self, file_path: str) -> None:
        query = """
        MATCH (d:Document {file_path: $file_path})
        SET d.entity_extracted = true
        """
        with self.driver.session() as session:
            session.run(query, file_path=file_path)

    def add_document_with_chunks(self, file_path: str, metadata: DocumentMetadata, chunks: list[TextChunk]) -> None:
        self.create_document_node(file_path, metadata)
        for chunk in chunks:
            self.create_chunk_node(chunk)
            self.create_has_chunk_relation(file_path, chunk.chunk_id)

    # ------------------------------------------------------------------
    # Phase 3.3 — Citation Graph
    # ------------------------------------------------------------------

    def create_reference_node(self, ref) -> None:
        """创建或更新 Reference 占位节点。

        Args:
            ref: CitationRecord 实例。
        """
        query = """
        MERGE (r:Reference {id: $ref_id})
        SET r.raw_text = $raw_text,
            r.title = $title,
            r.authors = $authors,
            r.year = $year,
            r.doi = $doi
        """
        with self.driver.session() as session:
            session.run(
                query,
                ref_id=ref.ref_id,
                raw_text=ref.raw_text,
                title=ref.title,
                authors=ref.authors,
                year=ref.year,
                doi=ref.doi,
            )

    def create_cites_relation(self, file_path: str, ref_id: str) -> None:
        """创建 Document -[:CITES]-> Reference 关系。"""
        query = """
        MATCH (d:Document {file_path: $file_path})
        MATCH (r:Reference {id: $ref_id})
        MERGE (d)-[:CITES]->(r)
        """
        with self.driver.session() as session:
            session.run(query, file_path=file_path, ref_id=ref_id)

    def get_document_citations(self, file_path: str) -> list[dict]:
        """查询文档的所有引用，返回 Reference 节点属性列表。"""
        query = """
        MATCH (d:Document {file_path: $file_path})-[:CITES]->(r:Reference)
        RETURN r.id AS ref_id, r.title AS title, r.authors AS authors,
               r.year AS year, r.doi AS doi, r.raw_text AS raw_text
        ORDER BY r.year DESC
        """
        with self.driver.session() as session:
            result = session.run(query, file_path=file_path)
            return [dict(record) for record in result]

    def delete_document_citations(self, file_path: str) -> int:
        """删除文档的所有 CITES 关系。

        Reference 节点本身保留，供其他文档共享引用。

        Returns:
            删除的关系数量。
        """
        query = """
        MATCH (d:Document {file_path: $file_path})-[c:CITES]->()
        WITH collect(c) AS rels, count(c) AS deleted
        FOREACH (r IN rels | DELETE r)
        RETURN deleted
        """
        with self.driver.session() as session:
            result = session.run(query, file_path=file_path)
            record = result.single()
            return record["deleted"] if record else 0

    # ------------------------------------------------------------------
    # Phase 12 — 社区检测支持
    # ------------------------------------------------------------------

    def get_entity_relations(self) -> list[dict]:
        """
        读取图中所有 Entity 节点间的 RELATES_TO 边（无向）。
        返回字段：source_id, target_id, source_name, target_name, source_desc, target_desc
        """
        query = """
        MATCH (s:Entity)-[:RELATES_TO]-(t:Entity)
        WHERE id(s) < id(t)
        RETURN s.id AS source_id, t.id AS target_id,
               s.name AS source_name, t.name AS target_name,
               s.description AS source_desc, t.description AS target_desc
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def create_community_node(self, community_id: str, label: int, summary: str, entity_count: int) -> None:
        """写入 Community 节点（幂等，MERGE），重复运行只更新摘要"""
        query = """
        MERGE (cm:Community {id: $community_id})
        SET cm.label = $label,
            cm.summary = $summary,
            cm.entity_count = $entity_count
        """
        with self.driver.session() as session:
            session.run(
                query,
                community_id=community_id,
                label=label,
                summary=summary,
                entity_count=entity_count,
            )

    def create_belongs_to_relation(self, entity_id: str, community_id: str) -> None:
        """写入 Entity-[:BELONGS_TO]->Community 关系"""
        query = """
        MATCH (e:Entity {id: $entity_id})
        MATCH (cm:Community {id: $community_id})
        MERGE (e)-[:BELONGS_TO]->(cm)
        """
        with self.driver.session() as session:
            session.run(query, entity_id=entity_id, community_id=community_id)

    def get_community_summaries(self, limit: int = 5) -> list[dict]:
        """按 entity_count 降序读取最大的若干 Community 摘要"""
        query = """
        MATCH (cm:Community)
        RETURN cm.id AS id,
               cm.label AS label,
               cm.summary AS summary,
               cm.entity_count AS entity_count
        ORDER BY cm.entity_count DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, limit=limit)
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Phase 5.2 — 图谱可视化支持
    # ------------------------------------------------------------------

    def get_graph_stats(self) -> dict:
        """返回图谱节点和关系统计。

        Returns:
            {
                "node_count": int,
                "relationship_count": int,
                "node_labels": [{"label": str, "count": int}],
                "relationship_types": [{"type": str, "count": int}],
            }
        """
        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            relationship_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            node_labels = [
                {"label": row["label"], "count": row["count"]}
                for row in session.run(
                    """
                    MATCH (n)
                    UNWIND labels(n) AS label
                    RETURN label, count(*) AS count
                    ORDER BY count DESC, label
                    """
                )
            ]
            relationship_types = [
                {"type": row["type"], "count": row["count"]}
                for row in session.run(
                    """
                    MATCH ()-[r]->()
                    RETURN type(r) AS type, count(*) AS count
                    ORDER BY count DESC, type
                    """
                )
            ]

        return {
            "node_count": int(node_count),
            "relationship_count": int(relationship_count),
            "node_labels": node_labels,
            "relationship_types": relationship_types,
        }

    def get_subgraph(
        self,
        limit: int = 100,
        node_types: list[str] | None = None,
        search: str | None = None,
    ) -> dict:
        """查询可视化子图。

        Args:
            limit: 最大节点数。
            node_types: 可选节点 label 过滤，如 ["Document", "Entity"]。
            search: 可选搜索词，匹配节点 name/title/id/file_path。

        Returns:
            {"nodes": [...], "edges": [...]}。
        """
        limit = max(1, min(int(limit), 500))
        node_types = [item for item in (node_types or []) if item]
        search_text = (search or "").strip().lower()

        query = """
        MATCH (n)
        WHERE ($node_types = [] OR any(label IN labels(n) WHERE label IN $node_types))
          AND (
            $search = ''
            OR toLower(coalesce(n.name, '')) CONTAINS $search
            OR toLower(coalesce(n.title, '')) CONTAINS $search
            OR toLower(coalesce(n.id, '')) CONTAINS $search
            OR toLower(coalesce(n.file_path, '')) CONTAINS $search
          )
        WITH n
        ORDER BY
          CASE
            WHEN 'Document' IN labels(n) THEN 0
            WHEN 'Entity' IN labels(n) THEN 1
            WHEN 'Community' IN labels(n) THEN 2
            WHEN 'Reference' IN labels(n) THEN 3
            ELSE 4
          END,
          coalesce(n.name, n.title, n.id, n.file_path, '')
        LIMIT $limit
        WITH collect(n) AS nodes
        UNWIND nodes AS n
        OPTIONAL MATCH (n)-[r]-(m)
        WHERE m IN nodes
        RETURN nodes, collect(DISTINCT {
            id: elementId(r),
            source: elementId(startNode(r)),
            target: elementId(endNode(r)),
            type: type(r),
            properties: properties(r)
        }) AS edges
        """
        with self.driver.session() as session:
            record = session.run(
                query,
                limit=limit,
                node_types=node_types,
                search=search_text,
            ).single()

        if record is None:
            return {"nodes": [], "edges": []}

        nodes = [self._format_visual_node(node) for node in record["nodes"]]
        edges = []
        seen_edges: set[str] = set()
        for edge in record["edges"]:
            edge_id = edge.get("id")
            if not edge_id or edge_id in seen_edges:
                continue
            seen_edges.add(edge_id)
            edges.append(
                {
                    "id": edge_id,
                    "source": edge.get("source"),
                    "target": edge.get("target"),
                    "type": edge.get("type"),
                    "label": edge.get("type"),
                    "properties": dict(edge.get("properties") or {}),
                }
            )

        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _format_visual_node(node) -> dict:
        labels = list(node.labels)
        properties = dict(node)
        primary_type = labels[0] if labels else "Node"
        label = (
            properties.get("name")
            or properties.get("title")
            or properties.get("id")
            or properties.get("file_path")
            or primary_type
        )
        return {
            "id": node.element_id,
            "label": str(label),
            "type": primary_type,
            "labels": labels,
            "properties": properties,
        }

    # ------------------------------------------------------------------
    # Phase 2.3 — 增量更新 / 精确删除支持
    # ------------------------------------------------------------------

    def delete_document_chunks(self, file_path: str) -> int:
        """删除文档的所有 Chunk 节点及其 MENTIONS 关系，返回删除数量。

        使用 DETACH DELETE 同时清除 HAS_CHUNK 和 MENTIONS 关系，
        实体节点本身不删除（由调用方决定是否清理孤立实体和孤立关系）。
        Document 节点本身不在此删除，由 delete_document_node() 负责。
        """
        query = """
        MATCH (d:Document {file_path: $file_path})-[:HAS_CHUNK]->(c:Chunk)
        WITH collect(c) AS chunks, size(collect(c)) AS cnt
        FOREACH (c IN chunks | DETACH DELETE c)
        RETURN cnt
        """
        with self.driver.session() as session:
            result = session.run(query, file_path=file_path).single()
            return int(result["cnt"]) if result else 0

    def delete_document_node(self, file_path: str) -> bool:
        """删除 Document 节点本身（及其所有残余关系），返回是否找到并删除。

        应在 Chunk 节点已清理后调用，确保 Document 节点不再有 HAS_CHUNK 关系。
        删除后同路径重新入库时不会复用旧节点的 entity_extracted 标志。
        """
        query = """
        MATCH (d:Document {file_path: $file_path})
        WITH d, count(d) AS cnt
        DETACH DELETE d
        RETURN cnt > 0 AS found
        """
        with self.driver.session() as session:
            result = session.run(query, file_path=file_path).single()
            return bool(result["found"]) if result else False

    def delete_stale_relations(self) -> int:
        """删除两端实体均存在但没有任何 Chunk 同时 MENTIONS 两端的 RELATES_TO 边。

        **语义说明（Phase 2.4 保守实现）**：
        判断依据是"共现"而非"关系来源"——只要另一篇文档的某个 Chunk 同时
        提及了两端实体，该关系就会被保留，即使该 Chunk 并未抽取出这条关系。
        这意味着此方法只能清理"完全无 Chunk 共现支撑"的关系，无法精确删除
        "仅由被删文档贡献、但两端实体仍被其他文档提及"的关系。

        若需要精确的关系来源追踪，需在后续阶段引入 relation_sources 表。
        """
        query = """
        MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
        WHERE NOT EXISTS {
            MATCH (c:Chunk)-[:MENTIONS]->(s)
            MATCH (c)-[:MENTIONS]->(t)
        }
        WITH collect(r) AS rels, size(collect(r)) AS cnt
        FOREACH (r IN rels | DELETE r)
        RETURN cnt
        """
        with self.driver.session() as session:
            result = session.run(query).single()
            return int(result["cnt"]) if result else 0

    def get_orphan_entity_ids(self) -> list[str]:
        """返回没有任何 MENTIONS 关系的实体 ID 列表（孤立实体）。"""
        query = """
        MATCH (e:Entity)
        WHERE NOT ()-[:MENTIONS]->(e)
        RETURN e.id AS entity_id
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [r["entity_id"] for r in result]

    def delete_entities_by_ids(self, entity_ids: list[str]) -> int:
        """批量删除指定 ID 的实体节点及其所有关系，返回删除数量。"""
        if not entity_ids:
            return 0
        query = """
        MATCH (e:Entity)
        WHERE e.id IN $ids
        WITH collect(e) AS entities, size(collect(e)) AS cnt
        FOREACH (e IN entities | DETACH DELETE e)
        RETURN cnt
        """
        with self.driver.session() as session:
            result = session.run(query, ids=entity_ids).single()
            return int(result["cnt"]) if result else 0

    def reset_document_entity_extracted(self, file_path: str) -> None:
        """重置文档的实体抽取标志，允许重新抽取。"""
        query = """
        MATCH (d:Document {file_path: $file_path})
        SET d.entity_extracted = false
        """
        with self.driver.session() as session:
            session.run(query, file_path=file_path)

    # ------------------------------------------------------------------
    # Phase 9-1 — LightRAG 双极检索支持
    # ------------------------------------------------------------------

    def search_by_relations(self, keywords: list[str], k: int = 5) -> list[dict]:
        """High-Level 检索：按关键词匹配关系描述，返回关系及关联实体信息。

        用于 LightRAG 双极检索的高层（Abstract）查询路径，
        匹配 RELATES_TO 关系的 description 字段，召回宏观语义关联。

        使用参数化 UNWIND 查询，不拼接关键词字符串，防止注入。
        ORDER BY score DESC 保证结果稳定排序（命中关键词越多排越前）。
        通过 Chunk→Document 反查年份，使 High-Level 结果能参与时间感知过滤。

        Args:
            keywords: 高层关键词列表（由 KeywordExtractor 提取的 hl_keywords）。
            k: 返回的最大关系数量。

        Returns:
            关系信息列表，每项包含：source_name, target_name, relation_type,
            description, source_id, target_id, year（可能为 None）。
        """
        if not keywords:
            return []

        # OPTIONAL MATCH Chunk→Document 回查年份；
        # 同一关系可能由多个 Document 贡献，取最大年份（最新文献优先）
        query = """
        UNWIND $keywords AS kw
        MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
        WHERE toLower(coalesce(r.description, '')) CONTAINS toLower(kw)
        WITH s, r, t, count(kw) AS score
        OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(s)
        OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c)
        WITH s, r, t, score, max(d.year) AS year
        RETURN s.id         AS source_id,
               s.name       AS source_name,
               t.id         AS target_id,
               t.name       AS target_name,
               r.relation_type AS relation_type,
               r.description   AS description,
               score,
               year
        ORDER BY score DESC, s.name
        LIMIT $k
        """
        with self.driver.session() as session:
            result = session.run(query, keywords=list(keywords[:8]), k=k)
            return [dict(record) for record in result]

    def get_one_hop_neighbors(self, entity_ids: set[str]) -> list[dict]:
        """LightRAG §3.2 高阶关联扩展：获取给定实体集合的一跳邻居实体信息。

        对应 LightRAG 论文中的 one-hop neighbor expansion：
            {v_i | v_i ∈ V ∧ (v_i ∈ N_v ∨ v_i ∈ N_e)}

        Args:
            entity_ids: 起始实体 ID 集合。

        Returns:
            邻居实体信息列表，每项包含：entity_id, name, description, relation_type。
        """
        if not entity_ids:
            return []

        query = """
        MATCH (seed:Entity)-[r:RELATES_TO]-(neighbor:Entity)
        WHERE seed.id IN $ids AND NOT neighbor.id IN $ids
        RETURN DISTINCT
            neighbor.id AS entity_id,
            neighbor.name AS name,
            coalesce(neighbor.description, '') AS description,
            coalesce(r.relation_type, '') AS relation_type
        LIMIT 50
        """
        with self.driver.session() as session:
            result = session.run(query, ids=list(entity_ids))
            return [dict(record) for record in result]
