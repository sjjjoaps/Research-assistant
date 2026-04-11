"""
Neo4j 图存储模块
负责创建约束、写入 Document/Chunk/Entity 节点及关系
"""
from neo4j import GraphDatabase

from src.chunker import TextChunk
from src.config import settings
from src.metadata_extractor import DocumentMetadata


class GraphStore:
    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )

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
            c.file_path = $file_path
        """
        with self.driver.session() as session:
            session.run(
                query,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                file_path=chunk.metadata.get("file_path"),
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
