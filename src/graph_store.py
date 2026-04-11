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
