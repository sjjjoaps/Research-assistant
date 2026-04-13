"""
实体与关系抽取模块
从 chunk 文本中抽取学术实体与关系，写入 Neo4j

Phase 2.2 变更：
- RelationItem 新增可选 description 字段
- 使用 upsert_entity_with_merge / upsert_relation_with_merge 替代直接覆盖写入
"""
from dataclasses import dataclass
from hashlib import md5

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.chunker import TextChunk
from src.graph_store import GraphStore
from src.ingestion.description_merger import DescriptionMerger
from src.llm_client import get_llm


class EntityItem(BaseModel):
    name: str = Field(description="实体名称")
    entity_type: str = Field(description="实体类型，如 Method/Model/Dataset/Task/Concept/Metric")
    description: str | None = Field(default=None, description="实体的简短描述")


class RelationItem(BaseModel):
    source_name: str = Field(description="关系起点实体名称")
    target_name: str = Field(description="关系终点实体名称")
    relation_type: str = Field(description="关系类型，如 USES/PROPOSES/EVALUATES_ON/APPLIES_TO/BASED_ON")
    description: str | None = Field(default=None, description="关系的简短描述（可选）")


class ExtractionResult(BaseModel):
    entities: list[EntityItem] = Field(default_factory=list)
    relations: list[RelationItem] = Field(default_factory=list)


_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            open("prompt/relation_extra_prompt.md", "r", encoding="utf-8").read(),
        ),
        (
            "human",
            "请从以下文本中抽取实体与关系：\n\n{text}",
        ),
    ]
)


@dataclass
class EntityExtractionStats:
    processed_chunks: int = 0
    created_entities: int = 0
    created_relations: int = 0


class EntityExtractor:
    def __init__(self, graph_store: GraphStore) -> None:
        self.graph_store = graph_store
        self._merger = DescriptionMerger()
        llm = get_llm(temperature=0.0)
        self._chain = _PROMPT | llm.with_structured_output(ExtractionResult)

    @staticmethod
    def _entity_id(file_path: str, name: str, entity_type: str) -> str:
        # Phase 2.1 将改为基于自然键（name + type）的 ID，届时跨文档合并才完全生效
        raw = f"{file_path}::{entity_type}::{name}".lower().strip()
        return md5(raw.encode("utf-8")).hexdigest()

    def extract_for_document(
        self,
        file_path: str,
        chunks: list[TextChunk],
        max_chunks: int | None = None,
    ) -> EntityExtractionStats:
        stats = EntityExtractionStats()

        effective_chunks = chunks[:max_chunks] if max_chunks is not None else chunks

        for chunk in effective_chunks:
            result = self._chain.invoke({"text": chunk.content})
            stats.processed_chunks += 1

            name_to_entity_id: dict[str, str] = {}
            for entity in result.entities:
                entity_id = self._entity_id(file_path, entity.name, entity.entity_type)
                # 使用合并写入，不覆盖已有描述；返回值表示是否为新建节点
                is_new = self.graph_store.upsert_entity_with_merge(
                    entity_id=entity_id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    description=entity.description,
                    merger=self._merger,
                )
                self.graph_store.create_mentions_relation(chunk.chunk_id, entity_id)
                name_to_entity_id[entity.name.strip().lower()] = entity_id
                if is_new:
                    stats.created_entities += 1

            for relation in result.relations:
                source_id = name_to_entity_id.get(relation.source_name.strip().lower())
                target_id = name_to_entity_id.get(relation.target_name.strip().lower())
                if not source_id or not target_id:
                    continue
                # 使用合并写入，不覆盖已有关系描述；返回值表示是否为新建关系
                is_new = self.graph_store.upsert_relation_with_merge(
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relation_type=relation.relation_type,
                    description=relation.description,
                    merger=self._merger,
                )
                if is_new:
                    stats.created_relations += 1

        self.graph_store.mark_document_entity_extracted(file_path)
        return stats
