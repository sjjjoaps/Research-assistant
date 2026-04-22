"""
实体与关系抽取模块
从 chunk 文本中抽取学术实体与关系，写入 Neo4j

Phase 2.2 变更：
- RelationItem 新增可选 description 字段
- 使用 upsert_entity_with_merge / upsert_relation_with_merge 替代直接覆盖写入

Phase 2.3 变更：
- _entity_id 改为基于自然键（normalized_name + entity_type），实现跨文档实体合并
- extract_for_document 返回真实 entity_ids / relation_ids 列表

Phase 2.5 变更：
- upsert_entity_with_merge / upsert_relation_with_merge 的并发安全由 GraphStore 内置细粒度锁保证
  此模块无需额外同步
"""
from dataclasses import dataclass, field
from hashlib import md5
import re
import unicodedata

from pydantic import BaseModel, Field

from src.agents.prompt_loader import load_prompt_pair
from src.ingestion.chunker import TextChunk
from src.storage.graph_store import GraphStore
from src.ingestion.description_merger import DescriptionMerger
from src.infrastructure.llm_client import get_native_llm
from src.infrastructure.json_utils import extract_json
from src.storage.relation_vector_store import RelationVectorRecord
from src.storage.entity_vector_store import EntityVectorRecord


def _normalize_extraction_json(raw) -> dict:
    """
    兼容 LLM 输出旧字段名或直接返回 list 的情况：
      - 若 raw 是 list，视为 entities 列表，relations 为空
      - entity: "type" → "entity_type"
      - relation: "source"/"relation"/"target" → "source_name"/"relation_type"/"target_name"
    """
    if isinstance(raw, list):
        raw = {"entities": raw, "relations": []}
    if not isinstance(raw, dict):
        return {"entities": [], "relations": []}

    entities = []
    for e in raw.get("entities", []):
        if not isinstance(e, dict):
            continue
        if "entity_type" not in e and "type" in e:
            e = {**e, "entity_type": e.pop("type")}
        entities.append(e)

    relations = []
    for r in raw.get("relations", []):
        if not isinstance(r, dict):
            continue
        if "source_name" not in r and "source" in r:
            r = {**r, "source_name": r.pop("source")}
        if "target_name" not in r and "target" in r:
            r = {**r, "target_name": r.pop("target")}
        if "relation_type" not in r and "relation" in r:
            r = {**r, "relation_type": r.pop("relation")}
        relations.append(r)

    return {"entities": entities, "relations": relations}


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


_sys, _human = load_prompt_pair("relation_extractor")


@dataclass
class EntityExtractionStats:
    processed_chunks: int = 0
    created_entities: int = 0
    created_relations: int = 0
    entity_ids: list[str] = field(default_factory=list)
    relation_keys: list[str] = field(default_factory=list)
    relation_records: list[RelationVectorRecord] = field(default_factory=list)
    entity_records: list[EntityVectorRecord] = field(default_factory=list)


class EntityExtractor:
    def __init__(self, graph_store: GraphStore) -> None:
        self.graph_store = graph_store
        self._merger = DescriptionMerger()
        self._llm = get_native_llm(temperature=0.0)

    @staticmethod
    def _normalize_name(name: str) -> str:
        """标准化实体名称，用于生成稳定的自然键。

        处理步骤：
        1. Unicode NFC 归一化（合并全角/半角、组合字符）
        2. 转小写
        3. 合并内部连续空白为单个空格
        4. 去除首尾空白
        """
        normalized = unicodedata.normalize("NFC", name)
        normalized = normalized.lower()
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _entity_id(name: str, entity_type: str) -> str:
        """基于自然键（normalized_name + entity_type）生成实体 ID。

        跨文档同名同类型实体共享同一 ID，实现跨文档合并。
        标准化处理内部多空格、全半角、Unicode 变体等常见脏数据。
        """
        normalized = EntityExtractor._normalize_name(name)
        raw = f"{entity_type.lower()}::{normalized}"
        return md5(raw.encode("utf-8")).hexdigest()

    def extract_for_document(
        self,
        file_path: str,
        chunks: list[TextChunk],
        max_chunks: int | None = None,
    ) -> EntityExtractionStats:
        stats = EntityExtractionStats()
        seen_entity_ids: set[str] = set()
        seen_relation_keys: set[str] = set()

        effective_chunks = chunks[:max_chunks] if max_chunks is not None else chunks

        for chunk in effective_chunks:
            resp = self._llm.invoke([
                {"role": "system", "content": _sys},
                {"role": "user",   "content": _human.format(text=chunk.content)},
            ])
            content = str(resp.get("content") or "").strip()
            result = ExtractionResult(**_normalize_extraction_json(extract_json(content)))
            stats.processed_chunks += 1

            name_to_entity_id: dict[str, str] = {}
            for entity in result.entities:
                entity_id = self._entity_id(entity.name, entity.entity_type)
                is_new = self.graph_store.upsert_entity_with_merge(
                    entity_id=entity_id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    description=entity.description,
                    merger=self._merger,
                )
                self.graph_store.create_mentions_relation(chunk.chunk_id, entity_id)
                name_to_entity_id[self._normalize_name(entity.name)] = entity_id
                if entity_id not in seen_entity_ids:
                    seen_entity_ids.add(entity_id)
                    if is_new:
                        stats.created_entities += 1
                    stats.entity_records.append(
                        EntityVectorRecord(
                            entity_id=entity_id,
                            entity_name=entity.name,
                            entity_type=entity.entity_type,
                            description=entity.description or "",
                            doc_id=str(chunk.metadata.get("doc_id", "")),
                            file_path=file_path,
                        )
                    )

            for relation in result.relations:
                source_id = name_to_entity_id.get(self._normalize_name(relation.source_name))
                target_id = name_to_entity_id.get(self._normalize_name(relation.target_name))
                if not source_id or not target_id:
                    continue
                is_new = self.graph_store.upsert_relation_with_merge(
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relation_type=relation.relation_type,
                    description=relation.description,
                    merger=self._merger,
                )
                rel_key = f"{source_id}::{relation.relation_type}::{target_id}"
                if rel_key not in seen_relation_keys:
                    seen_relation_keys.add(rel_key)
                    if is_new:
                        stats.created_relations += 1
                stats.relation_records.append(
                    RelationVectorRecord(
                        relation_key=rel_key,
                        source_entity_id=source_id,
                        source_name=relation.source_name,
                        relation_type=relation.relation_type,
                        target_entity_id=target_id,
                        target_name=relation.target_name,
                        description=relation.description or "",
                        doc_id=str(chunk.metadata.get("doc_id", "")),
                        file_path=file_path,
                    )
                )

        stats.entity_ids = list(seen_entity_ids)
        stats.relation_keys = list(seen_relation_keys)
        self.graph_store.mark_document_entity_extracted(file_path)
        return stats
