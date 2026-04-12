"""
社区检测模块
基于 Neo4j 中的 Entity-RELATES_TO 图构建 NetworkX 图，使用 Louvain 算法划分社区，
并为每个足够大的社区生成摘要后写回 Neo4j。
"""
from collections import defaultdict
from dataclasses import dataclass

import community as community_louvain
import networkx as nx
from langchain_core.prompts import ChatPromptTemplate

from src.graph_store import GraphStore
from src.llm_client import get_llm


def _load_prompt(filename: str) -> str:
    return open(f"prompt/{filename}", "r", encoding="utf-8").read()


_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _load_prompt("community_summary_system.md")),
        ("human", _load_prompt("community_summary_human.md")),
    ]
)


@dataclass
class CommunityStats:
    total_entities: int = 0
    detected_communities: int = 0
    written_communities: int = 0
    skipped_communities: int = 0


class CommunityDetector:
    def __init__(self, graph_store: GraphStore) -> None:
        self.graph_store = graph_store
        self.llm = get_llm(temperature=0.1)
        self.chain = _SUMMARY_PROMPT | self.llm

    @staticmethod
    def _community_id(label: int) -> str:
        return f"community-{label}"

    @staticmethod
    def _build_graph(relations: list[dict]) -> tuple[nx.Graph, dict[str, dict]]:
        graph = nx.Graph()
        entity_info: dict[str, dict] = {}

        for row in relations:
            source_id = row["source_id"]
            target_id = row["target_id"]

            graph.add_edge(source_id, target_id)
            entity_info[source_id] = {
                "name": row.get("source_name") or source_id,
                "description": row.get("source_desc") or "",
            }
            entity_info[target_id] = {
                "name": row.get("target_name") or target_id,
                "description": row.get("target_desc") or "",
            }

        return graph, entity_info

    @staticmethod
    def _group_entities(partition: dict[str, int]) -> dict[int, list[str]]:
        grouped: dict[int, list[str]] = defaultdict(list)
        for entity_id, label in partition.items():
            grouped[label].append(entity_id)
        return dict(grouped)

    @staticmethod
    def _format_entities(entity_ids: list[str], entity_info: dict[str, dict]) -> str:
        lines: list[str] = []
        for entity_id in entity_ids:
            info = entity_info.get(entity_id, {})
            name = info.get("name", entity_id)
            description = info.get("description", "")
            if description:
                lines.append(f"- {name}: {description}")
            else:
                lines.append(f"- {name}")
        return "\n".join(lines)

    def run(self, min_community_size: int = 3) -> CommunityStats:
        relations = self.graph_store.get_entity_relations()
        if not relations:
            return CommunityStats()

        graph, entity_info = self._build_graph(relations)
        if graph.number_of_nodes() == 0:
            return CommunityStats()

        partition = community_louvain.best_partition(graph)
        grouped_entities = self._group_entities(partition)

        stats = CommunityStats(
            total_entities=graph.number_of_nodes(),
            detected_communities=len(grouped_entities),
        )

        for label, entity_ids in grouped_entities.items():
            if len(entity_ids) < min_community_size:
                stats.skipped_communities += 1
                continue

            entities_text = self._format_entities(entity_ids, entity_info)
            message = self.chain.invoke({"entities": entities_text})
            summary = str(message.content).strip()
            community_id = self._community_id(label)

            self.graph_store.create_community_node(
                community_id=community_id,
                label=label,
                summary=summary,
                entity_count=len(entity_ids),
            )
            for entity_id in entity_ids:
                self.graph_store.create_belongs_to_relation(entity_id, community_id)

            stats.written_communities += 1

        return stats
