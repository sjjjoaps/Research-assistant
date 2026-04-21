"""
Phase 12 社区检测测试
用法：
python tests/test_community.py --min-size 3
"""

import sys
from pathlib import Path

# 把项目根目录加入 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

import argparse

from src.ingestion.community_detector import CommunityDetector
from src.storage.graph_store import GraphStore


def main() -> None:
    parser = argparse.ArgumentParser(description="测试社区检测与社区摘要生成")
    parser.add_argument("--min-size", type=int, default=3, help="社区最小实体数，小于该值则跳过")
    args = parser.parse_args()

    graph_store = GraphStore()
    graph_store.init_schema()

    detector = CommunityDetector(graph_store)
    stats = detector.run(min_community_size=args.min_size)

    print("=" * 80)
    print(f"总实体数: {stats.total_entities}")
    print(f"检测到社区数: {stats.detected_communities}")
    print(f"写入社区数: {stats.written_communities}")
    print(f"跳过社区数: {stats.skipped_communities}")

    if stats.total_entities == 0:
        print("无实体关系数据，请先执行开启实体抽取的入库流程。")
    else:
        print("请在 Neo4j Browser 中检查 (:Community) 与 [:BELONGS_TO]。")
    print("=" * 80)

    graph_store.close()


if __name__ == "__main__":
    main()
