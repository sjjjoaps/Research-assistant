"""
Phase 13 研究流程测试
用法：
python tests/test_research.py --question "AmpAgent解决了什么问题，其方法有哪些局限？"
"""

import sys
from pathlib import Path

# 把项目根目录加入 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

import argparse

from src.agents.deep_research_agent import DeepResearchAgent
from src.agents.idea_agent import IdeaAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="测试 Deep Research Agent")
    parser.add_argument("--question", required=True, help="研究问题")
    parser.add_argument("--thread-id", default="research-test", help="会话 ID")
    parser.add_argument("--top-k", type=int, default=5, help="每个子问题检索的 chunk 数")
    parser.add_argument("--max-subquestions", type=int, default=4, help="子问题上限")
    parser.add_argument(
        "--retriever-mode",
        choices=["semantic", "hybrid", "graph"],
        default="hybrid",
        help="检索模式：semantic / hybrid / graph",
    )
    parser.add_argument("--with-idea", action="store_true", help="追加测试 Idea 输出")
    parser.add_argument("--use-community", action="store_true", help="附加社区视角")
    args = parser.parse_args()

    agent = DeepResearchAgent(
        top_k=args.top_k,
        max_subquestions=args.max_subquestions,
        retriever_mode=args.retriever_mode,
        use_community=args.use_community,
    )
    report = agent.research(thread_id=args.thread_id, question=args.question)

    print("=" * 80)
    print(f"研究问题: {args.question}")
    print(f"检索模式: {args.retriever_mode}")
    print("子问题:")
    for i, q in enumerate(report.sub_questions, start=1):
        print(f"  [{i}] {q}")
    print("\n研究报告:\n")
    print(report.final_report)
    print("\nToken 汇总:")
    print(report.total_token_usage)

    if args.with_idea:
        idea = IdeaAgent().generate(args.question, report)
        print("\n" + "=" * 80)
        print("Idea 报告:\n")
        print(IdeaAgent.to_markdown(idea))

    print("=" * 80)


if __name__ == "__main__":
    main()
