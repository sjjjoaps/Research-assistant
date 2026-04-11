"""
研究 CLI 入口
用法：
python research.py --question "当前对比学习方法存在哪些不足？"
python research.py --question "..." --retriever-mode graph --with-idea --use-community
python research.py --question "..." --save-report --output reports/report_001.md
"""

import sys
from datetime import datetime
from pathlib import Path

# 把项目根目录加入 Python 路径
sys.path.append(str(Path(__file__).parent))

import argparse

from src.agents.deep_research_agent import DeepResearchAgent
from src.agents.idea_agent import IdeaAgent


def _default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).parent / "reports" / f"report_{timestamp}.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Research CLI")
    parser.add_argument("--question", required=True, help="研究问题")
    parser.add_argument("--thread-id", default="research", help="会话 ID")
    parser.add_argument("--top-k", type=int, default=5, help="每个子问题检索的 chunk 数")
    parser.add_argument("--max-subquestions", type=int, default=4, help="子问题上限")
    parser.add_argument(
        "--retriever-mode",
        choices=["semantic", "hybrid", "graph"],
        default="hybrid",
        help="检索模式：semantic / hybrid / graph",
    )
    parser.add_argument("--with-idea", action="store_true", help="追加 Idea 报告")
    parser.add_argument("--use-community", action="store_true", help="附加社区视角章节")
    parser.add_argument("--save-report", action="store_true", help="将研究报告保存为 Markdown 文件")
    parser.add_argument("--output", help="报告输出路径（默认 reports/report_<timestamp>.md）")
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
    print(f"子问题数量: {len(report.sub_questions)}")
    print("=" * 80)
    print("\n# 子问题列表")
    for i, q in enumerate(report.sub_questions, start=1):
        print(f"- [{i}] {q}")

    print("\n" + "=" * 80)
    print(report.final_report)
    print("=" * 80)

    print("\nToken 汇总:")
    print(report.total_token_usage)

    idea_markdown = ""
    if args.with_idea:
        idea_agent = IdeaAgent()
        idea = idea_agent.generate(args.question, report)
        idea_markdown = "\n\n" + IdeaAgent.to_markdown(idea)
        print(idea_markdown)

    if args.save_report:
        output_path = Path(args.output) if args.output else _default_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = report.final_report + idea_markdown
        output_path.write_text(content, encoding="utf-8")
        print(f"\n报告已保存到: {output_path}")


if __name__ == "__main__":
    main()
