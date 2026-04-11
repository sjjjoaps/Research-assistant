"""
Agent 测试（Phase10 / Phase11）
用法：
  # 语义检索（默认）
  python tests/test_agent.py --question "AmpAgent解决了什么问题？" --thread-id t1 --top-k 3

  # 混合检索
  python tests/test_agent.py --question "AmpAgent解决了什么问题？" --retriever-mode hybrid

  # 图检索
  python tests/test_agent.py --question "AmpAgent解决了什么问题？" --retriever-mode graph
"""

import sys
from pathlib import Path

# 把项目根目录加入 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

import argparse

from src.agents.qa_agent import QAAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="测试原生 QAAgent")
    parser.add_argument("--question", required=True, help="第一轮问题")
    parser.add_argument("--thread-id", default="test-thread", help="会话 ID")
    parser.add_argument("--top-k", type=int, default=3, help="每轮检索 chunk 数")
    parser.add_argument(
        "--retriever-mode",
        choices=["semantic", "hybrid", "graph"],
        default="semantic",
        help="检索模式：semantic / hybrid / graph",
    )
    args = parser.parse_args()

    agent = QAAgent(top_k=args.top_k, retriever_mode=args.retriever_mode)

    print("=" * 80)
    print("[1] 第一轮问答")
    result1 = agent.run_turn(thread_id=args.thread_id, user_input=args.question)
    print(f"问题: {args.question}")
    print(f"回答: {result1['answer']}")
    print("来源:")
    for i, source in enumerate(result1["sources"], start=1):
        print(f"  [{i}] {source}")

    print("\n" + "=" * 80)
    print("[2] 第二轮问答（测试上下文连续性）")
    follow_up_question = "请结合上一个回答，再总结得更简洁一点。"
    result2 = agent.run_turn(thread_id=args.thread_id, user_input=follow_up_question)
    print(f"问题: {follow_up_question}")
    print(f"回答: {result2['answer']}")
    print("来源:")
    for i, source in enumerate(result2["sources"], start=1):
        print(f"  [{i}] {source}")
    print("=" * 80)


if __name__ == "__main__":
    main()
