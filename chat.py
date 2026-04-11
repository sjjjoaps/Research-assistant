"""
交互式问答命令行入口（多轮对话）
用法：
python chat.py [--thread-id default] [--top-k 3]
"""

import sys
from pathlib import Path

# 把项目根目录加入 Python 路径
sys.path.append(str(Path(__file__).parent))

import argparse

from src.agents.qa_agent import QAAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="多轮文献问答交互 CLI")
    parser.add_argument("--thread-id", default="default", help="会话 ID，不同 ID 对应独立上下文")
    parser.add_argument("--top-k", type=int, default=3, help="每轮检索返回的 chunk 数")
    parser.add_argument("--max-history", type=int, default=5, help="携带的最大历史轮数")
    args = parser.parse_args()

    agent = QAAgent(top_k=args.top_k, max_history_turns=args.max_history)

    print("=" * 80)
    print(f"学术文献问答助手  (thread_id={args.thread_id})")
    print("输入 exit 或 quit 退出")
    print("=" * 80)

    while True:
        user_input = input("\n你：").strip()

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("\n已退出。")
            break

        result = agent.run_turn(thread_id=args.thread_id, user_input=user_input)

        print(f"\n助手：{result['answer']}")

        if result["sources"]:
            print("\n来源：")
            for i, source in enumerate(result["sources"], start=1):
                print(f"  [{i}] {source}")


if __name__ == "__main__":
    main()
