"""
基础问答测试
用法：
python tests/test_qa.py --question "什么是对比学习？"
"""

import sys
from pathlib import Path

# 把项目根目录加入 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

import argparse

from src.qa_chain import QAChain


def main() -> None:
    parser = argparse.ArgumentParser(description="测试基础 RAG 问答")
    parser.add_argument("--question", required=True, help="提问内容")
    parser.add_argument("--top-k", type=int, default=3, help="检索返回的 chunk 数")
    args = parser.parse_args()

    qa_chain = QAChain(top_k=args.top_k)
    result = qa_chain.ask(args.question)

    print("=" * 80)
    print(f"问题: {args.question}")
    print("\n回答:")
    print(result["answer"])

    print("\n来源:")
    if result["sources"]:
        for i, source in enumerate(result["sources"], start=1):
            print(f"[{i}] {source}")
    else:
        print("无")
    print("=" * 80)


if __name__ == "__main__":
    main()
