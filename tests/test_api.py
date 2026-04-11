"""
Phase 14 API 验证脚本
前提：main.py 已在 http://127.0.0.1:8000 运行

用法：
  F:/Anaconda/envs/llm_universe/python.exe tests/test_api.py
  F:/Anaconda/envs/llm_universe/python.exe tests/test_api.py --base-url http://127.0.0.1:8000
"""
import argparse
import json
import sys

import requests


def check(label: str, resp: requests.Response, expected_status: int = 200) -> dict:
    if resp.status_code != expected_status:
        print(f"[FAIL] {label} — HTTP {resp.status_code}: {resp.text[:200]}")
        sys.exit(1)
    data = resp.json()
    print(f"[OK]   {label}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="API 端到端验证")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--thread-id", default="api-test")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    # ── 健康检查 ──────────────────────────────────────────────────────────────
    data = check("GET /health", requests.get(f"{base}/health"))
    assert data["status"] == "ok", f"health status 异常: {data}"

    # ── 文献列表 ──────────────────────────────────────────────────────────────
    data = check("GET /documents", requests.get(f"{base}/documents"))
    print(f"       已入库文献数量: {len(data)}")

    # ── 问答（不依赖 LLM，仅验证接口可达性） ─────────────────────────────────
    chat_payload = {
        "message": "什么是对比学习？",
        "thread_id": args.thread_id,
        "top_k": 3,
        "max_history": 5,
        "retriever_mode": "hybrid",
    }
    data = check("POST /chat", requests.post(f"{base}/chat", json=chat_payload))
    assert "answer" in data, "chat 响应缺少 answer 字段"
    assert "sources" in data, "chat 响应缺少 sources 字段"
    print(f"       answer 前 80 字: {data['answer'][:80]}")

    # ── Idea 生成（使用占位报告，验证接口可达性） ─────────────────────────────
    idea_payload = {
        "question": "对比学习的局限是什么？",
        "report_markdown": "# 研究报告\n\n## 背景\n对比学习是一种自监督学习方法。\n\n## 局限\n需要大量负样本，计算开销大。",
    }
    data = check("POST /idea", requests.post(f"{base}/idea", json=idea_payload))
    assert "research_gaps" in data, "idea 响应缺少 research_gaps 字段"
    assert "markdown" in data, "idea 响应缺少 markdown 字段"
    print(f"       research_gaps 数量: {len(data['research_gaps'])}")

    print("\n所有接口验证通过。")
    print("如需测试 /research 接口（耗时较长），请手动执行：")
    print(f"""  curl -X POST {base}/research \\
    -H "Content-Type: application/json" \\
    -d '{json.dumps({"question": "AmpAgent解决了什么问题？", "thread_id": "r1", "top_k": 3, "max_subquestions": 3, "retriever_mode": "hybrid", "use_community": False})}'""")


if __name__ == "__main__":
    main()
