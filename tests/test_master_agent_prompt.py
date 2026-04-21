"""
测试 prompt_loader（Phase 8-4，Review 修复版）

测试策略：
- Prompt 加载层测试（不依赖外部服务）：
    1. 文件可被正常加载，返回非空字符串
    2. 返回内容包含所有 8 个必要章节标题（标题级精确匹配）
    3. load_master_agent_system_prompt() 是单例（@lru_cache 两次调用同一对象）
    4. 热更新测试：临时改文件内容 → reload 后获得新内容（真正验证缓存失效）
    5. 文件不存在时 fallback 到内置 Prompt，不抛出异常

- validate_system_prompt 验证函数：
    6. 完整 Prompt → ok=True，无 errors
    7. 缺少单个必要章节（回答策略）→ ok=False，errors 中含对应章节名
    8. 缺少单个必要章节（异常与空结果处理）→ ok=False，errors 中含对应章节名
    9. Prompt 过短 → ok=False，errors 中包含"过短"提示
   10. 传入未知工具名 → 进入 warnings，不影响 ok
   11. 8 个工具名均在 Prompt 中明确出现（内联列表，不依赖 tool_registry）

- fallback Prompt 结构验证：
   12. 内置 fallback Prompt 应通过 validate，ok=True（确保 fallback 结构与正式一致）

注意：
- 所有测试均不依赖 LLM / FAISS / Neo4j，可在任何环境运行。
- 不导入 tool_registry，保持 Prompt 测试层的独立性。
- 热更新测试结束后务必恢复原始文件内容，避免污染磁盘状态。
"""
import sys
import os
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.prompt_loader import (
    load_master_agent_system_prompt,
    reload_master_agent_system_prompt,
    validate_system_prompt,
    REQUIRED_SECTION_HEADERS,
    _FALLBACK_SYSTEM_PROMPT,
)

# ── 工具名内联定义（不依赖 tool_registry）─────────────────────────────────────
# 与 tool_registry.TOOL_DISPLAY_NAMES 的键保持一致；若工具改名，需同步更新此处。
_EXPECTED_TOOL_NAMES = [
    "retrieve_knowledge",
    "deep_research",
    "generate_research_ideas",
    "list_documents",
    "get_document_metadata",
    "search_by_entity",
    "get_knowledge_graph_stats",
    "save_user_memory",   # Phase 9-5-2 新增
]


# ══════════════════════════════════════════════════════════════════════════
# Case 1-5：加载层测试
# ══════════════════════════════════════════════════════════════════════════

def test_prompt_loads_successfully():
    """Prompt 文件应能被加载，返回非空字符串。"""
    load_master_agent_system_prompt.cache_clear()
    prompt = load_master_agent_system_prompt()
    assert isinstance(prompt, str), "应返回 str"
    assert len(prompt) > 500, f"Prompt 内容过短（{len(prompt)} 字符），疑似加载失败"
    print("[PASS] test_prompt_loads_successfully")


def test_prompt_contains_all_required_section_headers():
    """
    Prompt 必须包含 8 个必要章节的精确标题行。
    使用 REQUIRED_SECTION_HEADERS 而非内容关键词，防止宽松误判。
    """
    load_master_agent_system_prompt.cache_clear()
    prompt = load_master_agent_system_prompt()
    for section_name, header in REQUIRED_SECTION_HEADERS.items():
        assert header in prompt, (
            f"Prompt 缺少必要章节标题：'{header}'（章节：{section_name}）"
        )
    print("[PASS] test_prompt_contains_all_required_section_headers")


def test_prompt_singleton():
    """load_master_agent_system_prompt() 的两次调用应返回同一对象（@lru_cache）。"""
    load_master_agent_system_prompt.cache_clear()
    a = load_master_agent_system_prompt()
    b = load_master_agent_system_prompt()
    assert a is b, "两次调用返回不同对象，@lru_cache 失效"
    print("[PASS] test_prompt_singleton")


def test_reload_picks_up_file_changes():
    """
    热更新测试：临时修改 Prompt 文件后，reload 应返回新内容（而非缓存旧内容）。
    测试结束后必须恢复原始文件，避免污染磁盘。
    """
    # 优先使用新合并文件，与 prompt_loader 加载顺序一致
    prompt_path = Path("prompt/master_agent.md")
    if not prompt_path.exists():
        prompt_path = (
            Path(__file__).resolve().parent.parent
            / "prompt"
            / "master_agent.md"
        )
    if not prompt_path.exists():
        print("[SKIP] test_reload_picks_up_file_changes：Prompt 文件不存在，跳过热更新测试")
        return

    # 读取原始内容
    original_content = prompt_path.read_text(encoding="utf-8")

    # 第一次加载并缓存（内容为 system 段，非原始文件全文）
    load_master_agent_system_prompt.cache_clear()
    first_load = load_master_agent_system_prompt()
    assert len(first_load) > 0, "首次加载应返回非空内容"

    # 临时覆写文件（注入一个哨兵字符串）
    sentinel = "RELOAD_TEST_SENTINEL_XYZ_12345"
    try:
        prompt_path.write_text(
            original_content + f"\n\n<!-- {sentinel} -->",
            encoding="utf-8",
        )

        # reload 后应获得含哨兵的新内容
        reloaded = reload_master_agent_system_prompt()
        assert sentinel in reloaded, (
            f"reload 后未获得新内容，热更新未生效。期望含 '{sentinel}'，实际长度：{len(reloaded)}"
        )
        # 缓存也应已更新：再次调用与 reload 结果相同
        second_load = load_master_agent_system_prompt()
        assert second_load is reloaded, "reload 后缓存未更新，再次调用返回了旧对象"

    finally:
        # 无论测试是否通过，都恢复原始文件
        prompt_path.write_text(original_content, encoding="utf-8")
        load_master_agent_system_prompt.cache_clear()

    print("[PASS] test_reload_picks_up_file_changes")


def test_fallback_when_file_missing():
    """文件不存在时应 fallback 到内置 Prompt，不抛出异常，且内容非空。"""
    load_master_agent_system_prompt.cache_clear()
    try:
        with patch.object(Path, "exists", return_value=False):
            prompt = load_master_agent_system_prompt()

        assert isinstance(prompt, str), "fallback 应返回 str"
        assert len(prompt) > 200, f"fallback Prompt 过短（{len(prompt)} 字符）"
        assert "GraphAssistant" in prompt, "fallback Prompt 应包含角色名称"
        print("[PASS] test_fallback_when_file_missing")
    finally:
        load_master_agent_system_prompt.cache_clear()


# ══════════════════════════════════════════════════════════════════════════
# Case 6-10：validate_system_prompt 测试
# ══════════════════════════════════════════════════════════════════════════

def test_validate_full_prompt_ok():
    """完整的系统 Prompt 应通过验证，ok=True，无 errors。"""
    load_master_agent_system_prompt.cache_clear()
    prompt = load_master_agent_system_prompt()
    result = validate_system_prompt(prompt)
    assert result["ok"] is True, f"完整 Prompt 未通过验证，errors: {result['errors']}"
    assert len(result["errors"]) == 0, f"不应有 errors：{result['errors']}"
    print("[PASS] test_validate_full_prompt_ok")


def test_validate_missing_section_huida_celue():
    """缺少"回答策略"章节时，validate 应返回 ok=False，errors 中含对应章节名。"""
    # 构造去掉"# 回答策略"标题行及其内容的 Prompt
    full_prompt = load_master_agent_system_prompt()
    # 找到"# 回答策略"一节，截断
    marker = "# 回答策略"
    if marker not in full_prompt:
        print("[SKIP] test_validate_missing_section_huida_celue：正式 Prompt 缺少该节，跳过")
        return
    idx = full_prompt.index(marker)
    # 找到下一节开头位置
    next_section_idx = full_prompt.find("\n# ", idx + 1)
    if next_section_idx == -1:
        prompt_without = full_prompt[:idx]
    else:
        prompt_without = full_prompt[:idx] + full_prompt[next_section_idx:]

    result = validate_system_prompt(prompt_without)
    assert result["ok"] is False, "缺少'回答策略'章节时应返回 ok=False"
    assert any("回答策略" in e for e in result["errors"]), (
        f"errors 中应包含'回答策略'，实际：{result['errors']}"
    )
    print("[PASS] test_validate_missing_section_huida_celue")


def test_validate_missing_section_yichang():
    """缺少"异常与空结果处理"章节时，validate 应返回 ok=False，errors 中含对应章节名。"""
    full_prompt = load_master_agent_system_prompt()
    marker = "# 异常与空结果处理"
    if marker not in full_prompt:
        print("[SKIP] test_validate_missing_section_yichang：正式 Prompt 缺少该节，跳过")
        return
    idx = full_prompt.index(marker)
    next_section_idx = full_prompt.find("\n# ", idx + 1)
    if next_section_idx == -1:
        prompt_without = full_prompt[:idx]
    else:
        prompt_without = full_prompt[:idx] + full_prompt[next_section_idx:]

    result = validate_system_prompt(prompt_without)
    assert result["ok"] is False, "缺少'异常与空结果处理'章节时应返回 ok=False"
    assert any("异常" in e for e in result["errors"]), (
        f"errors 中应包含'异常'，实际：{result['errors']}"
    )
    print("[PASS] test_validate_missing_section_yichang")


def test_validate_too_short_prompt():
    """过短的 Prompt 应触发 ok=False 和"过短"提示。"""
    result = validate_system_prompt("你好")
    assert result["ok"] is False, "过短 Prompt 应返回 ok=False"
    assert any("过短" in e for e in result["errors"]), (
        f"errors 中应包含'过短'，实际：{result['errors']}"
    )
    print("[PASS] test_validate_too_short_prompt")


def test_validate_unknown_tool_warning():
    """传入未在 Prompt 中出现的工具名时，应进入 warnings，ok 仍为 True。"""
    load_master_agent_system_prompt.cache_clear()
    prompt = load_master_agent_system_prompt()
    result = validate_system_prompt(prompt, tool_names=["nonexistent_tool_xyz"])
    assert any("nonexistent_tool_xyz" in w for w in result["warnings"]), (
        f"未知工具应出现在 warnings，实际：{result['warnings']}"
    )
    print("[PASS] test_validate_unknown_tool_warning")


# ══════════════════════════════════════════════════════════════════════════
# Case 11：8 个工具名均在 Prompt 中出现（内联列表，不依赖 tool_registry）
# ══════════════════════════════════════════════════════════════════════════

def test_all_expected_tools_covered_in_prompt():
    """
    所有 8 个预期工具名均应在系统 Prompt 中有明确出现。
    使用内联 _EXPECTED_TOOL_NAMES，不导入 tool_registry，保持测试独立性。
    """
    load_master_agent_system_prompt.cache_clear()
    prompt = load_master_agent_system_prompt()
    missing = [name for name in _EXPECTED_TOOL_NAMES if name not in prompt]
    assert len(missing) == 0, (
        f"以下工具在 Prompt 中缺少说明：{missing}\n"
        "请在 prompt/master_agent.md 的'# 工具调用决策指南'节中补充对应工具的描述。"
    )
    print("[PASS] test_all_expected_tools_covered_in_prompt")


# ══════════════════════════════════════════════════════════════════════════
# Case 12：fallback Prompt 结构完整性
# ══════════════════════════════════════════════════════════════════════════

def test_fallback_prompt_passes_validation():
    """
    内置 fallback Prompt 应与正式 Prompt 同结构（8 节均存在），
    确保磁盘文件丢失时服务行为不会大幅退化。
    """
    result = validate_system_prompt(_FALLBACK_SYSTEM_PROMPT)
    assert result["ok"] is True, (
        f"内置 fallback Prompt 未通过结构验证，errors: {result['errors']}\n"
        "请修复 _FALLBACK_SYSTEM_PROMPT，使其与正式 Prompt 保持同等章节结构。"
    )
    print("[PASS] test_fallback_prompt_passes_validation")


# ── 主入口 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 8-4  prompt_loader 测试（Review 修复版）")
    print("=" * 60)

    tests = [
        # 加载层
        test_prompt_loads_successfully,
        test_prompt_contains_all_required_section_headers,
        test_prompt_singleton,
        test_reload_picks_up_file_changes,
        test_fallback_when_file_missing,
        # validate_system_prompt
        test_validate_full_prompt_ok,
        test_validate_missing_section_huida_celue,
        test_validate_missing_section_yichang,
        test_validate_too_short_prompt,
        test_validate_unknown_tool_warning,
        # 工具覆盖（内联列表）
        test_all_expected_tools_covered_in_prompt,
        # fallback 结构
        test_fallback_prompt_passes_validation,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"结果：{passed} 通过，{failed} 失败")
    print("=" * 60)
