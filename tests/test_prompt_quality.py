"""
Prompt 质量测试（P0-Step 4）

覆盖范围：
1. 合并格式文件存在性检查
2. 文件可被 prompt_loader 正确解析（system 段非空）
3. 结构硬性错误检查（via validate_prompt_structure）
4. 三层结构建议项汇报（suggestions 不阻断，仅打印）
5. 源码中不再存在旧式 _load_prompt / open("prompt/... 调用
"""
import sys
from pathlib import Path

import pytest

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.prompt_loader import load_prompt_pair, validate_prompt_structure

# ── 合并格式文件清单 ──────────────────────────────────────────────────────────

# pair prompt：同时包含 system + human 两段
PAIR_PROMPTS = [
    "community_summary",
    "deep_research_analyze",
    "deep_research_report",
    "description_merger",
    "idea_agent",
    "metadata_extractor",
    "modal_image",
    "modal_table",
    "qa_agent",
    "qa_chain",
    "relation_extractor",
    "deep_research_plan",
    "memory_extractor",
    "session_compact",
]

# system-only prompt：仅有 system 段，无 human 模板
SYSTEM_ONLY_PROMPTS = [
    "citation_extractor_fallback",
    "keyword_extractor_fallback",
    "master_agent",
]

ALL_MERGED_PROMPTS = PAIR_PROMPTS + SYSTEM_ONLY_PROMPTS

# ── 源码文件（需检查不含旧式加载） ────────────────────────────────────────────

SOURCE_FILES = [
    "src/community_detector.py",
    "src/entity_extractor.py",
    "src/qa_chain.py",
    "src/metadata_extractor.py",
    "src/agents/deep_research_agent.py",
    "src/agents/idea_agent.py",
    "src/agents/qa_agent.py",
    "src/agents/memory_extractor.py",
    "src/agents/session_manager.py",
    "src/ingestion/citation_extractor.py",
    "src/ingestion/description_merger.py",
    "src/ingestion/modal_processors.py",
    "src/retrieval/keyword_extractor.py",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. 文件存在性
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", ALL_MERGED_PROMPTS)
def test_merged_file_exists(name):
    path = Path(f"prompt/{name}.md")
    assert path.exists(), f"合并格式文件不存在: prompt/{name}.md"


# ══════════════════════════════════════════════════════════════════════════════
# 2. 解析正确性：system 段非空
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", ALL_MERGED_PROMPTS)
def test_system_prompt_non_empty(name):
    system, _ = load_prompt_pair(name)
    assert system, f"prompt/{name}.md 解析后 system 段为空"


# ══════════════════════════════════════════════════════════════════════════════
# 3. pair prompt：human 段非空
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", PAIR_PROMPTS)
def test_pair_prompt_has_human(name):
    _, human = load_prompt_pair(name)
    assert human, f"prompt/{name}.md 是 pair prompt 但 human 段为空"


# ══════════════════════════════════════════════════════════════════════════════
# 4. 结构硬性错误（validate_prompt_structure errors 为空）
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", ALL_MERGED_PROMPTS)
def test_no_structural_errors(name):
    result = validate_prompt_structure(name)
    assert result["ok"], (
        f"prompt/{name}.md 存在结构硬性错误:\n"
        + "\n".join(f"  - {e}" for e in result["errors"])
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. 三层结构建议项汇报（不阻断，仅打印）
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", ALL_MERGED_PROMPTS)
def test_structure_suggestions(name):
    result = validate_prompt_structure(name)
    if result.get("suggestions"):
        print(f"\n[建议] prompt/{name}.md:")
        for s in result["suggestions"]:
            print(f"  - {s}")
    # suggestions 不影响测试通过


# ══════════════════════════════════════════════════════════════════════════════
# 6. 源码中不再有旧式 Prompt 加载
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("source_file", SOURCE_FILES)
def test_no_legacy_load_prompt_function(source_file):
    content = Path(source_file).read_text(encoding="utf-8")
    assert "def _load_prompt(" not in content, (
        f"{source_file} 仍包含旧式 _load_prompt() 函数定义，请迁移至 prompt_loader"
    )


@pytest.mark.parametrize("source_file", SOURCE_FILES)
def test_no_direct_open_prompt(source_file):
    content = Path(source_file).read_text(encoding="utf-8")
    assert 'open("prompt/' not in content and "open('prompt/" not in content, (
        f"{source_file} 仍包含 open(\"prompt/...\") 直接读取，请迁移至 prompt_loader"
    )
