"""
全局 Prompt 加载器（P0-Step 1 重构）。

职责：
- 作为全项目统一 Prompt 加载入口，所有模块通过此模块读取 Prompt 文件
- 支持新合并格式（{name}.md）和旧拆分格式（{name}_system.md / {name}_human.md）兼容加载
- 提供带 lru_cache 的加载函数，避免重复磁盘 I/O
- 文件不存在时 fallback 并打印 WARNING，不中断启动

合并文件格式（新格式）：
    ## System Prompt
    ...system 内容...

    ---

    ## Human Prompt Template
    ...human 内容（含 {变量} 占位符）...

加载优先级（每次均按此顺序查找）：
    1. prompt/{name}.md            ← 新合并格式（优先）
    2. prompt/{name}_system.md     ← 旧 system 文件（兼容回退）
    3. 返回空字符串并打印 WARNING   ← 最终兜底

Public API：
    load_system_prompt(name)         → str
    load_human_template(name)        → str
    load_prompt_pair(name)           → tuple[str, str]
    reload_prompt(name)              → None  （清除指定 name 的缓存）
    load_master_agent_system_prompt() → str  （保留，供 master_agent 专用逻辑使用）
    reload_master_agent_system_prompt() → str
    validate_system_prompt(prompt)   → dict
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 路径常量 ──────────────────────────────────────────────────────────────────
_PROMPT_DIR = Path("prompt")
_PROMPT_DIR_ABS = Path(__file__).resolve().parent.parent.parent / "prompt"

# ── 分隔符与标题 ──────────────────────────────────────────────────────────────
_SYSTEM_HEADER = "## System Prompt"
_HUMAN_HEADER = "## Human Prompt Template"

# ── MasterAgent 章节校验 ──────────────────────────────────────────────────────
REQUIRED_SECTION_HEADERS: dict[str, str] = {
    "角色定义":       "# 角色定义",
    "能力边界":       "# 能力边界",
    "行为准则":       "# 行为准则",
    "工具调用决策指南": "# 工具调用决策指南",
    "回答策略":       "# 回答策略",
    "来源标注要求":   "# 来源标注要求",
    "异常与空结果处理": "# 异常与空结果处理",
    "输出格式要求":   "# 输出格式要求",
}

# ── MasterAgent Fallback Prompt ───────────────────────────────────────────────
_FALLBACK_SYSTEM_PROMPT = """\
# 角色定义
你是 GraphAssistant，一个专注于学术文献知识库的智能研究助手。
你的核心能力是帮助用户理解、分析和探索已入库的学术论文。

# 能力边界
- 优先基于知识库中的文献内容、文献元数据和知识图谱信息作答。
- 当问题依赖知识库证据时，必须先调用合适工具获取依据，不能凭空生成论文内容或引用。
- 如果知识库中没有足够证据，明确说明当前证据不足，建议用户先入库相关文献。
- 对于无需文献支撑的通用概念性问题，可以直接基于已有知识做简洁说明。

# 行为准则
1. 先判断后行动：先判断问题是否需要检索，再决定调用哪个工具。
2. 优先最小充分工具：能用 retrieve_knowledge 解决的不要直接调用 deep_research。
3. 分步推进：复杂问题拆解成多个子问题，必要时多轮调用不同工具。
4. 避免重复调用：已有足够结果时，避免用相同参数重复调用同一工具。
5. 空结果要友好：检索为空或工具失败时向用户解释现状，给出下一步建议。
6. 深度研究先告知：调用 deep_research 前，应先告知用户该步骤可能耗时较长。

# 工具调用决策指南
- retrieve_knowledge：适合回答文献中的具体内容、方法、实验、结论。
- list_documents：适合回答"知识库里有什么文献""有没有关于 XXX 的论文"。
- get_document_metadata：适合查看某篇文献的详细元数据（作者、年份、摘要等）。
- search_by_entity：适合探索具体实体（作者、方法、模型）及其关系网络。
- get_knowledge_graph_stats：适合查询知识图谱规模、文献数、实体数等状态信息。
- deep_research：适合复杂综述、多角度分析、系统性比较（耗时较长）。
- generate_research_ideas：适合在已有报告或主题上生成研究方向和创新点。

# 回答策略
- 如果用户只是寒暄，可以直接自然回复，不调用工具。
- 如果问题依赖知识库证据，应先获取证据再总结答案。
- 如果用户要求综述或全面分析，优先考虑 deep_research。
- 如果用户先问库里有无文献再问具体内容，先 list_documents，再 retrieve_knowledge。

# 来源标注要求
- 只在确实有文献依据时标注来源。
- 来源格式必须使用：[来源: 文件名#chunk-n]
- 不要伪造来源，不要引用当前回答中没有真正使用到的来源。

# 异常与空结果处理
- 当工具返回空结果时，明确说明"当前知识库中未检索到相关内容"。
- 当工具失败时，基于已有上下文决定是否换工具、换参数，或直接向用户说明限制。
- 不要因为单个工具失败就终止整轮任务；优先尝试合理的替代路径。

# 输出格式要求
- 使用 Markdown 输出。
- 优先先给结论，再给依据或展开说明。
- 对较复杂的问题，使用小标题、项目符号或编号列表提高可读性。
- 来源格式：[来源: 文件名#chunk-n]
- 输出语言必须与用户问题保持一致。
"""


# ══════════════════════════════════════════════════════════════════════════════
# 内部工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_prompt_path(filename: str) -> Path | None:
    """按优先级查找 prompt 文件，返回存在的 Path 或 None。"""
    for base in (_PROMPT_DIR, _PROMPT_DIR_ABS):
        p = base / filename
        if p.exists():
            return p
    return None


def _normalize_name(name: str) -> str:
    """
    将旧式 name 归一化为新式 name，支持旧调用方无感知兼容。

    规则：
      - "qa_agent_system"  → "qa_agent"（去掉 _system 后缀）
      - "qa_agent_human"   → "qa_agent"（去掉 _human 后缀）
      - "qa_agent"         → "qa_agent"（不变）
    """
    for suffix in ("_system", "_human"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.error("读取 Prompt 文件失败: %s", exc)
        return ""


def _split_merged(content: str) -> tuple[str, str]:
    """
    从合并文件中提取 system 和 human 两段正文。

    支持两种格式：
      格式A（标准）：文件含 `## System Prompt` 和 `## Human Prompt Template` 标题
      格式B（简单）：仅用 `---` 分隔，无明确标题

    返回 (system_text, human_text)，任一段缺失时返回空字符串。
    """
    # 格式A：按标题提取（优先，更精确）
    if _SYSTEM_HEADER in content:
        system_text = _extract_section(content, _SYSTEM_HEADER, _HUMAN_HEADER)
        human_text = _extract_section(content, _HUMAN_HEADER, None)
        return system_text, human_text

    # 格式B：按 `---` 分隔符切割
    parts = re.split(r"^\s*---\s*$", content, maxsplit=1, flags=re.MULTILINE)
    system_text = parts[0].strip()
    human_text = parts[1].strip() if len(parts) > 1 else ""
    return system_text, human_text


def _extract_section(content: str, start_header: str, end_header: str | None) -> str:
    """
    提取从 start_header 到 end_header（或文件末尾 / `---` 分隔符）之间的正文内容。
    不包含标题行本身。
    """
    start_idx = content.find(start_header)
    if start_idx == -1:
        return ""

    # 跳过标题行，从下一行开始
    after_header = content[start_idx + len(start_header):]

    # 找结束位置：end_header 或 `---` 分隔符（独立行）
    end_idx = len(after_header)
    if end_header:
        h_pos = after_header.find(end_header)
        if h_pos != -1:
            end_idx = h_pos

    sep_match = re.search(r"^\s*---\s*$", after_header[:end_idx], flags=re.MULTILINE)
    if sep_match:
        end_idx = min(end_idx, sep_match.start())

    return after_header[:end_idx].strip()


# ══════════════════════════════════════════════════════════════════════════════
# 核心加载逻辑（带 lru_cache）
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=64)
def _load_merged(name: str) -> tuple[str, str]:
    """
    缓存加载单个 prompt 的 (system, human) 对。

    name 会先经过 _normalize_name 归一化，因此旧调用方传入
    "qa_agent_system" 与传入 "qa_agent" 效果相同。

    加载优先级：
      1. prompt/{name}.md              新合并格式
      2. prompt/{name}_system.md       旧 system 文件（回退）
      human 部分：
      1. 合并文件中 --- 之后的内容
      2. prompt/{name}_human.md        旧 human 文件（回退）
      3. ""                            最终兜底
    """
    name = _normalize_name(name)

    # ── 尝试加载合并文件 ──────────────────────────────────────────────────────
    merged_path = _resolve_prompt_path(f"{name}.md")
    if merged_path:
        content = _read_file(merged_path)
        if content:
            system, human = _split_merged(content)
            if system:
                logger.debug("已加载合并 Prompt: %s（%d 字符）", merged_path, len(content))
                return system, human
            # 合并文件存在但解析不到 system 段，当作无 system 标题的 system-only 文件
            logger.debug("合并文件无 System 标题，整体作为 system prompt: %s", merged_path)
            return content, human

    # ── 回退：旧拆分文件 ──────────────────────────────────────────────────────
    system = ""
    system_path = _resolve_prompt_path(f"{name}_system.md")
    if system_path:
        system = _read_file(system_path)
        if system:
            logger.warning(
                "Prompt '%s' 使用旧拆分文件 %s，建议迁移至 %s.md",
                name, system_path.name, name,
            )

    human = ""
    human_path = _resolve_prompt_path(f"{name}_human.md")
    if human_path:
        human = _read_file(human_path)

    if not system:
        logger.warning("Prompt '%s' 未找到任何可用文件，返回空字符串", name)

    return system, human


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def load_system_prompt(name: str) -> str:
    """
    加载指定功能的 System Prompt。

    Args:
        name: Prompt 名称，不含扩展名，例如 "qa_agent"、"deep_research_analyze"

    Returns:
        System Prompt 文本；文件不存在时返回空字符串并打印 WARNING。
    """
    system, _ = _load_merged(name)
    return system


def load_human_template(name: str) -> str:
    """
    加载指定功能的 Human Prompt Template（含 {变量} 占位符）。

    Args:
        name: Prompt 名称，不含扩展名

    Returns:
        Human 模板文本；system-only prompt 或文件不存在时返回空字符串。
    """
    _, human = _load_merged(name)
    return human


def load_prompt_pair(name: str) -> tuple[str, str]:
    """
    一次加载 System Prompt 和 Human Template。

    Returns:
        (system_prompt, human_template) 元组。
    """
    return _load_merged(name)


def reload_prompt(name: str) -> None:
    """
    清除所有 Prompt 文件缓存，下次调用时重新读取磁盘。

    注意：lru_cache 不支持按单个 key 精确失效，因此本函数会清除全部缓存。
    传入 name 参数仅用于日志提示，实际清除范围是全部缓存条目。
    若需要精确失效单个 name，可考虑改用 dict 手动管理缓存。

    供热更新和测试使用。
    """
    _load_merged.cache_clear()
    logger.info("已清除全部 Prompt 缓存（触发原因：name=%s）", name)


def reload_all_prompts() -> None:
    """清除所有 Prompt 缓存。"""
    _load_merged.cache_clear()
    load_master_agent_system_prompt.cache_clear()
    logger.info("已清除全部 Prompt 缓存")


# ══════════════════════════════════════════════════════════════════════════════
# MasterAgent 专用接口（保持向后兼容）
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def load_master_agent_system_prompt() -> str:
    """
    加载 MasterAgent 系统 Prompt，并执行 8 节结构校验。

    加载优先级：
      1. prompt/master_agent_system.md（历史文件名，优先保持兼容）
      2. prompt/master_agent.md（新合并格式）
      3. 内置 _FALLBACK_SYSTEM_PROMPT

    Returns:
        str — 系统 Prompt 文本，可直接传入 SystemMessage(content=...)
    """
    # 尝试 master_agent_system.md（历史）
    for candidate in ("master_agent_system.md", "master_agent.md"):
        path = _resolve_prompt_path(candidate)
        if path:
            content = _read_file(path)
            if content:
                # 若是合并文件，只取 system 段
                if _SYSTEM_HEADER in content:
                    system, _ = _split_merged(content)
                    content = system if system else content
                result = validate_system_prompt(content)
                if not result["ok"]:
                    logger.warning(
                        "MasterAgent Prompt 结构校验失败，仍将使用该文件但请及时修复。errors: %s",
                        result["errors"],
                    )
                logger.info("已加载 MasterAgent Prompt（%d 字符，来源：%s）", len(content), path)
                return content

    logger.error(
        "MasterAgent Prompt 文件未找到（master_agent_system.md / master_agent.md），"
        "已降级使用内置 fallback Prompt。"
    )
    return _FALLBACK_SYSTEM_PROMPT


def reload_master_agent_system_prompt() -> str:
    """清除缓存并重新加载 MasterAgent Prompt（供热更新使用）。"""
    load_master_agent_system_prompt.cache_clear()
    return load_master_agent_system_prompt()


# ══════════════════════════════════════════════════════════════════════════════
# 校验工具
# ══════════════════════════════════════════════════════════════════════════════

def validate_system_prompt(prompt: str, tool_names: list[str] | None = None) -> dict:
    """
    验证 MasterAgent 系统 Prompt 的结构完整性（8 节标题级校验）。

    Args:
        prompt:     系统 Prompt 文本
        tool_names: 可选，当前注册的工具名列表（未出现进入 warnings）

    Returns:
        {"ok": bool, "warnings": list[str], "errors": list[str]}
    """
    warnings_list: list[str] = []
    errors_list:   list[str] = []

    for section_name, header in REQUIRED_SECTION_HEADERS.items():
        if header not in prompt:
            errors_list.append(f"缺少必要章节：{section_name}（期望标题行：'{header}'）")

    MIN_LENGTH = 500
    if len(prompt) < MIN_LENGTH:
        errors_list.append(
            f"Prompt 过短（仅 {len(prompt)} 字符，最低要求 {MIN_LENGTH} 字符），疑似内容缺失"
        )

    if tool_names:
        for name in tool_names:
            if name not in prompt:
                warnings_list.append(f"工具 '{name}' 未在 Prompt 中出现，LLM 可能缺少调用指引")

    return {"ok": len(errors_list) == 0, "warnings": warnings_list, "errors": errors_list}


def validate_prompt_structure(name: str) -> dict:
    """
    检查合并格式 Prompt 文件的质量，输出建议（suggestions）。

    三层结构（### 边界层 / ### 决策层 / ### 任务示例）是推荐写法，
    缺少任何一层只会进入 suggestions，不影响 ok 结果。

    硬性错误（会导致 ok=False）仅限于：
      - 文件不存在
      - 检测到 Human Prompt Template 标记但 human 段内容为空

    Returns:
        {
            "ok":          bool,        # False 仅表示有硬性错误
            "errors":      list[str],   # 硬性错误（影响运行）
            "suggestions": list[str],   # 建议改进项（不影响 ok）
            "is_pair":     bool,
        }
    """
    errors_list: list[str] = []
    suggestions: list[str] = []

    path = _resolve_prompt_path(f"{name}.md")
    if not path:
        return {"ok": False, "errors": [f"文件 {name}.md 不存在"], "suggestions": [], "is_pair": False}

    content = _read_file(path)
    system, human = _split_merged(content)

    # is_pair：从原始内容检测是否存在 human 分隔标记（独立于解析结果）
    has_human_marker = (
        _HUMAN_HEADER in content
        or bool(re.search(r"^\s*---\s*$", content, flags=re.MULTILINE))
    )
    is_pair = has_human_marker
    # 硬性错误：有 pair 标记但 human 段内容为空
    if has_human_marker and not human:
        errors_list.append("检测到 Human Prompt Template 标记但内容为空，请补充模板变量")

    # ── 以下均为建议项，不影响 ok ─────────────────────────────────────────────
    check_text = system if system else content

    # 建议：三层结构标题
    for section in ("### 边界层", "### 决策层", "### 任务示例"):
        if section not in check_text:
            suggestions.append(f"建议添加 '{section}' 章节，有助于模型更好地遵循规范")

    # 建议：边界层禁令数量
    boundary_items = [
        line for line in check_text.splitlines()
        if re.search(r"禁止", line) and line.strip().startswith("-")
    ]
    if len(boundary_items) < 2:
        suggestions.append(
            f"建议在边界层中添加更多禁令（当前 {len(boundary_items)} 条，推荐至少 2 条）"
        )

    # 建议：至少包含一个任务示例
    example_count = len(re.findall(
        r"(?:###\s*任务示例|###\s*示例|示例\s*[12]\b|\*\*示例\s*[12]\*\*|\*\*输入摘要\*\*)",
        check_text,
    ))
    if example_count < 1:
        suggestions.append("建议添加至少 1 个任务示例，帮助模型对齐输出格式")

    # 建议：最低长度
    if len(content) < 200:
        suggestions.append(f"文件内容较短（{len(content)} 字符），建议进一步补充")

    return {
        "ok": len(errors_list) == 0,
        "errors": errors_list,
        "suggestions": suggestions,
        "is_pair": is_pair,
    }
