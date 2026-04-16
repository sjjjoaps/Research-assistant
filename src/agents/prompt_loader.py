"""
系统 Prompt 加载工具（Phase 8-4）。

职责：
- 从 prompt/ 目录加载 MasterAgent 系统 Prompt（prompt/master_agent_system.md）
- 提供带缓存的加载函数，避免每次请求重复读取磁盘
- 若文件不存在，返回内置 fallback Prompt（与正式 Prompt 同结构，不裁减章节）并打印警告
- validate_system_prompt() 用标题级关键词覆盖全部 8 个必要章节，避免宽松误判

设计决策：
- 本模块不导入 tool_registry，保持 Prompt 层与工具注册层解耦
- validate_system_prompt() 使用标题级匹配（"# 章节名"格式），而非内容级关键词，
  防止章节恰好含关键词但结构缺失时被误判为合法

用法：
    from src.agents.prompt_loader import load_master_agent_system_prompt
    system_prompt = load_master_agent_system_prompt()
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Prompt 文件路径（相对于项目根目录）
_MASTER_AGENT_PROMPT_PATH = Path("prompt/master_agent_system.md")

# ── 必要章节清单（与 prompt/master_agent_system.md 的 8 节一一对应）────────────
# 值为"# 节名"的精确标题字符串，需在 Prompt 中以独立行出现。
# 若调整 Prompt 章节数量，同步更新此处。
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

# ── Fallback Prompt（文件不存在时使用）──────────────────────────────────────
# 与正式 Prompt 保持相同的 8 节结构，确保 MasterAgent 行为不因磁盘问题大幅退化。
# 内容是每节的最小保证子集，不可进一步删减。
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


@lru_cache(maxsize=1)
def load_master_agent_system_prompt() -> str:
    """
    加载 MasterAgent 系统 Prompt，使用 @lru_cache 确保全程只读一次磁盘。

    加载策略（优先级递减）：
    1. prompt/master_agent_system.md（相对于当前工作目录）
    2. <本文件所在目录>/../../.../prompt/master_agent_system.md（绝对路径回退）
    3. 内置 _FALLBACK_SYSTEM_PROMPT（最后兜底，同时发出 WARNING 级日志）

    加载成功后对内容做 validate 校验；若正式文件校验失败，发出 WARNING 但仍返回内容，
    不阻断启动——校验失败只是警告，不是错误。
    若回退到 fallback，发出 ERROR 级日志以明确通知运维人员。

    Returns:
        str — 系统 Prompt 文本，可直接传入 SystemMessage(content=...)
    """
    path = _MASTER_AGENT_PROMPT_PATH
    used_fallback = False

    if not path.exists():
        # 绝对路径回退：兼容工作目录不是项目根的情况
        alt_path = Path(__file__).resolve().parent.parent.parent / "prompt" / "master_agent_system.md"
        if alt_path.exists():
            path = alt_path
        else:
            logger.error(
                "系统 Prompt 文件未找到（%s 和 %s 均不存在），"
                "已降级使用内置 fallback Prompt。服务行为可能受限，请尽快恢复文件。",
                _MASTER_AGENT_PROMPT_PATH,
                alt_path,
            )
            used_fallback = True

    if not used_fallback:
        try:
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                logger.error(
                    "系统 Prompt 文件为空（%s），已降级使用内置 fallback Prompt。",
                    path,
                )
                used_fallback = True
            else:
                # 对正式文件做结构校验，发现问题只 WARNING 不中断
                result = validate_system_prompt(content)
                if not result["ok"]:
                    logger.warning(
                        "系统 Prompt 结构校验失败，仍将使用该文件但请及时修复。errors: %s",
                        result["errors"],
                    )
                logger.info(
                    "已加载系统 Prompt（%d 字符，来源：%s）", len(content), path
                )
                return content
        except OSError as exc:
            logger.error("读取系统 Prompt 文件失败: %s，已降级使用内置 fallback Prompt。", exc)
            used_fallback = True

    # fallback 路径：同样做校验，给出 WARNING（fallback 结构完整，正常不会触发 errors）
    if used_fallback:
        result = validate_system_prompt(_FALLBACK_SYSTEM_PROMPT)
        if not result["ok"]:
            logger.warning(
                "内置 fallback Prompt 结构校验失败（这是 bug，请修复代码）: %s",
                result["errors"],
            )
        return _FALLBACK_SYSTEM_PROMPT

    # 不可达分支，保证类型安全
    return _FALLBACK_SYSTEM_PROMPT  # pragma: no cover


def reload_master_agent_system_prompt() -> str:
    """
    清除缓存并重新加载 Prompt 文件（供热更新场景使用）。

    典型使用场景：
    - 修改 prompt/master_agent_system.md 后，不重启服务使新内容生效
    - 单元测试中验证不同 Prompt 内容时重置缓存状态

    Returns:
        str — 重新加载后的系统 Prompt 文本
    """
    load_master_agent_system_prompt.cache_clear()
    return load_master_agent_system_prompt()


def validate_system_prompt(prompt: str, tool_names: list[str] | None = None) -> dict:
    """
    验证系统 Prompt 的结构完整性。

    检查策略（严格于内容关键词匹配）：
    - 使用"# 节名"的标题级精确匹配，要求 8 个必要章节均以独立标题行出现
    - Prompt 最低长度 500 字符（正式 Prompt 约 2000 字符，fallback 约 900 字符）
    - 若传入 tool_names，检查每个工具名是否在 Prompt 中出现（未出现进入 warnings）

    Args:
        prompt:     系统 Prompt 文本
        tool_names: 可选，当前注册的工具名列表（用于检查工具覆盖率）

    Returns:
        dict — {
            "ok":       bool,         # True 表示无 errors（warnings 不影响 ok）
            "warnings": list[str],    # 非致命问题（如工具未覆盖）
            "errors":   list[str],    # 致命结构缺失（任意一条即 ok=False）
        }
    """
    warnings_list: list[str] = []
    errors_list:   list[str] = []

    # 1. 必要章节标题级检查（覆盖全部 8 节）
    for section_name, header in REQUIRED_SECTION_HEADERS.items():
        if header not in prompt:
            errors_list.append(
                f"缺少必要章节：{section_name}（期望标题行：'{header}'）"
            )

    # 2. 最低长度（正式 Prompt 约 2000 字，fallback 约 900 字，阈值取 500）
    MIN_LENGTH = 500
    if len(prompt) < MIN_LENGTH:
        errors_list.append(
            f"Prompt 过短（仅 {len(prompt)} 字符，最低要求 {MIN_LENGTH} 字符），疑似内容缺失"
        )

    # 3. 工具覆盖率（可选，未出现的工具名进入 warnings 而非 errors）
    if tool_names:
        for name in tool_names:
            if name not in prompt:
                warnings_list.append(
                    f"工具 '{name}' 未在 Prompt 中出现，LLM 可能缺少调用指引"
                )

    ok = len(errors_list) == 0
    return {"ok": ok, "warnings": warnings_list, "errors": errors_list}
