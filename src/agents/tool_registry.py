"""
八个核心工具的定义与注册（Phase 9-1；Phase 9-5 长期记忆集成）。

将现有 Agent 和检索器封装为 LLM 可调用的 @tool，对应 claw-code 的
`load_tool_snapshot()` + `@lru_cache` 模式：
- 启动时通过 build_tool_registry() 一次性构建工具列表（@lru_cache 保证单例）
- 工具内部复用 src/retrieval/ 和 src/agents/ 中已有的检索器/Agent，不重写业务逻辑
- 所有工具均返回格式化的 Markdown 字符串，供 LLM 直接读取并引用

工具清单：
    1. retrieve_knowledge        — 知识库多模式检索（Phase 9-1 新增 dual 双极检索模式）
    2. deep_research             — 多步深度研究报告
    3. generate_research_ideas   — 研究 Idea 生成
    4. list_documents            — 文献清单查询
    5. get_document_metadata     — 单篇文献元数据
    6. search_by_entity          — 知识图谱实体查询
    7. get_knowledge_graph_stats — 知识图谱统计信息
    8. save_user_memory          — 将用户偏好/规则写入长期记忆（Phase 9-5-2）

Phase 9-5 长期记忆集成：
    - _do_retrieve() 在 auto 模式下优先查询 LongTermMemory.get_best_mode()，
      有历史最优 mode 时直接使用，否则回退到启发式 _auto_select_mode()
    - 每次检索完成后异步调用 LongTermMemory.record_strategy_result() 记录效果

用法：
    from src.agents.tool_registry import build_tool_registry, TOOL_DISPLAY_NAMES
    tools = build_tool_registry()   # 返回 list[StructuredTool]
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, get_args, get_origin

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# P5-Step 1：原生 Tool 对象与工具注册表（并行实现，不修改旧路径）
# ════════════════════════════════════════════════════════════════════════════

# 手动维护枚举约束（函数签名中 str 类型但实际有枚举值的参数）
_PARAM_ENUMS: dict[str, list[str]] = {
    "mode": ["auto", "dual", "local", "global", "mix", "hybrid", "semantic"],
    "retriever_mode": ["semantic", "hybrid", "graph", "local", "global", "mix"],
    "section_filter": ["abstract", "introduction", "method", "experiment",
                       "conclusion", "related_work", "other", ""],
    "memory_type": ["preference", "rule", "note"],
}

_PY_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    bool: "boolean",
    float: "number",
}


def _annotation_to_prop(annotation: type, param_name: str = "") -> dict:
    """
    将单个类型注解递归转换为 JSON Schema property dict。

    处理顺序：
      1. Optional[X] / X | None  → 解包后递归处理 X
      2. list[X]                 → {"type": "array", "items": ...}
      3. Literal["a", "b"]       → {"type": "string", "enum": [...]}
      4. _PARAM_ENUMS 手动枚举表  → {"type": "string", "enum": [...]}
      5. 标量类型映射              → {"type": "string/integer/boolean/number"}
    """
    import types as _types

    origin = get_origin(annotation)
    args   = get_args(annotation)

    # 1. Union / Optional — 解包非 None 分支后递归
    if origin is not None:
        is_union = (
            origin is _types.UnionType
            or str(origin) in ("typing.Union", "<class 'typing.Union'>")
        )
        if is_union:
            non_none = [a for a in args if a is not type(None)]
            inner = non_none[0] if non_none else str
            return _annotation_to_prop(inner, param_name)

        # 2. list[X]
        if origin is list:
            item_type = args[0] if args else str
            item_prop = _annotation_to_prop(item_type)
            return {"type": "array", "items": item_prop}

        # 3. Literal["a", "b"]
        if str(origin) == "typing.Literal" or (
            hasattr(origin, "__name__") and origin.__name__ == "Literal"
        ):
            return {"type": "string", "enum": list(args)}

    # 3b. Literal 直接作为 annotation（Python 3.8 兼容路径）
    if get_origin(annotation) is not None:
        inner_origin = get_origin(annotation)
        if str(inner_origin) == "typing.Literal":
            return {"type": "string", "enum": list(get_args(annotation))}

    # 4. _PARAM_ENUMS 手动枚举表（优先于标量映射）
    if param_name and param_name in _PARAM_ENUMS:
        return {"type": "string", "enum": _PARAM_ENUMS[param_name]}

    # 5. 标量类型映射
    return {"type": _PY_TO_JSON.get(annotation, "string")}


def _infer_schema(func: Callable) -> dict:
    """从函数签名自动生成 JSON Schema（properties + required）。"""
    sig = inspect.signature(func)
    properties: dict = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        annotation  = param.annotation
        has_default = param.default is not inspect.Parameter.empty

        if annotation is inspect.Parameter.empty:
            annotation = str

        properties[name] = _annotation_to_prop(annotation, name)
        if not has_default:
            required.append(name)

    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


@dataclass
class Tool:
    """原生工具对象，替代 LangChain StructuredTool。"""
    name: str
    description: str
    func: Callable
    parameters: dict

    @property
    def args_schema(self) -> dict:
        """兼容现有测试中 t.args_schema 断言。"""
        return self.parameters

    def invoke(self, input_dict: dict) -> str:
        """统一调用入口，供 MasterAgent 工具循环使用。"""
        coerced = self._coerce_args(input_dict)
        return self.func(**coerced)

    def _coerce_args(self, input_dict: dict) -> dict:
        """将 LLM 传入的字符串参数按 JSON Schema 类型强制转换。"""
        sig = inspect.signature(self.func)
        result = {}
        for name, param in sig.parameters.items():
            if name not in input_dict:
                result[name] = input_dict.get(name, param.default)
                continue
            val = input_dict[name]
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                result[name] = val
                continue
            # unwrap Optional[X]
            import types as _types
            origin = get_origin(ann)
            args = get_args(ann)
            if origin is not None:
                is_union = (
                    origin is _types.UnionType
                    or str(origin) in ("typing.Union", "<class 'typing.Union'>")
                )
                if is_union:
                    non_none = [a for a in args if a is not type(None)]
                    ann = non_none[0] if non_none else str
            try:
                if ann is int and not isinstance(val, int):
                    result[name] = int(val)
                elif ann is float and not isinstance(val, float):
                    result[name] = float(val)
                elif ann is bool and not isinstance(val, bool):
                    result[name] = str(val).lower() in ("true", "1", "yes")
                else:
                    result[name] = val
            except (ValueError, TypeError):
                result[name] = val
        return result

    def to_openai_schema(self) -> dict:
        """导出 OpenAI function calling 所需的 JSON Schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _make_tool(name: str, func: Callable) -> Tool:
    """从函数名和函数体构造 Tool 对象，描述优先从 prompt/tools/{name}.md 加载。"""
    from src.agents.tool_prompt_loader import load_tool_description
    description = load_tool_description(name, fallback=func.__doc__ or "")
    parameters = _infer_schema(func)
    return Tool(name=name, description=description, func=func, parameters=parameters)

# ── 顶层导入（供测试 Mock 使用）─────────────────────────────────────────────
# 采用懒加载避免循环依赖，但同时在模块层面暴露引用，使 unittest.mock.patch 可用。
# 若外部服务（Neo4j / FAISS）未启动，导入本模块不会失败（import 在函数内部执行）。
# 以下别名仅供测试 patch，不在业务代码中直接使用。
try:
    from src.storage.database import MetadataDatabase          # noqa: F401
except Exception:
    MetadataDatabase = None  # type: ignore[assignment,misc]

try:
    from src.storage.graph_store import GraphStore              # noqa: F401
except Exception:
    GraphStore = None  # type: ignore[assignment,misc]

try:
    from src.retrieval.keyword_extractor import KeywordExtractor  # noqa: F401
except Exception:
    KeywordExtractor = None  # type: ignore[assignment,misc]

# ── 工具显示名称映射（用于 SSE ToolStartEvent.display_message）──────────────
TOOL_DISPLAY_NAMES: dict[str, str] = {
    "retrieve_knowledge":        "正在检索知识库...",
    "deep_research":             "正在进行深度研究（可能需要较长时间）...",
    "generate_research_ideas":   "正在生成研究 Idea...",
    "list_documents":            "正在查询文献列表...",
    "get_document_metadata":     "正在获取文献详情...",
    "search_by_entity":          "正在查询知识图谱...",
    "get_knowledge_graph_stats": "正在统计知识图谱...",
    "save_user_memory":          "正在保存长期记忆...",
}


# ════════════════════════════════════════════════════════════════════════════
# 工具 1：retrieve_knowledge
# ════════════════════════════════════════════════════════════════════════════

@tool("retrieve_knowledge")
def retrieve_knowledge(
    query: str,
    mode: str = "auto",
    top_k: int = 5,
    section_filter: str = "",
    year_from: int = 0,
    year_to: int = 0,
) -> str:
    """
    从已入库的学术文献知识库中检索相关内容。

    【使用场景】
    - 用户询问文献中的具体内容、方法、实验结果、结论
    - 需要引用文献支持某个论断时
    - 对比多篇文献的观点时

    【不应调用的场景】
    - 用户询问通用概念（如"什么是 Transformer"），不依赖库中文献时
    - 用户只是在闲聊或问候时
    - 已有足够检索结果，无需再次检索时

    【mode 参数选择指南】
    - auto: 系统自动判断（推荐，适合大多数情况）
    - dual: LightRAG 双极检索，Low-Level 实体精确 + High-Level 关系宏观 + one-hop 扩展
    - local: 适合包含具体实体名/方法名/数据集名的问题
    - global: 适合询问宏观趋势、综述、跨文献比较的问题
    - mix: 适合复杂的综合性问题
    - hybrid: 关键词+语义双路检索，适合精确术语查询
    - semantic: 纯语义向量检索

    【section_filter】可指定章节类型：abstract/method/experiment/conclusion/related_work/""（不限）

    【year_from/year_to】按发表年份过滤，0 表示不限制；也可留空由系统从查询文本自动提取时间约束
    """
    try:
        chunks = _do_retrieve(query=query, mode=mode, top_k=top_k,
                              section_filter=section_filter,
                              year_from=year_from, year_to=year_to)
        if not chunks:
            return (
                "【检索结果为空】\n"
                "在知识库中未找到与查询相关的内容。\n"
                "建议：1) 尝试换用其他 mode 参数；2) 通过文献管理页面入库相关文献后再检索。"
            )
        return _format_chunks(chunks)
    except Exception as exc:
        logger.error("retrieve_knowledge 执行失败: %s", exc, exc_info=True)
        return f"[检索失败] {exc}\n请尝试换用其他参数或稍后重试。"


def _do_retrieve(
    query: str,
    mode: str,
    top_k: int,
    section_filter: str,
    year_from: int,
    year_to: int,
) -> list:
    """
    根据 mode 路由到对应检索器，并按优先级应用时间过滤：
      1. LLM 显式传入 year_from/year_to → 精确过滤（_apply_year_filter）
      2. 两者均为 0 → 从 query 自动提取时间约束 → progressive_retrieve 渐进过滤
      3. 无时间约束 → 返回全量结果

    时间约束存在时，检索阶段扩大到 top_k * 4 再过滤，
    防止前 top_k 结果全为旧文献时新文献无法进入过滤阶段。

    Phase 9-5 长期记忆集成：
      - auto 模式优先查询 LongTermMemory.get_best_mode()，有历史最优时直接使用
      - 检索完成后异步调用 record_strategy_result() 记录本次策略效果
    """
    # Phase 9-5: 分类问题类型，供长期记忆记录和查询使用
    from src.infrastructure.long_term_memory import LongTermMemory, classify_question_type
    ltm = LongTermMemory.get_instance()
    question_type = classify_question_type(query)

    if mode == "auto":
        # Phase 9-5: temporal 查询跳过 LTM，时效性路由由启发式规则保证（[8]）
        if question_type != "temporal":
            history_mode = ltm.get_best_mode(question_type)
        else:
            history_mode = None

        if history_mode:
            mode = history_mode
            logger.debug("auto 模式 → 历史最优: %s (type=%s)", mode, question_type)
        else:
            mode = _auto_select_mode(query)
            logger.debug("auto 模式 → 启发式选择: %s", mode)

    # 判断是否需要扩大召回量：时间约束或 section_filter 均会导致候选被过滤，需提前扩容
    from src.retrieval.time_filter import extract_time_constraint
    has_time = (year_from != 0 or year_to != 0) or bool(extract_time_constraint(query))
    has_filter = bool(section_filter)
    fetch_k = top_k * 4 if (has_time or has_filter) else top_k

    if mode == "dual":
        # Phase 9-1: LightRAG 双极检索（Low-Level + High-Level + one-hop 扩展）
        # try/finally 保证 GraphRetriever / GraphStore 连接在异常时也能释放
        from src.retrieval.lightrag_retriever import LightRAGDualRetriever
        retriever = LightRAGDualRetriever(top_k=fetch_k)
        try:
            chunks = retriever.retrieve(query, top_k=fetch_k, section_filter=section_filter)
        finally:
            retriever.close()
    elif mode == "local":
        from src.retrieval.local_retriever import LocalRetriever
        retriever = LocalRetriever(top_k=fetch_k)
        try:
            chunks = retriever.retrieve(query, section_filter=section_filter)
        finally:
            retriever.close()
    elif mode == "global":
        # 主路径：LightRAGDualRetriever（High-Level 关系检索 + one-hop 扩展）
        # 备用路径：GlobalRetriever（RelationVectorStore，LightRAG 不可用时降级）
        from src.retrieval.lightrag_retriever import LightRAGDualRetriever
        retriever = LightRAGDualRetriever(top_k=fetch_k)
        try:
            chunks = retriever.retrieve(query, top_k=fetch_k, section_filter=section_filter)
        except Exception:
            from src.retrieval.global_retriever import GlobalRetriever
            fallback = GlobalRetriever(top_k=fetch_k)
            try:
                chunks = fallback.retrieve(query, section_filter=section_filter)
            finally:
                fallback.close()
        finally:
            retriever.close()
    elif mode == "mix":
        from src.retrieval.mix_retriever import MixRetriever
        retriever = MixRetriever(top_k=fetch_k)
        try:
            chunks = retriever.retrieve(query, section_filter=section_filter)
        finally:
            retriever.close()
    elif mode == "hybrid":
        from src.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(top_k=fetch_k)
        chunks = retriever.retrieve(query, section_filter=section_filter)
    elif mode == "semantic":
        from src.retrieval.retriever import SemanticRetriever
        retriever = SemanticRetriever(top_k=fetch_k)
        sf = section_filter if section_filter else None
        chunks = retriever.retrieve(query, section_type=sf)
        # SemanticRetriever 已支持 section_type 过滤，直接跳到时间过滤
        result = _apply_time_filter(chunks, query, year_from, year_to, top_k)
        # Phase 9-5: 记录策略效果；continued = 有结果则视为有效（[9]）
        ltm.record_strategy_result(question_type, "semantic", len(result),
                                   continued=len(result) > 0)
        return result
    else:
        # fallback: semantic；effective_mode 记录为 "semantic" 避免非法 mode 污染（[7]）
        logger.warning("未知 mode=%r，fallback 到 semantic", mode)
        from src.retrieval.retriever import SemanticRetriever
        retriever = SemanticRetriever(top_k=fetch_k)
        sf = section_filter if section_filter else None
        chunks = retriever.retrieve(query, section_type=sf)
        mode = "semantic"   # 统一用 mode 变量，后续记录时保持一致

    # Phase 10-2: 各检索器已在内部应用 section_filter（原生或后过滤），此处保留安全兜底
    # 仅 fallback 路径或未来新增检索器未处理时才会过滤到结果
    if section_filter:
        chunks = [c for c in chunks if getattr(c, "section_type", "") == section_filter]

    result = _apply_time_filter(chunks, query, year_from, year_to, top_k)
    # Phase 9-5: 记录策略效果；continued = 有结果视为有效（[9]），不再默认 True
    ltm.record_strategy_result(question_type, mode, len(result),
                               continued=len(result) > 0)
    return result


def _apply_time_filter(
    chunks: list, query: str, year_from: int, year_to: int, top_k: int = 0
) -> list:
    """
    Phase 9-2 统一时间过滤入口。

    优先级：
      1. LLM 显式传入 year_from/year_to（任一非 0）→ 精确过滤（_apply_year_filter）
      2. 两者均为 0 → 从 query 自动提取时间约束 → progressive_retrieve 渐进过滤
      3. query 无时间约束 → 直接返回结果

    top_k > 0 时，过滤后裁剪至 top_k（因检索阶段用了 fetch_k=top_k*4）。
    """
    # LLM 显式指定年份 → 精确过滤（不做渐进放宽，尊重 LLM 判断）
    if year_from != 0 or year_to != 0:
        result = _apply_year_filter(chunks, year_from, year_to)
        return result[:top_k] if top_k > 0 else result

    # 从 query 自动提取时间约束，渐进过滤
    from src.retrieval.time_filter import extract_time_constraint, progressive_retrieve
    constraint = extract_time_constraint(query)
    if constraint:
        logger.debug("时间感知过滤：constraint=%s", constraint)
        chunks = progressive_retrieve(chunks, constraint, min_count=3)

    return chunks[:top_k] if top_k > 0 else chunks


def _apply_year_filter(chunks: list, year_from: int, year_to: int) -> list:
    """
    按 LLM 显式传入的 year_from/year_to 精确过滤 chunk（0 表示不限）。
    Phase 9-2 后仅在 LLM 明确指定年份时调用，自动时间提取走 _apply_time_filter。
    """
    if year_from == 0 and year_to == 0:
        return chunks
    result = []
    for c in chunks:
        year = getattr(c, "year", None)
        if year is None:
            result.append(c)  # 无年份信息时不过滤
        else:
            try:
                y = int(year)
                if year_from and y < year_from:
                    continue
                if year_to and y > year_to:
                    continue
                result.append(c)
            except (ValueError, TypeError):
                result.append(c)
    return result


def _auto_select_mode(query: str) -> str:
    """
    Phase 9-1: 基于 LightRAG 双极关键词自动选择检索模式。

    路由规则（对应 LightRAG §3.2 Dual-Level Retrieval Paradigm）：
        has_ll ∧ has_hl → "dual"   双极检索（Low-Level + High-Level + one-hop 扩展）
        has_ll only     → "local"  Local 检索（实体精确）
        has_hl only     → "global" Global 检索（关系宏观）
        neither         → "mix"    Mix 检索（默认综合）

    时间词优先规则：含时间词的查询直接路由到 "mix"，避免时效性问题。
    """
    try:
        import src.agents.tool_registry as _m
        _KE = _m.KeywordExtractor
        if _KE is None:
            from src.retrieval.keyword_extractor import KeywordExtractor as _KE
        extractor = _KE()
        result = extractor.extract(query)
        hl_count = len(getattr(result, "hl_keywords", []))
        ll_count = len(getattr(result, "ll_keywords", []))

        # 含时间约束 → mix（时效性查询不适合图谱精确匹配，mix 覆盖更广）
        from src.retrieval.time_filter import extract_time_constraint
        if extract_time_constraint(query) is not None:
            return "mix"

        has_ll = ll_count >= 1
        has_hl = hl_count >= 1

        if has_ll and has_hl:
            return "dual"    # LightRAG 双极检索
        elif has_ll:
            return "local"   # 具体实体/方法
        elif has_hl:
            return "global"  # 宏观趋势/综述
        else:
            return "mix"     # 默认综合
    except Exception as exc:
        logger.warning("auto_select_mode 失败，fallback 到 mix: %s", exc)
        return "mix"


def _format_chunks(chunks: list) -> str:
    """将 RetrievedChunk 列表格式化为 Markdown 字符串。"""
    lines = [f"**检索到 {len(chunks)} 个相关片段：**\n"]
    for i, chunk in enumerate(chunks, start=1):
        source  = getattr(chunk, "file_path", "未知来源")
        idx     = getattr(chunk, "chunk_index", "?")
        sec     = getattr(chunk, "section_type", "")
        year    = getattr(chunk, "year", None)
        content = getattr(chunk, "content", str(chunk))
        source_tag = f"[来源: {source}#chunk-{idx}]"
        year_tag   = f"（年份: {year}）" if year else ""
        sec_tag    = f"（章节: {sec}）" if sec and sec not in ("unknown", "relation", "entity") else ""
        lines.append(f"**[{i}]** {source_tag}{year_tag}{sec_tag}")
        lines.append(content.strip())
        lines.append("")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 工具 2：deep_research
# ════════════════════════════════════════════════════════════════════════════

@tool("deep_research")
def deep_research(
    question: str,
    retriever_mode: str = "mix",
    max_subquestions: int = 3,
    use_community: bool = False,
) -> str:
    """
    对复杂研究问题进行多步骤深度研究，生成结构化研究报告。

    【使用场景】
    - 用户需要综述某个研究领域/方向
    - 问题复杂，需要从多个角度分析
    - 用户明确要求"深度研究"、"综述"、"全面分析"时

    【不应调用的场景】
    - 简单的事实查询（用 retrieve_knowledge 即可）
    - 用户只需要一个简短回答时

    【参数说明】
    - retriever_mode: 内部检索模式（mix/local/global/hybrid/semantic）
    - max_subquestions: 最多拆解多少个子问题（3-5 为佳）
    - use_community: 是否利用知识图谱社区摘要（需要已完成社区检测）

    注意：此工具耗时较长（通常 30-120 秒），调用前应告知用户。
    """
    try:
        from src.agents.deep_research_agent import DeepResearchAgent
        import uuid

        # RetrieverMode 为 Literal，需校验
        valid_modes = {"semantic", "hybrid", "graph", "local", "global", "mix"}
        rm = retriever_mode if retriever_mode in valid_modes else "mix"

        agent = DeepResearchAgent(
            max_subquestions=max_subquestions,
            retriever_mode=rm,            # type: ignore[arg-type]
            use_community=use_community,
        )

        thread_id = f"tool_{uuid.uuid4().hex[:8]}"
        report = agent.research(thread_id=thread_id, question=question)

        return report.final_report or "深度研究完成，但未能生成报告文本。"

    except Exception as exc:
        logger.error("deep_research 执行失败: %s", exc, exc_info=True)
        return (
            f"[深度研究失败] {exc}\n"
            "可能原因：知识库尚无相关文献，或检索器初始化失败。\n"
            "建议：先用 list_documents 确认知识库内容，再重试。"
        )


# ════════════════════════════════════════════════════════════════════════════
# 工具 3：generate_research_ideas
# ════════════════════════════════════════════════════════════════════════════

@tool("generate_research_ideas")
def generate_research_ideas(
    topic: str,
    research_report: str = "",
) -> str:
    """
    基于研究报告或主题生成创新研究 Idea。

    【使用场景】
    - 用户希望获得研究方向建议、研究 Idea
    - 通常在 deep_research 之后调用，基于报告生成 Idea
    - 用户说"给我一些想法/方向/创新点"时

    【建议】先调用 deep_research 获取报告，再将报告传入此工具，效果更好。

    【参数说明】
    - topic: 研究主题（必填）
    - research_report: deep_research 返回的报告文本（可选，填入后质量更高）
    """
    try:
        from src.agents.idea_agent import IdeaAgent

        agent = IdeaAgent()

        if research_report.strip():
            idea_report = agent.generate_from_markdown(
                question=topic,
                report_markdown=research_report,
            )
        else:
            # 无报告时先执行简单检索再生成
            from src.agents.deep_research_agent import DeepResearchAgent
            import uuid
            dr_agent = DeepResearchAgent(max_subquestions=2, retriever_mode="mix")
            thread_id = f"idea_{uuid.uuid4().hex[:8]}"
            dr_report = dr_agent.research(thread_id=thread_id, question=topic)
            idea_report = agent.generate(question=topic, report=dr_report)

        return IdeaAgent.to_markdown(idea_report)

    except Exception as exc:
        logger.error("generate_research_ideas 执行失败: %s", exc, exc_info=True)
        return (
            f"[Idea 生成失败] {exc}\n"
            "建议：先调用 deep_research 获取报告，再将报告传入此工具。"
        )


# ════════════════════════════════════════════════════════════════════════════
# 工具 4：list_documents
# ════════════════════════════════════════════════════════════════════════════

@tool("list_documents")
def list_documents(keyword: str = "", limit: int = 20) -> str:
    """
    列出知识库中已入库的文献清单，支持关键词筛选。

    【使用场景】
    - 用户问"库里有什么论文"、"有没有关于 XXX 的文献"
    - 需要先了解知识库内容再决定如何检索时

    【参数说明】
    - keyword: 关键词过滤（匹配标题/作者/摘要/关键词，为空则列出所有）
    - limit: 最多返回条数（默认 20）
    """
    try:
        import src.agents.tool_registry as _m
        _DB = _m.MetadataDatabase
        if _DB is None:
            from src.storage.database import MetadataDatabase as _DB

        db = _DB()
        all_docs = db.list_documents()

        # 关键词过滤
        if keyword.strip():
            kw = keyword.strip().lower()
            all_docs = [
                d for d in all_docs
                if (kw in (d.title or "").lower()
                    or kw in (d.authors or "").lower()
                    or kw in (d.abstract or "").lower()
                    or kw in (d.keywords or "").lower())
            ]

        if not all_docs:
            hint = f"（关键词 '{keyword}' 过滤后）" if keyword else ""
            return f"知识库{hint}中暂无文献。请通过文献管理页面入库相关文献。"

        total = len(all_docs)
        docs = all_docs[:limit]

        lines = [f"**知识库文献清单**（共 {total} 篇，显示前 {len(docs)} 篇）：\n"]
        for i, doc in enumerate(docs, start=1):
            title   = doc.title or "（无标题）"
            authors = doc.authors or "未知作者"
            year    = str(doc.year) if doc.year else "年份未知"
            doc_id  = doc.doc_id or doc.file_path or "?"
            lines.append(f"{i}. **{title}** — {authors}，{year}  \n   ID: `{doc_id}`")

        if total > limit:
            lines.append(f"\n_（还有 {total - limit} 篇未显示，可通过 keyword 参数缩小范围）_")

        return "\n".join(lines)

    except Exception as exc:
        logger.error("list_documents 执行失败: %s", exc, exc_info=True)
        return f"[查询失败] {exc}"


# ════════════════════════════════════════════════════════════════════════════
# 工具 5：get_document_metadata
# ════════════════════════════════════════════════════════════════════════════

@tool("get_document_metadata")
def get_document_metadata(doc_title_or_id: str) -> str:
    """
    获取特定文献的详细元数据（标题、作者、摘要、发表年份、入库时间等）。

    【使用场景】
    - 用户询问特定论文的详情
    - 需要确认某篇文献是否入库及其基本信息

    【参数说明】
    - doc_title_or_id: 文献标题（支持模糊匹配）或文献 doc_id
    """
    try:
        import src.agents.tool_registry as _m
        _DB = _m.MetadataDatabase
        if _DB is None:
            from src.storage.database import MetadataDatabase as _DB

        db = _DB()

        # 优先按 doc_id 精确查询
        doc = db.get_document_by_doc_id(doc_title_or_id)

        # 未找到则按标题模糊搜索
        if doc is None:
            kw = doc_title_or_id.strip().lower()
            all_docs = db.list_documents()
            matched = [d for d in all_docs if kw in (d.title or "").lower()]
            if not matched:
                return (
                    f"未找到与 '{doc_title_or_id}' 匹配的文献。\n"
                    "建议：先用 list_documents 查看已入库文献列表。"
                )
            doc = matched[0]  # 取最佳匹配
            if len(matched) > 1:
                others = "、".join(
                    (d.title or d.doc_id or "?") for d in matched[1:4]
                )
                # 附加提示，但继续返回第一个匹配
                extra = f"\n\n_（另找到 {len(matched)-1} 篇相似文献：{others}，如需查看请缩小关键词）_"
            else:
                extra = ""
        else:
            extra = ""

        lines = [f"## 文献详情：{doc.title or '（无标题）'}"]
        lines.append(f"- **作者**：{doc.authors or '未知'}")
        lines.append(f"- **年份**：{doc.year or '未知'}")
        lines.append(f"- **机构**：{doc.institution or '未知'}")
        lines.append(f"- **关键词**：{doc.keywords or '未知'}")
        lines.append(f"- **Doc ID**：`{doc.doc_id or '?'}`")
        lines.append(f"- **文件路径**：`{doc.file_path}`")
        if doc.abstract:
            lines.append(f"\n**摘要**：\n{doc.abstract}")

        return "\n".join(lines) + extra

    except Exception as exc:
        logger.error("get_document_metadata 执行失败: %s", exc, exc_info=True)
        return f"[查询失败] {exc}"


# ════════════════════════════════════════════════════════════════════════════
# 工具 6：search_by_entity
# ════════════════════════════════════════════════════════════════════════════

@tool("search_by_entity")
def search_by_entity(entity_name: str, relation_type: str = "") -> str:
    """
    在知识图谱中查询实体及其关联关系。

    【使用场景】
    - 用户询问"X 和 Y 有什么关系"
    - 用户询问某个具体实体（作者、方法、模型）的相关信息
    - 需要探索实体之间的网络关系时

    【参数说明】
    - entity_name: 实体名称（如论文名、方法名、作者名）
    - relation_type: 可选，过滤关系类型（如 "CITES"/"USES_METHOD"/"PROPOSES"）
    """
    try:
        import src.agents.tool_registry as _m
        _GS = _m.GraphStore
        if _GS is None:
            from src.storage.graph_store import GraphStore as _GS

        gs   = _GS()
        subgraph = gs.get_subgraph(
            entity_name=entity_name,
            relation_type=relation_type if relation_type else None,
        )
        gs.close()

        nodes     = subgraph.get("nodes", [])
        relations = subgraph.get("relations", [])

        if not nodes and not relations:
            return (
                f"未在知识图谱中找到实体 '{entity_name}' 的相关信息。\n"
                "建议：确认实体名称是否正确，或知识图谱尚未提取该实体。"
            )

        lines = [f"**实体查询结果：{entity_name}**\n"]

        if nodes:
            lines.append(f"**实体节点（{len(nodes)} 个）：**")
            for n in nodes[:10]:
                name  = n.get("name") or n.get("id", "?")
                ntype = n.get("type") or n.get("entity_type", "")
                desc  = n.get("description", "")
                tag   = f"（{ntype}）" if ntype else ""
                lines.append(f"- **{name}**{tag}" + (f"：{desc}" if desc else ""))

        if relations:
            lines.append(f"\n**关联关系（{len(relations)} 条）：**")
            for r in relations[:15]:
                src  = r.get("source") or r.get("from", "?")
                rtype = r.get("type") or r.get("relation_type", "关联")
                tgt  = r.get("target") or r.get("to", "?")
                desc = r.get("description", "")
                lines.append(f"- `{src}` —[{rtype}]→ `{tgt}`" + (f"：{desc}" if desc else ""))

        return "\n".join(lines)

    except Exception as exc:
        logger.error("search_by_entity 执行失败: %s", exc, exc_info=True)
        return f"[知识图谱查询失败] {exc}"


# ════════════════════════════════════════════════════════════════════════════
# 工具 7：get_knowledge_graph_stats
# ════════════════════════════════════════════════════════════════════════════

@tool("get_knowledge_graph_stats")
def get_knowledge_graph_stats() -> str:
    """
    获取知识图谱的统计信息（实体数、关系数、文档数、社区数等）。

    【使用场景】
    - 用户询问"知识库有多大"、"有多少实体"、"覆盖多少文献"
    - 系统状态查询、了解知识图谱整体规模
    """
    try:
        import src.agents.tool_registry as _m
        _GS = _m.GraphStore
        if _GS is None:
            from src.storage.graph_store import GraphStore as _GS
        _DB = _m.MetadataDatabase
        if _DB is None:
            from src.storage.database import MetadataDatabase as _DB

        gs    = _GS()
        stats = gs.get_graph_stats()
        gs.close()

        db        = _DB()
        doc_count = len(db.list_documents())

        lines = ["**📊 知识图谱统计信息**\n"]
        lines.append(f"| 指标 | 数量 |")
        lines.append(f"|------|------|")
        lines.append(f"| 已入库文献 | {doc_count} 篇 |")

        for key, val in stats.items():
            # 将 snake_case key 转为可读标签
            label = {
                "node_count":       "图谱节点总数",
                "entity_count":     "实体节点数",
                "document_count":   "文档节点数",
                "chunk_count":      "文本块节点数",
                "relation_count":   "关系总数",
                "community_count":  "社区数",
            }.get(key, key)
            lines.append(f"| {label} | {val} |")

        return "\n".join(lines)

    except Exception as exc:
        logger.error("get_knowledge_graph_stats 执行失败: %s", exc, exc_info=True)
        return f"[统计信息获取失败] {exc}\n知识图谱服务可能未启动，请检查 Neo4j 连接。"


# ════════════════════════════════════════════════════════════════════════════
# 工具 8：save_user_memory（Phase 9-5-2）
# ════════════════════════════════════════════════════════════════════════════

@tool("save_user_memory")
def save_user_memory(
    title: str,
    body: str,
    memory_type: str = "preference",
) -> str:
    """
    将用户明确要求持久化的偏好、规则或重要信息保存到长期记忆中。

    【使用场景】
    - 用户说"记住这个偏好"/"以后都这样做"/"记录一下这条规则"时立即调用
    - 用户明确要求持久化某个策略或行为准则时

    【不应调用的场景】
    - 普通对话内容（无需持久化的临时信息）
    - 用户没有明确要求记忆的内容
    - 单次检索结论、临时问题的答案

    【参数说明】
    - title: 记忆标题（简洁，10-30 字）
    - body: 记忆正文（详细描述，支持 Markdown）
    - memory_type: 类型标签（preference/rule/note，默认 preference）
    """
    # [建议] memory_type 白名单校验，主模型直写只允许 preference/rule/note
    _ALLOWED_TYPES = {"preference", "rule", "note"}
    if memory_type not in _ALLOWED_TYPES:
        memory_type = "preference"

    try:
        from src.infrastructure.long_term_memory import LongTermMemory, _slugify
        ltm  = LongTermMemory.get_instance()
        slug = _slugify(title)

        # [必须修复] description 从 body 首行提取，而非直接复制 title
        first_line = body.strip().splitlines()[0].strip() if body.strip() else title
        description = first_line[:100]  # 限制在 100 字内

        ltm.save_memory(
            title=title,
            slug=slug,
            description=description,
            body=body,
            metadata={"type": memory_type, "source": "user_explicit"},
        )
        logger.info("save_user_memory: 已写入记忆 %s.md", slug)
        return f"[记忆已保存] 标题：{title}，文件：{slug}.md"
    except Exception as exc:
        logger.error("save_user_memory 执行失败: %s", exc, exc_info=True)
        return f"[记忆保存失败] {exc}"


# ════════════════════════════════════════════════════════════════════════════
# 工具注册入口（对应 claw-code 的 @lru_cache + load_tool_snapshot）
# ════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def build_tool_registry() -> list:
    """
    一次性构建工具列表，通过 @lru_cache 保证全局单例。

    启动时调用一次，后续请求直接复用，避免重复初始化开销。

    Returns:
        list[StructuredTool] — 可直接传入 llm.bind_tools(tools) 的工具列表
    """
    tools = [
        retrieve_knowledge,
        deep_research,
        generate_research_ideas,
        list_documents,
        get_document_metadata,
        search_by_entity,
        get_knowledge_graph_stats,
        save_user_memory,
    ]
    logger.info("工具注册完成：%s", [t.name for t in tools])
    return tools


# ════════════════════════════════════════════════════════════════════════════
# P5-Step 1：原生工具业务函数（_xxx_func 命名，与 @tool 版本共存）
# ════════════════════════════════════════════════════════════════════════════

def _retrieve_knowledge_func(
    query: str,
    mode: str = "auto",
    top_k: int = 5,
    section_filter: str = "",
    year_from: int = 0,
    year_to: int = 0,
) -> str:
    """从已入库的学术文献知识库中检索相关内容。"""
    try:
        chunks = _do_retrieve(query=query, mode=mode, top_k=top_k,
                              section_filter=section_filter,
                              year_from=year_from, year_to=year_to)
        if not chunks:
            return (
                "【检索结果为空】\n"
                "在知识库中未找到与查询相关的内容。\n"
                "建议：1) 尝试换用其他 mode 参数；2) 通过文献管理页面入库相关文献后再检索。"
            )
        return _format_chunks(chunks)
    except Exception as exc:
        logger.error("retrieve_knowledge 执行失败: %s", exc, exc_info=True)
        return f"[检索失败] {exc}\n请尝试换用其他参数或稍后重试。"


def _deep_research_func(
    question: str,
    retriever_mode: str = "mix",
    max_subquestions: int = 3,
    use_community: bool = False,
) -> str:
    """对复杂研究问题进行多步骤深度研究，生成结构化研究报告。"""
    try:
        from src.agents.deep_research_agent import DeepResearchAgent
        import uuid
        valid_modes = {"semantic", "hybrid", "graph", "local", "global", "mix"}
        rm = retriever_mode if retriever_mode in valid_modes else "mix"
        agent = DeepResearchAgent(
            max_subquestions=max_subquestions,
            retriever_mode=rm,
            use_community=use_community,
        )
        thread_id = f"tool_{uuid.uuid4().hex[:8]}"
        report = agent.research(thread_id=thread_id, question=question)
        return report.final_report or "深度研究完成，但未能生成报告文本。"
    except Exception as exc:
        logger.error("deep_research 执行失败: %s", exc, exc_info=True)
        return (
            f"[深度研究失败] {exc}\n"
            "可能原因：知识库尚无相关文献，或检索器初始化失败。\n"
            "建议：先用 list_documents 确认知识库内容，再重试。"
        )


def _generate_research_ideas_func(topic: str, research_report: str = "") -> str:
    """基于研究报告或主题生成创新研究 Idea。"""
    try:
        from src.agents.idea_agent import IdeaAgent
        agent = IdeaAgent()
        if research_report.strip():
            idea_report = agent.generate_from_markdown(
                question=topic, report_markdown=research_report,
            )
        else:
            from src.agents.deep_research_agent import DeepResearchAgent
            import uuid
            dr_agent = DeepResearchAgent(max_subquestions=2, retriever_mode="mix")
            thread_id = f"idea_{uuid.uuid4().hex[:8]}"
            dr_report = dr_agent.research(thread_id=thread_id, question=topic)
            idea_report = agent.generate(question=topic, report=dr_report)
        return IdeaAgent.to_markdown(idea_report)
    except Exception as exc:
        logger.error("generate_research_ideas 执行失败: %s", exc, exc_info=True)
        return (
            f"[Idea 生成失败] {exc}\n"
            "建议：先调用 deep_research 获取报告，再将报告传入此工具。"
        )


def _list_documents_func(keyword: str = "", limit: int = 20) -> str:
    """列出知识库中已入库的文献清单，支持关键词筛选。"""
    try:
        import src.agents.tool_registry as _m
        _DB = _m.MetadataDatabase
        if _DB is None:
            from src.storage.database import MetadataDatabase as _DB
        db = _DB()
        all_docs = db.list_documents()
        if keyword.strip():
            kw = keyword.strip().lower()
            all_docs = [
                d for d in all_docs
                if (kw in (d.title or "").lower()
                    or kw in (d.authors or "").lower()
                    or kw in (d.abstract or "").lower()
                    or kw in (d.keywords or "").lower())
            ]
        if not all_docs:
            hint = f"（关键词 '{keyword}' 过滤后）" if keyword else ""
            return f"知识库{hint}中暂无文献。请通过文献管理页面入库相关文献。"
        total = len(all_docs)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 20
        docs = all_docs[:limit]
        lines = [f"**知识库文献清单**（共 {total} 篇，显示前 {len(docs)} 篇）：\n"]
        for i, doc in enumerate(docs, start=1):
            title   = doc.title or "（无标题）"
            authors = doc.authors or "未知作者"
            year    = str(doc.year) if doc.year else "年份未知"
            doc_id  = doc.doc_id or doc.file_path or "?"
            lines.append(f"{i}. **{title}** — {authors}，{year}  \n   ID: `{doc_id}`")
        if total > limit:
            lines.append(f"\n_（还有 {total - limit} 篇未显示，可通过 keyword 参数缩小范围）_")
        return "\n".join(lines)
    except Exception as exc:
        logger.error("list_documents 执行失败: %s", exc, exc_info=True)
        return f"[查询失败] {exc}"


def _get_document_metadata_func(doc_title_or_id: str) -> str:
    """获取特定文献的详细元数据。"""
    try:
        import src.agents.tool_registry as _m
        _DB = _m.MetadataDatabase
        if _DB is None:
            from src.storage.database import MetadataDatabase as _DB
        db = _DB()
        doc = db.get_document_by_doc_id(doc_title_or_id)
        if doc is None:
            kw = doc_title_or_id.strip().lower()
            all_docs = db.list_documents()
            matched = [d for d in all_docs if kw in (d.title or "").lower()]
            if not matched:
                return (
                    f"未找到与 '{doc_title_or_id}' 匹配的文献。\n"
                    "建议：先用 list_documents 查看已入库文献列表。"
                )
            doc = matched[0]
            extra = (
                f"\n\n_（另找到 {len(matched)-1} 篇相似文献，如需查看请缩小关键词）_"
                if len(matched) > 1 else ""
            )
        else:
            extra = ""
        lines = [f"## 文献详情：{doc.title or '（无标题）'}"]
        lines.append(f"- **作者**：{doc.authors or '未知'}")
        lines.append(f"- **年份**：{doc.year or '未知'}")
        lines.append(f"- **机构**：{doc.institution or '未知'}")
        lines.append(f"- **关键词**：{doc.keywords or '未知'}")
        lines.append(f"- **Doc ID**：`{doc.doc_id or '?'}`")
        lines.append(f"- **文件路径**：`{doc.file_path}`")
        if doc.abstract:
            lines.append(f"\n**摘要**：\n{doc.abstract}")
        return "\n".join(lines) + extra
    except Exception as exc:
        logger.error("get_document_metadata 执行失败: %s", exc, exc_info=True)
        return f"[查询失败] {exc}"


def _search_by_entity_func(entity_name: str, relation_type: str = "") -> str:
    """在知识图谱中查询实体及其关联关系。"""
    try:
        import src.agents.tool_registry as _m
        _GS = _m.GraphStore
        if _GS is None:
            from src.storage.graph_store import GraphStore as _GS
        gs = _GS()
        subgraph = gs.get_subgraph(
            search=entity_name,
            limit=50,
        )
        gs.close()
        nodes     = subgraph.get("nodes", [])
        relations = subgraph.get("edges", [])
        if not nodes and not relations:
            return (
                f"未在知识图谱中找到实体 '{entity_name}' 的相关信息。\n"
                "建议：确认实体名称是否正确，或知识图谱尚未提取该实体。"
            )
        lines = [f"**实体查询结果：{entity_name}**\n"]
        if nodes:
            lines.append(f"**实体节点（{len(nodes)} 个）：**")
            for n in nodes[:10]:
                name  = n.get("name") or n.get("id", "?")
                ntype = n.get("type") or n.get("entity_type", "")
                desc  = n.get("description", "")
                tag   = f"（{ntype}）" if ntype else ""
                lines.append(f"- **{name}**{tag}" + (f"：{desc}" if desc else ""))
        if relations:
            lines.append(f"\n**关联关系（{len(relations)} 条）：**")
            for r in relations[:15]:
                src   = r.get("source", "?")
                rtype = r.get("type") or r.get("label", "关联")
                tgt   = r.get("target", "?")
                desc  = r.get("description", "")
                lines.append(f"- `{src}` —[{rtype}]→ `{tgt}`" + (f"：{desc}" if desc else ""))
        return "\n".join(lines)
    except Exception as exc:
        logger.error("search_by_entity 执行失败: %s", exc, exc_info=True)
        return f"[知识图谱查询失败] {exc}"


def _get_knowledge_graph_stats_func() -> str:
    """获取知识图谱的统计信息。"""
    try:
        import src.agents.tool_registry as _m
        _GS = _m.GraphStore
        if _GS is None:
            from src.storage.graph_store import GraphStore as _GS
        _DB = _m.MetadataDatabase
        if _DB is None:
            from src.storage.database import MetadataDatabase as _DB
        gs    = _GS()
        stats = gs.get_graph_stats()
        gs.close()
        db        = _DB()
        doc_count = len(db.list_documents())
        lines = ["**知识图谱统计信息**\n", "| 指标 | 数量 |", "|------|------|",
                 f"| 已入库文献 | {doc_count} 篇 |"]
        label_map = {
            "node_count": "图谱节点总数", "entity_count": "实体节点数",
            "document_count": "文档节点数", "chunk_count": "文本块节点数",
            "relation_count": "关系总数", "community_count": "社区数",
        }
        for key, val in stats.items():
            lines.append(f"| {label_map.get(key, key)} | {val} |")
        return "\n".join(lines)
    except Exception as exc:
        logger.error("get_knowledge_graph_stats 执行失败: %s", exc, exc_info=True)
        return f"[统计信息获取失败] {exc}\n知识图谱服务可能未启动，请检查 Neo4j 连接。"


def _save_user_memory_func(title: str, body: str, memory_type: str = "preference") -> str:
    """将用户明确要求持久化的偏好、规则或重要信息保存到长期记忆中。"""
    _ALLOWED_TYPES = {"preference", "rule", "note"}
    if memory_type not in _ALLOWED_TYPES:
        memory_type = "preference"
    try:
        from src.infrastructure.long_term_memory import LongTermMemory, _slugify
        ltm  = LongTermMemory.get_instance()
        slug = _slugify(title)
        first_line = body.strip().splitlines()[0].strip() if body.strip() else title
        description = first_line[:100]
        ltm.save_memory(
            title=title, slug=slug, description=description, body=body,
            metadata={"type": memory_type, "source": "user_explicit"},
        )
        logger.info("save_user_memory: 已写入记忆 %s.md", slug)
        return f"[记忆已保存] 标题：{title}，文件：{slug}.md"
    except Exception as exc:
        logger.error("save_user_memory 执行失败: %s", exc, exc_info=True)
        return f"[记忆保存失败] {exc}"


@lru_cache(maxsize=1)
def build_native_tool_registry() -> list[Tool]:
    """原生工具注册表（P5-Step 1 新增，与 build_tool_registry() 并行存在）。"""
    _defs: list[tuple[str, Callable]] = [
        ("retrieve_knowledge",        _retrieve_knowledge_func),
        ("deep_research",             _deep_research_func),
        ("generate_research_ideas",   _generate_research_ideas_func),
        ("list_documents",            _list_documents_func),
        ("get_document_metadata",     _get_document_metadata_func),
        ("search_by_entity",          _search_by_entity_func),
        ("get_knowledge_graph_stats", _get_knowledge_graph_stats_func),
        ("save_user_memory",          _save_user_memory_func),
    ]
    tools = [_make_tool(name, func) for name, func in _defs]
    logger.info("原生工具注册完成：%s", [t.name for t in tools])
    return tools
