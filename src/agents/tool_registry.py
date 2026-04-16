"""
七个核心工具的定义与注册（Phase 8-3）。

将现有 Agent 和检索器封装为 LLM 可调用的 @tool，对应 claw-code 的
`load_tool_snapshot()` + `@lru_cache` 模式：
- 启动时通过 build_tool_registry() 一次性构建工具列表（@lru_cache 保证单例）
- 工具内部复用 src/retrieval/ 和 src/agents/ 中已有的检索器/Agent，不重写业务逻辑
- 所有工具均返回格式化的 Markdown 字符串，供 LLM 直接读取并引用

工具清单：
    1. retrieve_knowledge        — 知识库多模式检索
    2. deep_research             — 多步深度研究报告
    3. generate_research_ideas   — 研究 Idea 生成
    4. list_documents            — 文献清单查询
    5. get_document_metadata     — 单篇文献元数据
    6. search_by_entity          — 知识图谱实体查询
    7. get_knowledge_graph_stats — 知识图谱统计信息

用法：
    from src.agents.tool_registry import build_tool_registry, TOOL_DISPLAY_NAMES
    tools = build_tool_registry()   # 返回 list[StructuredTool]
"""
from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── 顶层导入（供测试 Mock 使用）─────────────────────────────────────────────
# 采用懒加载避免循环依赖，但同时在模块层面暴露引用，使 unittest.mock.patch 可用。
# 若外部服务（Neo4j / FAISS）未启动，导入本模块不会失败（import 在函数内部执行）。
# 以下别名仅供测试 patch，不在业务代码中直接使用。
try:
    from src.database import MetadataDatabase          # noqa: F401
except Exception:
    MetadataDatabase = None  # type: ignore[assignment,misc]

try:
    from src.graph_store import GraphStore              # noqa: F401
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
    - local: 适合包含具体实体名/方法名/数据集名的问题
    - global: 适合询问宏观趋势、综述、跨文献比较的问题
    - mix: 适合复杂的综合性问题
    - hybrid: 关键词+语义双路检索，适合精确术语查询
    - semantic: 纯语义向量检索

    【section_filter】可指定章节类型：abstract/method/experiment/conclusion/related_work/""（不限）

    【year_from/year_to】按发表年份过滤，0 表示不限制
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
    """根据 mode 路由到对应检索器，并应用 section_filter 后过滤。"""
    if mode == "auto":
        mode = _auto_select_mode(query)
        logger.debug("auto 模式选择: %s", mode)

    if mode == "local":
        from src.retrieval.local_retriever import LocalRetriever
        retriever = LocalRetriever(top_k=top_k)
        chunks = retriever.retrieve(query)
    elif mode == "global":
        from src.retrieval.global_retriever import GlobalRetriever
        retriever = GlobalRetriever(top_k=top_k)
        chunks = retriever.retrieve(query)
    elif mode == "mix":
        from src.retrieval.mix_retriever import MixRetriever
        retriever = MixRetriever(top_k=top_k)
        chunks = retriever.retrieve(query)
    elif mode == "hybrid":
        from src.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(top_k=top_k)
        chunks = retriever.retrieve(query)
    elif mode == "semantic":
        from src.retriever import SemanticRetriever
        retriever = SemanticRetriever(top_k=top_k)
        sf = section_filter if section_filter else None
        chunks = retriever.retrieve(query, section_type=sf)
        # SemanticRetriever 已支持 section_type 过滤，直接返回
        return _apply_year_filter(chunks, year_from, year_to)
    else:
        # fallback: semantic
        logger.warning("未知 mode=%r，fallback 到 semantic", mode)
        from src.retriever import SemanticRetriever
        retriever = SemanticRetriever(top_k=top_k)
        chunks = retriever.retrieve(query)

    # section_filter 后过滤（除 semantic 外，其余检索器不原生支持）
    if section_filter:
        chunks = [c for c in chunks if getattr(c, "section_type", "") == section_filter]

    return _apply_year_filter(chunks, year_from, year_to)


def _apply_year_filter(chunks: list, year_from: int, year_to: int) -> list:
    """按 year_from/year_to 过滤 chunk（0 表示不限）。"""
    if year_from == 0 and year_to == 0:
        return chunks
    result = []
    for c in chunks:
        # RetrievedChunk 暂无 year 字段，通过 metadata 获取（Phase 9-2 补充）
        # 此处预留兼容：有 year 字段时过滤，无则保留
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
    根据问题特征自动选择检索模式（复用 KeywordExtractor）。
    Phase 9-1 会将此逻辑迁移到 src/retrieval/auto_selector.py。
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

        # 含时间词 → mix
        time_words = ["最新", "最近", "近期", "近年", "latest", "recent",
                      "2023", "2024", "2025", "2026"]
        if any(w in query for w in time_words):
            return "mix"

        if hl_count > ll_count * 1.5 and hl_count >= 2:
            return "global"     # 宏观趋势/综述
        elif ll_count >= 1:
            return "local"      # 具体实体/方法
        else:
            return "mix"        # 默认综合
    except Exception as exc:
        logger.warning("auto_select_mode 失败，fallback 到 mix: %s", exc)
        return "mix"


def _format_chunks(chunks: list) -> str:
    """将 RetrievedChunk 列表格式化为 Markdown 字符串。"""
    lines = [f"**检索到 {len(chunks)} 个相关片段：**\n"]
    for i, chunk in enumerate(chunks, start=1):
        source = getattr(chunk, "file_path", "未知来源")
        idx    = getattr(chunk, "chunk_index", "?")
        sec    = getattr(chunk, "section_type", "")
        content = getattr(chunk, "content", str(chunk))
        source_tag = f"[来源: {source}#chunk-{idx}]"
        sec_tag    = f"（章节: {sec}）" if sec and sec != "unknown" else ""
        lines.append(f"**[{i}]** {source_tag}{sec_tag}")
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
            from src.database import MetadataDatabase as _DB

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
            from src.database import MetadataDatabase as _DB

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
            from src.graph_store import GraphStore as _GS

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
            from src.graph_store import GraphStore as _GS
        _DB = _m.MetadataDatabase
        if _DB is None:
            from src.database import MetadataDatabase as _DB

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
    ]
    logger.info("工具注册完成：%s", [t.name for t in tools])
    return tools
