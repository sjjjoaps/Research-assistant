"""
turn-end 后台记忆提取器（Phase 9-5-3）。

职责：
    分析单轮对话（user_input + AI 最终回答），通过 LLM 判断是否有值得归档的
    长期记忆（用户偏好、规则、重要结论等），若有则写入 data/memory/ 目录。

触发条件：
    MasterAgent._fire_memory_extractor() 在每轮 turn end 后台线程中调用本模块，
    仅在主模型本轮 **未主动调用 save_user_memory** 时触发。

去重逻辑（三层）：
    1. slug 碰撞：data/memory/{slug}.md 已存在 → 跳过
    2. title 匹配：MEMORY.md 索引中已有该 title → 跳过
    3. description 相似：同类型文件中词重叠率 > 0.7 → 跳过

权限约束：
    - 只允许写入 data/memory/ 目录（通过 LongTermMemory.save_memory()）
    - 不允许修改任何业务文件（src/、api/、tests/ 等）
    - LLM 调用使用 temperature=0.0 保证确定性
    - 全程 try/except 包裹，失败只记 WARNING，不影响主流程

LLM 判断输出格式（JSON）：
    {
        "should_save": true,
        "title": "偏好：优先查看实验部分",
        "body":  "用户希望在检索时优先返回 experiment/method 章节内容。",
        "type":  "preference"
    }
    或：
    {
        "should_save": false
    }
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from src.agents.prompt_loader import load_system_prompt

logger = logging.getLogger(__name__)

# 提取器 LLM Prompt（统一 Prompt 管理，fallback 内置于 prompt_loader）
_EXTRACTOR_TEMPLATE = load_system_prompt("memory_extractor")

# 词重叠去重阈值
_SIMILARITY_THRESHOLD = 0.7


def _word_overlap_ratio(text_a: str, text_b: str) -> float:
    """计算两段文本的词重叠率（Jaccard 相似度）。"""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _is_duplicate(slug: str, title: str, description: str, memory_type: str) -> bool:
    """
    三层去重检查：
        1. slug 碰撞：文件已存在
        2. title 精确匹配：MEMORY.md 索引中已有该 title
        3. description 相似：同类型文件中词重叠率 > _SIMILARITY_THRESHOLD
    """
    try:
        from src.core.long_term_memory import LongTermMemory
        ltm          = LongTermMemory.get_instance()
        memory_dir   = ltm._memory_dir

        # 1. slug 碰撞检查
        if (memory_dir / f"{slug}.md").exists():
            logger.debug("后台 extractor 去重：slug %s 已存在，跳过", slug)
            return True

        # 2. title 精确匹配（扫描 MEMORY.md 索引）
        index = ltm.list_memory_index()
        existing_titles = {entry.get("title", "") for entry in index}
        if title in existing_titles:
            logger.debug("后台 extractor 去重：title '%s' 已在索引中，跳过", title)
            return True

        # 3. description 相似度（扫描同类型文件）
        for md_file in memory_dir.glob("*.md"):
            if md_file.name == "MEMORY.md":
                continue
            meta, _ = ltm._read_file(md_file)
            if not meta:
                continue
            if meta.get("type") != memory_type:
                continue
            existing_desc = meta.get("description", "")
            if _word_overlap_ratio(description, existing_desc) > _SIMILARITY_THRESHOLD:
                logger.debug(
                    "后台 extractor 去重：description 与 %s 相似度过高，跳过",
                    md_file.name,
                )
                return True

    except Exception as exc:
        logger.warning("后台 extractor 去重检查失败: %s", exc)

    return False


def _call_extractor_llm(user_input: str, ai_response: str) -> Optional[dict]:
    """
    调用 LLM 判断是否有值得归档的记忆。

    Returns:
        解析后的 dict（含 should_save 字段），或 None（调用/解析失败）
    """
    try:
        from src.infrastructure.llm_client import get_llm

        prompt  = _EXTRACTOR_TEMPLATE.replace("{user_input}", user_input).replace("{ai_response}", ai_response)
        llm     = get_llm(temperature=0.0)
        result  = llm.invoke(prompt)
        content = str(result.content).strip()

        # 提取 JSON（兼容被 markdown 包裹的情况）
        json_match = re.search(r"\{.*?\}", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        logger.warning("后台 extractor LLM 返回内容无法解析为 JSON: %.100s", content)
        return None

    except Exception as exc:
        logger.warning("后台 extractor LLM 调用失败: %s", exc)
        return None


def extract_and_save_memory(
    user_input:  str,
    ai_response: str,
    session_id:  str,
) -> None:
    """
    在 daemon thread 中运行：分析对话，决定是否写入长期记忆。

    被 MasterAgent._fire_memory_extractor() 在后台线程中调用。
    全程 try/except 包裹，任何失败只记 WARNING，不影响主流程。

    Args:
        user_input:  本轮用户输入文本
        ai_response: 本轮 AI 最终回答文本（无 tool_calls 的最后一条 ai 消息）
        session_id:  会话 ID（仅用于日志）
    """
    try:
        result = _call_extractor_llm(user_input, ai_response)
        if result is None:
            return

        if not result.get("should_save", False):
            logger.debug("后台 extractor：本轮无需保存记忆 [session=%s]", session_id)
            return

        title       = str(result.get("title", "")).strip()
        body        = str(result.get("body",  "")).strip()
        memory_type = str(result.get("type",  "preference")).strip()

        # [必须修复] description 优先取 LLM 返回的专用字段，
        # 回退到 body 首行，最后才用 title，避免 description 等于 title
        raw_desc = str(result.get("description", "")).strip()
        if not raw_desc and body:
            raw_desc = body.strip().splitlines()[0].strip()[:100]
        description = raw_desc or title

        if not title or not body:
            logger.warning("后台 extractor：LLM 返回 title/body 为空，跳过")
            return

        from src.core.long_term_memory import LongTermMemory, _slugify
        slug = _slugify(title)

        # 去重检查（三层）：传入真实 description 而非 title
        if _is_duplicate(slug, title, description, memory_type):
            return

        ltm = LongTermMemory.get_instance()
        ltm.save_memory(
            title=title,
            slug=slug,
            description=description,
            body=body,
            metadata={"type": memory_type, "source": "auto_extractor"},
        )
        logger.info(
            "后台 extractor 写入记忆: %s.md [session=%s]",
            slug, session_id,
        )

    except Exception as exc:
        logger.warning("后台 extractor 执行失败 [session=%s]: %s", session_id, exc)
