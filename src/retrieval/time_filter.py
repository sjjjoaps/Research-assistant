"""
时间感知过滤器（Phase 9-2）

借鉴 AgenticRAG Meta Retrieval Agent 的渐进式召回机制，
为 retrieve_knowledge 增加时间维度的过滤与自适应放宽。

公开接口：
    extract_time_constraint(query)  — 从查询文本中解析时间约束
    filter_by_time(chunks, constraint) — 按时间约束过滤 chunk 列表
    progressive_retrieve(chunks, constraint, min_count) — 渐进式召回

使用方式：
    constraint = extract_time_constraint(query)
    filtered   = progressive_retrieve(chunks, constraint, min_count=3)
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retriever import RetrievedChunk

# 动态获取当前年份，不硬编码
def _current_year() -> int:
    return datetime.now().year

# "近N年"中 N 的默认值（当查询不含具体数字时）
_RECENT_YEARS_DEFAULT = 3

# 中文数字 → 阿拉伯数字映射（支持 近三年/近十年 等）
_CN_DIGIT = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "十一": 11, "十二": 12, "十三": 13,
    "十四": 14, "十五": 15, "二十": 20,
}


def _cn_to_int(s: str) -> int | None:
    """将中文数字字符串转为整数，失败返回 None。"""
    s = s.strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    return _CN_DIGIT.get(s)


def extract_time_constraint(query: str) -> dict | None:
    """
    从查询文本中提取时间约束。

    支持格式（按优先级匹配）：
        "2023-2024年"      → {"year_from": 2023, "year_to": 2024}
        "2023年至2025年"   → {"year_from": 2023, "year_to": 2025}
        "2023年以来"       → {"year_from": 2023, "year_to": None}
        "2023年之后"       → {"year_from": 2023, "year_to": None}
        "近5年/近五年"     → {"year_from": current-5+1, "year_to": None}
        "2023年"           → {"year_from": 2023, "year_to": 2023}
        最新/最近/近期/近年/latest/recent
                           → {"year_from": current-2, "year_to": None}

    Args:
        query: 用户原始查询字符串（不做预处理，函数内部统一 lower）。

    Returns:
        时间约束字典，或 None（无时间约束）。
        {"year_from": int, "year_to": int | None}
    """
    if not query:
        return None

    cur = _current_year()
    q = query  # 保留原文供中文匹配

    # ── 1. 近N年（含中文数字）：近3年/近三年/近几年 ──────────────────────────
    # 放最前，防止被 "近年" 模糊词匹配提前截断
    near_match = re.search(r"近([一二三四五六七八九十\d]{1,2})年", q)
    if near_match:
        raw = near_match.group(1)
        n = _cn_to_int(raw)
        if n and 1 <= n <= 50:
            return {"year_from": cur - n + 1, "year_to": None}

    # ── 2. 年份范围 YYYY-YYYY / YYYY年至YYYY年 / YYYY年到YYYY年 ─────────────
    # 允许年份后跟可选的 "年"
    range_match = re.search(
        r"(20\d{2})年?\s*[-~至到]\s*(20\d{2})年?",
        q,
    )
    if range_match:
        y_from = int(range_match.group(1))
        y_to   = int(range_match.group(2))
        if 2000 <= y_from <= cur + 5 and y_from <= y_to <= cur + 5:
            return {"year_from": y_from, "year_to": y_to}

    # ── 3. YYYY年以来 / YYYY年之后（开放上限）──────────────────────────────
    open_match = re.search(r"(20\d{2})年\s*(?:以来|之后|起|后)", q)
    if open_match:
        y = int(open_match.group(1))
        if 2000 <= y <= cur + 5:
            return {"year_from": y, "year_to": None}

    # ── 4. 单个精确年份：2023年 / 2023（需要词边界防止匹配到更长数字串）──────
    year_match = re.search(r"(?<!\d)(20\d{2})(?!\d)", q)
    if year_match:
        y = int(year_match.group(1))
        if 2000 <= y <= cur + 5:
            return {"year_from": y, "year_to": y}

    # ── 5. 模糊时间词（大小写不敏感的英文，精确的中文）──────────────────────
    fuzzy_zh = ["最新", "最近", "近期", "近年"]
    fuzzy_en = ["latest", "recent"]   # 对英文做 lower 比较
    if any(kw in q for kw in fuzzy_zh) or any(kw in q.lower() for kw in fuzzy_en):
        return {
            "year_from": cur - _RECENT_YEARS_DEFAULT + 1,
            "year_to":   None,
        }

    return None


def filter_by_time(
    chunks: list["RetrievedChunk"],
    constraint: dict | None,
) -> list["RetrievedChunk"]:
    """
    按时间约束过滤 chunk 列表。

    过滤策略：
    - chunk.year 为 None（年份未知）→ 保留（宁可召回，让 LLM 自行判断）
    - year_from 设置，chunk.year < year_from → 过滤掉
    - year_to 设置，chunk.year > year_to → 过滤掉

    Args:
        chunks:     RetrievedChunk 列表。
        constraint: extract_time_constraint() 返回的约束字典，或 None。

    Returns:
        过滤后的 chunk 列表（保序）。
    """
    if not constraint:
        return chunks

    year_from = constraint.get("year_from")
    year_to   = constraint.get("year_to")
    result: list["RetrievedChunk"] = []

    for chunk in chunks:
        year = getattr(chunk, "year", None)

        # 年份未知 → 保留（不因缺失 year 而丢弃）
        if year is None:
            result.append(chunk)
            continue

        try:
            y = int(year)
        except (ValueError, TypeError):
            result.append(chunk)
            continue

        if year_from is not None and y < year_from:
            continue
        if year_to is not None and y > year_to:
            continue
        result.append(chunk)

    return result


def progressive_retrieve(
    chunks: list["RetrievedChunk"],
    constraint: dict | None,
    min_count: int = 3,
) -> list["RetrievedChunk"]:
    """
    渐进式召回（借鉴 AgenticRAG Progressive Adaptive Retrieval）。

    策略（三档放宽，每档记日志便于追踪）：
        第一档：严格按 constraint 过滤
        第二档：若结果 < min_count，year_from 向前扩展 2 年（year_to 不变）
        第三档：若仍 < min_count，放弃时间约束，返回全部 chunks

    这样在时效性强的问题上优先召回最新文献，
    但当库中近期文献不足时不会静默返回空结果。

    Args:
        chunks:     候选 chunk 列表（已经过检索器返回）。
        constraint: 时间约束字典（可为 None）。
        min_count:  结果不足该数量时触发放宽，默认 3。

    Returns:
        过滤后的 chunk 列表（保序）。
    """
    import logging
    _log = logging.getLogger(__name__)

    if not constraint:
        return chunks

    # 第一档：严格过滤
    filtered = filter_by_time(chunks, constraint)
    _log.debug("时间过滤第一档：%d/%d chunks 通过 constraint=%s",
               len(filtered), len(chunks), constraint)
    if len(filtered) >= min_count:
        return filtered

    # 第二档：放宽 year_from -2 年
    year_from = constraint.get("year_from")
    if year_from is not None:
        relaxed = {**constraint, "year_from": year_from - 2}
        filtered = filter_by_time(chunks, relaxed)
        _log.debug("时间过滤第二档（放宽至 year_from=%d）：%d/%d chunks 通过",
                   relaxed["year_from"], len(filtered), len(chunks))
        if len(filtered) >= min_count:
            return filtered

    # 第三档：放弃时间约束
    _log.debug("时间过滤第三档：放弃约束，返回全量 %d chunks", len(chunks))
    return chunks
