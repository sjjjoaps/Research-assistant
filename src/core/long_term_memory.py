"""
文件系统长期记忆（Phase 9-5 重构版）

借鉴 Claude Code 自身的记忆系统设计，将跨会话策略经验与用户偏好存储为
独立 Markdown 文件，`MEMORY.md` 仅作索引，正文全部存入各自的 `{slug}.md`。

数据目录：data/memory/（路径来自 settings.data_dir）

目录结构：
    data/memory/
    ├─ MEMORY.md                      ← 索引，每行格式：- [title](slug.md) — hook
    ├─ strategy_specific_local.md     ← 策略类记忆文件
    ├─ strategy_temporal_mix.md
    └─ ...

单条记忆文件格式（frontmatter + 正文）：
    ---
    title: temporal 查询优先 mix
    type: strategy               # strategy / preference / qa / note
    question_type: temporal      # strategy 类专用
    mode: mix                    # strategy 类专用
    description: 时间敏感问题优先 mix，避免 local/global 对最新文献召回过窄。
    sample_count: 8              # strategy 类：总样本数
    success_count: 7             # strategy 类：成功样本数
    updated_at: 2026-04-17T10:00:00Z
    ---

    正文说明（Markdown）...

核心公开接口：
    classify_question_type(query)        → str（模块级，供 tool_registry 跨模块调用）
    _slugify(title)                      → str（模块级，供 tool 导入）
    LongTermMemory.save_memory(...)      → None（写单条记忆 + 更新 MEMORY.md）
    LongTermMemory.list_memory_index()  → list[dict]（解析 MEMORY.md 索引）
    LongTermMemory.find_relevant_memories(query, question_type) → list[dict]
    LongTermMemory.get_best_mode(question_type) → str | None
    LongTermMemory.record_strategy_result(...)  → None（后台更新策略记忆文件）
    LongTermMemory.record_qa_pair(...)          → None（兼容保留，暂为 stub）
    LongTermMemory.get_instance()               → LongTermMemory（单例）

设计要点：
    [1] 懒加载目录初始化（避免导入时即创建文件系统）
    [2] 写操作 try/except 包裹，失败只记 WARNING 不影响主流程
    [3] MEMORY.md 写入由 self._lock 保护（防多线程竞争）
    [4] _do_record_strategy() 整个读-改-写序列受 self._lock 保护（防计数竞争）
    [5] get_best_mode() 要求 sample_count >= MIN_SAMPLES 才给出建议
    [6] get_best_mode() 返回前校验 _VALID_MODES 白名单
    [7] temporal 类查询由 tool_registry 跳过 LTM（时效性路由不允许被历史覆盖）
    [8] 无 pyyaml 依赖，frontmatter 手动解析
    [9] classify_question_type() 保持原有签名，供 tool_registry 导入
   [10] MEMORY.md 绝不写正文，只写一行式索引
"""
from __future__ import annotations

import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import settings

logger = logging.getLogger(__name__)

# ── 常量 ─────────────────────────────────────────────────────────────────────

_MEMORY_DIR_NAME   = "memory"
_INDEX_FILE        = "MEMORY.md"
_MIN_SAMPLES       = 5            # get_best_mode 至少需要这么多样本才给出建议
_VALID_MODES       = frozenset({"dual", "local", "global", "mix", "hybrid", "semantic"})
_SLUG_RE           = re.compile(r"[^a-z0-9_-]")   # slug 字符白名单（外的字符替换为 _）
_FRONTMATTER_RE    = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


# ── 模块级工具函数 ────────────────────────────────────────────────────────────

def _slugify(title: str, max_len: int = 60) -> str:
    """
    将任意标题转换为安全的文件名 slug（仅含小写字母、数字、下划线、连字符）。

    示例：
        "temporal 查询优先 mix" → "temporal____mix"（最多 60 字符）
    """
    slug = title.lower()
    slug = slug.replace(" ", "_")
    slug = _SLUG_RE.sub("_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:max_len] if slug else "memory"


def classify_question_type(query: str) -> str:
    """
    根据查询文本中的关键词特征分类问题类型（公开接口，供 tool_registry 跨模块调用）。

    分类规则（优先级从高到低）：
        temporal  — 含时间约束（最优先判断，时效性最强）
        mixed     — 同时含 ll 和 hl 关键词
        specific  — 只含 ll 关键词（实体精确）
        abstract  — 只含 hl 关键词（宏观综述）
        general   — 无明显特征

    注意：temporal 类查询由 tool_registry 跳过 LTM 路由（[7]），
    分类结果仍会写入策略记忆供统计分析。
    """
    try:
        from src.retrieval.time_filter import extract_time_constraint
        if extract_time_constraint(query) is not None:
            return "temporal"
    except Exception:
        pass

    try:
        from src.retrieval.keyword_extractor import KeywordExtractor
        result = KeywordExtractor().extract(query)
        has_ll = bool(getattr(result, "ll_keywords", []))
        has_hl = bool(getattr(result, "hl_keywords", []))
        if has_ll and has_hl:
            return "mixed"
        if has_ll:
            return "specific"
        if has_hl:
            return "abstract"
    except Exception:
        pass

    return "general"


# ── 内部工具函数 ──────────────────────────────────────────────────────────────

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    解析 Markdown 文件的 YAML-like frontmatter（无 pyyaml 依赖）[8]。

    格式：
        ---
        key: value
        ---
        正文...

    Returns:
        (meta_dict, body_str)  — 若无合法 frontmatter 则返回 ({}, text)
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    meta: dict = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()

    body = text[m.end():]
    return meta, body


def _format_memory_file(meta: dict, body: str) -> str:
    """将 frontmatter dict 和正文组合为 Markdown 文件内容。

    [建议] frontmatter value 做单行化处理，防止 value 中含换行或 '---' 破坏解析。
    """
    lines = ["---"]
    for k, v in meta.items():
        # 单行化：去掉换行符、压缩多余空白，防止 frontmatter 被截断
        safe_v = str(v).replace("\r", "").replace("\n", " ").strip()
        lines.append(f"{k}: {safe_v}")
    lines.append("---")
    lines.append("")
    if body.strip():
        lines.append(body.strip())
    return "\n".join(lines) + "\n"


# ── LongTermMemory ────────────────────────────────────────────────────────────

class LongTermMemory:
    """
    文件系统长期记忆（Phase 9-5 重构版）。

    每条记忆存储为 data/memory/{slug}.md，MEMORY.md 仅作索引。
    strategy 类记忆同时负责记录检索策略效果，供 auto 模式参考。

    用法：
        ltm = LongTermMemory.get_instance()
        ltm.save_memory("标题", "my_slug", "描述", "正文", {"type": "preference"})
        ltm.record_strategy_result("specific", "local", 5, continued=True)
        best = ltm.get_best_mode("specific")   # → "local" 或 None
    """

    def __init__(self, memory_dir: Optional[Path] = None) -> None:
        self._memory_dir  = memory_dir or (settings.data_dir / _MEMORY_DIR_NAME)
        self._lock        = threading.Lock()   # 保护 MEMORY.md 写入及策略文件读-改-写 [3][4]
        self._initialized = False

    # ── 初始化 ────────────────────────────────────────────────────────────────

    def _ensure_init(self) -> None:
        """懒加载：首次访问时创建 data/memory/ 目录和空 MEMORY.md（[1]）。"""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            try:
                self._memory_dir.mkdir(parents=True, exist_ok=True)
                index_path = self._index_path
                if not index_path.exists():
                    index_path.write_text("", encoding="utf-8")
                self._initialized = True
                logger.debug("LongTermMemory 初始化完成: %s", self._memory_dir)
            except Exception as exc:
                logger.warning("LongTermMemory 初始化失败: %s", exc)

    @property
    def _index_path(self) -> Path:
        return self._memory_dir / _INDEX_FILE

    # ── 文件读写工具 ──────────────────────────────────────────────────────────

    def _read_file(self, file_path: Path) -> tuple[dict, str]:
        """读取记忆文件并解析 frontmatter，不存在时返回 ({}, '')。"""
        if not file_path.exists():
            return {}, ""
        try:
            text = file_path.read_text(encoding="utf-8")
            return _parse_frontmatter(text)
        except Exception as exc:
            logger.warning("读取记忆文件失败 %s: %s", file_path.name, exc)
            return {}, ""

    def _write_file(self, file_path: Path, meta: dict, body: str) -> None:
        """原子写入记忆文件（write tmp → os.replace，减少部分写风险）。"""
        content  = _format_memory_file(meta, body)
        tmp_path = file_path.with_suffix(".md.tmp")
        try:
            tmp_path.write_text(content, encoding="utf-8")
            os.replace(str(tmp_path), str(file_path))
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    # ── MEMORY.md 索引管理 ────────────────────────────────────────────────────

    def _update_index(self, title: str, slug: str, hook: str) -> None:
        """
        在 MEMORY.md 中追加或更新该 slug 对应的索引行（[3] 调用方持锁时调用）。

        格式：- [title](slug.md) — hook

        如果已存在该 slug 的行则原地替换，否则追加。
        注意：调用方必须已经持有 self._lock，本方法不再加锁。
        """
        index_path = self._index_path
        new_line   = f"- [{title}]({slug}.md) — {hook}"

        try:
            existing = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
            lines    = existing.splitlines(keepends=True)

            # 查找该 slug 是否已存在
            target    = f"({slug}.md)"
            updated   = False
            new_lines = []
            for line in lines:
                if target in line:
                    new_lines.append(new_line + "\n")
                    updated = True
                else:
                    new_lines.append(line)

            if not updated:
                if new_lines and not new_lines[-1].endswith("\n"):
                    new_lines[-1] += "\n"
                new_lines.append(new_line + "\n")

            index_path.write_text("".join(new_lines), encoding="utf-8")
        except Exception as exc:
            logger.warning("更新 MEMORY.md 索引失败: %s", exc)

    # ── 公开写接口 ────────────────────────────────────────────────────────────

    def save_memory(
        self,
        title:       str,
        slug:        str,
        description: str,
        body:        str,
        metadata:    dict,
    ) -> None:
        """
        写入单条记忆文件并更新 MEMORY.md 索引（两步写入）[10]。

        Args:
            title:       显示标题（用于 MEMORY.md 索引行）
            slug:        文件名（不含 .md 后缀，仅含 [a-z0-9_-]）；内部强制 _slugify 防路径穿越
            description: 一行式描述（用于 MEMORY.md 钩子行 + 召回过滤）
            body:        记忆正文（Markdown）
            metadata:    frontmatter 中的额外字段（如 type、source）
        """
        self._ensure_init()
        try:
            # [必须修复] 内部强制 _slugify，防止调用方传入 "../xxx" 造成路径穿越
            slug      = _slugify(slug)
            now_iso   = datetime.now(timezone.utc).isoformat()
            meta = {
                "title":       title,
                "description": description,
                "updated_at":  now_iso,
                **metadata,
            }
            file_path = self._memory_dir / f"{slug}.md"

            with self._lock:
                self._write_file(file_path, meta, body)
                self._update_index(title, slug, description)

            logger.debug("LongTermMemory 保存记忆: %s.md", slug)
        except Exception as exc:
            logger.warning("LongTermMemory save_memory 失败: %s", exc)

    # ── 公开读接口 ────────────────────────────────────────────────────────────

    def list_memory_index(self) -> list[dict]:
        """
        解析 MEMORY.md 索引，返回所有条目的 title/file/hook。

        Returns:
            list of {title: str, file: str, hook: str}
        """
        self._ensure_init()
        try:
            text    = self._index_path.read_text(encoding="utf-8")
            pattern = re.compile(r"^-\s+\[([^\]]+)\]\(([^)]+)\)\s+—\s+(.+)$")
            result  = []
            for line in text.splitlines():
                m = pattern.match(line.strip())
                if m:
                    result.append({
                        "title": m.group(1),
                        "file":  m.group(2),
                        "hook":  m.group(3),
                    })
            return result
        except Exception as exc:
            logger.warning("LongTermMemory list_memory_index 失败: %s", exc)
            return []

    def find_relevant_memories(
        self,
        query:         str,
        question_type: str,
    ) -> list[dict]:
        """
        基于 description / type / question_type frontmatter 做轻量相关性召回。

        评分规则：
            +3  question_type 精确匹配
            +1  query 词与 description 中的词有重叠（每个词 +1）

        Returns:
            按相关性分降序排列的记忆条目列表（含 meta + body），最多返回 5 条。
        """
        self._ensure_init()
        query_words = set(query.lower().split())
        results: list[tuple[int, dict]] = []

        try:
            for md_file in self._memory_dir.glob("*.md"):
                if md_file.name == _INDEX_FILE:
                    continue
                meta, body = self._read_file(md_file)
                if not meta:
                    continue

                score = 0
                if meta.get("question_type") == question_type:
                    score += 3
                desc_words = set(meta.get("description", "").lower().split())
                score += len(query_words & desc_words)

                if score > 0:
                    results.append((score, {**meta, "body": body, "file": md_file.name}))
        except Exception as exc:
            logger.warning("LongTermMemory find_relevant_memories 失败: %s", exc)

        results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in results[:5]]

    def get_best_mode(self, question_type: str) -> Optional[str]:
        """
        从 strategy 类型记忆文件中，找出对该问题类型成功率最高的检索模式。

        成功率 = success_count / sample_count。
        至少需要 MIN_SAMPLES(5) 条记录才给出建议（[5]），防止少量噪声误导。
        返回前校验 _VALID_MODES 白名单（[6]），过滤历史中的非法 mode。

        Returns:
            合法的最优 mode 字符串，或 None（样本不足 / 无合格记忆 / 读取失败）。
        """
        self._ensure_init()
        try:
            best_rate: float         = -1.0
            best_mode: Optional[str] = None

            for md_file in self._memory_dir.glob("strategy_*.md"):
                meta, _ = self._read_file(md_file)
                if not meta:
                    continue
                if meta.get("type") != "strategy":
                    continue
                if meta.get("question_type") != question_type:
                    continue

                try:
                    sample_count  = int(meta.get("sample_count",  0))
                    success_count = int(meta.get("success_count", 0))
                except (ValueError, TypeError):
                    continue

                if sample_count < _MIN_SAMPLES:
                    continue

                mode = meta.get("mode", "")
                if mode not in _VALID_MODES:
                    logger.debug("LongTermMemory 过滤非法 mode=%r，跳过", mode)
                    continue

                rate = success_count / sample_count
                if rate > best_rate:
                    best_rate = rate
                    best_mode = mode

            if best_mode:
                logger.debug(
                    "LongTermMemory 历史最优: type=%s mode=%s rate=%.2f",
                    question_type, best_mode, best_rate,
                )
            return best_mode

        except Exception as exc:
            logger.warning("LongTermMemory get_best_mode 失败: %s", exc)
            return None

    # ── 策略记录 ──────────────────────────────────────────────────────────────

    def record_strategy_result(
        self,
        question_type: str,
        mode:          str,
        result_count:  int,
        continued:     bool = True,
    ) -> None:
        """
        记录一次检索策略执行结果（后台线程，不阻塞主流程）。

        Args:
            question_type: 问题类型（"specific"/"abstract"/"mixed"/"temporal"/"general"）
            mode:          实际使用的检索模式（"local"/"global"/"dual"/"mix" 等）
            result_count:  检索到的 chunk 数量
            continued:     隐式反馈——建议用 `len(result) > 0`，而非默认 True
        """
        t = threading.Thread(
            target=self._do_record_strategy,
            args=(question_type, mode, result_count, continued),
            daemon=True,
            name="ltm-record",
        )
        t.start()

    def _do_record_strategy(
        self,
        question_type: str,
        mode:          str,
        result_count:  int,
        continued:     bool,
    ) -> None:
        """
        读取或新建 strategy_{question_type}_{mode}.md，更新计数，写回。

        整个读-改-写序列受 self._lock 保护（[4]），防止并发线程产生计数竞争。
        """
        self._ensure_init()
        slug      = f"strategy_{question_type}_{mode}"
        file_path = self._memory_dir / f"{slug}.md"
        now_iso   = datetime.now(timezone.utc).isoformat()

        try:
            with self._lock:
                meta, body = self._read_file(file_path)

                if meta:
                    # 更新已有文件
                    try:
                        sample_count  = int(meta.get("sample_count",  0))
                        success_count = int(meta.get("success_count", 0))
                    except (ValueError, TypeError):
                        sample_count = success_count = 0
                else:
                    # 首次创建
                    sample_count  = 0
                    success_count = 0
                    body = (
                        "该策略记忆由系统自动维护，记录"
                        f" [{question_type}] 类问题使用 [{mode}] 模式的检索效果历史。"
                    )

                sample_count  += 1
                success_count += 1 if (continued and result_count > 0) else 0

                new_meta = {
                    "title":         f"{question_type} 类查询优先 {mode}",
                    "type":          "strategy",
                    "question_type": question_type,
                    "mode":          mode,
                    "description": (
                        f"{question_type} 类问题使用 {mode} 模式的历史成功率："
                        f"{success_count}/{sample_count}"
                    ),
                    "sample_count":  str(sample_count),
                    "success_count": str(success_count),
                    "updated_at":    now_iso,
                }

                self._write_file(file_path, new_meta, body)
                # [必须修复] 每次更新策略文件都同步刷新 MEMORY.md 索引，
                # 否则 hook 会长期停留在首次写入的旧成功率
                self._update_index(
                    new_meta["title"],
                    slug,
                    new_meta["description"],
                )

            logger.debug(
                "LongTermMemory 策略更新: type=%s mode=%s count=%d/%d",
                question_type, mode, success_count, sample_count,
            )
        except Exception as exc:
            logger.warning("LongTermMemory 策略记录失败: %s", exc)

    # ── QA 对记录（兼容保留）────────────────────────────────────────────────

    def record_qa_pair(
        self,
        question: str,
        answer:   str,
        sources:  list[str],
        quality:  float,
    ) -> None:
        """
        记录高质量问答对（Phase 9-5 预留接口，当前仅作兼容存根，不参与召回）。

        后续可扩展为写入 qa_{slug}.md 文件并接入 Few-shot。
        """
        # 当前保留为空实现（兼容外部可能的调用）
        pass

    # ── 全局单例 ──────────────────────────────────────────────────────────────

    _instance:      Optional["LongTermMemory"] = None
    _instance_lock: threading.Lock              = threading.Lock()

    @classmethod
    def get_instance(cls) -> "LongTermMemory":
        """返回全局单例（线程安全双重检查锁）。"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
