"""
JSONL 会话管理器。

实现跨进程的会话持久化，采用"增量追加 + covers_turn_ids 集合判定"方案，
避免旧版字符串 turn_id 比较带来的顺序脆弱性问题。

数据结构：
    data/sessions/{session_id}.jsonl       ← 会话事件/消息日志（仅追加）
    data/sessions/{session_id}.meta.json   ← 会话元数据（标题、时间、token 统计）

JSONL 行格式（两种类型）：

    type=messages  本轮新增消息（增量写入，不重复保存历史）
    {
      "type": "messages",
      "turn_id": "t1713254400123456789",    ← 纳秒时间戳，全局唯一
      "timestamp": "2026-04-16T10:00:00Z",
      "user_input": "RAG 和 GraphRAG 有什么区别？",
      "new_messages": [...],
      "sources": ["paper.pdf#chunk-3"],
      "token_usage": {"prompt": 1200, "completion": 350, "total": 1550},
      "cost_cny": 0.000048                  ← Phase 9-3: 本轮估算人民币费用（round 6位）
    }

    type=summary  压缩摘要（追加写入，不重写主 JSONL）
    {
      "type": "summary",
      "covers_turn_ids": ["t...", "t..."],  ← 被此摘要覆盖的 turn_id 集合
      "summary_text": "...",
      "compressed_at": "2026-04-16T11:00:00Z"
    }

设计要点（对应 Review 意见）：
    [1] load() 用 covers_turn_ids 集合成员判断跳过旧轮，不依赖字符串顺序比较
    [2] 多次压缩时保留所有 summary，按时序合并注入 SystemMessage
    [3] turn_id 用 time.time_ns() 生成，纳秒精度，实际无碰撞风险
    [4] 摘要以 SystemMessage 注入，不伪造 HumanMessage/AIMessage
    [5] 删除未使用的 SYSTEM_PROMPT_PATH 常量
    [6] 压缩 Prompt 抽取到 prompt/session_compact_system.md 统一管理

Phase 9-4 新增：
    [7] _build_history_text()：从 new_messages 提取 AI 最终回答，
        生成完整"用户-助手"对话文本，摘要质量优于仅传 user_input
    [8] _compact_if_needed()：压缩后重置 meta.total_tokens 为保留轮次的
        token 之和，防止压缩后立即再次触发压缩
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from src.infrastructure.config import settings
from src.agents.prompt_loader import load_system_prompt

logger = logging.getLogger(__name__)

# 压缩摘要 Prompt（统一 Prompt 管理）
_COMPACT_TEMPLATE = load_system_prompt("session_compact")


# ── 消息序列化 / 反序列化 ──────────────────────────────────────────────────────

def _serialize_message(msg: BaseMessage) -> dict:
    """将 LangChain BaseMessage 序列化为可 JSON 存储的 dict。"""
    base: dict = {"role": msg.type, "content": msg.content}

    if isinstance(msg, AIMessage) and msg.tool_calls:
        base["tool_calls"] = [
            {
                "id":   tc.get("id", ""),
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
            }
            for tc in msg.tool_calls
        ]

    if isinstance(msg, ToolMessage):
        base["tool_call_id"] = msg.tool_call_id

    return base


def _deserialize_message(data: dict) -> BaseMessage:
    """将 JSON dict 反序列化为 LangChain BaseMessage。"""
    role    = data.get("role", "human")
    content = data.get("content", "")

    if role == "human":
        return HumanMessage(content=content)
    if role == "ai":
        return AIMessage(content=content, tool_calls=data.get("tool_calls", []))
    if role == "system":
        return SystemMessage(content=content)
    if role == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=data.get("tool_call_id", ""),
        )
    # 未知类型：fallback 到 HumanMessage，避免崩溃
    logger.warning("未知消息 role=%r，已 fallback 为 HumanMessage", role)
    return HumanMessage(content=content)


# ── SessionManager ─────────────────────────────────────────────────────────

class SessionManager:
    """
    跨进程会话持久化管理器。

    设计原则：
    - JSONL 文件只追加，永不覆盖历史记录（append-only）
    - load() 通过 covers_turn_ids 集合跳过已压缩轮次，不依赖字符串顺序比较
    - 多次压缩的所有 summary 按时序合并为一条 SystemMessage 注入，不丢失历史

    用法示例：
        sm = SessionManager()
        sm.save_turn(session_id, user_input, new_messages, sources, token_usage)
        history = sm.load(session_id)   # 进程重启后仍可恢复
    """

    SESSION_DIR: Path            = settings.data_dir / "sessions"
    COMPACT_TOKEN_THRESHOLD: int = 8000   # 超过此 token 数触发压缩
    KEEP_RECENT_TURNS: int       = 3      # 压缩时保留最近 N 轮原始记录

    # [Fix-1] session_id 安全校验正则（与 api/routers/agent.py 保持一致）
    _SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

    def __init__(self) -> None:
        self.SESSION_DIR.mkdir(parents=True, exist_ok=True)

    # ── session_id 防御校验 ────────────────────────────────────────────────────

    def _assert_safe_session_id(self, session_id: str) -> None:
        """
        [Fix-1] 防御式校验 session_id，防止路径穿越攻击。

        SessionManager 是内部基础设施，路由层已做校验；
        此处作为第二道防线，保证即使上层遗漏也不会写出目录外文件。
        抛出 ValueError（由调用方决定是否向上传播）。
        """
        if not self._SESSION_ID_RE.match(session_id):
            raise ValueError(
                f"session_id 格式非法（路径穿越防护）：{session_id!r}"
            )

    # ── 路径工具 ──────────────────────────────────────────────────────────────

    def _jsonl_path(self, session_id: str) -> Path:
        self._assert_safe_session_id(session_id)   # [Fix-1]
        return self.SESSION_DIR / f"{session_id}.jsonl"

    def _meta_path(self, session_id: str) -> Path:
        self._assert_safe_session_id(session_id)   # [Fix-1]
        return self.SESSION_DIR / f"{session_id}.meta.json"

    # ── 元数据操作 ────────────────────────────────────────────────────────────

    def _load_meta(self, session_id: str) -> dict:
        """读取 meta.json；不存在时返回空 dict。"""
        meta_path = self._meta_path(session_id)
        if not meta_path.exists():
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_meta(self, session_id: str, meta: dict) -> None:
        """覆写 meta.json。"""
        with open(self._meta_path(session_id), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _update_meta(
        self,
        session_id:    str,
        *,
        title:         Optional[str]   = None,
        token_delta:   Optional[dict]  = None,
        cost_delta:    Optional[float] = None,
        increment_turn: bool           = False,
    ) -> dict:
        """
        部分更新 meta.json，未传入的字段保持不变。

        Phase 9-3 新增：
        - cost_delta: 本轮估算费用（元），累计到 total_cost_cny 字段

        Note: 压缩游标（compressed_until_turn_id）已废弃。
              load() 改从 JSONL 中的 covers_turn_ids 字段判断旧轮，
              meta 不再承担压缩状态职责。
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        meta    = self._load_meta(session_id)

        if not meta:
            meta = {
                "session_id":     session_id,
                "title":          title or "新会话",
                "created_at":     now_iso,
                "updated_at":     now_iso,
                "turn_count":     0,
                "total_tokens":   0,
                "total_cost_cny": 0.0,
            }

        meta["updated_at"] = now_iso

        if title:
            meta["title"] = title
        if increment_turn:
            meta["turn_count"] = meta.get("turn_count", 0) + 1
        if token_delta:
            meta["total_tokens"] = (
                meta.get("total_tokens", 0) + token_delta.get("total", 0)
            )
        if cost_delta is not None:
            meta["total_cost_cny"] = round(
                meta.get("total_cost_cny", 0.0) + cost_delta, 6
            )

        self._save_meta(session_id, meta)
        return meta

    # ── JSONL 操作 ────────────────────────────────────────────────────────────

    def _append_jsonl(self, session_id: str, record: dict) -> None:
        """原子追加一行 JSONL（a 模式，多进程安全的基础保证）。"""
        line = json.dumps(record, ensure_ascii=False)
        with open(self._jsonl_path(session_id), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _read_all_records(self, session_id: str) -> list[dict]:
        """读取 JSONL 文件中所有行，容错跳过损坏行。"""
        jsonl_path = self._jsonl_path(session_id)
        if not jsonl_path.exists():
            return []

        records: list[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(
                        "会话 %s 第 %d 行 JSON 损坏，已跳过: %.80s",
                        session_id, lineno, line,
                    )
        return records

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def load(self, session_id: str) -> list[BaseMessage]:
        """
        从 JSONL 恢复消息历史，进程重启后可恢复。

        恢复逻辑（两次扫描）：
        1. 第一次扫描：收集所有 summary 记录中的 covers_turn_ids，
           构建"已压缩 turn_id 集合"（O(1) 成员判定，不依赖字符串顺序）
        2. 第二次扫描：跳过 turn_id 在集合中的旧轮，反序列化其余 messages
        3. 将所有 summary 文本按时序合并为单条 SystemMessage 注入到列表头部
           （[4] 不伪造 HumanMessage，[2] 保留多次压缩的完整历史）

        Returns:
            list[BaseMessage] — 可直接追加到 MasterAgent 消息列表的历史记录
                                （不含 MasterAgent 自身的 SystemMessage）
        """
        records = self._read_all_records(session_id)

        # ── 第一次扫描：建立 covered 集合 + 收集有序 summaries ────────────────
        covered_turn_ids: set[str]  = set()
        ordered_summaries: list[dict] = []

        for record in records:
            if record.get("type") == "summary":
                ordered_summaries.append(record)
                covered_turn_ids.update(record.get("covers_turn_ids", []))

        # ── 第二次扫描：跳过已压缩轮次，反序列化有效消息 ────────────────────
        messages: list[BaseMessage] = []
        for record in records:
            if record.get("type") != "messages":
                continue
            turn_id = record.get("turn_id", "")
            if turn_id in covered_turn_ids:   # [1] 集合成员判定，不用字符串比较
                continue
            for msg_data in record.get("new_messages", []):
                try:
                    messages.append(_deserialize_message(msg_data))
                except Exception as exc:
                    logger.warning(
                        "消息反序列化失败（已跳过）: %s | 数据: %s", exc, msg_data
                    )

        # ── 构建 summary SystemMessage（[4] 干净语义，[2] 多次压缩不丢失）────
        if ordered_summaries:
            if len(ordered_summaries) == 1:
                summary_content = ordered_summaries[0]["summary_text"]
            else:
                # 多次压缩：按时序拼接，标注各轮覆盖范围
                parts = [
                    f"[第 {i} 次压缩（涵盖 {len(s.get('covers_turn_ids', []))} 轮）]\n{s['summary_text']}"
                    for i, s in enumerate(ordered_summaries, start=1)
                ]
                summary_content = "\n\n".join(parts)

            prefix: list[BaseMessage] = [
                SystemMessage(
                    content=(
                        "以下是之前对话的历史摘要，请基于此继续回答：\n\n"
                        + summary_content
                    )
                )
            ]
            return prefix + messages

        return messages

    def save_turn(
        self,
        session_id:   str,
        user_input:   str,
        new_messages: list[BaseMessage],
        sources:      list[str],
        token_usage:  dict,
        cost_cny:     float = 0.0,
    ) -> None:
        """
        追加写入一行增量 JSONL 记录（原子操作，不可变）。

        Args:
            session_id:   会话 ID
            user_input:   本轮用户输入（用于生成摘要 / 会话标题）
            new_messages: 本轮新增的 LangChain 消息（不含历史）
            sources:      本轮引用的文献来源列表
            token_usage:  {"prompt": int, "completion": int, "total": int}
            cost_cny:     本轮估算人民币费用（Phase 9-3）
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        # [3] 纳秒时间戳：即使同毫秒并发也不会碰撞
        turn_id = f"t{_time.time_ns()}"
        # Phase 9-3: 统一 round，保证 JSONL 单轮记录与 meta 累计值精度一致
        normalized_cost = round(cost_cny, 6)

        # 序列化消息
        serialized: list[dict] = []
        for msg in new_messages:
            try:
                serialized.append(_serialize_message(msg))
            except Exception as exc:
                logger.warning("消息序列化失败（已跳过）: %s", exc)

        record = {
            "type":         "messages",
            "turn_id":      turn_id,
            "timestamp":    now_iso,
            "user_input":   user_input,
            "new_messages": serialized,
            "sources":      sources,
            "token_usage":  token_usage,
            "cost_cny":     normalized_cost,
        }

        self._append_jsonl(session_id, record)

        # 新会话第一轮：用用户输入前 30 字作为会话标题
        is_first_turn = not self._meta_path(session_id).exists()
        title = user_input[:30] if is_first_turn else None

        self._update_meta(
            session_id,
            title=title,
            token_delta=token_usage,
            cost_delta=normalized_cost,
            increment_turn=True,
        )

        # 超出阈值时触发压缩（daemon thread，不阻塞调用方 async 事件循环）
        t = threading.Thread(
            target=self._compact_if_needed,
            args=(session_id,),
            daemon=True,
            name=f"compact-{session_id[:8]}",
        )
        t.start()

    def list_sessions(self) -> list[dict]:
        """
        列举所有会话的元数据，按更新时间倒序排列（供前端会话列表展示）。
        """
        sessions: list[dict] = []
        for meta_file in self.SESSION_DIR.glob("*.meta.json"):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    sessions.append(json.load(f))
            except Exception as exc:
                logger.warning("读取会话元数据失败 %s: %s", meta_file.name, exc)

        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """
        删除指定会话的所有持久化文件（JSONL + meta）。

        Returns:
            True  — 至少删除了一个文件
            False — 会话文件不存在
        """
        deleted = False
        for path in [self._jsonl_path(session_id), self._meta_path(session_id)]:
            if path.exists():
                path.unlink()
                deleted = True
                logger.info("已删除会话文件: %s", path.name)
        return deleted

    def get_session_meta(self, session_id: str) -> Optional[dict]:
        """获取单个会话的元数据；不存在时返回 None。"""
        meta = self._load_meta(session_id)
        return meta if meta else None

    # ── 会话压缩（Phase 9-4 完整实现）───────────────────────────────────────

    def _estimate_total_tokens(self, session_id: str) -> int:
        """从 meta 获取已累计的 token 总量（估算值）。"""
        return self._load_meta(session_id).get("total_tokens", 0)

    def _build_history_text(self, turns: list[dict]) -> str:
        """
        将 turn records 转为可读的对话历史文本，供 LLM 摘要使用。

        Phase 9-4：从 new_messages 提取 AI 最终回答（最后一条 role=ai 且无
        tool_calls 的消息），与 user_input 配对，生成完整"用户-助手"对话格式。
        仅取 user_input 的旧版已废弃，新版包含助手回答，摘要质量更高。

        截断策略（head+tail）：取前 250 字 + "…" + 后 150 字，保留开头上下文
        和结论段落，比单纯截头更能覆盖 AI 回答的核心内容。
        若某轮无合格 AI 消息（全为中间推理或工具错误），该轮只保留用户问题，
        此为设计意图：宁缺失助手内容也不引入错误信息。

        格式示例：
            [第 1 轮]
            用户：RAG 和 GraphRAG 有什么区别？
            助手：RAG 基于向量检索……（前250字）…（后150字）
        """
        _HEAD = 250
        _TAIL = 150

        parts: list[str] = []
        for idx, turn in enumerate(turns, start=1):
            user_input = turn.get("user_input", "").strip()
            # 从 new_messages 找最后一条 role=ai 且无 tool_calls 的消息（最终回答）
            ai_answer = ""
            for msg in reversed(turn.get("new_messages", [])):
                if (
                    msg.get("role") == "ai"
                    and not msg.get("tool_calls")
                    and msg.get("content", "").strip()
                ):
                    ai_answer = msg["content"].strip()
                    break

            block = f"[第 {idx} 轮]\n用户：{user_input}"
            if ai_answer:
                total = _HEAD + _TAIL
                if len(ai_answer) <= total:
                    truncated = ai_answer
                else:
                    truncated = ai_answer[:_HEAD] + "…" + ai_answer[-_TAIL:]
                block += f"\n助手：{truncated}"
            parts.append(block)
        return "\n\n".join(parts)

    def _compact_if_needed(self, session_id: str) -> None:
        """
        Token 超过阈值时触发压缩（Phase 9-4 完整实现）：

        1. 从 JSONL 中找出尚未被任何 summary 覆盖的旧轮
        2. 保留最近 KEEP_RECENT_TURNS 轮原始记录，其余送入 LLM 生成摘要
        3. 追加 summary 记录（covers_turn_ids 记录本次覆盖的 turn_id 集合）
        4. 重置 meta.total_tokens = 保留轮次的 token 之和，防止反复触发压缩
        5. 不再写 meta 压缩游标（load() 直接读 JSONL 中的 covers_turn_ids）
        """
        if self._estimate_total_tokens(session_id) <= self.COMPACT_TOKEN_THRESHOLD:
            return

        all_records = self._read_all_records(session_id)

        # 收集已被覆盖的 turn_id
        covered_turn_ids: set[str] = set()
        for r in all_records:
            if r.get("type") == "summary":
                covered_turn_ids.update(r.get("covers_turn_ids", []))

        # 找出尚未被覆盖的轮次
        uncovered_turns = [
            r for r in all_records
            if r.get("type") == "messages"
            and r.get("turn_id", "") not in covered_turn_ids
        ]

        if len(uncovered_turns) <= self.KEEP_RECENT_TURNS:
            return  # 可压缩轮次不足，跳过

        turns_to_compress = uncovered_turns[: -self.KEEP_RECENT_TURNS]
        turns_to_keep     = uncovered_turns[-self.KEEP_RECENT_TURNS:]

        try:
            summary_text = self._generate_summary(turns_to_compress)
        except Exception as exc:
            logger.warning("会话 %s 压缩失败，跳过: %s", session_id, exc)
            return

        now_iso = datetime.now(timezone.utc).isoformat()
        summary_record = {
            "type":            "summary",
            "covers_turn_ids": [t["turn_id"] for t in turns_to_compress],
            "summary_text":    summary_text,
            "compressed_at":   now_iso,
        }
        self._append_jsonl(session_id, summary_record)

        # 重置 total_tokens = 保留轮次 token 之和 + summary 注入上下文的粗估 token
        # summary 注入时会作为 SystemMessage 出现在下一轮上下文中，需计入，
        # 否则下次触发点会偏晚。粗估：字符数 / 1.5（中文约 1.5 字符/token）
        retained_tokens = sum(
            t.get("token_usage", {}).get("total", 0)
            for t in turns_to_keep
        )
        summary_token_est = max(1, int(len(summary_text) / 1.5))
        meta = self._load_meta(session_id)
        if meta:
            meta["total_tokens"] = retained_tokens + summary_token_est
            self._save_meta(session_id, meta)

        logger.info(
            "会话 %s 已压缩 %d 轮 → %d 轮保留，"
            "retained_tokens=%d summary_token_est=%d summary_len=%d",
            session_id,
            len(turns_to_compress),
            len(turns_to_keep),
            retained_tokens,
            summary_token_est,
            len(summary_text),
        )

    def _generate_summary(self, turns: list[dict]) -> str:
        """
        调用 LLM 生成多轮对话摘要（Phase 9-4 升级版）。

        改进：
        - 使用 _build_history_text() 将完整"用户-助手"对话文本传入 LLM，
          而非仅传 user_input，使摘要包含助手结论，质量显著提升。
        - Prompt 从 prompt/session_compact_system.md 加载（统一 Prompt 管理），
          文件不存在时使用内置 fallback。
        - temperature=0.0 保证摘要确定性。
        """
        from src.infrastructure.llm_client import get_llm

        history_text = self._build_history_text(turns)
        prompt = _COMPACT_TEMPLATE.replace("{history}", history_text)

        llm    = get_llm(temperature=0.0)
        result = llm.invoke(prompt)
        return str(result.content)
