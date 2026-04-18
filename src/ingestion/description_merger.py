"""
描述合并器

实体/关系被多文档命中时，合并描述而不是反复覆盖。

合并策略：
- 先去重：完全相同的描述只保留一条
- 短描述（总字符 ≤ max_direct_chars）：直接用分隔符拼接
- 长描述（总字符 > max_direct_chars）：调用 LLM 生成摘要，失败时回退到直接拼接

注意：跨文档描述合并依赖实体自然键（Phase 2.1 完成后生效）。
      Phase 2.2 在同一实体 ID 下已可正确合并同文档多次命中的描述。
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.prompt_loader import load_prompt_pair
from src.infrastructure.llm_client import get_llm

logger = logging.getLogger(__name__)

_SUMMARIZE_SYSTEM, _SUMMARIZE_HUMAN = load_prompt_pair("description_merger")

class DescriptionMerger:
    """实体/关系描述合并器。

    Args:
        max_direct_chars: 直接拼接的字符数上限，超过则触发 LLM 摘要。默认 2000。
        separator: 直接拼接时的分隔符。
    """

    def __init__(
        self,
        max_direct_chars: int = 2000,
        separator: str = "\n---\n",
    ) -> None:
        self.max_direct_chars = max_direct_chars
        self.separator = separator
        self._llm = None  # 懒加载，避免无谓 LLM 实例化

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def merge(self, descriptions: list[str]) -> str:
        """合并描述列表，返回合并后的单条描述。

        Args:
            descriptions: 原始描述列表（可含重复、空值）。

        Returns:
            合并后的描述字符串；输入全为空时返回空字符串。
        """
        unique = self._deduplicate(descriptions)
        if not unique:
            return ""
        if len(unique) == 1:
            return unique[0]

        total_chars = sum(len(d) for d in unique)
        if total_chars <= self.max_direct_chars:
            return self.separator.join(unique)

        logger.info("描述总长 %d 字符（%d 条），触发 LLM 摘要", total_chars, len(unique))
        return self._llm_summarize(unique)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(descriptions: list[str]) -> list[str]:
        """去重并过滤空描述，保持原始顺序。"""
        seen: set[str] = set()
        result: list[str] = []
        for d in descriptions:
            d = d.strip()
            if d and d not in seen:
                seen.add(d)
                result.append(d)
        return result

    def _llm_summarize(self, descriptions: list[str]) -> str:
        """调用 LLM 将多条描述合并为一条摘要；失败时回退到直接拼接。"""
        if self._llm is None:
            self._llm = get_llm(temperature=0.0)

        combined = self.separator.join(descriptions)
        messages = [
            SystemMessage(content=_SUMMARIZE_SYSTEM),
            HumanMessage(content=_SUMMARIZE_HUMAN.format(combined=combined)),
        ]
        try:
            response = self._llm.invoke(messages)
            return response.content.strip()
        except Exception as exc:
            logger.warning("LLM 摘要失败，回退到直接拼接: %s", exc)
            return self.separator.join(descriptions)
