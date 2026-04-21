"""
工具描述文件加载器（P0-Step 7）。

职责：
- 从 prompt/tools/{tool_name}.md 加载工具描述文本
- 提供带 lru_cache 的加载函数，避免重复磁盘 I/O
- 文件不存在时 fallback 到传入的 docstring（保证启动不中断）

Public API：
    load_tool_description(tool_name, fallback="") → str
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_TOOL_PROMPT_DIR = Path("prompt/tools")
_TOOL_PROMPT_DIR_ABS = Path(__file__).resolve().parent.parent.parent / "prompt" / "tools"


def _resolve_tool_prompt_path(tool_name: str) -> Path | None:
    for base in (_TOOL_PROMPT_DIR, _TOOL_PROMPT_DIR_ABS):
        p = base / f"{tool_name}.md"
        if p.exists():
            return p
    return None


@lru_cache(maxsize=32)
def load_tool_description(tool_name: str, fallback: str = "") -> str:
    """
    加载指定工具的描述文本。

    Args:
        tool_name: 工具名称，对应 prompt/tools/{tool_name}.md
        fallback:  文件不存在时使用的兜底描述（通常为函数原始 docstring）

    Returns:
        工具描述字符串；文件不存在时返回 fallback（若 fallback 为空则返回 "Tool: {tool_name}"）
    """
    path = _resolve_tool_prompt_path(tool_name)
    if path:
        try:
            content = path.read_text(encoding="utf-8").strip()
            if content:
                logger.debug("已加载工具描述: %s（%d 字符，来源：%s）", tool_name, len(content), path)
                return content
        except OSError as exc:
            logger.error("读取工具描述文件失败: %s", exc)

    if fallback:
        logger.warning("工具描述文件未找到: prompt/tools/%s.md，使用 fallback docstring", tool_name)
        return fallback[:200]  # 取 docstring 前 200 字，避免描述过长

    logger.warning("工具描述文件未找到: prompt/tools/%s.md，使用最小 fallback", tool_name)
    return f"Tool: {tool_name}"
