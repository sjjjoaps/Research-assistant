"""
JSON 解析工具

提供对 LLM 输出的健壮 JSON 提取，兼容以下常见格式：
  1. 纯 JSON：直接解析
  2. Markdown fenced block：```json ... ```
  3. 先文本说明后 JSON："如下所示：\n{...}"
  4. JSON 嵌入段落中间：从文本中定位首个 { 或 [ 提取
"""
from __future__ import annotations

import json
import re


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.IGNORECASE)
_OBJ_RE   = re.compile(r"\{[\s\S]+\}")
_ARR_RE   = re.compile(r"\[[\s\S]+\]")


def extract_json(text: str) -> any:
    """从 LLM 输出文本中健壮地提取第一个 JSON 对象或数组。

    解析优先级：
    1. 整体直接解析（纯 JSON）
    2. 提取 fenced code block 内容解析
    3. 用正则定位首个 { ... } 解析
    4. 用正则定位首个 [ ... ] 解析

    Args:
        text: LLM 返回的原始文本。

    Returns:
        解析得到的 Python 对象（dict / list）。

    Raises:
        ValueError: 所有方式均失败时。
    """
    text = (text or "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    obj_match = _OBJ_RE.search(text)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass

    arr_match = _ARR_RE.search(text)
    if arr_match:
        try:
            return json.loads(arr_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法从文本中提取 JSON: {text[:200]!r}")
