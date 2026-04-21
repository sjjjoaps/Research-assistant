"""
测试 P5-Step 1.5：ToolCallLimiter（滑动窗口频率熔断器）

测试覆盖：
1. 正常调用在限制内允许（check 返回 True）
2. 超过 max_calls 次后触发熔断（check 返回 False）
3. 触发熔断时 reject_message 包含工具名和限制信息
4. 窗口滑动后旧记录过期，计数恢复（模拟时间推进）
5. 不同 session_id 的计数相互独立
6. 不同 tool_name 的计数相互独立
7. reset() — 重置指定 session+tool 后可重新调用
8. reset() — 重置全部后所有计数清零
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.tool_call_limiter import ToolCallLimiter


# ══════════════════════════════════════════════════════════════════════════
# Case 1：正常调用允许
# ══════════════════════════════════════════════════════════════════════════

def test_allow_within_limit():
    """在 max_calls 次内，check() 应持续返回 True。"""
    limiter = ToolCallLimiter(max_calls=3, window_seconds=60.0)
    for _ in range(3):
        assert limiter.check("session_a", "tool_x"), "未超限时应允许调用"
    print("[PASS] test_allow_within_limit")


# ══════════════════════════════════════════════════════════════════════════
# Case 2：超限触发熔断
# ══════════════════════════════════════════════════════════════════════════

def test_trigger_limit():
    """超过 max_calls 次后，check() 应返回 False。"""
    limiter = ToolCallLimiter(max_calls=3, window_seconds=60.0)
    for _ in range(3):
        limiter.check("session_a", "tool_x")
    assert not limiter.check("session_a", "tool_x"), "超限后应拒绝"
    print("[PASS] test_trigger_limit")


# ══════════════════════════════════════════════════════════════════════════
# Case 3：reject_message 包含关键信息
# ══════════════════════════════════════════════════════════════════════════

def test_reject_message_content():
    """reject_message 应包含工具名、窗口时长、次数上限。"""
    limiter = ToolCallLimiter(max_calls=5, window_seconds=60.0)
    msg = limiter.reject_message("retrieve_knowledge")
    assert "retrieve_knowledge" in msg, "拒绝消息未包含工具名"
    assert "5" in msg,                 "拒绝消息未包含次数上限"
    assert "60" in msg,                "拒绝消息未包含窗口时长"
    print("[PASS] test_reject_message_content")


# ══════════════════════════════════════════════════════════════════════════
# Case 4：窗口滑动后旧记录过期
# ══════════════════════════════════════════════════════════════════════════

def test_window_expiry():
    """使用极短窗口，等窗口过期后计数应重置，check 重新允许。"""
    limiter = ToolCallLimiter(max_calls=2, window_seconds=0.1)
    limiter.check("s", "t")
    limiter.check("s", "t")
    assert not limiter.check("s", "t"), "超限后应拒绝"
    time.sleep(0.15)
    assert limiter.check("s", "t"), "窗口过期后应重新允许"
    print("[PASS] test_window_expiry")


# ══════════════════════════════════════════════════════════════════════════
# Case 5：不同 session_id 计数独立
# ══════════════════════════════════════════════════════════════════════════

def test_cross_session_independence():
    """session_a 触发熔断，session_b 仍应可以调用。"""
    limiter = ToolCallLimiter(max_calls=2, window_seconds=60.0)
    limiter.check("session_a", "tool_x")
    limiter.check("session_a", "tool_x")
    assert not limiter.check("session_a", "tool_x"), "session_a 应触发熔断"
    assert limiter.check("session_b", "tool_x"),     "session_b 应独立，仍允许"
    print("[PASS] test_cross_session_independence")


# ══════════════════════════════════════════════════════════════════════════
# Case 6：不同 tool_name 计数独立
# ══════════════════════════════════════════════════════════════════════════

def test_cross_tool_independence():
    """tool_x 触发熔断，tool_y 仍应可以调用。"""
    limiter = ToolCallLimiter(max_calls=2, window_seconds=60.0)
    limiter.check("session_a", "tool_x")
    limiter.check("session_a", "tool_x")
    assert not limiter.check("session_a", "tool_x"), "tool_x 应触发熔断"
    assert limiter.check("session_a", "tool_y"),     "tool_y 应独立，仍允许"
    print("[PASS] test_cross_tool_independence")


# ══════════════════════════════════════════════════════════════════════════
# Case 7：reset 指定 session+tool
# ══════════════════════════════════════════════════════════════════════════

def test_reset_specific():
    """reset(session, tool) 后该组合计数清零，可重新调用。"""
    limiter = ToolCallLimiter(max_calls=2, window_seconds=60.0)
    limiter.check("s", "t")
    limiter.check("s", "t")
    assert not limiter.check("s", "t"), "超限后应拒绝"
    limiter.reset("s", "t")
    assert limiter.check("s", "t"), "reset 后应重新允许"
    print("[PASS] test_reset_specific")


# ══════════════════════════════════════════════════════════════════════════
# Case 8：reset 全部
# ══════════════════════════════════════════════════════════════════════════

def test_reset_all():
    """reset() 全部清零后所有组合均可重新调用。"""
    limiter = ToolCallLimiter(max_calls=1, window_seconds=60.0)
    limiter.check("s1", "t1")
    limiter.check("s2", "t2")
    assert not limiter.check("s1", "t1"), "s1/t1 应超限"
    assert not limiter.check("s2", "t2"), "s2/t2 应超限"
    limiter.reset()
    assert limiter.check("s1", "t1"), "全量 reset 后 s1/t1 应允许"
    assert limiter.check("s2", "t2"), "全量 reset 后 s2/t2 应允许"
    print("[PASS] test_reset_all")


# ── 主入口 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("P5-Step 1.5  ToolCallLimiter 测试")
    print("=" * 60)

    tests = [
        test_allow_within_limit,
        test_trigger_limit,
        test_reject_message_content,
        test_window_expiry,
        test_cross_session_independence,
        test_cross_tool_independence,
        test_reset_specific,
        test_reset_all,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"结果：{passed} 通过，{failed} 失败")
    print("=" * 60)
