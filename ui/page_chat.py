"""
多轮问答页
- 聊天气泡式对话界面
- 侧边栏配置 thread_id / top_k / retriever_mode
- 每轮展示来源与 token 用量
"""
import streamlit as st
import requests


def render(api_base: str) -> None:
    st.header("多轮问答")

    # ── 侧边栏配置 ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("问答配置")
        thread_id = st.text_input("会话 ID", value="chat-default", key="chat_thread_id")
        top_k = st.slider("检索 chunk 数 (top_k)", 1, 10, 3, key="chat_top_k")
        retriever_mode = st.selectbox(
            "检索模式",
            ["mix", "local", "global", "hybrid", "semantic", "graph"],
            key="chat_retriever_mode",
        )
        max_history = st.slider("携带历史轮数", 1, 10, 5, key="chat_max_history")
        if st.button("清空对话", key="chat_clear"):
            st.session_state["chat_messages"] = []

    # ── 初始化消息列表 ────────────────────────────────────────────────────────
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # ── 渲染历史消息 ──────────────────────────────────────────────────────────
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("来源"):
                    for src in msg["sources"]:
                        st.caption(src)
            if msg.get("token_usage"):
                tu = msg["token_usage"]
                st.caption(
                    f"tokens: prompt={tu.get('prompt_tokens',0)}  "
                    f"completion={tu.get('completion_tokens',0)}  "
                    f"total={tu.get('total_tokens',0)}  "
                    f"费用≈¥{tu.get('estimated_cost_cny',0):.5f}"
                )

    # ── 输入框 ────────────────────────────────────────────────────────────────
    user_input = st.chat_input("输入问题…")
    if user_input:
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("检索中…"):
                try:
                    resp = requests.post(
                        f"{api_base}/chat",
                        json={
                            "message": user_input,
                            "thread_id": thread_id,
                            "top_k": top_k,
                            "max_history": max_history,
                            "retriever_mode": retriever_mode,
                        },
                        timeout=120,
                    )
                    if resp.ok:
                        data = resp.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])
                        token_usage = data.get("token_usage")
                    else:
                        answer = f"请求失败：{resp.json().get('detail', resp.text)}"
                        sources = []
                        token_usage = None
                except Exception as e:
                    answer = f"连接 API 失败：{e}"
                    sources = []
                    token_usage = None

            st.markdown(answer)
            if sources:
                with st.expander("来源"):
                    for src in sources:
                        st.caption(src)
            if token_usage:
                st.caption(
                    f"tokens: prompt={token_usage.get('prompt_tokens',0)}  "
                    f"completion={token_usage.get('completion_tokens',0)}  "
                    f"total={token_usage.get('total_tokens',0)}  "
                    f"费用≈¥{token_usage.get('estimated_cost_cny',0):.5f}"
                )

        st.session_state["chat_messages"].append(
            {"role": "assistant", "content": answer, "sources": sources, "token_usage": token_usage}
        )
