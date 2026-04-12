"""
深度研究页
- 研究问题输入
- 侧边栏配置检索参数
- 展示子问题列表 + Markdown 研究报告
- 一键生成 Idea 报告
- 下载报告为 .md 文件
"""
import streamlit as st
import requests


def render(api_base: str) -> None:
    st.header("深度研究")

    # ── 侧边栏配置 ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("研究配置")
        thread_id = st.text_input("会话 ID", value="research-default", key="res_thread_id")
        retriever_mode = st.selectbox(
            "检索模式",
            ["hybrid", "semantic", "graph"],
            key="res_retriever_mode",
        )
        top_k = st.slider("每子问题检索 chunk 数", 1, 10, 5, key="res_top_k")
        max_subq = st.slider("子问题上限", 2, 5, 4, key="res_max_subq")
        use_community = st.checkbox("附加社区视角", key="res_use_community")

    # ── 研究问题输入 ──────────────────────────────────────────────────────────
    question = st.text_area(
        "研究问题",
        placeholder="例如：AmpAgent 解决了什么问题，其方法有哪些局限？",
        height=80,
        key="res_question",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        run_btn = st.button("开始研究", type="primary", key="res_run")

    # ── 执行研究 ──────────────────────────────────────────────────────────────
    if run_btn:
        if not question.strip():
            st.warning("请输入研究问题")
        else:
            st.session_state.pop("res_result", None)
            st.session_state.pop("res_idea", None)
            with st.spinner("研究中，请稍候（通常需要 1~3 分钟）…"):
                try:
                    resp = requests.post(
                        f"{api_base}/research",
                        json={
                            "question": question.strip(),
                            "thread_id": thread_id,
                            "top_k": top_k,
                            "max_subquestions": max_subq,
                            "retriever_mode": retriever_mode,
                            "use_community": use_community,
                        },
                        timeout=600,
                    )
                    if resp.ok:
                        st.session_state["res_result"] = resp.json()
                    else:
                        st.error(f"研究失败：{resp.json().get('detail', resp.text)}")
                except Exception as e:
                    st.error(f"连接 API 失败：{e}")

    # ── 展示研究结果 ──────────────────────────────────────────────────────────
    result = st.session_state.get("res_result")
    if result:
        st.subheader("子问题列表")
        for i, sq in enumerate(result.get("sub_questions", []), start=1):
            st.write(f"{i}. {sq}")

        st.subheader("研究报告")
        st.markdown(result["final_report"])

        if result.get("all_sources"):
            with st.expander("参考来源"):
                for src in result["all_sources"]:
                    st.caption(src)

        tu = result.get("total_token_usage", {})
        if tu:
            st.caption(
                f"总 token：{tu.get('total', 0)}  |  "
                f"费用≈¥{tu.get('cost_cny', 0):.5f}"
            )

        # 下载报告
        report_md = result["final_report"]
        st.download_button(
            label="下载研究报告 (.md)",
            data=report_md.encode("utf-8"),
            file_name="research_report.md",
            mime="text/markdown",
            key="res_download",
        )

        st.divider()

        # ── Idea 生成 ─────────────────────────────────────────────────────────
        if st.button("生成 Idea 报告", key="res_idea_btn"):
            st.session_state.pop("res_idea", None)
            with st.spinner("生成 Idea 报告中…"):
                try:
                    resp = requests.post(
                        f"{api_base}/idea",
                        json={
                            "question": result["question"],
                            "report_markdown": result["final_report"],
                        },
                        timeout=120,
                    )
                    if resp.ok:
                        st.session_state["res_idea"] = resp.json()
                    else:
                        st.error(f"Idea 生成失败：{resp.json().get('detail', resp.text)}")
                except Exception as e:
                    st.error(f"连接 API 失败：{e}")

        idea = st.session_state.get("res_idea")
        if idea:
            st.subheader("Idea 报告")
            st.markdown(idea["markdown"])
            st.download_button(
                label="下载 Idea 报告 (.md)",
                data=idea["markdown"].encode("utf-8"),
                file_name="idea_report.md",
                mime="text/markdown",
                key="res_idea_download",
            )
