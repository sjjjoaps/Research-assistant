"""
社区检测页
- 触发 Louvain 社区检测
- 展示已有社区摘要列表
"""
import streamlit as st
import requests


def render(api_base: str) -> None:
    st.header("社区检测")
    st.caption("基于 Neo4j 实体关系图，使用 Louvain 算法划分研究社区并生成摘要。前提：已完成实体抽取。")

    # ── 触发检测 ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("检测配置")
        min_size = st.slider("社区最小实体数", 2, 10, 3, key="comm_min_size")

    if st.button("运行社区检测", type="primary", key="comm_run"):
        with st.spinner("社区检测中，请稍候…"):
            try:
                resp = requests.post(
                    f"{api_base}/community/detect",
                    json={"min_community_size": min_size},
                    timeout=300,
                )
                if resp.ok:
                    stats = resp.json()
                    st.success(
                        f"检测完成  |  实体总数：{stats['total_entities']}  |  "
                        f"检测到社区：{stats['detected_communities']}  |  "
                        f"写入：{stats['written_communities']}  |  "
                        f"跳过（过小）：{stats['skipped_communities']}"
                    )
                    st.session_state.pop("comm_list", None)
                else:
                    st.error(f"检测失败：{resp.json().get('detail', resp.text)}")
            except Exception as e:
                st.error(f"连接 API 失败：{e}")

    st.divider()

    # ── 社区列表 ──────────────────────────────────────────────────────────────
    st.subheader("已有社区摘要")
    col1, col2 = st.columns([1, 5])
    with col1:
        limit = st.number_input("显示数量", min_value=1, max_value=100, value=20, key="comm_limit")
    with col2:
        if st.button("刷新列表", key="comm_refresh"):
            st.session_state.pop("comm_list", None)

    if "comm_list" not in st.session_state:
        try:
            resp = requests.get(f"{api_base}/community/list", params={"limit": limit}, timeout=10)
            st.session_state["comm_list"] = resp.json() if resp.ok else []
        except Exception as e:
            st.error(f"获取社区列表失败：{e}")
            st.session_state["comm_list"] = []

    communities = st.session_state.get("comm_list", [])
    if communities:
        for cm in communities:
            with st.expander(f"社区 {cm['label']}（{cm['entity_count']} 个实体）"):
                st.write(cm["summary"] or "（暂无摘要）")
    else:
        st.info("暂无社区数据，请先运行社区检测。")
