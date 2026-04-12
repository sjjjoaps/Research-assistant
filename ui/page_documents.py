"""
文献管理页
- 展示已入库文献列表
- 支持单文件 / 目录入库
"""
import streamlit as st
import requests


def render(api_base: str) -> None:
    st.header("文献管理")

    # ── 已入库文献列表 ────────────────────────────────────────────────────────
    st.subheader("已入库文献")
    if st.button("刷新列表", key="doc_refresh"):
        st.session_state.pop("doc_list", None)

    if "doc_list" not in st.session_state:
        try:
            resp = requests.get(f"{api_base}/documents", timeout=10)
            st.session_state["doc_list"] = resp.json() if resp.ok else []
        except Exception as e:
            st.error(f"获取文献列表失败：{e}")
            st.session_state["doc_list"] = []

    docs = st.session_state.get("doc_list", [])
    if docs:
        st.dataframe(
            [
                {
                    "ID": d["id"],
                    "标题": d["title"],
                    "作者": d["authors"],
                    "年份": d["year"],
                    "关键词": d["keywords"],
                }
                for d in docs
            ],
            use_container_width=True,
        )
    else:
        st.info("暂无已入库文献，请先执行入库操作。")

    st.divider()

    # ── 入库操作 ──────────────────────────────────────────────────────────────
    st.subheader("入库操作")
    tab_file, tab_dir = st.tabs(["单文件入库", "目录入库"])

    with tab_file:
        file_path = st.text_input(
            "文件路径（PDF / DOCX / TXT）",
            placeholder="C:/Users/.../paper.pdf",
            key="ingest_file_path",
        )
        enable_entity = st.checkbox("启用实体抽取（耗时较长）", key="ingest_file_entity")
        if st.button("开始入库", key="ingest_file_btn"):
            if not file_path.strip():
                st.warning("请输入文件路径")
            else:
                with st.spinner("入库中，请稍候…"):
                    try:
                        resp = requests.post(
                            f"{api_base}/documents/ingest-file",
                            json={"file_path": file_path.strip(), "enable_entity_extraction": enable_entity},
                            timeout=300,
                        )
                        if resp.ok:
                            r = resp.json()
                            st.success(
                                f"入库成功：{r['title']}  |  chunk 数：{r['chunk_count']}  |  实体数：{r['entity_count']}"
                            )
                            st.session_state.pop("doc_list", None)
                        else:
                            st.error(f"入库失败：{resp.json().get('detail', resp.text)}")
                    except Exception as e:
                        st.error(f"请求失败：{e}")

    with tab_dir:
        dir_path = st.text_input(
            "目录路径",
            placeholder="C:/Users/.../raw_data",
            key="ingest_dir_path",
        )
        enable_entity_dir = st.checkbox("启用实体抽取（耗时较长）", key="ingest_dir_entity")
        if st.button("开始批量入库", key="ingest_dir_btn"):
            if not dir_path.strip():
                st.warning("请输入目录路径")
            else:
                with st.spinner("批量入库中，请稍候…"):
                    try:
                        resp = requests.post(
                            f"{api_base}/documents/ingest-directory",
                            json={"directory": dir_path.strip(), "enable_entity_extraction": enable_entity_dir},
                            timeout=600,
                        )
                        if resp.ok:
                            results = resp.json()
                            st.success(f"批量入库完成，共处理 {len(results)} 个文件")
                            for r in results:
                                st.write(f"- {r['title']}（chunk: {r['chunk_count']}）")
                            st.session_state.pop("doc_list", None)
                        else:
                            st.error(f"入库失败：{resp.json().get('detail', resp.text)}")
                    except Exception as e:
                        st.error(f"请求失败：{e}")
