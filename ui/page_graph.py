"""
知识图谱可视化页

展示图谱统计、子图筛选和节点详情。
"""
from __future__ import annotations

import json

import requests
import streamlit as st
import streamlit.components.v1 as components


_NODE_COLORS = {
    "Document": "#2F80ED",
    "Chunk": "#56CCF2",
    "Entity": "#27AE60",
    "Community": "#F2994A",
    "Reference": "#9B51E0",
}


def _render_pyvis(nodes: list[dict], edges: list[dict]) -> str:
    from pyvis.network import Network

    net = Network(height="640px", width="100%", bgcolor="#0f172a", font_color="#e5e7eb")
    net.barnes_hut(gravity=-24000, central_gravity=0.25, spring_length=150)

    for node in nodes:
        node_type = node.get("type", "Node")
        props = node.get("properties", {})
        title = "<br>".join(
            f"<b>{k}</b>: {v}"
            for k, v in props.items()
            if v is not None and len(str(v)) <= 300
        )
        net.add_node(
            node["id"],
            label=node.get("label", node["id"])[:40],
            title=title or node.get("label", ""),
            color=_NODE_COLORS.get(node_type, "#BDBDBD"),
            group=node_type,
        )

    for edge in edges:
        if edge.get("source") and edge.get("target"):
            net.add_edge(
                edge["source"],
                edge["target"],
                label=edge.get("label", ""),
                title=edge.get("type", ""),
                color="#94a3b8",
            )

    net.set_options(
        """
        {
          "nodes": {"shape": "dot", "size": 18, "font": {"size": 14}},
          "edges": {"font": {"size": 10, "align": "middle"}, "smooth": true},
          "physics": {"stabilization": true},
          "interaction": {"hover": true, "navigationButtons": true}
        }
        """
    )
    return net.generate_html(notebook=False)


def _render_fallback(nodes: list[dict], edges: list[dict]) -> None:
    st.info("未安装 pyvis，已切换为表格展示。可运行 pip install pyvis>=0.3.2 启用交互图。")
    st.dataframe(nodes, use_container_width=True)
    st.dataframe(edges, use_container_width=True)


def render(api_base: str) -> None:
    st.header("知识图谱")

    with st.sidebar:
        st.subheader("图谱筛选")
        limit = st.slider("最大节点数", 10, 500, 120, step=10, key="graph_limit")
        node_types = st.multiselect(
            "节点类型",
            ["Document", "Chunk", "Entity", "Community", "Reference"],
            default=["Document", "Entity", "Community", "Reference"],
            key="graph_node_types",
        )
        search = st.text_input("搜索节点", value="", key="graph_search")
        refresh = st.button("刷新图谱", key="graph_refresh")

    try:
        stats_resp = requests.get(f"{api_base}/graph/stats", timeout=30)
        stats_resp.raise_for_status()
        stats = stats_resp.json()
    except Exception as exc:
        st.error(f"读取图谱统计失败：{exc}")
        return

    col1, col2 = st.columns(2)
    col1.metric("节点数", stats.get("node_count", 0))
    col2.metric("关系数", stats.get("relationship_count", 0))

    with st.expander("统计详情", expanded=False):
        st.write("节点类型")
        st.dataframe(stats.get("node_labels", []), use_container_width=True)
        st.write("关系类型")
        st.dataframe(stats.get("relationship_types", []), use_container_width=True)

    params: list[tuple[str, str | int]] = [("limit", limit), ("search", search)]
    for node_type in node_types:
        params.append(("node_types", node_type))

    try:
        subgraph_resp = requests.get(f"{api_base}/graph/subgraph", params=params, timeout=60)
        subgraph_resp.raise_for_status()
        subgraph = subgraph_resp.json()
    except Exception as exc:
        st.error(f"读取子图失败：{exc}")
        return

    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])
    st.caption(f"当前子图：{len(nodes)} 个节点，{len(edges)} 条关系")

    if not nodes:
        st.warning("当前筛选条件下没有图谱节点。")
        return

    try:
        html = _render_pyvis(nodes, edges)
        components.html(html, height=680, scrolling=True)
    except Exception:
        _render_fallback(nodes, edges)

    st.subheader("节点详情")
    node_options = {
        f"{node.get('label')} ({node.get('type')})": node
        for node in nodes
    }
    selected_label = st.selectbox("选择节点查看属性", list(node_options.keys()))
    selected_node = node_options[selected_label]
    st.json(json.dumps(selected_node.get("properties", {}), ensure_ascii=False, indent=2))
