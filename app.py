"""
Streamlit 应用入口（Phase 8-7/8-8 兼容版）。

启动方式：
  F:/Anaconda/envs/llm_universe/python.exe -m streamlit run c:/Users/Administrator/Desktop/GraphAssitant/app.py

前提：FastAPI 服务已在 http://127.0.0.1:8000 运行
  F:/Anaconda/envs/llm_universe/python.exe c:/Users/Administrator/Desktop/GraphAssitant/main.py

导航结构：
  🤖 智能对话  ← Phase 8-7 MasterAgent 统一对话页（新增，置顶）
  ─────────────────────────────────── 兼容层（保留旧页面）
  文献管理
  多轮问答    ← 旧 QAAgent（保留，Phase 8 兼容层）
  深度研究    ← 旧 DeepResearchAgent（保留，Phase 8 兼容层）
  社区检测
  知识图谱
"""
import streamlit as st

from ui import page_agent, page_chat, page_community, page_documents, page_graph, page_research

st.set_page_config(
    page_title="学术文献知识库助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 侧边栏导航 ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 GraphAssistant")
    st.caption("学术文献知识库助手")
    st.divider()

    page = st.radio(
        "导航",
        ["🤖 智能对话", "文献管理", "多轮问答", "深度研究", "社区检测", "知识图谱"],
        key="nav_page",
    )

    st.divider()
    api_base = st.text_input(
        "API 地址",
        value="http://127.0.0.1:8000",
        key="api_base",
    )

    # 健康检查指示灯（顺带展示 API 版本）
    import requests
    try:
        r = requests.get(f"{api_base}/health", timeout=3)
        if r.ok:
            health = r.json()
            st.success(f"API 已连接  v{health.get('version', '')}")
        else:
            st.warning("API 响应异常")
    except Exception:
        st.error("API 未连接，请先启动 main.py")

# ── 页面路由 ──────────────────────────────────────────────────────────────────
if page == "🤖 智能对话":
    page_agent.render(api_base)           # Phase 8-7：MasterAgent 统一对话页
elif page == "文献管理":
    page_documents.render(api_base)
elif page == "多轮问答":
    page_chat.render(api_base)            # 旧 QAAgent，Phase 8 兼容层
elif page == "深度研究":
    page_research.render(api_base)        # 旧 DeepResearchAgent，Phase 8 兼容层
elif page == "社区检测":
    page_community.render(api_base)
elif page == "知识图谱":
    page_graph.render(api_base)
