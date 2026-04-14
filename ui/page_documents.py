"""
文献管理页
- 展示已入库文献列表（含处理状态）
- 支持单文件 / 目录入库
- 支持查看单个文档的详细状态
- 单文件入库支持实时追踪阶段状态
"""
import time

import streamlit as st
import requests

# 状态 → 显示标签映射
_STATUS_LABEL = {
    "pending":    "⏳ 等待",
    "parsing":    "📄 解析中",
    "chunking":   "✂️ 切块中",
    "metadata":   "🔍 提取元数据",
    "indexing":   "💾 写入索引",
    "graph":      "🕸️ 写入图谱",
    "extracting": "🧠 实体抽取",
    "citations":  "📚 引用抽取",
    "deleting":   "🗑️ 删除中",
    "delete_failed": "⚠️ 删除失败",
    "processed":  "✅ 完成",
    "failed":     "❌ 失败",
}

_STATUS_FLOW = ["pending", "parsing", "chunking", "metadata", "indexing", "graph", "citations", "extracting", "processed"]
_TRACKABLE_FLOW = _STATUS_FLOW[:-1]


def _format_status(status: str | None) -> str:
    return _STATUS_LABEL.get(status or "", status or "—")


def _status_progress(status: str | None) -> int:
    if status == "processed":
        return 100
    if status in {"failed", "delete_failed"}:
        return 100
    if status not in _STATUS_FLOW:
        return 0
    index = _STATUS_FLOW.index(status)
    return max(5, int(index / (len(_STATUS_FLOW) - 1) * 100))


def _inject_tracker_styles() -> None:
    st.markdown(
        """
        <style>
        .ingest-tracker-wrap {
            border: 1px solid rgba(33, 37, 41, 0.14);
            border-radius: 18px;
            padding: 18px 18px 10px 18px;
            background:
                radial-gradient(circle at top right, rgba(255, 196, 0, 0.18), transparent 28%),
                linear-gradient(135deg, #fffdf6 0%, #f7fbff 100%);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
            margin-bottom: 14px;
        }
        .ingest-tracker-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 4px;
        }
        .ingest-tracker-subtitle {
            font-size: 0.88rem;
            color: #475569;
            margin-bottom: 12px;
            word-break: break-all;
        }
        .ingest-current-pill {
            display: inline-block;
            padding: 8px 12px;
            border-radius: 999px;
            background: #0f172a;
            color: white;
            font-weight: 700;
            margin-bottom: 12px;
            animation: ingestPulse 1.6s ease-in-out infinite;
        }
        .ingest-stage-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 10px 0 8px 0;
        }
        .ingest-stage-card {
            border-radius: 14px;
            padding: 12px 10px;
            border: 1px solid #dbe4ea;
            background: #ffffff;
            min-height: 84px;
        }
        .ingest-stage-card.done {
            background: #ecfdf5;
            border-color: #86efac;
        }
        .ingest-stage-card.current {
            background: #fff7ed;
            border-color: #fb923c;
            box-shadow: 0 0 0 2px rgba(251, 146, 60, 0.18);
            transform: translateY(-2px);
        }
        .ingest-stage-card.pending {
            background: #f8fafc;
            border-color: #e2e8f0;
        }
        .ingest-stage-index {
            font-size: 0.78rem;
            font-weight: 700;
            color: #64748b;
            margin-bottom: 6px;
        }
        .ingest-stage-name {
            font-size: 0.95rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 4px;
        }
        .ingest-stage-state {
            font-size: 0.82rem;
            color: #475569;
        }
        @keyframes ingestPulse {
            0% { box-shadow: 0 0 0 0 rgba(15, 23, 42, 0.22); }
            70% { box-shadow: 0 0 0 12px rgba(15, 23, 42, 0); }
            100% { box-shadow: 0 0 0 0 rgba(15, 23, 42, 0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _build_stage_cards(status: str | None) -> str:
    current_index = _TRACKABLE_FLOW.index(status) if status in _TRACKABLE_FLOW else -1
    cards = []
    for idx, step in enumerate(_TRACKABLE_FLOW, start=1):
        if status == "processed" or idx - 1 < current_index:
            state_class = "done"
            state_text = "已完成"
        elif step == status:
            state_class = "current"
            state_text = "进行中"
        else:
            state_class = "pending"
            state_text = "等待中"
        cards.append(
            f"""
            <div class="ingest-stage-card {state_class}">
                <div class="ingest-stage-index">STEP {idx}</div>
                <div class="ingest-stage-name">{_STATUS_LABEL.get(step, step)}</div>
                <div class="ingest-stage-state">{state_text}</div>
            </div>
            """
        )
    return "".join(cards)


def _render_ingest_tracker(container, status_payload: dict, file_path: str, doc_id: str) -> None:
    progress = _status_progress(status_payload.get("status"))
    current_step = status_payload.get("current_step") or "—"
    with container.container():
        _inject_tracker_styles()
        st.markdown(
            f"""
            <div class="ingest-tracker-wrap">
                <div class="ingest-tracker-title">单文献入库实时追踪</div>
                <div class="ingest-tracker-subtitle">{file_path}</div>
                <div class="ingest-current-pill">
                    {_format_status(status_payload.get("status"))} | 当前步骤：{current_step}
                </div>
                <div class="ingest-stage-grid">
                    {_build_stage_cards(status_payload.get("status"))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(progress, text=f"{_format_status(status_payload.get('status'))} | 当前步骤：{current_step}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Chunk数", status_payload.get("chunk_count", 0))
        col2.metric("实体数", status_payload.get("entity_count", 0))
        col3.metric("关系数", status_payload.get("relation_count", 0))
        st.caption(f"doc_id: {doc_id}")
        if status_payload.get("error_message"):
            if status_payload.get("status") in {"failed", "delete_failed"}:
                st.error(f"最近错误：{status_payload['error_message']}")
            else:
                st.warning(f"最近错误：{status_payload['error_message']}")


def _track_single_ingest(api_base: str, doc_id: str, file_path: str, timeout_seconds: int = 600) -> dict | None:
    tracker = st.empty()
    started_at = time.time()
    last_status = None
    while time.time() - started_at < timeout_seconds:
        try:
            resp = requests.get(f"{api_base}/documents/{doc_id}/status", timeout=10)
        except Exception as e:
            tracker.warning(f"状态轮询失败：{e}")
            time.sleep(1)
            continue

        if resp.status_code == 404:
            tracker.info("入库任务已启动，等待状态记录刷新…")
            time.sleep(1)
            continue
        if not resp.ok:
            tracker.error(f"状态查询失败：{resp.text}")
            return None

        payload = resp.json()
        last_status = payload
        _render_ingest_tracker(tracker, payload, file_path=file_path, doc_id=doc_id)

        if payload.get("status") == "processed":
            tracker.success("文献入库完成。")
            return payload
        if payload.get("status") in {"failed", "delete_failed"}:
            tracker.error(f"文献入库失败，停留在步骤：{payload.get('current_step') or '—'}")
            return payload

        time.sleep(1)

    tracker.warning("等待入库状态超时，可稍后点击“刷新列表”或使用详细状态查询继续查看。")
    return last_status


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
        failed_docs = [d for d in docs if d.get("status") in {"failed", "delete_failed"}]
        running_docs = [d for d in docs if d.get("status") not in {None, "processed", "failed", "delete_failed"}]

        if running_docs:
            st.info(f"当前有 {len(running_docs)} 篇文档处于处理中状态，可点击“刷新列表”查看最新进度。")
        if failed_docs:
            st.warning(f"当前有 {len(failed_docs)} 篇文档存在最近错误，请关注“最近错误”列或展开查看详情。")

        st.dataframe(
            [
                {
                    "ID": d["id"],
                    "标题": d["title"],
                    "作者": d["authors"],
                    "年份": d["year"],
                    "关键词": d["keywords"],
                    "状态": _format_status(d.get("status")),
                    "当前步骤": d.get("current_step") or "—",
                    "Chunk数": d.get("chunk_count", 0),
                    "实体数": d.get("entity_count", 0),
                    "关系数": d.get("relation_count", 0),
                    "最近错误": d.get("error_message") or "—",
                    "更新时间": d.get("updated_at") or "—",
                    "doc_id": d.get("doc_id") or "—",
                }
                for d in docs
            ],
            use_container_width=True,
        )

        # ── 查看单个文档详细状态 ──────────────────────────────────────────────
        with st.expander("查看文档详细状态"):
            doc_ids = [d.get("doc_id") for d in docs if d.get("doc_id")]
            if doc_ids:
                selected = st.selectbox("选择 doc_id", doc_ids, key="status_doc_id")
                if st.button("查询状态", key="status_query_btn"):
                    try:
                        resp = requests.get(f"{api_base}/documents/{selected}/status", timeout=10)
                        if resp.ok:
                            s = resp.json()
                            st.write(f"状态：{_format_status(s.get('status'))}")
                            st.write(f"当前步骤：{s.get('current_step') or '—'}")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Chunk数", s.get("chunk_count", 0))
                            col2.metric("实体数", s.get("entity_count", 0))
                            col3.metric("关系数", s.get("relation_count", 0))
                            st.caption(f"更新时间：{s.get('updated_at') or '—'}")

                            if s.get("status") in {"failed", "delete_failed"}:
                                st.error(
                                    f"最近错误：{s.get('error_message') or '未提供错误信息'}"
                                )
                            elif s.get("error_message"):
                                st.warning(f"最近错误：{s['error_message']}")

                            with st.expander("查看原始状态 JSON"):
                                st.json(s)
                        else:
                            st.error(f"查询失败：{resp.text}")
                    except Exception as e:
                        st.error(f"请求失败：{e}")
            else:
                st.info("暂无带 doc_id 的文档记录。")
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
                try:
                    resp = requests.post(
                        f"{api_base}/documents/ingest-file/start",
                        json={"file_path": file_path.strip(), "enable_entity_extraction": enable_entity},
                        timeout=30,
                    )
                    if not resp.ok:
                        st.error(f"入库启动失败：{resp.json().get('detail', resp.text)}")
                    else:
                        start_payload = resp.json()
                        if start_payload.get("accepted"):
                            st.info("后台入库任务已启动，下面实时显示处理阶段。")
                        else:
                            st.info(start_payload.get("message", "当前文档已有可复用状态，直接展示最新进度。"))

                        final_status = _track_single_ingest(
                            api_base=api_base,
                            doc_id=start_payload["doc_id"],
                            file_path=start_payload["file_path"],
                        )
                        st.session_state.pop("doc_list", None)

                        if final_status and final_status.get("status") == "processed":
                            refreshed = requests.get(f"{api_base}/documents", timeout=10)
                            docs_after = refreshed.json() if refreshed.ok else []
                            matched = next((d for d in docs_after if d.get("doc_id") == start_payload["doc_id"]), None)
                            if matched:
                                st.success(
                                    f"入库成功：{matched['title']}  |  chunk 数：{matched.get('chunk_count', 0)}  |  实体数：{matched.get('entity_count', 0)}"
                                )
                            else:
                                st.success("入库成功。")
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
                                st.write(f"- {r['title']}（chunk: {r['chunk_count']}，doc_id: {r.get('doc_id', '—')}）")
                            st.session_state.pop("doc_list", None)
                        else:
                            st.error(f"入库失败：{resp.json().get('detail', resp.text)}")
                    except Exception as e:
                        st.error(f"请求失败：{e}")
