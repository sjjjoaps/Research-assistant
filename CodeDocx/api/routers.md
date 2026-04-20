# routers.md

## 所在层次
API 层 `api/routers/`

## 主体功能
FastAPI 路由层，将 HTTP 请求路由到对应的业务逻辑。

## 路由文件清单

| 文件 | 前缀 | 主要接口 |
|---|---|---|
| `documents.py` | `/documents` | 文献管理（列表/状态/引用/入库/删除） |
| `agent.py` | `/agent` | MasterAgent SSE 流式对话、会话管理 |
| `graph.py` | `/graph` | 图谱统计、子图查询 |
| `chat.py` | `/chat` | 传统 QA 接口（非流式） |
| `research.py` | `/research` | 深度研究接口 |
| `community.py` | `/community` | 社区检测触发接口 |

## 关键接口

**documents.py**
- `GET /documents` — 列出所有已入库文献（含处理状态）
- `GET /documents/{doc_id}/status` — 查询单文档处理状态（前端轮询用）
- `POST /documents/ingest-file/start` — 异步启动入库，返回 `doc_id` 供轮询
- `DELETE /documents/{doc_id}` — 精确删除文档

**agent.py**
- `POST /agent/chat` — SSE 流式对话（`text/event-stream`）
- `GET /agent/sessions` — 列出所有会话（返回 `SessionMeta[]`）
- `GET /agent/sessions/{session_id}/history` — 获取会话历史

## 调用关系
- **被调用方**：`main.py`（`app.include_router()`）
- **依赖方**：`api/schemas.py`（请求/响应模型）、`src/workflows/`、`src/agents/`、`src/storage/`

## 注意事项
- `session_id` 路径穿越防御：正则 `^[A-Za-z0-9_-]{1,64}$`，非法值返回 HTTP 400
- SSE 协议：`POST + text/event-stream`（非浏览器原生 GET EventSource）
- `GET /agent/sessions` 返回 `Session[]` 纯数组（不是 `{sessions: [...]}`）
- `DocumentStatusResponse` 是部分字段，前端需用 `upsert` 合并而非替换完整文档对象
