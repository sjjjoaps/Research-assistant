# API 层（api）

目录：`api/`

## 模块清单

| 文件 | 主要内容 | 职责 |
|---|---|---|
| `routers/agent.py` | `POST /agent/chat`, `GET/DELETE /agent/sessions/*` | MasterAgent SSE 流式接口 |
| `routers/documents.py` | `GET/POST/DELETE /documents/*` | 文献管理（上传、状态、删除、引用） |
| `routers/community.py` | `POST /community/detect`, `GET /community/list` | 社区检测触发与结果查询 |
| `routers/graph.py` | `GET /graph/stats`, `GET /graph/subgraph` | 图谱统计与子图查询 |
| `routers/chat.py` | `POST /chat` | 传统问答接口（兼容层） |
| `routers/research.py` | `POST /research`, `POST /idea` | 深度研究与 Idea 生成（兼容层） |
| `schemas.py` | Pydantic 模型 | 请求/响应数据结构定义 |

## 主要接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/agent/chat` | MasterAgent SSE 流式对话 |
| GET | `/agent/sessions` | 会话列表 |
| DELETE | `/agent/sessions/{id}` | 删除会话 |
| GET | `/agent/sessions/{id}/history` | 会话历史 |
| GET | `/documents` | 文献列表 |
| POST | `/documents/ingest-file` | 同步入库 |
| POST | `/documents/ingest-file/start` | 异步入库（后台） |
| DELETE | `/documents/{doc_id}` | 精确删除文档 |
| GET | `/documents/{doc_id}/status` | 文档详细状态 |
| GET | `/documents/{doc_id}/citations` | 文档引用列表 |
| POST | `/community/detect` | 触发 Louvain 社区检测 |
| GET | `/community/list` | 社区摘要列表 |
| GET | `/graph/stats` | 图谱节点/关系统计 |
| GET | `/graph/subgraph` | 子图查询（支持节点类型过滤、关键词搜索） |
| GET | `/health` | 健康检查 |

## SSE 事件协议

`POST /agent/chat` 返回 W3C SSE 格式事件流：

```
session_start → thinking → [tool_start → tool_end]* → text_delta* → sources → usage → done
```

错误通过 `error` 事件推送，不返回 HTTP 500。

## 安全约束

- `session_id` 格式校验：`^[A-Za-z0-9_-]{1,64}$`
- CORS 允许 `localhost:5173`（React 开发服务器）
